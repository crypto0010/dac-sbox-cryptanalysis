"""Second-revision studies: generator ablation, affine-class statistics, annealing parameters.

Writes ``output/results_studies.json`` (schema :data:`RESULTS_SCHEMA`) and the
archive tables of ``output/tables``; ``verify_latex.py`` merges this file with
the two others.

STEP 1  Metric identities of every analysed S-box (``W = 2 LAT``,
        ``C_max = 2 LAP``, ``NL(S) = 2^(n-1) - 2^n LAP``, ``delta = 2^n DP``).
        The runner stops before writing anything if one fails.
STEP 2  Lookup table, DDT, LAT and BCT of every analysed S-box as CSV files in
        ``output/tables`` (the raw tables behind every metric of the paper).
STEP 3  Generator ablation of the affine-equivalence search: chaotic
        tent-logistic generator vs NumPy PCG64 vs a deterministic Weyl
        sequence (and lexicographic enumeration), same budget.
STEP 4  Invariant vs representative-dependent criteria over random
        representatives of the inversion and the Gold class.
STEP 5  Swap-move annealing as a function of the move budget N (including
        N = 1) and of the initial temperature T0 (including T0 = 0).
STEP 6  Exact count of the affine representatives with an ideal SAC matrix
        (admissible input bases, their orbits, the exact probability).
STEP 7  The executable specification of Algorithm 2 (src.algorithm_spec),
        run from seed 0.5, must reproduce both published tables and every
        recorded search statistic; plus the worked evaluation of S(6).

Deterministic: two runs give byte-identical files apart from ``timings``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import DEFAULT_OUTPUT_DIR, section_header, timed  # noqa: E402
from src.comparative_analysis import (all_sbox_tables, analyze_all,  # noqa: E402
                                      require_metric_identities, write_results)
from src.sbox import DAC_SBOX  # noqa: E402
from src.sbox_tables import TABLES  # noqa: E402
from src import algorithm_spec  # noqa: E402
from src.search_studies import (EXPONENTS, affine_class_statistics,  # noqa: E402
                                annealing_parameter_study, generator_ablation,
                                ideal_sac_counting, locate_basis)
from src.sbox_optimizer import optimize_dac_sbox  # noqa: E402
from src.security_analysis import (check_metric_identities, compute_bct,  # noqa: E402
                                   compute_ddt, compute_lat)

RESULTS_FILENAME = "results_studies.json"
RESULTS_SCHEMA = "dac-sbox-results-studies/1"
TABLES_DIRNAME = "tables"


def run_identities(tables: Dict[str, List[int]]) -> Dict[str, Any]:
    """STEP 1: the normalisation identities, re-derived for every table."""
    section_header("STEP 1: METRIC IDENTITIES (W = 2 LAT, C = 2 LAP, "
                   "NL = 16 - 32 LAP, delta = 32 DP)")
    require_metric_identities(analyze_all(tables))
    report = {name: check_metric_identities(table) for name, table in tables.items()}
    for name, entry in report.items():
        v = entry["values"]
        print(f"  {name:14s} LAP {v['lap']:.4f}  C_max {v['correlation_max']:.4f}  "
              f"NL {v['nl_sbox']} = 16 - 32 LAP = {v['nl_from_lap']:g}  "
              f"delta {v['differential_uniformity']} = 32 DP = {v['du_from_dp']:g}  "
              f"-> {'all hold' if entry['all_hold'] else 'VIOLATED'}")
    return report


def _write_matrix(path: str, matrix: Sequence[Sequence[int]], label: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([label] + list(range(len(matrix[0]))))
        for index, row in enumerate(matrix):
            writer.writerow([index] + [int(v) for v in row])


def export_tables(tables: Dict[str, List[int]], output_dir: str) -> List[str]:
    """STEP 2: lookup table, DDT, LAT and BCT of every S-box as CSV."""
    section_header("STEP 2: ARCHIVE TABLES (lookup table, DDT, LAT, BCT as CSV)")
    directory = os.path.join(output_dir, TABLES_DIRNAME)
    os.makedirs(directory, exist_ok=True)
    written: List[str] = []
    for name, table in tables.items():
        stem = name.replace(" ", "_")
        lookup = os.path.join(directory, f"{stem}_sbox.csv")
        with open(lookup, "w", encoding="utf-8", newline="\n") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["x", "S(x)"])
            writer.writerows([x, int(y)] for x, y in enumerate(table))
        written.append(lookup)
        for kind, matrix, label in (
                ("ddt", compute_ddt(table)["table"], "dx\\dy"),
                ("lat", compute_lat(table)["table"], "a\\b"),
                ("bct", compute_bct(table)["table"], "dx\\dy")):
            path = os.path.join(directory, f"{stem}_{kind}.csv")
            _write_matrix(path, matrix.tolist(), label)
            written.append(path)
    print(f"  {len(written)} files in {directory}")
    return [os.path.relpath(p, output_dir).replace(os.sep, "/") for p in written]


def run_ablation(runs: int) -> Dict[str, Any]:
    """STEP 3: the affine search under three decision sources."""
    section_header("STEP 3: GENERATOR ABLATION OF THE AFFINE SEARCH (same budget)")
    report = generator_ablation(runs)
    for cls, entry in report["classes"].items():
        print(f"  class {cls} (budget {entry['budget']} candidates; class optimum key "
              f"{entry['class_optimum_key']}; lexicographic order reaches an ideal SAC "
              f"at candidate {entry['lexicographic']['candidates_to_perfect_sac']}):")
        for kind, gen in entry["generators"].items():
            s = gen["summary"]
            reach = s["candidates_to_perfect_sac"]
            reach_text = "never" if reach is None else f"median {reach['median']:g}"
            print(f"    {kind:8s} ideal SAC in {s['runs_reaching_perfect_sac']}/{runs} runs; "
                  f"D_mean median {s['sac_d_mean']['median']:.4f}; "
                  f"BIC-SAC D_max median {s['bic_sac_d_max']['median']:.4f}; "
                  f"FP max {s['fixed_points']['max']:g}, OFP max "
                  f"{s['opposite_fixed_points']['max']:g}; candidates to ideal SAC "
                  f"{reach_text}")
    return report


def run_class_statistics(samples: int) -> Dict[str, Any]:
    """STEP 4: what the affine class fixes and what the representative decides."""
    section_header("STEP 4: INVARIANT vs REPRESENTATIVE-DEPENDENT CRITERIA")
    report = affine_class_statistics(
        samples, reference={"inverse": TABLES["DAC-OPT"].table,
                            "gold": TABLES["DAC-OPT-GOLD"].table})
    for cls, entry in report["classes"].items():
        print(f"  class {cls}: invariant {sorted(entry['invariant'])}")
        for key, value in entry["variable"].items():
            zero = (f", at zero {value['share_at_zero']:.3f}"
                    if "share_at_zero" in value else "")
            print(f"    {key:22s} range [{value['min']:g}, {value['max']:g}] "
                  f"median {value['median']:g}{zero}")
    return report


#: ``--quick``: a smoke-test scale of STEP 5 (the paper uses the defaults).
QUICK_SA = {"budgets": (1, 100, 1000), "seeds": (0.25, 0.5), "study_budget": 1000}


def run_annealing_study(quick: bool = False) -> Dict[str, Any]:
    """STEP 5: annealing budget N and initial temperature T0."""
    section_header("STEP 5: ANNEALING BUDGET N AND INITIAL TEMPERATURE T0")
    report = (annealing_parameter_study(DAC_SBOX, **QUICK_SA) if quick
              else annealing_parameter_study(DAC_SBOX))
    for row in report["budget_sweep"]:
        print(f"  N = {row['iterations']:>6}: best energy median "
              f"{row['best_energy']['median']:>9.1f} [{row['best_energy']['min']:g}, "
              f"{row['best_energy']['max']:g}], delta {row['differential_uniformity']}")
    for row in report["temperature_sweep"]:
        print(f"  T0 = {row['factor']!s:>10} x T0_cal: best energy median "
              f"{row['best_energy']['median']:>9.1f}, accepted median "
              f"{row['accepted_moves']['median']:g}, delta {row['differential_uniformity']}")
    print(f"  median within 1% of the best median from N = "
          f"{report['plateau_iterations_within_1pct']}")
    return report


#: Published tables, their class and the search budget that produced them.
SPEC_RUNS = (("DAC-OPT", "inverse", "affine_inverse", 4096),
             ("DAC-OPT-GOLD", "gold", "affine_gold", 512))
#: Input of the worked example printed in the paper.
WORKED_EXAMPLE_INPUT = 6


def run_sac_counting() -> Dict[str, Any]:
    """STEP 6: exact count of ideal-SAC representatives per class."""
    section_header("STEP 6: EXACT COUNT OF IDEAL-SAC AFFINE REPRESENTATIVES")
    report: Dict[str, Any] = {}
    for name, cls, strategy, budget in SPEC_RUNS:
        entry = ideal_sac_counting(EXPONENTS[cls])
        b_rows = optimize_dac_sbox(strategy, 0.5, budget)["provenance"]["affine"]["B_rows"]
        entry["published_table"] = {"name": name, **locate_basis(entry, b_rows)}
        report[cls] = entry
        print(f"  class {cls}: {entry['admissible_bases']} of {entry['input_bases']} input "
              f"bases admissible; |Z| {entry['zero_set_sizes']}; P(ideal SAC) = "
              f"{entry['probability_ideal_sac']:.6g} (one in {entry['one_in']:.6g}); "
              f"orbits {[o['size'] for o in entry['orbits']]}")
    return report


def run_specification() -> Dict[str, Any]:
    """STEP 7: the printed pseudocode, executed, against the optimizer."""
    section_header("STEP 7: EXECUTABLE SPECIFICATION OF ALGORITHM 2 (independent code)")
    report: Dict[str, Any] = {"seed": 0.5, "runs": {}}
    for name, cls, strategy, budget in SPEC_RUNS:
        spec = algorithm_spec.affine_select(EXPONENTS[cls], 0.5, budget)
        opt = optimize_dac_sbox(strategy, 0.5, budget)
        prov, search = opt["provenance"], opt["provenance"]["search"]
        checks = {
            "table": spec["sbox"] == list(TABLES[name].table) == list(opt["sbox"]),
            "A_rows": spec["A_rows"] == list(prov["affine"]["A_rows"]),
            "B_rows": spec["B_rows"] == list(prov["affine"]["B_rows"]),
            "constants": (spec["a"], spec["b"]) == (prov["affine"]["a"], prov["affine"]["b"]),
            "best_candidate_index": spec["best_candidate_index"] == search["best_candidate_index"],
            "ideal_sac_candidates": spec["ideal_sac_candidates"] == search["perfect_sac_candidates"],
            "refine_accepted": spec["refine_accepted"] == search["refine_accepted"],
            "draws": spec["draws"] == prov["rng"]["draws"],
        }
        report["runs"][name] = {"candidates": budget, **{k: v for k, v in spec.items()
                                                          if k != "sbox"},
                                "checks": checks, "reproduces": all(checks.values())}
        print(f"  {name}: specification reproduces the optimizer: {all(checks.values())} "
              f"(kept candidate {spec['best_candidate_index']}, {spec['draws']} draws)")
        if not all(checks.values()):
            raise RuntimeError(f"Algorithm 2 specification disagrees for {name}: {checks}")
    affine = report["runs"]["DAC-OPT"]
    report["worked_example"] = algorithm_spec.worked_example(
        EXPONENTS["inverse"], affine["A_rows"], affine["a"], affine["B_rows"], affine["b"],
        WORKED_EXAMPLE_INPUT)
    if report["worked_example"]["S(x)"] != TABLES["DAC-OPT"].table[WORKED_EXAMPLE_INPUT]:
        raise RuntimeError("worked example disagrees with the DAC-OPT table")
    gen = algorithm_spec.Gen(0.5)
    report["generator_first_variates"] = [gen.uniform() for _ in range(3)]
    print(f"  worked example: {report['worked_example']}")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the runner."""
    parser = argparse.ArgumentParser(
        description="Second-revision studies (ablation, class statistics, annealing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--runs", type=int, default=32, help="runs per generator in STEP 3")
    parser.add_argument("--samples", type=int, default=1000,
                        help="representatives per affine class in STEP 4")
    parser.add_argument("--quick", action="store_true",
                        help="smoke-test scale for STEP 5 (not the paper's numbers)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="directory for results_studies.json and tables/")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the studies; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.runs < 1 or args.samples < 1:
        parser.error("--runs and --samples must be >= 1")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    start = time.perf_counter()
    timings: Dict[str, float] = {}
    tables = all_sbox_tables()
    identities = timed(timings, "identities", run_identities, tables)
    exported = timed(timings, "tables", export_tables, tables, output_dir)
    ablation = timed(timings, "ablation", run_ablation, args.runs)
    classes = timed(timings, "class_statistics", run_class_statistics, args.samples)
    annealing = timed(timings, "annealing", run_annealing_study, args.quick)
    counting = timed(timings, "sac_counting", run_sac_counting)
    specification = timed(timings, "specification", run_specification)
    results = {
        "schema": RESULTS_SCHEMA,
        "runner": "run_studies.py",
        "arguments": {"runs": args.runs, "samples": args.samples, "quick": args.quick},
        "metric_identities": identities,
        "archive_tables": exported,
        "generator_ablation": ablation,
        "affine_class_statistics": classes,
        "annealing_study": annealing,
        "ideal_sac_counting": counting,
        "algorithm_specification": specification,
        "timings": {"sections_s": timings, "total_s": time.perf_counter() - start},
    }
    path = write_results(os.path.join(output_dir, RESULTS_FILENAME), results)
    print(f"\n  studies finished in {time.perf_counter() - start:.1f} s -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
