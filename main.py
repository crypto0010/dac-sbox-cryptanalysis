"""Main runner for the DAC (Dynamic Airy Chaotic) S-box implementation.

Runs the paper pipeline on the frozen tables of :mod:`src.sbox_tables`:

1. Chaotic map analysis (Lyapunov exponents, orbit entropy)
2. S-box construction and verification (paper Table 3)
3. Boolean function analysis (ANF, algebraic degree, balance)
4. Complete security analysis of the DAC S-box
5. Paper figures (Fig. 2, 4, 5, 6, DDT / LAT heatmaps, S-box mapping)
6. Block cipher demonstration
7. Cipher-level security analysis (round function, branch numbers, minimum
   active S-boxes, single-characteristic trail bounds, bypass-bit trails,
   test vectors, birthday bound)
8. Seeded software benchmarks
9. Key-dependent dynamic S-box demonstration
10. Summary table (DAC family vs computed competitors) and ``results.json``

Every step is deterministic: the only random inputs (benchmark lookups,
plaintexts and keys) are seeded by ``--seed``.  ``results.json`` is
byte-identical between runs with the same arguments except for its
``timings`` entry.  The ``comparison_*`` figures are produced by
``run_optimized.py`` and are not regenerated here.  Works from any working
directory; see ``python main.py --help``.

This runner owns ``output/results.json`` (schema
:data:`RESULTS_SCHEMA`) and is its only writer.  ``run_optimized.py``
writes a *different* file, ``output/results_optimized.json``, so neither
runner can silently destroy the other's sections; ``verify_latex.py``
reads whichever of the two exist and merges them.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.block_cipher import DEMO_PLAINTEXT, demo_cipher  # noqa: E402
from src.boolean_function import verify_boolean_properties  # noqa: E402
from src.chaotic_maps import (canonical_seed, compare_maps,  # noqa: E402
                              default_maps)
from src.cipher_analysis import (DATA_LIMIT_LOG2, analyze_cipher,  # noqa: E402
                                 format_active_sbox_table)
from src.comparative_analysis import (SUMMARY_FOOTNOTE, benchmark_table,  # noqa: E402
                                      collect_benchmarks, comparative_results,
                                      core_results, frozen_table,
                                      partition_benchmarks, summary_table,
                                      write_results)
from src.sbox import DACSBox, generate_dynamic_sbox  # noqa: E402
from src.security_analysis import (compute_ddt, compute_lat, compute_sac,  # noqa: E402
                                   full_security_analysis)
from src.visualization import generate_all_figures  # noqa: E402

#: The original DAC S-box (paper Table 3), taken from the frozen tables.
#: A *copy* of the frozen paper table: never hand out the frozen list itself,
#: or one in-place write anywhere in the pipeline rewrites paper Table 3 for
#: ``TABLES``, ``all_sbox_tables()`` and every results file.
DAC_TABLE: List[int] = frozen_table("DAC")
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
#: Results file of *this* runner.  ``run_optimized.py`` writes its own file
#: under its own schema, so the two runners never overwrite each other.
RESULTS_FILENAME = "results.json"
RESULTS_SCHEMA = "dac-sbox-results-main/1"
#: Key seeds of the dynamic S-box demonstration.  0.7 folds onto
#: 1 - 0.7 = 0.30000000000000004, a key distinct from 0.3: the pair shows
#: that near-mirror seeds are different keys.
DYNAMIC_DEMO_SEEDS = (0.3, 0.5, 0.7, 0.500001)
#: Section 6 test vector (recorded in ``results.json`` under ``cipher_demo``).
#: Re-exported from :mod:`src.block_cipher`, which both prints and returns
#: it, so the console walk-through and the recorded vector share one source.
#: Display titles of the maps returned by :func:`src.chaotic_maps.default_maps`.
MAP_TITLES: Dict[str, str] = {
    "logistic": "Logistic Map (Eq. 1)",
    "tent": "Tent Map",
    "tent_logistic": "Tent-Logistic Hybrid Map (Eq. 3, 4, 5)",
}
#: Attributes printed as the parameter string of a chaotic map, in order.
MAP_PARAMETER_NAMES = ("r_logistic", "r_tent", "r", "seed")
#: Round count up to which Section 7 extends the active-S-box MILP chain
#: beyond the cipher's own 16 rounds, so that the round count at which each
#: trail bound reaches the 2^-32 data limit can be reported (16 rounds do not
#: get there for the DAC or the E-DAC S-box).
CIPHER_EXTENDED_ROUNDS = 32
#: The same cap under ``--quick``.  27 is the smallest value at which every
#: threshold of this study is still attained (DAC needs 27 rounds, the worst
#: case), so ``--quick`` shortens the MILP chain without changing a single
#: reported round count: only ``searched_up_to_rounds`` and the length of the
#: per-round count lists differ between the two scales.
CIPHER_EXTENDED_ROUNDS_QUICK = 27
#: Number of leading test vectors printed per S-box in Section 7; every
#: vector is recorded in ``results.json``.
PRINTED_TEST_VECTORS = 4


def section_header(title: str) -> None:
    """Print a banner line for a pipeline section."""
    print(f"\n{'#' * 72}\n# {title}\n{'#' * 72}")


def timed(timings: Dict[str, float], name: str, func: Callable[..., Any], *args: Any) -> Any:
    """Call ``func(*args)`` and store its wall-clock time under ``timings[name]``."""
    start = time.perf_counter()
    result = func(*args)
    timings[name] = time.perf_counter() - start
    return result


def map_parameters(chaotic_map: Any) -> str:
    """Parameter string read off the map instance itself.

    Derived from the object rather than hand-written, so the banner cannot
    drift from :func:`src.chaotic_maps.default_maps`.
    """
    return ", ".join(f"{name}={getattr(chaotic_map, name)!r}"
                     for name in MAP_PARAMETER_NAMES if hasattr(chaotic_map, name))


def run_chaotic_map_analysis() -> Dict[str, Dict[str, float]]:
    """Section 1: Lyapunov exponents and orbit entropy of the three paper maps.

    The maps come from :func:`src.chaotic_maps.default_maps` and every
    Lyapunov exponent is computed exactly once, inside
    :func:`src.chaotic_maps.compare_maps`; the banner below and the
    ``chaotic_maps`` record of ``results.json`` are printed from that one
    value, so they cannot disagree.
    """
    section_header("SECTION 1: CHAOTIC MAP ANALYSIS")
    maps = default_maps()
    comparison = compare_maps(n_iterations=5000)

    results: Dict[str, Dict[str, float]] = {}
    for name, chaotic_map in maps.items():
        data = comparison[name]
        results[name] = {
            "entropy_bits": float(data["entropy"]),
            "entropy_normalized": float(data["entropy_normalized"]),
            "lyapunov": float(data["lyapunov"]),
        }
        print(f"\n--- {MAP_TITLES.get(name, name)} ---")
        print(f"  Parameters: {map_parameters(chaotic_map)}")
        print(f"  Lyapunov Exponent: {results[name]['lyapunov']:.6f}")
        print(f"  Chaotic: {'Yes' if results[name]['lyapunov'] > 0 else 'No'}")

    print("\n--- Entropy Comparison (5000 iterations, 50 bins) ---")
    for name, record in results.items():
        print(f"  {name:15s}: Entropy={record['entropy_bits']:.6f} bits, "
              f"Lyapunov={record['lyapunov']:.6f}")
    return results


def run_sbox_construction() -> DACSBox:
    """Section 2: S-box construction and verification (paper Table 3)."""
    section_header("SECTION 2: S-BOX CONSTRUCTION (Table 3)")
    sbox = DACSBox(DAC_TABLE)

    print("\n--- DAC S-Box Table ---")
    print(sbox.to_binary_table())
    print(f"\n--- Hex: {sbox.to_hex_string()}")
    print(f"--- Bijective: {sbox.is_bijective()}")

    print("\n--- Coordinate Boolean Functions (MSB first) ---")
    for i, truth_table in enumerate(sbox.get_component_functions()):
        weight = sum(truth_table)
        print(f"  f{i}: weight={weight}/32, balanced={'Yes' if weight == 16 else 'No'}")
    return sbox


def run_boolean_analysis() -> List[Dict[str, Any]]:
    """Section 3: ANF, degree and balance of the coordinate functions."""
    section_header("SECTION 3: BOOLEAN FUNCTION ANALYSIS (ANF)")
    return [
        {"bit_position": c["bit_position"], "anf": c["anf_string"],
         "algebraic_degree": c["algebraic_degree"], "hamming_weight": c["hamming_weight"],
         "is_balanced": c["is_balanced"]}
        for c in verify_boolean_properties(DAC_TABLE)
    ]


def run_security_analysis() -> Dict[str, Any]:
    """Section 4: complete security analysis of the DAC S-box (printed report)."""
    section_header("SECTION 4: COMPREHENSIVE SECURITY ANALYSIS")
    return full_security_analysis(DAC_TABLE, name="DAC")


def run_visualization(security_results: Dict[str, Any], output_dir: str) -> List[str]:
    """Section 5: the paper figures, reusing Section 4's SAC / DDT / LAT.

    Returns the written figure paths (recorded in ``results.json``).
    """
    section_header("SECTION 5: PAPER FIGURES")
    return generate_all_figures(DAC_TABLE, output_dir=output_dir,
                                security_results=security_results)


#: Fields of :func:`src.block_cipher.demo_cipher`'s record kept in
#: ``results.json`` (the cipher object itself is not serialisable).
CIPHER_DEMO_KEYS = ("plaintext", "ciphertext", "round_trip_ok",
                    "avalanche_bits_lsb_flip", "block_bits")


def run_cipher_demo() -> Dict[str, Any]:
    """Section 6: block cipher demonstration; returns its test vector.

    :func:`src.block_cipher.demo_cipher` prints the walk-through and
    returns every value it computed, so the record below is a projection
    of what was printed rather than a second, independent evaluation of
    the same test vector: Section 6 cannot print one set of numbers and
    store another.
    """
    section_header("SECTION 6: LIGHTWEIGHT BLOCK CIPHER (SPN)")
    demo = demo_cipher()
    record = {key: demo[key] for key in CIPHER_DEMO_KEYS}
    print(f"\n  Recorded test vector (results.json -> cipher_demo, "
          f"plaintext 0x{DEMO_PLAINTEXT:08X}):")
    for key, value in record.items():
        print(f"    {key:24s}: {value}")
    return record


def run_cipher_analysis(quick: bool) -> Dict[str, Any]:
    """Section 7: round-level security analysis of the 32-bit SPN.

    Answers reviewer 2's major points 4 and 7 and reviewer 4's cipher-level
    request with computed numbers: the exact round function (re-derived by
    probing :mod:`src.block_cipher` rather than restated by hand), the
    differential and linear branch numbers of every frozen S-box, the minimum
    number of active S-boxes per round from the Mouha-style MILP, the
    resulting single-characteristic trail bounds, the two state bits that
    bypass the S-layer, reproducible test vectors and the birthday
    limitation of a 32-bit block.

    Everything printed here is a projection of the returned dictionary, which
    is recorded verbatim under ``cipher_analysis`` in ``results.json``, so the
    console and the results file cannot disagree.  The analysis is a LOWER
    BOUND from a branch-number relaxation and a single-characteristic bound
    under the independent-round heuristic, never a proof of resistance; the
    ``caveats`` list that says so is printed at the end of the section.

    ``quick`` shortens the MILP chain to
    :data:`CIPHER_EXTENDED_ROUNDS_QUICK` rounds, which changes no reported
    round count (see that constant).
    """
    section_header("SECTION 7: CIPHER-LEVEL SECURITY ANALYSIS (32-bit SPN)")
    extended = CIPHER_EXTENDED_ROUNDS_QUICK if quick else CIPHER_EXTENDED_ROUNDS
    analysis = analyze_cipher(extended_rounds=extended)

    print("\n--- Round function (re-derived from src.block_cipher) ---")
    print(analysis["round_function"]["text"])
    checks = analysis["round_function_checks"]
    print(f"  description re-derived from the code: "
          f"{sum(checks.values())}/{len(checks)} checks pass")

    print("\n--- Branch numbers (differential / linear; 2 is the worst possible) ---")
    for name, branch in analysis["branch_numbers"].items():
        print(f"  {name:14s}: {branch['differential']} / {branch['linear']}")

    bypass = analysis["bypass_analysis"]
    print("\n--- Bypass bits (state bits 1 and 0 skip the S-layer) ---")
    print(f"  max consecutive free rounds   : {bypass['max_consecutive_free_rounds']}")
    print(f"  multi-round bypass-only trail : "
          f"{bypass['multi_round_bypass_only_trail_exists']}")

    print("\n--- Minimum active S-boxes and single-characteristic bounds ---")
    print(format_active_sbox_table(analysis))
    findings = analysis["findings"]
    rounds = analysis["active_sboxes"]["reported_rounds"]
    print(f"\n  at the cipher's {rounds} rounds: "
          f"{findings['min_active_sboxes_at_cipher_rounds']} active S-boxes "
          f"({findings['free_rounds_at_cipher_rounds']} rounds crossed for free)")
    print(f"  rounds needed to reach the 2^-{int(DATA_LIMIT_LOG2)} data limit "
          f"(searched to {extended} rounds):")
    for name, bound in analysis["bounds"].items():
        limit = bound["rounds_to_data_limit"]
        print(f"    {name:14s}: differential {limit['differential']}, "
              f"linear {limit['linear']}")

    print("\n--- Exact trail search with the real DDT / LAT (Matsui-style) ---")
    for name, entry in analysis["exact_trails"].items():
        for model in ("differential", "linear"):
            row = entry[model]
            print(f"  {name:14s} {model:12s}: exact active "
                  f"{row['exact_min_active_sboxes'][:rounds]} | MILP tight from r=1 to "
                  f"{(row['first_round_milp_not_tight'] or rounds + 1) - 1} | best "
                  f"weight at {rounds} rounds {row['weight_at_cipher_rounds']:.3f} | "
                  f"data limit at r = {row['rounds_to_data_limit_exact_weight']}")

    print("\n--- Test vectors (every one verified by decryption) ---")
    for name, vectors in analysis["test_vectors"].items():
        print(f"  {name}:")
        for vector in vectors[:PRINTED_TEST_VECTORS]:
            print(f"    key {vector['key_hex']}  pt {vector['plaintext_hex']}  "
                  f"ct {vector['ciphertext_hex']}  "
                  f"round-trip {vector['round_trip_ok']}")
        print(f"    ({len(vectors)} vectors recorded in results.json)")

    print("\n--- Generic 32-bit block limitation ---")
    print(f"  {analysis['birthday_bound']['note']}")

    print("\n--- Findings ---")
    for statement in findings["statements"]:
        print(f"  * {statement}")
    print("\n--- Scope and caveats ---")
    for caveat in analysis["caveats"]:
        print(f"  ! {caveat}")
    return analysis


def run_benchmarks(seed: int, quick: bool,
                   title: str = "SECTION 8: SOFTWARE BENCHMARKS (seeded)") -> Dict[str, Any]:
    """Section 7: seeded software benchmarks of the DAC S-box and the cipher.

    ``title`` is the banner line, so callers with their own numbering
    (``run_optimized.py``) can label the step correctly.

    The machine, OS, interpreter and timer description recorded under
    ``benchmarks.methodology`` is printed as well (reviewer 2, additional
    point 4, and reviewer 4): these timings are single-threaded interpreted
    CPython and are ILLUSTRATIVE ONLY, not a performance claim and not
    comparable with optimised C, assembly or hardware figures.
    """
    section_header(title)
    benchmarks = collect_benchmarks(seed, quick=quick)
    sizes = benchmarks["sizes"]
    print(f"\n  scale: {benchmarks['scale']} ({sizes['n_lookups']:,} lookups, "
          f"{sizes['n_blocks']:,} blocks, {sizes['n_keys']:,} key schedules, "
          f"{sizes['n_avalanche_plaintexts']:,} avalanche plaintexts; "
          f"minimum of {sizes['repeat']} repeats; input seed {seed})")
    method = benchmarks["methodology"]
    print(f"  machine: {method['cpu_model']} ({method['cpu_architecture']}, "
          f"{method['physical_cores']} physical / {method['logical_cores']} logical cores)")
    print(f"  software: {method['os']} {method['os_release']} {method['os_version']}, "
          f"{method['python_implementation']} {method['python_version']}, "
          f"numpy {method['numpy_version']}")
    print(f"  timer: {method['timer']}, resolution "
          f"{method['timer_resolution_ns']:g} ns; both the minimum and the median "
          "of the repeats are reported")
    print("  NOTE: these software timings are ILLUSTRATIVE ONLY (single-threaded "
          "interpreted CPython).")
    print(benchmark_table(benchmarks))
    return benchmarks


def run_dynamic_sbox_demo() -> List[Dict[str, Any]]:
    """Section 9: key-dependent dynamic S-boxes (improvement); returns their metrics."""
    section_header("SECTION 9: KEY-DEPENDENT DYNAMIC S-BOX (IMPROVEMENT)")
    print("\nGenerating S-boxes with different key seeds...")
    print("  (the tent-logistic map satisfies f(1 - s) = f(s): the key space is")
    print("   (0, 1/2] and a seed above 1/2 is folded to 1 - seed)")
    records: List[Dict[str, Any]] = []
    for seed in DYNAMIC_DEMO_SEEDS:
        dynamic = generate_dynamic_sbox(key_seed=seed)
        sac = compute_sac(dynamic.table)
        lat = compute_lat(dynamic.table)
        ddt = compute_ddt(dynamic.table)
        print(f"\n  Seed={seed} (canonical seed {canonical_seed(seed)!r}):")
        print(f"    First 8 values: {dynamic.table[:8]}")
        print(f"    Bijective: {dynamic.is_bijective()}")
        print(f"    SAC={sac['average']:.4f}, LAP={lat['lap']:.4f}, DP={ddt['dp']:.4f}")
        records.append({
            "seed": seed, "canonical_seed": canonical_seed(seed), "table": list(dynamic.table),
            "is_bijective": dynamic.is_bijective(), "sac_avg": sac["average"],
            "lap": lat["lap"], "dp": ddt["dp"],
        })
    return records


def run_summary() -> Dict[str, Any]:
    """Section 10: summary table of the DAC family and the computed competitors."""
    section_header("SECTION 10: SUMMARY - DAC FAMILY vs COMPUTED COMPETITORS")
    comparison = comparative_results()
    print(summary_table(comparison["metrics"]))
    print(SUMMARY_FOOTNOTE)
    print("(pairwise tables, literature comparison and comparison figures: run_optimized.py)")
    return comparison


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the runner."""
    parser = argparse.ArgumentParser(
        description="DAC S-box paper pipeline (frozen tables, seeded benchmarks).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="integer seed of the benchmark inputs (lookups, plaintexts, keys)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke-test benchmark sizes (10x fewer samples)")
    parser.add_argument("--no-figures", action="store_true", help="skip the paper figures")
    parser.add_argument("--no-cipher-analysis", action="store_true",
                        help="skip Section 7, the MILP active-S-box chain and the "
                             "slowest step (about 12 s); results.json then carries no "
                             "'cipher_analysis' key and verify_latex.py reports every "
                             "cipher-level table as unchecked")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="directory for the figures and results.json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the complete pipeline; returns the process exit code."""
    args = build_parser().parse_args(argv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    arguments = {"seed": args.seed, "quick": args.quick,
                 "figures": not args.no_figures,
                 "cipher_analysis": not args.no_cipher_analysis,
                 "output_dir": output_dir}

    print("=" * 72)
    print("  DAC (Dynamic Airy Chaotic) SUBSTITUTION BOX")
    print("  Complete Implementation Pipeline")
    print("  Based on: Sarmila & Manisekaran, EIJ 2025")
    print("=" * 72)
    print(f"  arguments: {arguments}")

    start = time.perf_counter()
    timings: Dict[str, float] = {}
    chaos = timed(timings, "chaotic_maps", run_chaotic_map_analysis)
    timed(timings, "sbox_construction", run_sbox_construction)
    boolean = timed(timings, "boolean_functions", run_boolean_analysis)
    security = timed(timings, "security_analysis", run_security_analysis)
    figures: List[str] = []
    if not args.no_figures:
        figures = timed(timings, "figures", run_visualization, security, output_dir)
    cipher_demo = timed(timings, "cipher_demo", run_cipher_demo)
    cipher_analysis: Optional[Dict[str, Any]] = None
    if not args.no_cipher_analysis:
        cipher_analysis = timed(timings, "cipher_analysis", run_cipher_analysis, args.quick)
    benchmarks = timed(timings, "benchmarks", run_benchmarks, args.seed, args.quick)
    dynamic = timed(timings, "dynamic_sbox_demo", run_dynamic_sbox_demo)
    comparison = timed(timings, "summary", run_summary)

    benchmark_values, benchmark_timings = partition_benchmarks(benchmarks)
    results = core_results("main.py", arguments, comparison)
    results.update({
        "schema": RESULTS_SCHEMA,
        "chaotic_maps": chaos,
        "boolean_functions": {"DAC": boolean},
        "cipher_demo": cipher_demo,
        "benchmarks": benchmark_values,
        "dynamic_sbox_demo": dynamic,
        "figures": figures,
        "timings": {"sections_s": timings, "benchmarks": benchmark_timings,
                    "total_s": time.perf_counter() - start},
    })
    if cipher_analysis is not None:
        results["cipher_analysis"] = cipher_analysis
    results_path = write_results(os.path.join(output_dir, RESULTS_FILENAME), results)

    print(f"\n{'=' * 72}")
    print(f"  PIPELINE COMPLETE in {time.perf_counter() - start:.2f} seconds")
    print(f"  Results: {results_path}")
    print(f"  Figures: {output_dir}")
    print(f"  Verilog HDL: {os.path.join(PROJECT_ROOT, 'hdl')}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
