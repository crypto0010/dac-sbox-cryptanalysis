"""Optimised DAC S-box pipeline: paper baseline, chaos-driven search, comparison.

STEP 1  Security and Boolean analysis of the original DAC S-box (paper Table 3).
STEP 2  Chaos-driven S-box search, ``optimize_dac_sbox(strategy="all")``,
        seeded by ``--optimizer-seed``.  With the default seed it reproduces
        the frozen DAC-OPT table bit for bit.  The search result is recorded
        in ``results_optimized.json``; everything below uses the FROZEN
        tables of :mod:`src.sbox_tables`, never a fresh search result.
STEP 3  Comparative analysis of the DAC family against computed competitors
        (ASCON, Keccak chi, PRIMATE), the literature values of the base paper
        and the ``comparison_*`` figures.  This step also prints the
        reviewer-requested discriminating quantities that the headline
        metrics do not separate -- the BCT spectrum, the symmetric SAC
        deviations D_max / D_mean and the algebraic immunity -- all of which
        are recorded per S-box under ``metrics`` in the results file.
STEP 4  Seeded software benchmarks, with the machine / OS / interpreter /
        timer description recorded under ``benchmarks.methodology``.
STEP 5  Summary table and ``results_optimized.json``.

The paper figures (Fig. 2-6, heatmaps, mapping) belong to ``main.py``;
``--all-figures`` regenerates them here as well.
``results_optimized.json`` is byte-identical between runs with the same
arguments except for its ``timings`` entry.  Works from any working
directory.

This runner writes ``output/results_optimized.json`` (schema
:data:`RESULTS_SCHEMA`) and never touches ``output/results.json``, which
belongs to ``main.py``: the two files carry different sections (this one
has ``optimizer``, ``main.py`` has ``chaotic_maps``, ``boolean_functions``,
``cipher_demo``, ``cipher_analysis`` and ``dynamic_sbox_demo``) and neither
runner can destroy the other's numbers.  The round-level cipher analysis is
deliberately run once, by ``main.py``, because it is the slowest step of the
pipeline (about 12 s of MILP) and nothing here depends on it.
``verify_latex.py`` reads whichever of the two exist and merges them.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import (DAC_TABLE, DEFAULT_OUTPUT_DIR, DEFAULT_SEED,  # noqa: E402
                  run_benchmarks, section_header, timed)
from src.boolean_function import verify_boolean_properties  # noqa: E402
from src.chaotic_maps import canonical_seed  # noqa: E402
from src.comparative_analysis import (SUMMARY_FOOTNOTE, core_results,  # noqa: E402
                                      frozen_match, optimizer_record,
                                      optimizer_runtimes, partition_benchmarks,
                                      run_comparative_analysis, summary_table,
                                      write_results)
from src.sbox_optimizer import STRATEGIES, optimize_dac_sbox  # noqa: E402
from src.sbox_tables import DAC_OPT_SEED  # noqa: E402
from src.security_analysis import full_security_analysis  # noqa: E402
from src.visualization import generate_all_figures  # noqa: E402

#: Results file of *this* runner; ``main.py`` owns ``output/results.json``.
RESULTS_FILENAME = "results_optimized.json"
RESULTS_SCHEMA = "dac-sbox-results-optimized/1"


def run_baseline() -> Dict[str, Any]:
    """STEP 1: security and Boolean analysis of the original DAC S-box."""
    section_header("STEP 1: ORIGINAL DAC S-BOX ANALYSIS (paper implementation)")
    results = full_security_analysis(DAC_TABLE, name="DAC (Original)")
    verify_boolean_properties(DAC_TABLE)
    return results


def run_optimizer(seed: float) -> Dict[str, Any]:
    """STEP 2: run every search strategy from ``seed`` and report the winner."""
    section_header("STEP 2: CHAOS-DRIVEN S-BOX SEARCH (improvement B5)")
    print(f"  key seed {seed!r} (canonical {canonical_seed(seed)!r}); "
          f"strategies: {', '.join(STRATEGIES)}")
    result = optimize_dac_sbox(strategy="all", seed=seed, verbose=True)
    print(f"  winner: {result['provenance']['strategy']} (objective {result['objective']})")
    match = frozen_match(result["sbox"])
    if match is None:
        print("  WARNING: the winner differs from every frozen table of src.sbox_tables;")
        print("           the comparison, figures and summary below use the frozen tables.")
    else:
        print(f"  reproduces the frozen table {match} of src.sbox_tables bit for bit")
    return result


def run_comparison(output_dir: str, figures: bool) -> Dict[str, Any]:
    """STEP 3: DAC family vs computed competitors (frozen tables)."""
    section_header("STEP 3: COMPARATIVE ANALYSIS (frozen tables, computed competitors)")
    return run_comparative_analysis(output_dir, figures=figures)


def run_paper_figures(security_results: Dict[str, Any], output_dir: str) -> List[str]:
    """Optional: the paper figures normally produced by ``main.py``."""
    section_header("PAPER FIGURES (--all-figures)")
    return generate_all_figures(DAC_TABLE, output_dir=output_dir,
                                security_results=security_results)


def run_summary(metrics: Dict[str, Any]) -> None:
    """STEP 5: final summary table of every analysed S-box."""
    section_header("STEP 5: SUMMARY - DAC FAMILY vs COMPUTED COMPETITORS")
    print(summary_table(metrics))
    print(SUMMARY_FOOTNOTE)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the runner."""
    parser = argparse.ArgumentParser(
        description="DAC S-box optimisation pipeline (search, comparison, benchmarks).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="integer seed of the benchmark inputs (lookups, plaintexts, keys)")
    parser.add_argument("--optimizer-seed", type=float, default=DAC_OPT_SEED,
                        help="key seed in (0, 1) of the chaotic generator driving the "
                             "S-box search (the default reproduces the frozen DAC-OPT table)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke-test benchmark sizes (10x fewer samples)")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip the comparison figures (the paper figures of "
                             "--all-figures are unaffected)")
    parser.add_argument("--all-figures", action="store_true",
                        help="also regenerate the paper figures of main.py "
                             "(independent of --no-figures: the two flags control "
                             "disjoint figure sets)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="directory for the figures and results_optimized.json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the optimisation pipeline; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        canonical_seed(args.optimizer_seed)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    figures = not args.no_figures
    arguments = {"seed": args.seed, "optimizer_seed": args.optimizer_seed, "quick": args.quick,
                 "figures": figures, "all_figures": args.all_figures,
                 "output_dir": output_dir}

    print("=" * 72)
    print("  DAC S-BOX: PAPER IMPLEMENTATION + OPTIMISED IMPROVEMENTS")
    print("  Based on: Sarmila & Manisekaran, EIJ 2025")
    print("=" * 72)
    print(f"  arguments: {arguments}")

    start = time.perf_counter()
    timings: Dict[str, float] = {}
    security = timed(timings, "baseline", run_baseline)
    result = timed(timings, "optimizer", run_optimizer, args.optimizer_seed)
    comparison = timed(timings, "comparison", run_comparison, output_dir, figures)
    figure_paths: List[str] = list(comparison["figures"])
    if args.all_figures:
        figure_paths += timed(timings, "paper_figures", run_paper_figures,
                              security, output_dir)
    benchmarks = timed(timings, "benchmarks", run_benchmarks, args.seed, args.quick,
                       "STEP 4: SOFTWARE BENCHMARKS (seeded)")
    timed(timings, "summary", run_summary, comparison["metrics"])

    benchmark_values, benchmark_timings = partition_benchmarks(benchmarks)
    results = core_results("run_optimized.py", arguments, comparison)
    results.update({
        "schema": RESULTS_SCHEMA,
        "optimizer": optimizer_record(result),
        "benchmarks": benchmark_values,
        "figures": figure_paths,
        "timings": {"sections_s": timings, "optimizer": optimizer_runtimes(result),
                    "benchmarks": benchmark_timings, "total_s": time.perf_counter() - start},
    })
    results_path = write_results(os.path.join(output_dir, RESULTS_FILENAME), results)

    print(f"\n{'=' * 72}")
    print(f"  COMPLETE PIPELINE FINISHED in {time.perf_counter() - start:.1f} seconds")
    print(f"  Results: {results_path}")
    print(f"  Figures: {output_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
