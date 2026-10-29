"""Common-condition hardware evaluation of the seven 5-bit S-boxes.

Writes ``output/results_hardware.json`` (schema :data:`RESULTS_SCHEMA`) and
the mapped gate-level netlists in ``output/netlists``.  For every S-box and
both open cell libraries of :mod:`src.hardware_synthesis` the lookup table is
synthesised with Yosys/ABC, the mapped netlist is verified on all 32 inputs
against the table, and area (um^2 and GE), logic depth, critical-path delay,
dynamic energy per evaluation and leakage are computed from the Liberty data.
The ASCON and Keccak-chi S-boxes are additionally synthesised from their
hand-written structural descriptions as a reference for the lookup-table flow.

Needs ``yowasp-yosys`` (``requirements.txt``) and, on the first run, network
access to download the two Liberty files (SHA-256 checked, cached under
``hdl/synth/libs``).  Deterministic: two runs give identical files apart from
``timings``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import DEFAULT_OUTPUT_DIR, section_header  # noqa: E402
from src.comparative_analysis import all_sbox_tables, write_results  # noqa: E402
from src.hardware_synthesis import (LIBRARIES, ORFS_COMMIT,  # noqa: E402
                                    STRUCTURAL_REFERENCES, default_cache_dir,
                                    library_record, synthesize, yosys_version)

RESULTS_FILENAME = "results_hardware.json"
RESULTS_SCHEMA = "dac-sbox-results-hardware/1"
NETLIST_DIRNAME = "netlists"


def tag(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_")


#: Representatives synthesised per class and kind in the class area study.
CLASS_AREA_SAMPLES = 48
CLASS_AREA_LIBRARY = "nangate45"
CLASS_AREA_SEED = 2026


def _ideal_sac_representative(rng, choices) -> list:
    """A uniformly random representative of the inversion class with ideal SAC.

    ``choices`` lists every (input basis, ordered output rows) pair with an
    ideal SAC matrix; a uniform pick, a uniform column order of ``B`` and
    uniform constants give a uniform member of the ideal-SAC subset.
    """
    import numpy as np

    from src.sbox_optimizer import INVERSION_EXPONENT, AffineParams, _transpose
    pick = choices[int(rng.integers(len(choices)))]
    cols = list(pick[0])
    rng.shuffle(cols)
    a, b = (int(v) for v in rng.integers(0, 32, size=2))
    params = AffineParams(INVERSION_EXPONENT, tuple(int(v) for v in pick[1]), a,
                          tuple(int(v) for v in _transpose(np.array(cols))), b)
    return [int(v) for v in params.sbox()]


def _gf2_rank(vectors) -> int:
    basis = []
    for v in vectors:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return len(basis)


def class_area_study(tables: Dict[str, list], samples: int = CLASS_AREA_SAMPLES,
                     lib_key: str = CLASS_AREA_LIBRARY, seed: int = CLASS_AREA_SEED
                     ) -> Dict[str, Any]:
    """Area of random representatives of each class, under the same flow.

    Area is not one of the criteria Algorithm 2 selects, so it varies with the
    representative like the SAC matrix does.  For the inversion class three
    samples are synthesised: uniformly random representatives, uniformly
    random representatives *with an ideal SAC matrix* (drawn from the exactly
    counted set of STEP 6 of run_studies.py), and for the Gold class uniformly
    random representatives.  The published tables are placed in these
    distributions.
    """
    import itertools
    from statistics import median

    import numpy as np

    from src.search_studies import EXPONENTS, _random_representative
    from src.sbox_optimizer import _is_invertible, autocorrelation_table
    rng = np.random.Generator(np.random.PCG64(seed))
    ac = autocorrelation_table(EXPONENTS["inverse"])
    subsets = np.array(list(itertools.combinations(range(1, 32), 5)), dtype=np.int64)
    bases = subsets[_is_invertible(subsets)]
    choices = []
    for basis in bases:
        zero = [c for c in range(1, 32) if all(ac[c, d] == 0 for d in basis)]
        if len(zero) >= 5 and _gf2_rank(zero) == 5:
            choices.extend((tuple(int(v) for v in basis), rows)
                           for rows in itertools.permutations(zero, 5) if _gf2_rank(rows) == 5)
    if len(choices) != 40920:
        raise RuntimeError(f"expected 40920 ideal-SAC (basis, A) pairs, found {len(choices)}")
    kinds = {
        "inverse_random": lambda: _random_representative(EXPONENTS["inverse"], rng),
        "inverse_ideal_sac": lambda: _ideal_sac_representative(rng, choices),
        "gold_random": lambda: _random_representative(EXPONENTS["gold"], rng),
    }
    reference = {"inverse_random": "DAC-OPT", "inverse_ideal_sac": "DAC-OPT",
                 "gold_random": "DAC-OPT-GOLD"}
    report: Dict[str, Any] = {"library": lib_key, "samples": samples, "seed": seed,
                              "generator": "numpy.random.PCG64", "kinds": {}}
    for kind, draw in kinds.items():
        areas = []
        for index in range(samples):
            table = draw()
            result = synthesize(table, lib_key, f"class_{kind}_{index}")
            areas.append(result["area_ge"])
        ref_name = reference[kind]
        ref_area = synthesize(tables[ref_name], lib_key, tag(ref_name))["area_ge"]
        report["kinds"][kind] = {
            "min_ge": min(areas), "median_ge": float(median(areas)), "max_ge": max(areas),
            "reference": ref_name, "reference_ge": ref_area,
            "share_at_most_reference": float(np.mean(np.asarray(areas) <= ref_area)),
            "areas_ge": areas,
        }
        print(f"  {kind:18s} GE min {min(areas):.2f} median {median(areas):.2f} "
              f"max {max(areas):.2f}; {ref_name} {ref_area:.2f}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--class-samples", type=int, default=CLASS_AREA_SAMPLES,
                        help="representatives synthesised per class in the area study")
    args = parser.parse_args(argv)
    if args.class_samples < 1:
        parser.error("--class-samples must be >= 1")
    output_dir = os.path.abspath(args.output_dir)
    netlist_dir = os.path.join(output_dir, NETLIST_DIRNAME)
    os.makedirs(netlist_dir, exist_ok=True)
    start = time.perf_counter()
    tables = all_sbox_tables()
    report: Dict[str, Any] = {"libraries": {}, "lookup": {}, "structural": {}}
    for key in LIBRARIES:
        section_header(f"SYNTHESIS: {LIBRARIES[key].label}")
        report["libraries"][key] = library_record(key)
        report["lookup"][key] = {}
        report["structural"][key] = {}
        for name, table in tables.items():
            result = synthesize(table, key, tag(name))
            report["lookup"][key][name] = result
            print(f"  {name:13s} {result['cells']:3d} cells  {result['area_ge']:6.2f} GE  "
                  f"depth {result['logic_depth']}  {result['delay_ps']:6.1f} ps  "
                  f"{result['energy_fj_per_evaluation']:7.2f} fJ  "
                  f"leak {result['leakage_nw']:.4g} nW  (verified on 32 inputs)")
            src = os.path.join(default_cache_dir(), "work", f"{tag(name)}_{key}.json")
            shutil.copyfile(src, os.path.join(netlist_dir, f"{tag(name)}_{key}.json"))
        for name, body in STRUCTURAL_REFERENCES.items():
            result = synthesize(tables[name], key, tag(name) + "_structural", structural=body)
            report["structural"][key][name] = result
            print(f"  {name + ' (struct.)':24s} {result['area_ge']:6.2f} GE  depth "
                  f"{result['logic_depth']}  {result['delay_ps']:6.1f} ps  "
                  f"{result['energy_fj_per_evaluation']:7.2f} fJ")
    section_header(f"AREA ACROSS THE AFFINE CLASSES ({CLASS_AREA_LIBRARY}, "
                   f"{args.class_samples} representatives per kind)")
    report["class_area"] = class_area_study(tables, args.class_samples)
    results = {
        "schema": RESULTS_SCHEMA,
        "runner": "run_hardware.py",
        "tool": {"synthesis": yosys_version(),
                 "flow": "synth -flatten; abc -liberty (delay-oriented mapping, area recovery)",
                 "liberty_source_commit": ORFS_COMMIT},
        "conditions": {
            "description": "32-entry case lookup per S-box; same script for every S-box",
            "input_drive": "reference inverter of the library",
            "output_load": "one reference-inverter input per output",
            "energy": "mean over all 1024 ordered pairs of consecutive inputs, zero-delay "
                      "activity, internal + switching energy; glitches excluded",
            "delay": "arc-based static timing, worst of rise and fall, NLDM interpolation",
            "pre_layout": True,
        },
        "hardware": report,
        "timings": {"total_s": time.perf_counter() - start},
    }
    path = write_results(os.path.join(output_dir, RESULTS_FILENAME), results)
    print(f"\n  hardware evaluation finished in {time.perf_counter() - start:.1f} s -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
