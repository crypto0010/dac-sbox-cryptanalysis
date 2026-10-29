"""Tiered verification report of the computational record.

Recomputes the archived results and states, tier by tier, what was
reproduced, how, and how long it took:

Tier A -- direct table enumeration (seconds).
    A self-contained re-implementation *in this file* (nothing imported from
    ``src``) of the DDT, LAT, BCT, SAC matrix, vectorial nonlinearity, LAP,
    differential and boomerang uniformity, algebraic degree and fixed
    points recomputes every S-box metric from the seven lookup tables and
    compares it with ``output/results.json``; the CSV tables of
    ``output/tables`` are checked against the same enumeration; the four
    metric identities are re-derived; the exact count of ideal-SAC
    representatives of the inversion class is recounted by brute force; and
    the executable specification of Algorithm 2 (``src/algorithm_spec.py``)
    is re-run and must reproduce both published tables.
Tier B -- MILP and exact trail search (about a minute).
    ``src.cipher_analysis.analyze_cipher`` is re-run (HiGHS MILP for the
    branch-number bound, Matsui-style branch and bound over the real DDT and
    LAT) and its output must equal ``cipher_analysis`` of ``results.json``.
Tier C -- logic synthesis (``--hardware``; needs yowasp-yosys and, once,
    network access for the two Liberty files).  Every S-box is synthesised
    again and must reproduce ``output/results_hardware.json``.
Tier D -- manuscript (when ``paper.tex`` is present).
    ``verify_latex.py`` compares every tabulated value with the result files.

Writes ``output/verification_report.md``, ``output/verification_report.json``
and ``output/MANIFEST.json`` (SHA-256 of every archived output file, tool and
library versions, git commit).  Exit status 0 only if every tier run passes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(PROJECT_ROOT, "output")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

N = 5
Q = 32


# ------------------------------------------------------------------
# Tier A: self-contained enumeration (deliberately independent of src/)
# ------------------------------------------------------------------
def _parity(v: int) -> int:
    return bin(v).count("1") & 1


def enum_ddt(s: Sequence[int]) -> List[List[int]]:
    t = [[0] * Q for _ in range(Q)]
    for dx in range(Q):
        for x in range(Q):
            t[dx][s[x] ^ s[x ^ dx]] += 1
    return t


def enum_lat(s: Sequence[int]) -> List[List[int]]:
    """Signed count ``#{x : a.x = b.S(x)} - 16``."""
    return [[sum(1 for x in range(Q) if _parity(a & x) == _parity(b & s[x])) - Q // 2
             for b in range(Q)] for a in range(Q)]


def enum_bct(s: Sequence[int]) -> List[List[int]]:
    inv = [0] * Q
    for x, y in enumerate(s):
        inv[y] = x
    return [[sum(1 for x in range(Q)
                 if inv[s[x] ^ dy] ^ inv[s[x ^ dx] ^ dy] == dx) for dy in range(Q)]
            for dx in range(Q)]


def enum_degree(f: Sequence[int]) -> int:
    anf = list(f)
    for i in range(N):
        for x in range(Q):
            if x & (1 << i):
                anf[x] ^= anf[x ^ (1 << i)]
    return max((bin(u).count("1") for u in range(Q) if anf[u]), default=0)


def enum_metrics(s: Sequence[int]) -> Dict[str, Any]:
    ddt, lat, bct = enum_ddt(s), enum_lat(s), enum_bct(s)
    delta = max(ddt[dx][dy] for dx in range(1, Q) for dy in range(Q))
    max_lat = max(abs(lat[a][b]) for a in range(Q) for b in range(Q) if (a, b) != (0, 0))
    max_walsh_nontrivial = max(abs(2 * lat[a][b]) for a in range(Q) for b in range(1, Q))
    sac = [[sum(1 for x in range(Q) if ((s[x] ^ s[x ^ (1 << (N - 1 - i))]) >> (N - 1 - j)) & 1)
            / Q for j in range(N)] for i in range(N)]
    devs = [abs(p - 0.5) for row in sac for p in row]
    degrees = [enum_degree([_parity(b & s[x]) for x in range(Q)]) for b in range(1, Q)]
    return {
        "differential_uniformity": delta,
        "dp": delta / Q,
        "lap": max_lat / Q,
        "nl_sbox": Q // 2 - max_walsh_nontrivial // 2,
        "boomerang_uniformity": max(bct[dx][dy] for dx in range(1, Q) for dy in range(1, Q)),
        "min_degree": min(degrees),
        "max_degree": max(degrees),
        "sac_avg": sum(p for row in sac for p in row) / (N * N),
        "sac_d_max": max(devs),
        "sac_d_mean": sum(devs) / (N * N),
        "fixed_points": sum(1 for x in range(Q) if s[x] == x),
        "opposite_fixed_points": sum(1 for x in range(Q) if s[x] == x ^ (Q - 1)),
        "_ddt": ddt, "_lat": lat, "_bct": bct,
    }


def _gf_mul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a & Q:
            a ^= 0b100101
    return r


def _rank(vectors: Sequence[int]) -> int:
    basis: List[int] = []
    for v in vectors:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return len(basis)


def enum_ideal_sac_count() -> Dict[str, Any]:
    """Brute-force recount for F = x^-1 (= x^30) over GF(2^5)."""
    f = [0] * Q
    for z in range(1, Q):
        f[z] = next(y for y in range(1, Q) if _gf_mul(z, y) == 1)
    ac = [[sum(1 - 2 * _parity(c & (f[y] ^ f[y ^ d])) for y in range(Q)) for d in range(Q)]
          for c in range(Q)]
    bases = [c for c in itertools.combinations(range(1, Q), N) if _rank(c) == N]
    admissible, sizes, ordered = 0, {}, 0
    for basis in bases:
        zero = [c for c in range(1, Q) if all(ac[c][d] == 0 for d in basis)]
        if len(zero) >= N and _rank(zero) == N:
            admissible += 1
            sizes[len(zero)] = sizes.get(len(zero), 0) + 1
            ordered += sum(1 for rows in itertools.permutations(zero, N) if _rank(rows) == N)
    return {"input_bases": len(bases), "admissible_bases": admissible,
            "zero_set_sizes": {str(k): v for k, v in sorted(sizes.items())},
            "ordered_pairs_with_ideal_sac": 120 * ordered}


# ------------------------------------------------------------------
# Report plumbing
# ------------------------------------------------------------------
class Tier:
    def __init__(self, key: str, title: str, method: str) -> None:
        self.key, self.title, self.method = key, title, method
        self.items: List[Dict[str, Any]] = []
        self.seconds = 0.0
        self.skipped: Optional[str] = None

    def check(self, what: str, ok: bool, detail: str = "", compared: int = 1) -> None:
        self.items.append({"check": what, "pass": bool(ok), "detail": detail,
                           "values_compared": int(compared)})

    @property
    def passed(self) -> bool:
        return self.skipped is None and all(item["pass"] for item in self.items)

    def as_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "method": self.method, "seconds": round(self.seconds, 1),
                "skipped": self.skipped, "passed": self.passed,
                "values_compared": sum(i["values_compared"] for i in self.items),
                "checks": self.items}


def _load(name: str) -> Dict[str, Any]:
    with open(os.path.join(OUTPUT, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _close(a: Any, b: Any) -> bool:
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))


def _diff_paths(a: Any, b: Any, path: str = "") -> List[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}/{key} (missing on one side)")
            else:
                out.extend(_diff_paths(a[key], b[key], f"{path}/{key}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} vs {len(b)})"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(_diff_paths(x, y, f"{path}[{i}]"))
        return out
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if _close(a, b) else [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def _leaves(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_leaves(v) for v in value.values())
    if isinstance(value, list):
        return sum(_leaves(v) for v in value)
    return 1


def _timed(tier: Tier, body: Callable[[], None]) -> None:
    start = time.perf_counter()
    try:
        body()
    except Exception as exc:  # a crash is a failed check, not a silent pass
        tier.check("tier completed without an exception", False, repr(exc))
    tier.seconds = time.perf_counter() - start


# ------------------------------------------------------------------
# Tiers
# ------------------------------------------------------------------
METRIC_KEYS = ("differential_uniformity", "dp", "lap", "nl_sbox", "boomerang_uniformity",
               "min_degree", "max_degree", "sac_avg", "sac_d_max", "sac_d_mean",
               "fixed_points", "opposite_fixed_points")


def tier_a(tier: Tier) -> None:
    results = _load("results.json")
    studies = _load("results_studies.json")
    for name, table in sorted(results["tables"].items()):
        m = enum_metrics(table)
        stored = results["metrics"][name]
        bad = [k for k in METRIC_KEYS if not _close(m[k], stored[k])]
        tier.check(f"{name}: {len(METRIC_KEYS)} metrics by direct enumeration", not bad,
                   ", ".join(f"{k}: {m[k]} vs {stored[k]}" for k in bad), len(METRIC_KEYS))
        ids = (m["nl_sbox"] == 16 - 32 * m["lap"]
               and m["differential_uniformity"] == 32 * m["dp"])
        tier.check(f"{name}: NL = 16 - 32 LAP and delta = 32 DP", ids, "", 2)
        stem = name.replace(" ", "_")
        for kind in ("ddt", "lat", "bct"):
            path = os.path.join(OUTPUT, "tables", f"{stem}_{kind}.csv")
            with open(path, newline="", encoding="utf-8") as handle:
                rows = [[int(v) for v in row[1:]] for row in list(csv.reader(handle))[1:]]
            tier.check(f"{name}: tables/{stem}_{kind}.csv", rows == m["_" + kind], "", Q * Q)
    counting = enum_ideal_sac_count()
    stored = studies["ideal_sac_counting"]["inverse"]
    keys = ("input_bases", "admissible_bases", "zero_set_sizes", "ordered_pairs_with_ideal_sac")
    bad = [k for k in keys if counting[k] != stored[k]]
    tier.check("ideal-SAC count of the inversion class (brute force: 83,328 bases, "
               "186 admissible, 4,910,400 (A, B) pairs)", not bad,
               ", ".join(f"{k}: {counting[k]} vs {stored[k]}" for k in bad), len(keys))
    from src import algorithm_spec
    from src.sbox_tables import TABLES
    for name, exponent, budget in (("DAC-OPT", 30, 4096), ("DAC-OPT-GOLD", 3, 512)):
        run = algorithm_spec.affine_select(exponent, 0.5, budget)
        rec = studies["algorithm_specification"]["runs"][name]
        same = (run["sbox"] == list(TABLES[name].table)
                and all(run[k] == rec[k] for k in ("A_rows", "B_rows", "a", "b",
                                                   "best_candidate_index", "draws")))
        tier.check(f"{name}: executable Algorithm 2 specification from seed 0.5", same,
                   f"kept candidate {run['best_candidate_index']}, {run['draws']} draws", Q + 6)


def tier_b(tier: Tier) -> None:
    from src.cipher_analysis import analyze_cipher
    stored = _load("results.json")["cipher_analysis"]
    fresh = json.loads(json.dumps(analyze_cipher()))
    diffs = _diff_paths(fresh, stored)
    tier.check("cipher_analysis (MILP bounds, exact trail search, bypass analysis, "
               "test vectors) identical to results.json", not diffs, "; ".join(diffs[:5]),
               _leaves(stored))


def tier_c(tier: Tier) -> None:
    from src.hardware_synthesis import synthesize
    stored = _load("results_hardware.json")["hardware"]["lookup"]
    tables = _load("results.json")["tables"]
    for lib, entries in stored.items():
        for name, record in entries.items():
            fresh = synthesize(tables[name], lib, "verify_" + name.replace("-", "_"))
            diffs = _diff_paths(fresh, record)
            tier.check(f"{lib}/{name}: synthesis, gate-level check on 32 inputs, area, "
                       "depth, delay, energy, leakage", not diffs, "; ".join(diffs[:3]),
                       _leaves(record))


def tier_d(tier: Tier) -> None:
    tex = os.path.join(PROJECT_ROOT, "paper.tex")
    if not os.path.exists(tex):
        tier.skipped = "paper.tex is not part of the archive"
        return
    import verify_latex
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        errors, _, mismatches = verify_latex.verify(tex)
    timing_rows = {f"tab:benchmarks/software/{verify_latex.slug(r)}"
                   for r in verify_latex.BENCHMARK_TIMING_ROWS}
    unexpected = [m for m in mismatches if verify_latex.mismatch_key(m) not in
                  {t.replace(' ', '') for t in timing_rows}]
    compared = int(next((line.split(":")[1] for line in buffer.getvalue().splitlines()
                         if "Values compared:" in line), "0"))
    tier.check("every tabulated value of the manuscript equals the result files "
               "(machine-dependent wall-clock rows excepted)", not errors and not unexpected,
               "; ".join((errors + unexpected)[:5]), compared)


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------
def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def _versions() -> Dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "matplotlib", "prettytable", "pytest",
                    "yowasp-yosys", "yowasp-runtime", "wasmtime"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = "not installed"
    return out


def manifest() -> Dict[str, Any]:
    files: Dict[str, str] = {}
    for root, _, names in os.walk(OUTPUT):
        for name in sorted(names):
            rel = os.path.relpath(os.path.join(root, name), OUTPUT).replace(os.sep, "/")
            if rel.endswith((".json", ".csv")) and not rel.startswith("verification_report") \
                    and rel != "MANIFEST.json":
                files[rel] = _sha256(os.path.join(root, name))
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_describe": _git("describe", "--tags", "--always", "--dirty"),
        "working_tree_clean": status == "" if status is not None else None,
        "versions": _versions(),
        "seeds": {"chaotic search (DAC-OPT, DAC-OPT-GOLD)": 0.5,
                  "annealing study": "x0 in {0.05, ..., 0.50}",
                  "generator ablation": "chaotic k/64, PCG64 0..31, Weyl k/32",
                  "affine-class statistics": "PCG64 2026",
                  "hardware class-area study": "PCG64 2026",
                  "benchmarks and test vectors (main.py --seed)": 42},
        "commands": ["python main.py", "python run_optimized.py", "python run_studies.py",
                     "python run_hardware.py", "python verify_results.py --hardware",
                     "python verify_latex.py"],
        "files_sha256": dict(sorted(files.items())),
    }


def render_markdown(tiers: Sequence[Tier], info: Dict[str, Any]) -> str:
    lines = ["# Verification report", "",
             f"Generated by `verify_results.py` at commit `{info['git_commit']}` "
             f"(`{info['git_describe']}`), Python {info['versions']['python']}, "
             f"NumPy {info['versions']['numpy']}, SciPy {info['versions']['scipy']}, "
             f"{info['versions']['yowasp-yosys']} (yowasp-yosys).", "",
             "| Tier | What is reproduced | Values compared | Time | Result |",
             "|---|---|---:|---:|---|"]
    for tier in tiers:
        d = tier.as_dict()
        result = ("skipped: " + d["skipped"]) if d["skipped"] else (
            "PASS" if d["passed"] else "FAIL")
        lines.append(f"| {tier.key} | {tier.title} | {d['values_compared']} | "
                     f"{d['seconds']:.1f} s | {result} |")
    for tier in tiers:
        lines += ["", f"## Tier {tier.key}: {tier.title}", "", tier.method, ""]
        if tier.skipped:
            lines.append(f"Skipped: {tier.skipped}.")
            continue
        for item in tier.items:
            mark = "pass" if item["pass"] else "FAIL"
            detail = f" -- {item['detail']}" if item["detail"] else ""
            lines.append(f"- [{mark}] {item['check']} ({item['values_compared']} values){detail}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tiered verification of the archived results.")
    parser.add_argument("--hardware", action="store_true",
                        help="also run tier C (logic synthesis; needs yowasp-yosys)")
    parser.add_argument("--skip-trails", action="store_true",
                        help="skip tier B (MILP and exact trail search)")
    args = parser.parse_args(argv)
    tiers = [
        Tier("A", "Direct table enumeration",
             "Self-contained enumeration in verify_results.py (no code shared with src/) of "
             "every S-box metric, the archived DDT/LAT/BCT CSV files, the metric identities "
             "and the exact ideal-SAC count; plus the executable specification of "
             "Algorithm 2 (src/algorithm_spec.py)."),
        Tier("B", "MILP bounds and exact trail search",
             "src.cipher_analysis.analyze_cipher re-run: HiGHS MILP (branch-number bound) and "
             "Matsui-style branch and bound over the real DDT/LAT; compared leaf by leaf."),
        Tier("C", "Logic synthesis",
             "Yosys/ABC synthesis of every S-box on NanGate45 and SKY130 (Liberty files "
             "SHA-256 checked), gate-level evaluation on all 32 inputs, area, depth, delay, "
             "energy and leakage recomputed."),
        Tier("D", "Manuscript against the result files",
             "verify_latex.py: every number printed in the tables of paper.tex."),
    ]
    bodies = {"A": tier_a, "B": tier_b, "C": tier_c, "D": tier_d}
    for tier in tiers:
        if tier.key == "B" and args.skip_trails:
            tier.skipped = "--skip-trails"
            continue
        if tier.key == "C" and not args.hardware:
            tier.skipped = "run with --hardware"
            continue
        print(f"Tier {tier.key}: {tier.title} ...", flush=True)
        _timed(tier, lambda t=tier: bodies[t.key](t))
        print(f"  {'PASS' if tier.passed else 'FAIL'} ({tier.seconds:.1f} s)")
    info = manifest()
    report = {"tiers": {t.key: t.as_dict() for t in tiers}, **{k: info[k] for k in (
        "git_commit", "git_describe", "working_tree_clean", "versions")}}
    with open(os.path.join(OUTPUT, "verification_report.json"), "w", encoding="utf-8",
              newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(OUTPUT, "verification_report.md"), "w", encoding="utf-8",
              newline="\n") as handle:
        handle.write(render_markdown(tiers, info))
    with open(os.path.join(OUTPUT, "MANIFEST.json"), "w", encoding="utf-8",
              newline="\n") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
        handle.write("\n")
    ran = [t for t in tiers if t.skipped is None]
    return 0 if all(t.passed for t in ran) else 1


if __name__ == "__main__":
    sys.exit(main())
