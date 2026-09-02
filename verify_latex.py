"""Structural verification of ``paper.tex`` and numeric cross-check of its tables.

Structural checks: brace balance, environment matching, document structure,
sections, formal elements, tables and figures, figure files, labels and
references, citations, packages, common errors and Springer Nature template
conformance.

The manuscript is a Springer Nature submission (Scientific Reports, a Nature
Portfolio journal), so :func:`check_template` requires
``\\documentclass[pdflatex,sn-nature]{sn-jnl}`` exactly -- a missing class, a
different class and a different reference-style option are separate errors --
and reports every IEEEtran leftover (``\\markboth``, ``\\IEEEPARstart``,
``\\IEEEkeywords``, ``\\IEEEauthorblock``, starred ``table*``/``figure*``
floats, ``\\columnwidth``, an embedded ``thebibliography``) as an error.
The reference list comes from BibTeX: ``\\bibliography{paper}`` plus
``paper.bib``, whose entries are read directly, so an unresolved ``\\cite``
and an entry that nothing cites are both errors.

Numeric check: every number printed in the metric tables of the paper
(``tab:corrected_metrics``, ``tab:comparison``, ``tab:improvement``,
``tab:nl_dac``, ``tab:nl_comparison``, ``tab:lyapunov``,
``tab:benchmarks``, ``tab:gate_count``), the tables added for the revision
(``tab:active_sboxes``, ``tab:cipher_bounds``, ``tab:test_vectors``,
``tab:bct_spectrum``, ``tab:algebraic_immunity``, ``tab:sac_deviation``,
``tab:benchmark_env``), the SAC matrix ``eq:sac_matrix``,
the lookup tables ``tab:dac_sbox`` / ``tab:edac_sbox`` /
``tab:opt_sbox`` and the bit permutation ``tab:perm`` is compared with
the computed values written by the runners: ``output/results.json``
(``main.py``) and ``output/results_optimized.json`` (``run_optimized.py``).
Whichever of the two exist are merged, first file wins on a conflicting
leaf; a value that is in neither file is an ERROR, not a silent skip.  The
paper's ``NL (min)`` is compared with the true S-box nonlinearity
``nl_sbox`` over all 31 components, so the E-DAC row (paper: 10, computed:
8) is reported as a mismatch; the paper's quoted LAP claim of 0.125 is
checked against ``lap_half``, the base paper's normalisation.  The ``Wins``
row of ``tab:improvement`` is recounted from the table's own metric rows,
judged by ``COMPARISON_ROWS`` / ``judge`` of
:mod:`src.comparative_analysis` -- the single source of the wins semantics,
so no quantity is double-weighted -- and must sum to the number of judged
metric rows.

No check may fail silently: a table label that matches no ``tabular``, a
spec row label that appears in no table row, a results key that no results
file provides, a regex that misses and any registered check that ends up
comparing zero values are all reported as ERRORS.  A check whose table is
simply *absent* from ``paper.tex`` -- the normal state while the paper is
being written -- reports one clear ``table not found`` error naming the
missing ``\\label`` instead of crashing or passing quietly, so the gate
exists before the table does.

The revision tables are registered by ``\\label`` name.  Their expected
shapes, for whoever writes them:

``tab:active_sboxes``
    Sixteen rows, one per round; columns ``r``, ``N_d``, ``N_l`` and then one
    ``log2 P`` column per S-box of :data:`ACTIVE_SBOX_BOUND_COLUMNS`.
``tab:cipher_bounds``
    One row per label of :data:`CIPHER_BOUND_ROWS`; the value is read from
    the second column.
``tab:test_vectors``
    Rows ``S-box & key & plaintext & ciphertext`` in hexadecimal; every row
    must match a recorded vector of ``cipher_analysis.test_vectors``.
``tab:bct_spectrum``
    Rows ``S-box & BU & entries at the maximum & mean non-trivial entry &
    spectrum``, the spectrum written as space- or comma-separated
    ``value:count`` pairs.
``tab:algebraic_immunity`` and ``tab:sac_deviation``
    One row per S-box; the columns are the metric keys of
    :data:`ALGEBRAIC_IMMUNITY_COLUMNS` resp. :data:`SAC_DEVIATION_COLUMNS`.
``tab:benchmark_env``
    Rows ``label & value`` over :data:`BENCHMARK_ENV_ROWS`; text rows are
    compared as text, numeric rows as numbers.

Importable without side effects.  ``python verify_latex.py [--tex PATH]
[--results PATH ...]`` prints the report and exits with status 1 when there
are structural errors, numeric-check errors or numeric mismatches.  A
documented, deliberate mismatch (such as the two E-DAC NL(S) disclosures,
or the machine-dependent rows of ``tab:benchmarks``) can be excluded from
the exit code with ``--allow WHERE/SBOX/KEY`` (repeatable); allowed
mismatches are still printed, and new regressions still fail.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.comparative_analysis import COMPARISON_ROWS, judge  # noqa: E402

DEFAULT_TEX = os.path.join(PROJECT_ROOT, "paper.tex")
#: Results files written by the two runners, in merge order (first wins).
DEFAULT_RESULTS_FILES: Tuple[str, ...] = (
    os.path.join(PROJECT_ROOT, "output", "results.json"),            # main.py
    os.path.join(PROJECT_ROOT, "output", "results_optimized.json"),  # run_optimized.py
)
#: Backwards-compatible single default (``main.py``'s file).
DEFAULT_RESULTS = DEFAULT_RESULTS_FILES[0]

#: Packages ``paper.tex`` must load itself.  ``hyperref`` is deliberately not
#: on this list: ``sn-jnl.cls`` loads and configures it, and loading it again
#: in the preamble is an option clash.
REQUIRED_PACKAGES = ("amsmath", "graphicx", "booktabs", "tikz", "array",
                     "algorithm", "algorithmic", "multirow")

#: Document class the manuscript must use, and the reference-style option it
#: must carry.  Scientific Reports is a Nature Portfolio journal, so the
#: Springer Nature class must be used with the ``sn-nature`` option.
REQUIRED_CLASS = "sn-jnl"
REQUIRED_CLASS_OPTION = "sn-nature"
#: Class options that would silently select a different reference style.
WRONG_CLASS_OPTIONS = ("sn-basic", "sn-mathphys-num", "sn-mathphys-ay", "sn-aps",
                       "sn-vancouver-num", "sn-vancouver-ay", "sn-apa", "sn-chicago")
#: IEEEtran leftovers that must not survive the Springer Nature conversion.
FORBIDDEN_MARKUP = (
    ("\\markboth", "\\markboth (prints an IEEE running head on every page)"),
    ("\\IEEEPARstart", "\\IEEEPARstart (IEEEtran drop cap)"),
    ("\\IEEEpeerreviewmaketitle", "\\IEEEpeerreviewmaketitle"),
    ("\\IEEEauthorblock", "\\IEEEauthorblockN / \\IEEEauthorblockA"),
    ("\\begin{IEEEkeywords}", "\\begin{IEEEkeywords} (use \\keywords{...})"),
    ("\\begin{table*}", "\\begin{table*} (sn-jnl is single column)"),
    ("\\begin{figure*}", "\\begin{figure*} (sn-jnl is single column)"),
    ("\\columnwidth", "\\columnwidth (sn-jnl is single column; use \\textwidth)"),
    ("\\begin{thebibliography}", "\\begin{thebibliography} "
                                 "(SN production requires BibTeX: \\bibliography{paper})"),
)
#: The .bib database the manuscript must cite, relative to the .tex file.
BIB_BASENAME = "paper"

# ------------------------------------------------------------------
# Numeric check specification: paper table -> results.json metrics keys
# ------------------------------------------------------------------
#: ``tab:corrected_metrics``: row label -> key checked in the ``Computed`` column (DAC).
CORRECTED_METRICS_ROWS: Dict[str, str] = {
    "SAC (average)": "sac_avg", "Hamming dist. (avg)": "hd_avg", "LAP": "lap", "DP": "dp",
    "BIC-SAC": "bic_sac_avg", "NL (min, via WHT)": "nl_sbox",
    "NL (avg, via WHT)": "nl_coord_avg", "Boomerang uniformity": "boomerang_uniformity",
    "Algebraic degree": "min_degree",
}
#: ``tab:comparison``: keys of the columns after the S-box name.
COMPARISON_COLUMNS: Tuple[str, ...] = (
    "nl_sbox", "nl_coord_avg", "sac_avg", "lap", "dp", "bic_sac_avg", "bic_nl_avg",
    "boomerang_uniformity", "min_degree",
)
#: ``tab:comparison`` / ``tab:nl_comparison``: row label -> S-box name in results.json.
ROW_SBOX_NAMES: Dict[str, str] = {
    "DAC (original)": "DAC", "E-DAC (proposed)": "E-DAC", "DAC": "DAC", "E-DAC": "E-DAC",
}
#: ``tab:improvement``: row label -> key, columns DAC and E-DAC.
IMPROVEMENT_ROWS: Dict[str, str] = {
    "DP": "dp", "DP (max DDT)": "differential_uniformity", "NL (min)": "nl_sbox",
    "NL (avg)": "nl_coord_avg", "SAC (avg)": "sac_avg", "SAC (min)": "sac_min",
    "Hd (avg)": "hd_avg", "LAP": "lap", "BIC-SAC": "bic_sac_avg", "BIC-NL": "bic_nl_avg",
    "Alg. degree": "min_degree", "Boomerang": "boomerang_uniformity",
}
#: Scoring of :data:`src.comparative_analysis.COMPARISON_ROWS`, the single
#: source of the wins semantics: metric key -> (direction, target).
COMPARISON_SCORING: Dict[str, Tuple[str, Optional[float]]] = {
    key: (direction, target) for _, key, direction, target in COMPARISON_ROWS}
#: ``tab:improvement``: row label -> the ``COMPARISON_ROWS`` key that judges it.
#: The ``Wins`` recount is driven entirely by this mapping plus
#: :data:`COMPARISON_SCORING`, so a quantity that ``COMPARISON_ROWS``
#: deliberately scores once is not scored twice here.  ``DP (max DDT)`` is
#: therefore absent (it is the integer twin of ``DP`` -- the library scores
#: ``dp`` and not ``differential_uniformity``), and so is ``Balanced funcs.``
#: (the paper prints ``5/5`` coordinates, the library counts balanced
#: components out of 31).  Rows the paper prints and this mapping does not
#: judge are reported alongside the recount rather than silently dropped.
IMPROVEMENT_WIN_KEYS: Dict[str, str] = {
    "Bijective": "is_bijective", "DP": "dp", "NL (min)": "nl_sbox",
    "NL (avg)": "nl_coord_avg", "SAC (avg)": "sac_avg", "SAC (min)": "sac_min",
    "Hd (avg)": "hd_avg", "LAP": "lap", "BIC-SAC": "bic_sac_avg",
    "BIC-NL": "bic_nl_avg", "Alg. degree": "min_degree",
    "Boomerang": "boomerang_uniformity",
}
#: Lookup-table labels -> S-box name, for tables with a single ``S(x)`` row.
SBOX_TABLES: Dict[str, str] = {"tab:dac_sbox": "DAC", "tab:edac_sbox": "E-DAC"}
#: ``tab:opt_sbox`` prints two tables in one tabular, one row label per S-box:
#: row label in the paper -> S-box name under ``tables`` in the results.
OPT_SBOX_ROWS: Dict[str, str] = {"DAC-OPT": "DAC-OPT", "GOLD": "DAC-OPT-GOLD"}
#: ``tab:perm``: the ``P(i)`` rows are the 32 entries of the bit permutation,
#: compared with ``cipher_analysis.round_function.perm_table``.
PERM_TABLE_PATH = "cipher_analysis.round_function.perm_table"
PERM_ROW_LABEL = "P(i)"
PERM_INDEX_LABEL = "i"
#: ``tab:lyapunov``: row label -> map name under ``chaotic_maps`` (main.py).
LYAPUNOV_ROWS: Dict[str, str] = {
    "Logistic (r=3.9)": "logistic",
    "Tent (r=1.8)": "tent",
    "Tent--Logistic (r=8)": "tent_logistic",
}
#: ``tab:benchmarks``: row label -> (dotted results path, factor).
#: ``computed = factor * value``; the factors convert the recorded SI units
#: (lookups/s, MB/s) into the units the paper prints (M lookups/s, KB/s).
BENCHMARK_ROWS: Dict[str, Tuple[str, float]] = {
    "S-box lookup throughput":
        ("timings.benchmarks.sbox.table_lookup.lookups_per_sec", 1e-6),
    "S-box lookup latency":
        ("timings.benchmarks.sbox.table_lookup.ns_per_lookup_min", 1.0),
    "Cipher encryption throughput": ("timings.benchmarks.cipher.enc.mb_per_sec", 1e3),
    "Cipher decryption throughput": ("timings.benchmarks.cipher.dec.mb_per_sec", 1e3),
    "Encryption latency (per block)":
        ("timings.benchmarks.cipher.enc.us_per_block_min", 1.0),
    "Key schedule generation":
        ("timings.benchmarks.key_schedule.us_per_key_schedule_min", 1.0),
    "Cipher avalanche ratio": ("benchmarks.avalanche.avg_avalanche_ratio", 1.0),
}
#: Rows of :data:`BENCHMARK_ROWS` that are wall-clock measurements: they
#: depend on the machine that ran the benchmark, so a mismatch there means
#: "the paper quotes another machine", not "the paper is wrong".  They are
#: still compared (and individually whitelistable via ``--allow``).
BENCHMARK_TIMING_ROWS = frozenset(BENCHMARK_ROWS) - {"Cipher avalanche ratio"}
#: ``tab:gate_count``: row label -> key of the paper-reported gate equivalents.
GATE_COUNT_ROWS: Dict[str, str] = {
    "DAC / E-DAC": "DAC", "Thakor et al.": "Thakor_et_al", "SKINNY": "SKINNY",
    "RECTANGLE": "RECTANGLE", "Hummingbird-2": "Hummingbird-2", "PRESENT": "PRESENT",
}
GATE_EQUIVALENTS_PATH = "benchmarks.memory.paper_reported.gate_equivalents"

# ---- revision tables: cipher-level analysis (main.py, cipher_analysis) ----
#: Root of the round-level analysis written by ``main.py``.
CIPHER_PATH = "cipher_analysis"
#: ``tab:active_sboxes``: the two active-S-box columns, in table order after
#: the round number.  Both are read from the per-round rows of the first
#: S-box of :data:`ACTIVE_SBOX_BOUND_COLUMNS` (the counts are a property of
#: the wiring and the branch number, so they are the same for every S-box of
#: the family -- a fact the check itself re-verifies).
ACTIVE_SBOX_COUNT_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("N_d", "min_active_sboxes_differential"),
    ("N_l", "min_active_sboxes_linear"),
)
#: ``tab:active_sboxes``: the remaining columns, one per S-box.
ACTIVE_SBOX_BOUND_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("DAC", "log2_max_characteristic_probability"),
    ("E-DAC", "log2_max_characteristic_probability"),
    ("DAC-OPT", "log2_max_characteristic_probability"),
)
#: Rounds ``tab:active_sboxes`` must tabulate (the cipher's own round count).
ACTIVE_SBOX_ROUNDS: Tuple[int, ...] = tuple(range(1, 17))

#: ``tab:cipher_bounds``: row label -> dotted path under ``cipher_analysis``.
#: Every one of these is a headline number of the revision's cipher-level
#: answer to reviewer 2 (major points 4 and 7) and reviewer 4.
CIPHER_BOUND_ROWS: Dict[str, str] = {
    "Differential branch number": "branch_numbers.DAC.differential",
    "Linear branch number": "branch_numbers.DAC.linear",
    "Min active S-boxes (16 rounds)": "findings.min_active_sboxes_at_cipher_rounds",
    "Free rounds (16 rounds)": "findings.free_rounds_at_cipher_rounds",
    "Max consecutive bypass rounds": "bypass_analysis.max_consecutive_free_rounds",
    "Log2 characteristic probability, DAC (16 rounds)":
        "bounds.DAC.at_cipher_rounds.log2_max_characteristic_probability",
    "Log2 characteristic probability, E-DAC (16 rounds)":
        "bounds.E-DAC.at_cipher_rounds.log2_max_characteristic_probability",
    "Log2 characteristic probability, DAC-OPT (16 rounds)":
        "bounds.DAC-OPT.at_cipher_rounds.log2_max_characteristic_probability",
    "Rounds to data limit, DAC differential": "bounds.DAC.rounds_to_data_limit.differential",
    "Rounds to data limit, DAC linear": "bounds.DAC.rounds_to_data_limit.linear",
    "Rounds to data limit, E-DAC differential": "bounds.E-DAC.rounds_to_data_limit.differential",
    "Rounds to data limit, E-DAC linear": "bounds.E-DAC.rounds_to_data_limit.linear",
    "Rounds to data limit, DAC-OPT differential":
        "bounds.DAC-OPT.rounds_to_data_limit.differential",
    "Rounds to data limit, DAC-OPT linear": "bounds.DAC-OPT.rounds_to_data_limit.linear",
    "Birthday bound (blocks)": "birthday_bound.birthday_blocks",
    "Birthday bound (log2 blocks)": "birthday_bound.log2_birthday_blocks",
}
#: ``tab:test_vectors``: S-boxes whose vectors the table must carry.
TEST_VECTOR_SBOXES: Tuple[str, ...] = ("DAC", "E-DAC", "DAC-OPT")

# ---- revision tables: per-S-box metric columns (both runners, metrics) ----
#: ``tab:bct_spectrum``: numeric columns after the S-box name.  The fifth
#: column is the spectrum itself and is parsed separately.
BCT_SPECTRUM_COLUMNS: Tuple[str, ...] = (
    "boomerang_uniformity", "bct_max_count", "bct_mean_nontrivial")
#: ``tab:algebraic_immunity``: numeric columns after the S-box name.
ALGEBRAIC_IMMUNITY_COLUMNS: Tuple[str, ...] = ("ai_min", "ai_max", "ai_avg")
#: ``tab:sac_deviation``: numeric columns after the S-box name.  ``sac_d_max``
#: and ``sac_d_mean`` are the reviewer's D_max / D_mean notation.
SAC_DEVIATION_COLUMNS: Tuple[str, ...] = (
    "sac_avg", "sac_min", "sac_max", "sac_d_max", "sac_d_mean")
#: S-boxes every per-S-box revision table must carry a row for.
REVISION_TABLE_SBOXES: Tuple[str, ...] = ("DAC", "E-DAC", "DAC-OPT")
#: Row label -> S-box name for the revision tables (a superset of
#: :data:`ROW_SBOX_NAMES`, which only covers the two-row paper tables).
REVISION_ROW_SBOX_NAMES: Dict[str, str] = {
    "DAC": "DAC", "DAC (original)": "DAC", "E-DAC": "E-DAC",
    "E-DAC (proposed)": "E-DAC", "DAC-OPT": "DAC-OPT",
    "DAC-OPT-GOLD": "DAC-OPT-GOLD", "ASCON": "ASCON",
    "Keccak-chi": "Keccak-chi", "Keccak-": "Keccak-chi", "PRIMATE": "PRIMATE",
}

# ---- revision table: benchmark methodology (both runners, benchmarks) ----
#: ``tab:benchmark_env``: row label -> (dotted results path, kind).  ``kind``
#: is ``"text"`` (compared case-insensitively after whitespace collapsing) or
#: ``"number"``.  Answers reviewer 2's additional point 4 and reviewer 4's
#: request that the Python throughput be labelled and reproducible.
BENCHMARK_ENV_ROWS: Dict[str, Tuple[str, str]] = {
    "Processor": ("benchmarks.methodology.cpu_model", "text"),
    "Architecture": ("benchmarks.methodology.cpu_architecture", "text"),
    "Physical cores": ("benchmarks.methodology.physical_cores", "number"),
    "Logical cores": ("benchmarks.methodology.logical_cores", "number"),
    "Operating system": ("benchmarks.methodology.os_platform", "text"),
    "Python": ("benchmarks.methodology.python_version", "text"),
    "NumPy": ("benchmarks.methodology.numpy_version", "text"),
    "Timer": ("benchmarks.methodology.timer", "text"),
    "Repetitions": ("benchmarks.sizes.repeat", "number"),
    "S-box lookups timed": ("benchmarks.sizes.n_lookups", "number"),
    "Cipher blocks timed": ("benchmarks.sizes.n_blocks", "number"),
    "Key schedules timed": ("benchmarks.sizes.n_keys", "number"),
    "Avalanche plaintexts": ("benchmarks.sizes.n_avalanche_plaintexts", "number"),
}


# ------------------------------------------------------------------
# LaTeX helpers
# ------------------------------------------------------------------
def read_tex(path: str) -> Tuple[str, List[str]]:
    """Return the file content and its lines."""
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    return content, content.split("\n")


#: An unescaped ``%`` and everything after it on the same line.
COMMENT_RE = re.compile(r"(?<!\\)%.*")


def strip_comments(content: str) -> str:
    """``content`` with LaTeX comments blanked out.

    Used by the template checks so that a comment *explaining* why a command
    must not be used (``%% Do NOT add \\bibliographystyle{sn-nature}``) is not
    mistaken for the command itself.  Lines are kept, only their commented
    tail is removed, so reported line numbers stay meaningful.
    """
    return "\n".join(COMMENT_RE.sub("", line) for line in content.split("\n"))


def clean_cell(cell: str) -> str:
    """Plain text of a table cell: citations, bold, math and commands removed."""
    cell = re.sub(r"~?\\cite\{[^}]*\}", "", cell)
    for _ in range(3):   # unwrap nested \textbf{...} / \text{...}
        cell = re.sub(r"\\(?:textbf|text|mathbf|emph)\{([^{}]*)\}", r"\1", cell)
    cell = cell.replace("$", "").replace("\\%", "%").replace("---", "")
    cell = re.sub(r"\\[A-Za-z]+", " ", cell)   # \beta, \midrule, \checkmark, \uparrow ...
    cell = re.sub(r"\\[ ,;!]", " ", cell)       # control spaces such as "dist.\ (avg)"
    return " ".join(cell.split())


def tabular_rows(content: str, label: str) -> List[List[str]]:
    """Cleaned cell rows of the ``tabular`` that follows ``\\label{label}``."""
    match = re.search(r"\\label\{" + re.escape(label) + r"\}.*?\\begin\{tabular\}\{[^}]*\}(.*?)"
                      r"\\end\{tabular\}", content, re.DOTALL)
    if match is None:
        return []
    rows = []
    for raw in match.group(1).split("\\\\"):
        if "&" not in raw or "\\multicolumn" in raw:
            continue
        rows.append([clean_cell(cell) for cell in raw.split("&")])
    return rows


def parse_number(text: str) -> Optional[float]:
    """``float`` of a cleaned cell, ``None`` for non-numeric cells."""
    try:
        return float(text)
    except ValueError:
        return None


def tolerance(text: str) -> float:
    """Half a unit in the last printed decimal place (plus rounding slack)."""
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * 10.0 ** (-decimals) + 1e-9


#: First numeric token of a cell such as ``15.45 M lookups/s``.
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def first_number(text: str) -> Optional[str]:
    """The first numeric token of a cell, or ``None`` when there is none.

    Returned as text so :func:`tolerance` can read its printed precision:
    ``"0.5023 (ideal: 0.5)"`` -> ``"0.5023"``.
    """
    match = NUMBER_RE.search(text)
    return match.group() if match else None


def value_at(results: Mapping[str, Any], path: str) -> Any:
    """``results`` walked along a dotted ``path``; ``None`` when absent."""
    node: Any = results
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


# ------------------------------------------------------------------
# Structural checks
# ------------------------------------------------------------------
def check_braces(content: str, errors: List[str]) -> None:
    open_b, close_b = content.count("{"), content.count("}")
    print("\n--- Brace Balance ---")
    print(f"  Open braces:  {open_b}")
    print(f"  Close braces: {close_b}")
    if open_b == close_b:
        print("  Status: PASS (balanced)")
    else:
        errors.append(f"Brace imbalance: {open_b} open vs {close_b} close")
        print(f"  Status: FAIL (imbalanced by {abs(open_b - close_b)})")


def check_environments(lines: Sequence[str], errors: List[str]) -> None:
    print("\n--- Environment Matching ---")
    stack: List[Tuple[str, int]] = []
    counts: Dict[str, int] = {}
    for number, line in enumerate(lines, 1):
        for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
            action, name = match.group(1), match.group(2)
            if action == "begin":
                stack.append((name, number))
                counts[name] = counts.get(name, 0) + 1
            elif stack and stack[-1][0] == name:
                stack.pop()
            elif stack:
                errors.append(f"Line {number}: \\end{{{name}}} expected "
                              f"\\end{{{stack[-1][0]}}} (opened line {stack[-1][1]})")
            else:
                errors.append(f"Line {number}: \\end{{{name}}} without matching \\begin")
    if stack:
        for name, number in stack:
            errors.append(f"Unclosed environment: {name} opened at line {number}")
        print(f"  Status: FAIL ({len(stack)} unclosed)")
    else:
        print("  Status: PASS (all matched)")
    print("  Environments used:")
    for name, count in sorted(counts.items()):
        print(f"    {name}: {count}")


def check_document_structure(content: str, errors: List[str]) -> None:
    """Springer Nature front matter, back matter and BibTeX hook.

    ``\\title`` carries a short-title option, ``\\author`` is starred and
    numbered (``\\author*[1]{...}``), the abstract is a *command* and not an
    environment, and the reference list comes from BibTeX rather than an
    embedded ``thebibliography``.
    """
    print("\n--- Document Structure ---")
    checks = (
        ("\\begin{document}", "\\begin{document}" in content),
        ("\\end{document}", "\\end{document}" in content),
        ("\\title", bool(re.search(r"\\title(\[[^\]]*\])?\{", content))),
        ("\\author", bool(re.search(r"\\author\*?(\[[^\]]*\])?\{", content))),
        ("\\affil", bool(re.search(r"\\affil\*?(\[[^\]]*\])?\{", content))),
        ("\\abstract", "\\abstract{" in content),
        ("\\keywords", "\\keywords{" in content),
        ("\\maketitle", "\\maketitle" in content),
        ("\\backmatter", "\\backmatter" in content),
        ("bibliography", bool(re.search(r"\\bibliography\{[^}]*\}", content))),
    )
    for name, found in checks:
        print(f"  {name:20s}: {'PASS' if found else 'FAIL'}")
        if not found:
            errors.append(f"Missing required element: {name}")
    # A corresponding author and a second author: the reviewers address
    # "the authors", plural, and the response letter says so too.
    authors = len(re.findall(r"\\author\*?(?:\[[^\]]*\])?\{", content))
    corresponding = len(re.findall(r"\\author\*(?:\[[^\]]*\])?\{", content))
    print(f"  {'authors':20s}: {authors} ({corresponding} corresponding)")
    if corresponding != 1:
        errors.append(f"Exactly one \\author* (corresponding author) is required; "
                      f"found {corresponding}")
    if authors < 2:
        errors.append(f"At least two \\author entries are required; found {authors}")
    # Springer Nature back matter required by Scientific Reports.
    for heading in ("Data availability", "Code availability", "Competing interests",
                    "Author contributions", "Funding"):
        found = f"\\bmhead{{{heading}}}" in content
        print(f"  declaration {heading:24s}: {'PASS' if found else 'FAIL'}")
        if not found:
            errors.append(f"Missing SN declaration: \\bmhead{{{heading}}}")


def check_sections(content: str) -> Dict[str, List[str]]:
    print("\n--- Sections ---")
    found = {
        "sections": re.findall(r"\\section\{([^}]+)\}", content),
        "subsections": re.findall(r"\\subsection\{([^}]+)\}", content),
        "subsubsections": re.findall(r"\\subsubsection\{([^}]+)\}", content),
    }
    for kind, names in found.items():
        print(f"  {kind.capitalize()} ({len(names)}):")
        for name in names:
            print(f"    - {name}")
    return found


def count_elements(content: str) -> Dict[str, int]:
    """Counts of the formal environments, tables, figures and TikZ pictures."""
    counts = {name: len(re.findall(r"\\begin\{" + name + r"\}", content))
              for name in ("definition", "lemma", "theorem", "proposition", "remark",
                           "proof", "equation", "align", "algorithm", "tikzpicture")}
    counts["table"] = len(re.findall(r"\\begin\{table\*?\}", content))
    counts["figure"] = len(re.findall(r"\\begin\{figure\*?\}", content))
    return counts


def report_elements(counts: Mapping[str, int]) -> None:
    print("\n--- Mathematical/Scientific Elements ---")
    for name in ("definition", "lemma", "theorem", "proposition", "remark", "proof",
                 "equation", "align", "algorithm"):
        print(f"  {name.capitalize() + 's:':15s}{counts[name]}")
    formal = sum(counts[name] for name in ("definition", "lemma", "theorem", "proposition",
                                           "remark", "equation", "align"))
    print(f"  Total formal elements: {formal}")
    print("\n--- Tables and Figures ---")
    print(f"  Tables:      {counts['table']}")
    print(f"  Figures:     {counts['figure']}")
    print(f"  TikZ pics:   {counts['tikzpicture']}")


def check_figure_files(content: str, tex_dir: str, errors: List[str]) -> None:
    print("\n--- Figure Files ---")
    for figure in re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", content):
        exists = os.path.exists(os.path.join(tex_dir, figure))
        if not exists:
            errors.append(f"Missing figure: {figure}")
        print(f"  {figure}: {'FOUND' if exists else 'MISSING'}")


def bib_paths(content: str, tex_dir: str) -> List[str]:
    """``.bib`` files named by ``\\bibliography{...}``, resolved next to the .tex."""
    paths: List[str] = []
    for group in re.findall(r"\\bibliography\{([^}]+)\}", content):
        for name in group.split(","):
            name = name.strip()
            if not name:
                continue
            if not name.endswith(".bib"):
                name += ".bib"
            paths.append(os.path.join(tex_dir, name))
    return paths


def bib_keys(paths: Sequence[str], errors: List[str]) -> List[str]:
    """Citation keys defined in the ``.bib`` databases; a missing file is an ERROR."""
    keys: List[str] = []
    for path in paths:
        if not os.path.exists(path):
            errors.append(f"Missing bibliography database: {os.path.basename(path)} "
                          "(named by \\bibliography{} but not present)")
            print(f"  {os.path.basename(path)}: MISSING")
            continue
        with open(path, "r", encoding="utf-8") as handle:
            database = handle.read()
        found = re.findall(r"^@\w+\s*\{\s*([^,\s]+)\s*,", database, re.MULTILINE)
        keys.extend(found)
        print(f"  {os.path.basename(path)}: {len(found)} entries")
    return keys


def check_references(content: str, errors: List[str], warnings: List[str],
                     tex_dir: str = PROJECT_ROOT) -> Tuple[int, int]:
    """Labels vs references and citations vs the ``.bib``; returns (bib entries, cited).

    The bibliography now lives in ``paper.bib`` rather than in an embedded
    ``thebibliography``, so the citation keys are read from every database
    named by ``\\bibliography{...}``.  A ``.bib`` file that is named but
    absent, a ``\\cite`` that no entry defines and an entry that nothing
    cites are all reported.
    """
    print("\n--- Labels and References ---")
    labels = set(re.findall(r"\\label\{([^}]+)\}", content))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", content))
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", content)
             for key in group.split(",")}
    databases = bib_paths(content, tex_dir)
    if not databases:
        errors.append("No \\bibliography{...} in the paper: BibTeX has no database to "
                      "read, so no \\cite can resolve")
    bibkey_list = bib_keys(databases, errors)
    bibkeys = set(bibkey_list)
    duplicates = sorted({key for key in bibkey_list if bibkey_list.count(key) > 1})
    if duplicates:
        errors.append("Duplicate citation key(s) in the .bib: " + ", ".join(duplicates))

    print(f"  Labels defined:      {len(labels)}")
    print(f"  References used:     {len(refs)}")
    _report_set("Undefined references", sorted(refs - labels), errors, "Undefined reference")
    _report_set("Unused labels", sorted(labels - refs), warnings, "Unused label")
    print(f"\n  Citation keys used:     {len(cited)}")
    print(f"  Bibliography entries:   {len(bibkeys)}")
    _report_set("Undefined citations", sorted(cited - bibkeys), errors, "Undefined citation")
    # BibTeX drops an uncited entry silently, so an entry nothing cites is a
    # reference the authors believe they made and the PDF does not contain.
    _report_set("Uncited bibliography entries", sorted(bibkeys - cited), errors,
                "Uncited bibliography entry", is_error=True)
    return len(bibkeys), len(cited)


def _report_set(title: str, items: Sequence[str], sink: List[str], prefix: str,
                is_error: Optional[bool] = None) -> None:
    """Print ``items`` under ``title`` and append ``"prefix: item"`` to ``sink``.

    Items that go to the error list are tagged ``[ERROR]``, others ``[WARN]``.
    """
    if is_error is None:
        is_error = prefix.startswith("Undefined")
    if not items:
        print(f"  {title}: 0{' (PASS)' if is_error else ''}")
        return
    print(f"  {title} ({len(items)}):")
    for item in items:
        sink.append(f"{prefix}: {item}")
        print(f"    [{'ERROR' if is_error else 'WARN'}] {item}")


def check_packages(content: str, warnings: List[str]) -> None:
    print("\n--- Package Verification ---")
    loaded = [name.strip()
              for group in re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", content)
              for name in group.split(",")]
    for package in REQUIRED_PACKAGES:
        found = package in loaded
        print(f"  {package:15s}: {'FOUND' if found else 'MISSING'}")
        if not found:
            warnings.append(f"Missing recommended package: {package}")


def check_common_errors(content: str, lines: Sequence[str], errors: List[str],
                        warnings: List[str]) -> None:
    print("\n--- Common LaTeX Error Checks ---")
    double_dollar = content.count("$$")
    print(f"  Double dollar signs: {double_dollar} "
          f"{'(good)' if double_dollar == 0 else '(consider equation env)'}")
    # The bibliography is produced by BibTeX from paper.bib; \bibliography must
    # sit in the document body (after \backmatter), not in the preamble.
    match = re.search(r"\\bibliography\{[^}]*\}", content)
    if match is None:
        print("  \\bibliography placement: SKIPPED (no \\bibliography)")
    elif "\\begin{document}" not in content:
        print("  \\bibliography placement: SKIPPED (no \\begin{document})")
    elif match.start() < content.index("\\begin{document}"):
        errors.append("\\bibliography{} is in the preamble; it belongs in the document "
                      "body where the reference list is to appear")
        print("  \\bibliography placement: FAIL (in the preamble)")
    else:
        print("  \\bibliography placement: OK")
    for number, line in enumerate(lines, 1):
        if "\\being{" in line:
            errors.append(f"Line {number}: Probable typo \\being (should be \\begin)")
        if "\\ned{" in line:
            errors.append(f"Line {number}: Probable typo \\ned (should be \\end)")


def check_template(content: str, errors: List[str], warnings: List[str],
                   tex_dir: str = PROJECT_ROOT) -> None:
    """The manuscript must be the Springer Nature class in its Nature style.

    Replaces the old ``check_ieeetran``.  The class line is matched exactly:
    a missing ``\\documentclass``, a class other than ``sn-jnl`` (IEEEtran,
    elsarticle, article, ...) and ``sn-jnl`` carrying some other reference
    style option are each a separate ERROR, so the gate cannot be satisfied
    by a wrong class that merely happens to compile.  Every IEEEtran-only
    construct that the conversion removed is then checked to be gone.
    """
    print("\n--- Springer Nature Template ---")
    code = strip_comments(content)     # a comment about \markboth is not \markboth
    match = re.search(r"\\documentclass(?:\[([^\]]*)\])?\{([^}]+)\}", code)
    if match is None:
        errors.append("No \\documentclass in the paper")
        print("  \\documentclass:   MISSING (ERROR)")
        return
    options = [option.strip() for option in (match.group(1) or "").split(",") if option.strip()]
    cls = match.group(2).strip()
    print(f"  class:             {cls}")
    print(f"  options:           {', '.join(options) or '(none)'}")
    if cls != REQUIRED_CLASS:
        errors.append(f"Wrong document class {cls!r}: Scientific Reports requires "
                      f"{REQUIRED_CLASS!r} (the Springer Nature template class)")
    if REQUIRED_CLASS_OPTION not in options:
        errors.append(f"Missing class option {REQUIRED_CLASS_OPTION!r}: Scientific Reports "
                      "is a Nature Portfolio journal, so sn-jnl must be loaded as "
                      "\\documentclass[pdflatex,sn-nature]{sn-jnl}")
    for option in WRONG_CLASS_OPTIONS:
        if option in options:
            errors.append(f"Class option {option!r} selects a non-Nature reference style; "
                          f"use {REQUIRED_CLASS_OPTION!r}")
    print(f"  sn-nature option:  {'PASS' if REQUIRED_CLASS_OPTION in options else 'FAIL'}")

    print("  IEEEtran leftovers:")
    leftovers = 0
    for needle, description in FORBIDDEN_MARKUP:
        if needle in code:
            leftovers += 1
            errors.append(f"IEEEtran leftover in the Springer Nature manuscript: "
                          f"{description}")
            print(f"    [ERROR] {description}")
    if not leftovers:
        print("    none (PASS)")

    # sn-jnl.cls loads and configures hyperref itself; loading it again is an
    # option clash, and it also issues \bibliographystyle{sn-nature}, so a
    # second \bibliographystyle makes BibTeX report "Illegal, another
    # \bibstyle command".
    reloads_hyperref = bool(
        re.search(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\bhyperref\b[^}]*\}", code))
    if reloads_hyperref:
        errors.append("paper.tex loads hyperref, but sn-jnl.cls already loads and "
                      "configures it (option clash)")
    if "\\bibliographystyle" in code:
        errors.append("\\bibliographystyle in paper.tex: sn-jnl.cls already issues "
                      "\\bibliographystyle{sn-nature} for the sn-nature option, and a "
                      "second one makes BibTeX abort with 'Illegal, another \\bibstyle'")
    print(f"  hyperref not reloaded: {'FAIL' if reloads_hyperref else 'PASS'}")

    # Entries the authors still have to check must stay visible in the typeset
    # reference list; if none remain, prove they were completed, not dropped.
    verify_markers = 0
    for path in bib_paths(content, tex_dir):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                database = handle.read()
            verify_markers += len(re.findall(r"^\s*note\s*=.*\[VERIFY\]", database,
                                             re.MULTILINE))
    print(f"  [VERIFY] notes in the .bib: {verify_markers}")
    if verify_markers == 0:
        # A marker can vanish two ways: the entry was completed, or the marker was
        # quietly deleted while the entry stayed incomplete.  Distinguish them by
        # checking completeness directly, so "0 markers" can never mean "hidden".
        incomplete = []
        for path in bib_paths(content, tex_dir):
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                database = handle.read()
            for match in re.finditer(r"@(\w+)\{(\w+),(.*?)\n\}", database, re.S):
                key, body = match.group(2), match.group(3)
                if key in DOI_EXEMPT:
                    continue
                if not re.search(r"\n\s*doi\s*=", body, re.I):
                    incomplete.append(key)
        if incomplete:
            errors.append("No [VERIFY] note left in the .bib, but these entries are "
                          "still incomplete (no DOI): " + ", ".join(sorted(incomplete)))
        else:
            print("  every entry carries a DOI except the "
                  f"{len(DOI_EXEMPT)} that have never been assigned one "
                  f"({', '.join(sorted(DOI_EXEMPT))})")


# ------------------------------------------------------------------
# Numeric consistency against results.json
# ------------------------------------------------------------------
#: Bibliography keys that have genuinely never been assigned a DOI: the Keccak
#: reference document and a CAESAR competition submission.  Every other entry
#: must carry one, so a silently dropped [VERIFY] marker cannot hide.
DOI_EXEMPT: Tuple[str, ...] = ("ref_keccak", "ref_primate")


#: Truth values a boolean-judged table cell may print.
BOOLEAN_CELLS: Dict[str, bool] = {"yes": True, "no": False,
                                  "true": True, "false": False}


def row_values(first_cell: str, second_cell: str, direction: str
               ) -> Optional[Tuple[Any, Any]]:
    """The two comparable cell values, or ``None`` when a cell is unreadable."""
    if direction == "bool":
        first = BOOLEAN_CELLS.get(first_cell.strip().lower())
        second = BOOLEAN_CELLS.get(second_cell.strip().lower())
    else:
        first, second = parse_number(first_cell), parse_number(second_cell)
    if first is None or second is None:
        return None
    return first, second


def slug(label: str) -> str:
    """``--allow`` friendly identity of a row label: ``S-box lookup`` -> ``s_box_lookup``."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def normalise_text(text: str) -> str:
    """Case- and whitespace-insensitive form of a text cell, for text comparisons."""
    return " ".join(str(text).split()).casefold()


#: A ``value:count`` pair of a printed BCT spectrum cell.
SPECTRUM_PAIR_RE = re.compile(r"(\d+)\s*:\s*(\d+)")


def parse_spectrum(cell: str) -> Optional[List[List[int]]]:
    """``"0:438 2:247"`` -> ``[[0, 438], [2, 247]]``; ``None`` when no pair is found."""
    pairs = [[int(value), int(count)] for value, count in SPECTRUM_PAIR_RE.findall(cell)]
    return pairs or None


def parse_hex(cell: str) -> Optional[str]:
    """Upper-case hexadecimal digits of a cell, or ``None`` when it is not hex.

    Accepts ``0x``-prefixed, spaced and lower-case forms so that a paper may
    print ``0x31C8AEDC``, ``31c8aedc`` or ``31C8 AEDC`` for one value.
    """
    text = re.sub(r"\s+", "", cell).strip()
    if text.lower().startswith("0x"):
        text = text[2:]
    if text and all(character in "0123456789abcdefABCDEF" for character in text):
        return text.upper()
    return None


class NumericCheck:
    """Compare the numbers printed in the paper with the runners' results.

    Nothing is skipped quietly.  A table label that matches no ``tabular``,
    a spec row label that no table row carries, a results key that no
    results file provides, a regex that misses and a registered check that
    ends up comparing zero values all append to :attr:`errors`, which the
    driver treats exactly like a structural error.
    """

    #: Registered checks, in report order.  Each must compare >= 1 value.
    CHECKS: Tuple[str, ...] = ("corrected_metrics", "comparison", "improvement",
                               "nl_tables", "sac_matrix", "lookup_tables",
                               "opt_lookup_tables", "perm_table",
                               "lyapunov", "benchmarks", "gate_count",
                               "active_sboxes", "cipher_bounds", "test_vectors",
                               "bct_spectrum", "algebraic_immunity",
                               "sac_deviation", "benchmark_env")
    #: ``\\label`` name each check reads, for the "table not found" report.
    CHECK_LABELS: Dict[str, str] = {
        "corrected_metrics": "tab:corrected_metrics", "comparison": "tab:comparison",
        "improvement": "tab:improvement", "nl_tables": "tab:nl_dac / tab:nl_comparison",
        "sac_matrix": "eq:sac_matrix",
        "lookup_tables": "tab:dac_sbox / tab:edac_sbox",
        "opt_lookup_tables": "tab:opt_sbox", "perm_table": "tab:perm",
        "lyapunov": "tab:lyapunov", "benchmarks": "tab:benchmarks",
        "gate_count": "tab:gate_count", "active_sboxes": "tab:active_sboxes",
        "cipher_bounds": "tab:cipher_bounds", "test_vectors": "tab:test_vectors",
        "bct_spectrum": "tab:bct_spectrum",
        "algebraic_immunity": "tab:algebraic_immunity",
        "sac_deviation": "tab:sac_deviation", "benchmark_env": "tab:benchmark_env",
    }

    def __init__(self, content: str, results: Mapping[str, Any]) -> None:
        self.content = content
        self.results = results
        self.compared = 0
        self.mismatches: List[str] = []
        self.errors: List[str] = []
        #: ``\\label`` names for which no ``tabular`` was found, so that
        #: :meth:`run` can say "the table is absent" rather than repeating
        #: the generic "compared 0 values" complaint.
        self.missing_tables: List[str] = []
        self.metrics: Mapping[str, Mapping[str, Any]] = self._require("metrics") or {}
        self.tables: Mapping[str, Sequence[int]] = self._require("tables") or {}

    # -- access helpers -------------------------------------------------
    def _require(self, path: str) -> Any:
        """``results`` value at a dotted ``path``; a miss is an error, not a skip."""
        value = value_at(self.results, path)
        if value is None:
            self.errors.append(f"results / {path}: no results file provides this key "
                               "(run main.py and run_optimized.py)")
        return value

    def _rows(self, label: str) -> List[List[str]]:
        """Rows of the tabular labelled ``label``; an unmatched label is an error.

        The label is also recorded in :attr:`missing_tables` so that
        :meth:`run` reports "table not found" once, naming the ``\\label``,
        instead of adding a second, generic "compared 0 values" error for a
        table the paper simply does not have yet.
        """
        rows = tabular_rows(self.content, label)
        if not rows:
            self.missing_tables.append(label)
            self.errors.append(f"{label}: table not found -- no tabular follows "
                               f"\\label{{{label}}}, so nothing in it was checked "
                               "(add the table, or the label was renamed or moved)")
        return rows

    def _require_labels(self, where: str, expected: Sequence[str],
                        seen: Sequence[str]) -> None:
        missing = [label for label in expected if label not in set(seen)]
        if missing:
            self.errors.append(f"{where}: row label(s) present in the check specification but "
                               f"not in the table: {', '.join(repr(m) for m in missing)} -- "
                               "those values were not checked")

    # -- comparison primitives ------------------------------------------
    def _compare(self, where: str, sbox: str, key: str, text: str) -> None:
        """Record a mismatch when the paper value ``text`` differs from ``metrics[sbox][key]``."""
        paper = parse_number(text)
        if paper is None:
            return          # the paper prints no number here ("Not reported", "---")
        if sbox not in self.metrics:
            self.errors.append(f"{where} / {sbox}: no computed metrics for this S-box")
            return
        if key not in self.metrics[sbox]:
            self.errors.append(f"{where} / {sbox} / {key}: metric missing from the results")
            return
        computed = self.metrics[sbox][key]
        self.compared += 1
        if abs(paper - float(computed)) <= tolerance(text):
            return
        note = ""
        if key == "nl_sbox":
            note = (f" (minimum over the 5 coordinates nl_coord_min = "
                    f"{self.metrics[sbox]['nl_coord_min']}; NL(S) over all 31 components "
                    f"is {computed})")
        self.mismatches.append(f"{where} / {sbox} / {key}: paper {text}, computed {computed}{note}")

    def _compare_number(self, where: str, group: str, key: str, cell: str,
                        computed: Optional[float], note: str = "") -> None:
        """Compare the first number of ``cell`` with ``computed`` (both must exist)."""
        text = first_number(cell)
        if text is None:
            self.errors.append(f"{where} / {group} / {key}: cell {cell!r} holds no number")
            return
        if computed is None:
            self.errors.append(f"{where} / {group} / {key}: no computed value available")
            return
        self.compared += 1
        if abs(float(text) - float(computed)) > tolerance(text):
            self.mismatches.append(f"{where} / {group} / {key}: paper {text}, "
                                   f"computed {float(computed):.6g}{note}")

    # -- individual tables ----------------------------------------------
    def corrected_metrics(self) -> None:
        rows = self._rows("tab:corrected_metrics")
        seen = []
        for row in rows:
            key = CORRECTED_METRICS_ROWS.get(row[0])
            if key is None or len(row) < 3:
                continue
            seen.append(row[0])
            self._compare("tab:corrected_metrics (Computed)", "DAC", key, row[2])
            if key == "lap":   # the base paper's claim uses max_bias / 2^(n+1)
                self._compare("tab:corrected_metrics (Paper claim, lap_half)", "DAC",
                              "lap_half", row[1])
        if rows:
            self._require_labels("tab:corrected_metrics", list(CORRECTED_METRICS_ROWS), seen)

    def comparison(self) -> None:
        rows = self._rows("tab:comparison")
        seen = []
        for row in rows:
            sbox = ROW_SBOX_NAMES.get(row[0])
            if sbox is None:
                continue
            seen.append(sbox)
            for key, text in zip(COMPARISON_COLUMNS, row[1:]):
                self._compare("tab:comparison", sbox, key, text)
        if rows:
            self._require_labels("tab:comparison", ["DAC", "E-DAC"], seen)

    def improvement(self) -> None:
        rows = self._rows("tab:improvement")
        seen = []
        for row in rows:
            key = IMPROVEMENT_ROWS.get(row[0])
            if key is None or len(row) < 3:
                continue
            seen.append(row[0])
            self._compare("tab:improvement", "DAC", key, row[1])
            self._compare("tab:improvement", "E-DAC", key, row[2])
        if rows:
            self._require_labels("tab:improvement", list(IMPROVEMENT_ROWS), seen)
        self.improvement_wins(rows)

    def improvement_wins(self, rows: Sequence[Sequence[str]]) -> None:
        """Recount the ``Wins`` row of ``tab:improvement`` from its metric rows.

        The judging is delegated to :data:`COMPARISON_SCORING` and
        :func:`judge` of :mod:`src.comparative_analysis` -- the same single
        source the library's own wins tallies use, so no quantity is scored
        twice.  Rows the paper prints that the library does not judge (the
        integer twin ``DP (max DDT)``, the ``5/5`` ``Balanced funcs.`` row)
        are named in the mismatch instead of being silently absorbed.
        """
        printed: Optional[Tuple[int, ...]] = None
        tally = {"first": 0, "second": 0, "tie": 0}
        judged = 0
        unjudged: List[str] = []
        for row in rows:
            if len(row) < 3 or row[0] in ("", "Metric"):
                continue
            if row[0] == "Wins":
                numbers = [re.search(r"\d+", cell) for cell in row[1:4]]
                if len(numbers) == 3 and all(numbers):
                    printed = tuple(int(match.group()) for match in numbers)
                else:
                    self.errors.append("tab:improvement / Wins: the printed tally could not "
                                       f"be read from {list(row[1:4])}")
                continue
            key = IMPROVEMENT_WIN_KEYS.get(row[0])
            if key is None:
                unjudged.append(row[0])
                continue
            scoring = COMPARISON_SCORING.get(key)
            if scoring is None:
                self.errors.append(f"tab:improvement / {row[0]}: key {key!r} is no longer "
                                   "scored by COMPARISON_ROWS of src.comparative_analysis")
                continue
            direction, target = scoring
            values = row_values(row[1], row[2], direction)
            if values is None:
                self.errors.append(f"tab:improvement / {row[0]}: cells {row[1]!r} / {row[2]!r} "
                                   f"are not readable as '{direction}' values")
                continue
            _, sign = judge(values[0], values[1], direction, target)
            tally[{1: "second", -1: "first", 0: "tie"}[sign]] += 1
            judged += 1
        if judged == 0:
            return                       # run() reports the no-op
        self.compared += 1
        if printed is None:
            self.errors.append("tab:improvement: no Wins row found; the tally was not verified")
            return
        note = (f" [rows printed but not judged, because COMPARISON_ROWS does not score them: "
                f"{', '.join(unjudged)}]") if unjudged else ""
        computed = (tally["first"], tally["second"], tally["tie"])
        if printed != computed or sum(printed) != judged:
            self.mismatches.append(
                f"tab:improvement / Wins / tally: paper "
                f"{printed[0]}/{printed[1]}/{printed[2]} (sum {sum(printed)}), computed "
                f"{computed[0]}/{computed[1]}/{computed[2]} over {judged} judged metric "
                f"rows{note}")

    def _coordinate_nl(self, where: str, sbox: str, cells: Sequence[str]) -> None:
        if sbox not in self.metrics:
            self.errors.append(f"{where} / {sbox}: no computed metrics for this S-box")
            return
        per_bit = self.metrics[sbox]["nl_per_bit"]
        for i, (text, computed) in enumerate(zip(cells, per_bit)):
            paper = parse_number(text)
            if paper is None:
                continue
            self.compared += 1
            if abs(paper - computed) > tolerance(text):
                self.mismatches.append(f"{where} / {sbox} / nl_per_bit[{i}]: "
                                       f"paper {text}, computed {computed}")

    def nl_tables(self) -> None:
        dac_rows = self._rows("tab:nl_dac")
        matched = 0
        for row in dac_rows:
            if parse_number(row[0]) is None or len(row) < 6:
                continue
            matched += 1
            self._coordinate_nl("tab:nl_dac", "DAC", row[:5])
            summary = [part.strip() for part in row[5].split("/")]
            if len(summary) == 2:
                self._compare("tab:nl_dac (Min)", "DAC", "nl_coord_min", summary[0])
                self._compare("tab:nl_dac (Avg)", "DAC", "nl_coord_avg", summary[1])
            else:
                self.errors.append("tab:nl_dac: the 'Min / Avg' cell "
                                   f"{row[5]!r} is not a 'min / avg' pair")
        if dac_rows and matched == 0:
            self.errors.append("tab:nl_dac: no numeric coordinate row found")
        comparison_rows = self._rows("tab:nl_comparison")
        seen = []
        for row in comparison_rows:
            sbox = ROW_SBOX_NAMES.get(row[0])
            if sbox is None or len(row) < 7:
                continue
            seen.append(sbox)
            self._coordinate_nl("tab:nl_comparison", sbox, row[1:6])
            self._compare("tab:nl_comparison (Min of coordinates)", sbox, "nl_coord_min", row[6])
        if comparison_rows:
            self._require_labels("tab:nl_comparison", ["DAC", "E-DAC"], seen)

    def sac_matrix(self) -> None:
        match = re.search(r"\\mathbf\{P\}_\{\\text\{DAC\}\}\s*=\s*\\begin\{pmatrix\}(.*?)"
                          r"\\end\{pmatrix\}", self.content, re.DOTALL)
        if match is None:
            self.errors.append("eq:sac_matrix: the \\mathbf{P}_{\\text{DAC}} pmatrix was not "
                               "found; the SAC matrix was not checked")
            return
        if "DAC" not in self.metrics:
            self.errors.append("eq:sac_matrix / DAC: no computed metrics for this S-box")
            return
        computed = self.metrics["DAC"]["sac_matrix"]
        rows = [[cell.strip() for cell in line.split("&")]
                for line in match.group(1).split("\\\\") if "&" in line]
        if len(rows) != len(computed):
            self.errors.append(f"eq:sac_matrix: the paper prints {len(rows)} row(s), the "
                               f"computed matrix has {len(computed)}")
        for i, (row, computed_row) in enumerate(zip(rows, computed)):
            for j, (text, value) in enumerate(zip(row, computed_row)):
                paper = parse_number(text)
                if paper is None:
                    self.errors.append(f"eq:sac_matrix / DAC / sac_matrix[{i}][{j}]: "
                                       f"cell {text!r} holds no number")
                    continue
                self.compared += 1
                if abs(paper - value) > tolerance(text):
                    self.mismatches.append(f"eq:sac_matrix / DAC / sac_matrix[{i}][{j}]: "
                                           f"paper {text}, computed {value:.4f}")

    def lookup_tables(self) -> None:
        for label, sbox in SBOX_TABLES.items():
            values: List[int] = []
            for row in self._rows(label):
                if row[0] == "S(x)":
                    values.extend(int(cell) for cell in row[1:] if parse_number(cell) is not None)
            if not values:
                self.errors.append(f"{label}: no 'S(x)' row with numeric entries was found")
                continue
            if sbox not in self.tables:
                self.errors.append(f"{label} / {sbox}: no computed table for this S-box")
                continue
            self.compared += 1
            if values != list(self.tables[sbox]):
                self.mismatches.append(f"{label} / {sbox} / table: paper {values}, "
                                       f"computed {list(self.tables[sbox])}")

    def opt_lookup_tables(self) -> None:
        """``tab:opt_sbox`` vs the frozen ``tables`` for DAC-OPT and DAC-OPT-GOLD.

        The table prints both proposed S-boxes in one tabular, two blocks of
        sixteen columns, with one row per S-box in each block.  The rows of a
        given label are concatenated in document order, so a row that moved
        or a digit that changed is a mismatch, not a silent pass.
        """
        rows = self._rows("tab:opt_sbox")
        if not rows:
            return
        collected: Dict[str, List[int]] = {label: [] for label in OPT_SBOX_ROWS}
        for row in rows:
            if row[0] not in collected:
                continue
            collected[row[0]].extend(int(cell) for cell in row[1:]
                                     if parse_number(cell) is not None)
        self._require_labels("tab:opt_sbox", list(OPT_SBOX_ROWS),
                             [label for label, values in collected.items() if values])
        for label, sbox in OPT_SBOX_ROWS.items():
            values = collected[label]
            if not values:
                continue
            if sbox not in self.tables:
                self.errors.append(f"tab:opt_sbox / {sbox}: no computed table for this S-box")
                continue
            self.compared += 1
            if values != list(self.tables[sbox]):
                self.mismatches.append(f"tab:opt_sbox / {sbox} / table: paper {values}, "
                                       f"computed {list(self.tables[sbox])}")

    def perm_table(self) -> None:
        """``tab:perm`` vs ``cipher_analysis.round_function.perm_table``.

        All 32 entries are compared, and so are the 32 index values of the
        ``i`` rows, so a mis-ordered block cannot hide a wrong permutation.
        """
        computed = self._require(PERM_TABLE_PATH)
        rows = self._rows("tab:perm")
        if not rows:
            return
        indices: List[int] = []
        images: List[int] = []
        for row in rows:
            target = {PERM_INDEX_LABEL: indices, PERM_ROW_LABEL: images}.get(row[0])
            if target is None:
                continue
            target.extend(int(cell) for cell in row[1:] if parse_number(cell) is not None)
        seen = [label for label, values in ((PERM_INDEX_LABEL, indices),
                                            (PERM_ROW_LABEL, images)) if values]
        self._require_labels("tab:perm", [PERM_INDEX_LABEL, PERM_ROW_LABEL], seen)
        if not images:
            self.errors.append("tab:perm: no 'P(i)' row with numeric entries was found")
            return
        if computed is None:
            return          # _require already reported the missing results key
        if isinstance(computed, str) or not isinstance(computed, Sequence):
            self.errors.append(f"tab:perm: {PERM_TABLE_PATH} is not a list of indices")
            return
        expected = [int(value) for value in computed]
        if indices:
            self.compared += 1
            if indices != list(range(len(expected))):
                self.mismatches.append(
                    f"tab:perm / cipher / index row: paper {indices}, "
                    f"expected {list(range(len(expected)))}")
        self.compared += 1
        if images != expected:
            self.mismatches.append(f"tab:perm / cipher / perm_table: paper {images}, "
                                   f"computed {expected}")

    def lyapunov(self) -> None:
        """``tab:lyapunov`` vs ``chaotic_maps.*.lyapunov`` (written by main.py)."""
        exponents = self._require("chaotic_maps")
        rows = self._rows("tab:lyapunov")
        if not isinstance(exponents, Mapping):
            return
        seen = []
        for row in rows:
            name = LYAPUNOV_ROWS.get(row[0])
            if name is None or len(row) < 2:
                continue
            seen.append(row[0])
            record = exponents.get(name)
            if not isinstance(record, Mapping) or "lyapunov" not in record:
                self.errors.append(f"tab:lyapunov / {name}: chaotic_maps.{name}.lyapunov "
                                   "is missing from the results")
                continue
            self._compare_number("tab:lyapunov", name, "lyapunov", row[1], record["lyapunov"])
        if rows:
            self._require_labels("tab:lyapunov", list(LYAPUNOV_ROWS), seen)

    def benchmarks(self) -> None:
        """``tab:benchmarks`` vs the recorded benchmark values and timings."""
        rows = self._rows("tab:benchmarks")
        seen = []
        for row in rows:
            spec = BENCHMARK_ROWS.get(row[0])
            if spec is None or len(row) < 2:
                continue
            seen.append(row[0])
            path, factor = spec
            raw = value_at(self.results, path)
            if raw is None:
                self.errors.append(f"tab:benchmarks / {row[0]}: no results file provides "
                                   f"{path!r}")
                continue
            note = (" (wall-clock measurement: machine-dependent, not a paper error as such)"
                    if row[0] in BENCHMARK_TIMING_ROWS else "")
            self._compare_number("tab:benchmarks", "software", slug(row[0]), row[1],
                                 factor * float(raw), note)
        if rows:
            self._require_labels("tab:benchmarks", list(BENCHMARK_ROWS), seen)

    def gate_count(self) -> None:
        """``tab:gate_count`` vs the gate equivalents recorded in the results.

        These are the *base paper's* reported literature figures (kept in
        ``src.benchmarks.PAPER_REPORTED_GATE_EQUIVALENTS``, not synthesised
        by this code), so the check verifies that the paper's table and the
        code's copy of the same claim still agree.
        """
        reported = self._require(GATE_EQUIVALENTS_PATH)
        rows = self._rows("tab:gate_count")
        if not isinstance(reported, Mapping):
            return
        seen = []
        for row in rows:
            key = GATE_COUNT_ROWS.get(row[0])
            if key is None or len(row) < 3:
                continue
            seen.append(row[0])
            if key not in reported:
                self.errors.append(f"tab:gate_count / {key}: missing from "
                                   f"{GATE_EQUIVALENTS_PATH}")
                continue
            self._compare_number("tab:gate_count", "gate_equivalents", key, row[2],
                                 reported[key])
        if rows:
            self._require_labels("tab:gate_count", list(GATE_COUNT_ROWS), seen)

    # -- revision tables ------------------------------------------------
    def _cipher(self, path: str) -> Any:
        """Value at ``cipher_analysis.<path>``; a miss is an error, not a skip."""
        return self._require(f"{CIPHER_PATH}.{path}")

    def _sbox_metric_table(self, label: str, columns: Sequence[str]) -> None:
        """One row per S-box, ``columns`` read from the cells after the name.

        Shared by ``tab:bct_spectrum``, ``tab:algebraic_immunity`` and
        ``tab:sac_deviation``: every row whose first cell names a known S-box
        is compared column by column against ``metrics[name]``, and every
        S-box of :data:`REVISION_TABLE_SBOXES` must appear.
        """
        rows = self._rows(label)
        seen: List[str] = []
        for row in rows:
            sbox = REVISION_ROW_SBOX_NAMES.get(row[0])
            if sbox is None or len(row) < 1 + len(columns):
                continue
            seen.append(sbox)
            for key, text in zip(columns, row[1:]):
                self._compare(label, sbox, key, text)
        if rows:
            self._require_labels(label, list(REVISION_TABLE_SBOXES), seen)

    def active_sboxes(self) -> None:
        """``tab:active_sboxes`` vs ``cipher_analysis.bounds.<sbox>.rows``."""
        bounds = self._cipher("bounds")
        rows = self._rows("tab:active_sboxes")
        if not isinstance(bounds, Mapping):
            return
        names = [name for name, _ in ACTIVE_SBOX_BOUND_COLUMNS]
        missing = [name for name in names if not isinstance(bounds.get(name), Mapping)]
        if missing:
            self.errors.append("tab:active_sboxes: no computed bounds for "
                               f"{', '.join(missing)} in {CIPHER_PATH}.bounds")
            return
        by_round = {name: {int(entry["rounds"]): entry
                           for entry in bounds[name]["rows"]} for name in names}
        seen: List[str] = []
        for row in rows:
            rounds = parse_number(row[0])
            if rounds is None or len(row) < 3 + len(ACTIVE_SBOX_BOUND_COLUMNS):
                continue
            rounds = int(rounds)
            entry = by_round[names[0]].get(rounds)
            if entry is None:
                self.errors.append(f"tab:active_sboxes: the paper prints a row for r = "
                                   f"{rounds}, which {CIPHER_PATH} does not tabulate")
                continue
            seen.append(str(rounds))
            for offset, (label, key) in enumerate(ACTIVE_SBOX_COUNT_COLUMNS, start=1):
                self._compare_number("tab:active_sboxes", f"r={rounds}", label,
                                     row[offset], entry[key])
            offset = 1 + len(ACTIVE_SBOX_COUNT_COLUMNS)
            for index, (name, key) in enumerate(ACTIVE_SBOX_BOUND_COLUMNS):
                cell_round = by_round[name].get(rounds)
                if cell_round is None:
                    self.errors.append(f"tab:active_sboxes / {name}: no computed row for "
                                       f"r = {rounds}")
                    continue
                self._compare_number("tab:active_sboxes", f"r={rounds}",
                                     f"{name} log2P", row[offset + index], cell_round[key])
        if rows:
            self._require_labels("tab:active_sboxes",
                                 [str(r) for r in ACTIVE_SBOX_ROUNDS], seen)

    def cipher_bounds(self) -> None:
        """``tab:cipher_bounds`` vs the headline scalars of ``cipher_analysis``."""
        rows = self._rows("tab:cipher_bounds")
        if not rows:
            return
        seen: List[str] = []
        for row in rows:
            path = CIPHER_BOUND_ROWS.get(row[0])
            if path is None or len(row) < 2:
                continue
            seen.append(row[0])
            computed = value_at(self.results, f"{CIPHER_PATH}.{path}")
            if computed is None:
                self.errors.append(f"tab:cipher_bounds / {row[0]}: no results file provides "
                                   f"{CIPHER_PATH}.{path} (run main.py without "
                                   "--no-cipher-analysis)")
                continue
            self._compare_number("tab:cipher_bounds", "cipher", slug(row[0]), row[1],
                                 float(computed))
        self._require_labels("tab:cipher_bounds", list(CIPHER_BOUND_ROWS), seen)

    def test_vectors(self) -> None:
        """``tab:test_vectors`` vs ``cipher_analysis.test_vectors``.

        Each row names an S-box and prints key, plaintext and ciphertext in
        hexadecimal.  The (key, plaintext) pair selects the recorded vector
        and the ciphertext is compared with it, so a wrong ciphertext -- or a
        (key, plaintext) pair the pipeline never evaluated -- is an error, not
        an unnoticed row.
        """
        recorded = self._cipher("test_vectors")
        rows = self._rows("tab:test_vectors")
        if not isinstance(recorded, Mapping):
            return
        index: Dict[str, Dict[Tuple[str, str], str]] = {
            name: {(vector["key_hex"].upper(), vector["plaintext_hex"].upper()):
                   vector["ciphertext_hex"].upper() for vector in vectors}
            for name, vectors in recorded.items()}
        seen: List[str] = []
        for row in rows:
            sbox = REVISION_ROW_SBOX_NAMES.get(row[0])
            if sbox is None or len(row) < 4:
                continue
            key, plaintext, ciphertext = (parse_hex(row[1]), parse_hex(row[2]),
                                          parse_hex(row[3]))
            if key is None or plaintext is None or ciphertext is None:
                self.errors.append(f"tab:test_vectors / {sbox}: cells "
                                   f"{row[1]!r} / {row[2]!r} / {row[3]!r} are not all "
                                   "hexadecimal")
                continue
            if sbox not in index:
                self.errors.append(f"tab:test_vectors / {sbox}: no recorded test vectors "
                                   "for this S-box")
                continue
            seen.append(sbox)
            computed = index[sbox].get((key, plaintext))
            self.compared += 1
            if computed is None:
                self.errors.append(f"tab:test_vectors / {sbox}: the pipeline recorded no "
                                   f"vector for key {key} plaintext {plaintext}")
            elif computed != ciphertext:
                self.mismatches.append(f"tab:test_vectors / {sbox} / {key}_{plaintext}: "
                                       f"paper {ciphertext}, computed {computed}")
        if rows:
            self._require_labels("tab:test_vectors", list(TEST_VECTOR_SBOXES), seen)

    def bct_spectrum(self) -> None:
        """``tab:bct_spectrum`` vs the BCT columns and the full spectrum."""
        self._sbox_metric_table("tab:bct_spectrum", BCT_SPECTRUM_COLUMNS)
        for row in tabular_rows(self.content, "tab:bct_spectrum"):
            sbox = REVISION_ROW_SBOX_NAMES.get(row[0])
            if sbox is None or len(row) < 2 + len(BCT_SPECTRUM_COLUMNS):
                continue
            printed = parse_spectrum(row[1 + len(BCT_SPECTRUM_COLUMNS)])
            if printed is None:
                self.errors.append(f"tab:bct_spectrum / {sbox}: the spectrum cell "
                                   f"{row[1 + len(BCT_SPECTRUM_COLUMNS)]!r} holds no "
                                   "'value:count' pair")
                continue
            if sbox not in self.metrics or "bct_spectrum" not in self.metrics[sbox]:
                self.errors.append(f"tab:bct_spectrum / {sbox}: bct_spectrum is missing "
                                   "from the results")
                continue
            computed = [[int(value), int(count)]
                        for value, count in self.metrics[sbox]["bct_spectrum"]]
            self.compared += 1
            if printed != computed:
                self.mismatches.append(f"tab:bct_spectrum / {sbox} / bct_spectrum: "
                                       f"paper {printed}, computed {computed}")

    def algebraic_immunity(self) -> None:
        """``tab:algebraic_immunity`` vs the AI columns of ``metrics``."""
        self._sbox_metric_table("tab:algebraic_immunity", ALGEBRAIC_IMMUNITY_COLUMNS)

    def sac_deviation(self) -> None:
        """``tab:sac_deviation`` vs the SAC average and the D_max / D_mean columns."""
        self._sbox_metric_table("tab:sac_deviation", SAC_DEVIATION_COLUMNS)

    def benchmark_env(self) -> None:
        """``tab:benchmark_env`` vs the recorded benchmark methodology.

        Text rows are compared case- and whitespace-insensitively; numeric
        rows go through the usual printed-precision tolerance.  A value the
        runner could not determine (``None``, e.g. the physical core count
        without psutil) is an error rather than a pass: the paper must not
        print a number the pipeline never measured.
        """
        rows = self._rows("tab:benchmark_env")
        if not rows:
            return
        seen: List[str] = []
        for row in rows:
            spec = BENCHMARK_ENV_ROWS.get(row[0])
            if spec is None or len(row) < 2:
                continue
            path, kind = spec
            seen.append(row[0])
            computed = value_at(self.results, path)
            if computed is None:
                self.errors.append(f"tab:benchmark_env / {row[0]}: no results file provides "
                                   f"{path!r} (or the runner recorded no value for it)")
                continue
            if kind == "number":
                self._compare_number("tab:benchmark_env", "methodology", slug(row[0]),
                                     row[1], float(computed))
                continue
            self.compared += 1
            if normalise_text(row[1]) != normalise_text(computed):
                self.mismatches.append(f"tab:benchmark_env / methodology / {slug(row[0])}: "
                                       f"paper {row[1]!r}, computed {computed!r}")
        self._require_labels("tab:benchmark_env", list(BENCHMARK_ENV_ROWS), seen)

    def run(self) -> None:
        """Run every registered check; a check comparing nothing is an error.

        A check whose table is simply absent from the paper -- the normal
        state while the paper is still being written -- has already reported
        one clear ``table not found`` error naming the missing ``\\label``
        from :meth:`_rows`, and is not given a second, generic "silent no-op"
        complaint on top of it.  Every other way of comparing nothing (a
        renamed row label, a missing results key, a regex that missed) still
        is one.
        """
        for name in self.CHECKS:
            before = self.compared
            missing_before = len(self.missing_tables)
            getattr(self, name)()
            if self.compared != before or len(self.missing_tables) > missing_before:
                continue
            self.errors.append(f"numeric check {name!r} compared 0 values -- it is a "
                               "silent no-op (renamed label, moved table or missing "
                               f"results key), not a pass; it reads "
                               f"{self.CHECK_LABELS.get(name, name)}")


def load_results(path: str) -> Optional[Dict[str, Any]]:
    """The parsed results file or ``None`` when it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_results(loaded: Sequence[Tuple[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    """Recursively merge the results files in order; the first file wins a clash.

    ``main.py`` and ``run_optimized.py`` write different files with
    different sections (``chaotic_maps`` / ``cipher_demo`` here,
    ``optimizer`` there) over a shared core, so the union is what the paper
    has to be checked against.
    """
    merged: Dict[str, Any] = {}
    for _, results in loaded:
        _merge_into(merged, results)
    return merged


def _merge_into(target: Dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key not in target:
            target[key] = dict(value) if isinstance(value, Mapping) else value
        elif isinstance(target[key], dict) and isinstance(value, Mapping):
            _merge_into(target[key], value)
        # otherwise the earlier file keeps the value


def check_numbers(content: str, results_paths: Sequence[str], errors: List[str],
                  warnings: List[str]) -> List[str]:
    """Run :class:`NumericCheck` over the merged results; returns the mismatches.

    A missing results file is an error, never a silent pass: with nothing
    on disk the check verifies nothing and must not report success.
    """
    print("\n--- Numeric Consistency vs the runners' results files ---")
    loaded: List[Tuple[str, Mapping[str, Any]]] = []
    for path in results_paths:
        results = load_results(path)
        if results is None:
            print(f"  {path}: NOT FOUND")
            warnings.append(f"results file not found: {path}")
            continue
        loaded.append((path, results))
        print(f"  {path}: written by {results.get('runner', '?')} "
              f"(schema {results.get('schema', '?')})")
    if not loaded:
        errors.append("No results file found (" + ", ".join(results_paths) + "); the numeric "
                      "check verified NOTHING -- run main.py and run_optimized.py first")
        print("  No results file found; the numeric check verified NOTHING (ERROR).")
        return []
    check = NumericCheck(content, merge_results(loaded))
    check.run()
    print(f"  Values compared: {check.compared}")
    if check.errors:
        print(f"  Check errors ({len(check.errors)}):")
        for item in check.errors:
            print(f"    [ERROR] {item}")
        errors.extend(f"Numeric check: {item}" for item in check.errors)
    else:
        print("  Check errors: 0 (every registered check compared at least one value)")
    if check.mismatches:
        print(f"  Mismatches ({len(check.mismatches)}):")
        for item in check.mismatches:
            print(f"    [MISMATCH] {item}")
    else:
        print("  Mismatches: 0 (PASS)")
    return check.mismatches


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------
def verify(tex_path: str = DEFAULT_TEX,
           results_paths: Union[str, Sequence[str]] = DEFAULT_RESULTS_FILES
           ) -> Tuple[List[str], List[str], List[str]]:
    """Print the full report; returns ``(errors, warnings, mismatches)``.

    ``results_paths`` is the list of runner output files to merge (a single
    path is accepted for convenience).
    """
    if isinstance(results_paths, str):
        results_paths = (results_paths,)
    content, lines = read_tex(tex_path)
    errors: List[str] = []
    warnings: List[str] = []

    print("=" * 70)
    print("  LATEX STRUCTURAL VERIFICATION REPORT")
    print("=" * 70)
    print(f"  File: {tex_path}")
    print(f"  Total lines: {len(lines)}")
    print(f"  Total characters: {len(content)}")

    check_braces(content, errors)
    check_environments(lines, errors)
    check_document_structure(content, errors)
    sections = check_sections(content)
    counts = count_elements(content)
    report_elements(counts)
    tex_dir = os.path.dirname(os.path.abspath(tex_path))
    check_figure_files(content, tex_dir, errors)
    bib_entries, cited = check_references(content, errors, warnings, tex_dir)
    check_packages(content, warnings)
    check_common_errors(content, lines, errors, warnings)
    check_template(content, errors, warnings, tex_dir)
    mismatches = check_numbers(content, list(results_paths), errors, warnings)

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Lines:       {len(lines)}")
    print(f"  Sections:    {len(sections['sections'])}")
    print(f"  Subsections: {len(sections['subsections'])}")
    print(f"  Tables:      {counts['table']}")
    print(f"  Figures:     {counts['figure']}")
    print(f"  Equations:   {counts['equation']} + {counts['align']} align blocks")
    print(f"  Definitions: {counts['definition']}")
    print(f"  Lemmas:      {counts['lemma']}")
    print(f"  Theorems:    {counts['theorem']}")
    print(f"  Algorithms:  {counts['algorithm']}")
    print(f"  TikZ:        {counts['tikzpicture']}")
    print(f"  References:  {bib_entries} entries, {cited} cited")
    print()
    print(f"  Errors:   {len(errors)}")
    for item in errors:
        print(f"    [ERROR] {item}")
    print(f"  Warnings: {len(warnings)}")
    for item in warnings:
        print(f"    [WARN]  {item}")
    print(f"  Numeric mismatches vs the results files: {len(mismatches)}")
    for item in mismatches:
        print(f"    [MISMATCH] {item}")

    if errors:
        print(f"\n  VERDICT: {len(errors)} error(s) need fixing "
              "(structural and/or numeric-check).")
    else:
        print("\n  VERDICT: Paper structure is VALID")
        print("  The paper should compile with pdflatex + bibtex + sn-jnl.cls "
              "(sn-nature)")
    if mismatches:
        print(f"  {len(mismatches)} table value(s) disagree with the computed results.")
    print("=" * 70)
    return errors, warnings, mismatches


def mismatch_key(mismatch: str) -> str:
    """``WHERE/SBOX/KEY`` identity of a mismatch line (spaces removed)."""
    return mismatch.split(": paper ", 1)[0].replace(" ", "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify paper.tex structure and its numbers against results.json.")
    parser.add_argument("--tex", default=DEFAULT_TEX, help="LaTeX source (default: paper.tex)")
    parser.add_argument("--results", action="append", default=[], metavar="PATH",
                        help="results file written by main.py / run_optimized.py "
                             "(repeatable; they are merged, first wins). Default: "
                             + ", ".join(os.path.basename(p) for p in DEFAULT_RESULTS_FILES))
    parser.add_argument("--allow", action="append", default=[], metavar="WHERE/SBOX/KEY",
                        help="documented mismatch to tolerate in the exit code, e.g. "
                             "tab:comparison/E-DAC/nl_sbox (repeatable); tolerated "
                             "mismatches are still printed, new ones still fail")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the report; exit status 1 on errors or unexpected numeric mismatches.

    Errors include the structural ones, a numeric check that verified
    nothing and a missing results file.  Mismatches whose
    ``WHERE/SBOX/KEY`` identity is named by ``--allow`` are reported but do
    not affect the exit status.
    """
    args = build_parser().parse_args(argv)
    results_paths = args.results or list(DEFAULT_RESULTS_FILES)
    errors, _, mismatches = verify(args.tex, results_paths)
    allowed = {entry.replace(" ", "") for entry in args.allow}
    unexpected = [item for item in mismatches if mismatch_key(item) not in allowed]
    tolerated = len(mismatches) - len(unexpected)
    if tolerated:
        print(f"  ({tolerated} mismatch(es) tolerated via --allow, "
              f"{len(unexpected)} unexpected)")
    return 1 if errors or unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
