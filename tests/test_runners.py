"""Tests of the two runners and of ``verify_latex.py``.

Covered here:

* ``main.py`` and ``run_optimized.py`` write *separate* results files under
  separate schema strings, so running one cannot destroy the sections the
  other recorded, and both stay deterministic apart from ``timings``;
* ``main.py``'s chaotic-map section is built from
  :func:`src.chaotic_maps.default_maps` and prints the same Lyapunov
  exponent it records, and its cipher-demo record is the one it prints;
* ``verify_latex.py``'s numeric check: it merges whichever results files
  exist, verifies the three tables that used to be uncovered
  (``tab:lyapunov``, ``tab:benchmarks``, ``tab:gate_count``), drives the
  ``tab:improvement`` wins recount from ``COMPARISON_ROWS`` /``judge`` of
  :mod:`src.comparative_analysis` (so no quantity is double-weighted), and
  reports an ERROR -- never a silent PASS -- for a renamed table label, a
  renamed row label, a missing results key, a missing results file and any
  registered check that would compare nothing;
* the checks added for the revision (``tab:active_sboxes``,
  ``tab:cipher_bounds``, ``tab:test_vectors``, ``tab:bct_spectrum``,
  ``tab:algebraic_immunity``, ``tab:sac_deviation``, ``tab:benchmark_env``).
  Those gates are exercised against a *self-contained synthetic* Springer
  Nature document built here from the runners' own results
  (:func:`synthetic_document`): a faithful table must pass, a doctored cell
  must mismatch and an absent table must produce one clear "table not found"
  error.  The synthetic document carries *only* the synthetic tables, so a
  table of the same name in ``paper.tex`` cannot shadow it -- ``paper.tex``
  prints the standard-scale sample sizes while these fixtures come from a
  ``--quick`` run, and an earlier version of this module compared the two by
  accident.  ``paper.tex`` itself is checked separately, as the finished
  manuscript that carries every table.

The runners are executed once per module with ``--quick --no-figures``
into a temporary directory (about twenty seconds in total, most of it the
MILP active-S-box chain of ``main.py``'s Section 7).
"""

import copy
import json
import os
import re
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as main_runner
import run_optimized
import verify_latex as vl
from src.chaotic_maps import default_maps
from src.comparative_analysis import COMPARISON_ROWS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_TEX = os.path.join(PROJECT_ROOT, "paper.tex")

#: Checks whose tables ``paper.tex`` did not carry before this revision.
#: The paper now carries all seven; the gates themselves are additionally
#: verified against :func:`synthetic_document`, whose cells are built from the
#: results dict so that a faithful table provably cannot mismatch.
REVISION_CHECKS = ("active_sboxes", "cipher_bounds", "test_vectors",
                   "bct_spectrum", "algebraic_immunity", "sac_deviation",
                   "benchmark_env")


# ------------------------------------------------------------
# A synthetic LaTeX document carrying the revision tables
# ------------------------------------------------------------
def _cell(value):
    """Render a results value the way a paper table would print it."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _tabular(label, rows):
    """A ``table``/``tabular`` block labelled ``label`` holding ``rows``."""
    width = max(len(row) for row in rows)
    body = "\\\\\n".join(" & ".join(_cell(cell) for cell in row) for row in rows)
    return ("\\begin{table}\n\\caption{synthetic}\n\\label{%s}\n"
            "\\begin{tabular}{%s}\n%s\\\\\n"
            "\\end{tabular}\n\\end{table}\n" % (label, "l" * width, body))


def revision_tables_tex(merged):
    """The seven revision tables, filled from a merged results dict.

    Everything printed comes out of ``merged`` itself, so the document is
    correct by construction: any mismatch the checks then report is a bug in
    ``verify_latex.py``, not a disagreement about the numbers.
    """
    cipher = merged["cipher_analysis"]
    metrics = merged["metrics"]
    blocks = []

    names = [name for name, _ in vl.ACTIVE_SBOX_BOUND_COLUMNS]
    by_round = {name: {int(row["rounds"]): row
                       for row in cipher["bounds"][name]["rows"]} for name in names}
    rows = [["r", "N_d", "N_l"] + [f"{name} log2 P" for name in names]]
    for rounds in vl.ACTIVE_SBOX_ROUNDS:
        first = by_round[names[0]][rounds]
        rows.append([rounds,
                     first["min_active_sboxes_differential"],
                     first["min_active_sboxes_linear"]]
                    + [by_round[name][rounds][key]
                       for name, key in vl.ACTIVE_SBOX_BOUND_COLUMNS])
    blocks.append(_tabular("tab:active_sboxes", rows))

    rows = [["Quantity", "Value"]]
    for label, path in vl.CIPHER_BOUND_ROWS.items():
        rows.append([label, vl.value_at(merged, f"{vl.CIPHER_PATH}.{path}")])
    blocks.append(_tabular("tab:cipher_bounds", rows))

    rows = [["S-box", "Key", "Plaintext", "Ciphertext"]]
    for name in vl.TEST_VECTOR_SBOXES:
        for vector in cipher["test_vectors"][name]:
            rows.append([name, vector["key_hex"], vector["plaintext_hex"],
                         vector["ciphertext_hex"]])
    blocks.append(_tabular("tab:test_vectors", rows))

    rows = [["S-box", "BU", "At max", "Mean", "Spectrum"]]
    for name in vl.REVISION_TABLE_SBOXES:
        m = metrics[name]
        spectrum = " ".join(f"{int(v)}:{int(c)}" for v, c in m["bct_spectrum"])
        rows.append([name] + [m[key] for key in vl.BCT_SPECTRUM_COLUMNS] + [spectrum])
    blocks.append(_tabular("tab:bct_spectrum", rows))

    for label, columns in (("tab:algebraic_immunity", vl.ALGEBRAIC_IMMUNITY_COLUMNS),
                           ("tab:sac_deviation", vl.SAC_DEVIATION_COLUMNS)):
        rows = [["S-box"] + list(columns)]
        rows += [[name] + [metrics[name][key] for key in columns]
                 for name in vl.REVISION_TABLE_SBOXES]
        blocks.append(_tabular(label, rows))

    rows = [["Item", "Value"]]
    for label, (path, _kind) in vl.BENCHMARK_ENV_ROWS.items():
        rows.append([label, vl.value_at(merged, path)])
    blocks.append(_tabular("tab:benchmark_env", rows))
    return "\n".join(blocks)


#: Front matter of the synthetic document.  It is a real Springer Nature
#: manuscript skeleton -- the ``sn-jnl``/``sn-nature`` class line and the SN
#: front matter, with the back matter and the BibTeX hook below -- so the
#: fixture cannot drift away from what ``verify_latex`` demands of a
#: manuscript; :func:`test_the_synthetic_document_satisfies_the_template_checks`
#: asserts exactly that.
SYNTHETIC_PREAMBLE = "\n".join((
    r"\documentclass[pdflatex,sn-nature]{sn-jnl}",
    r"\usepackage{graphicx}",
    r"\usepackage{multirow}",
    r"\usepackage{amsmath}",
    r"\usepackage{booktabs}",
    r"\usepackage{array}",
    r"\usepackage{algorithm}",
    r"\usepackage{algorithmic}",
    r"\usepackage{tikz}",
    "",
    r"\begin{document}",
    "",
    r"\title[Synthetic fixture]{Synthetic fixture carrying the revision tables}",
    "",
    r"\author*[1]{\fnm{[FIRST NAME]} \sur{[SURNAME]}}"
    r"\email{[CORRESPONDING AUTHOR EMAIL]}",
    r"\author[1]{\fnm{[SECOND FIRST NAME]} \sur{[SECOND SURNAME]}}"
    r"\email{[SECOND AUTHOR EMAIL]}",
    r"\affil*[1]{\orgdiv{[DEPARTMENT]}, \orgname{[INSTITUTION]}}",
    "",
    r"\abstract{This document is a test fixture, not a manuscript.  Every cell",
    r"of every table below is printed from the runners' own results dict, so a",
    r"faithful table cannot disagree with the results and any mismatch is a bug",
    r"in the verifier.}",
    "",
    r"\keywords{test fixture, S-box, verification}",
    "",
    r"\maketitle",
    "",
))

#: Back matter of the synthetic document.  ``verify_latex`` requires the five
#: Scientific Reports declarations and a ``\bibliography`` in the *body*; the
#: bodies point at ``paper.tex`` rather than declaring anything, because this
#: file is a fixture and must never read as a real declaration.
SYNTHETIC_BACKMATTER = "\n".join((
    "",
    r"\backmatter",
    "",
    r"\section*{Declarations}",
    "",
    r"\bmhead{Funding} Test fixture: see paper.tex for the manuscript's own"
    r" declarations.",
    "",
    r"\bmhead{Competing interests} Test fixture: see paper.tex.",
    "",
    r"\bmhead{Data availability} Test fixture: see paper.tex.",
    "",
    r"\bmhead{Code availability} Test fixture: see paper.tex.",
    "",
    r"\bmhead{Author contributions} Test fixture: see paper.tex.",
    "",
    r"\bibliography{paper}",
    "",
    r"\end{document}",
    "",
))


def synthetic_document(merged):
    """A self-contained Springer Nature document carrying only those tables.

    Self-contained on purpose.  ``paper.tex`` now carries all seven revision
    tables itself, and :func:`verify_latex.tabular_rows` reads the *first*
    ``tabular`` after a label, so appending these tables to the paper would
    leave every check reading the paper's copy: the paper prints the
    standard-scale sample sizes of the recorded run while ``merged`` comes
    from a ``--quick`` run, and five ``tab:benchmark_env`` rows disagreed for
    that reason alone.  With nothing but the synthetic tables in the document
    the checks can only see them.
    """
    return SYNTHETIC_PREAMBLE + revision_tables_tex(merged) + SYNTHETIC_BACKMATTER


# ------------------------------------------------------------
# Fixtures: one run of each runner into a temporary output dir
# ------------------------------------------------------------
@pytest.fixture(scope="module")
def runner_output(tmp_path_factory):
    """Run ``main.py`` twice and ``run_optimized.py`` once into a temp dir.

    Returns ``(output_dir, first_main_bytes)``: ``main.py`` runs first, then
    a second time (for the determinism assertion), then ``run_optimized.py``
    -- the order that used to overwrite ``main.py``'s sections.
    """
    output_dir = str(tmp_path_factory.mktemp("runner_output"))
    argv = ["--quick", "--no-figures", "--output-dir", output_dir]
    assert main_runner.main(argv) == 0
    with open(os.path.join(output_dir, main_runner.RESULTS_FILENAME), "rb") as handle:
        first_main = handle.read()
    assert main_runner.main(argv) == 0
    assert run_optimized.main(argv) == 0
    return output_dir, first_main


def _results_paths(output_dir):
    return [os.path.join(output_dir, main_runner.RESULTS_FILENAME),
            os.path.join(output_dir, run_optimized.RESULTS_FILENAME)]


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def merged(runner_output):
    output_dir, _ = runner_output
    return vl.merge_results([(path, _load(path)) for path in _results_paths(output_dir)])


@pytest.fixture(scope="module")
def tex_content():
    return vl.read_tex(PAPER_TEX)[0]


@pytest.fixture(scope="module")
def recorded_results():
    """The committed results files ``paper.tex`` was written from.

    The ``merged`` fixture comes from a ``--quick`` run, whose reduced sample
    sizes legitimately move the sample-dependent rows (the cipher avalanche
    ratio over 20 plaintexts rather than 100, the ``tab:benchmark_env`` counts).
    Assertions about what the *paper* prints therefore have to be made against
    the run the paper reports, which is what ``verify_latex`` itself checks by
    default.  A missing file here is a real failure: it means the paper's
    numbers can no longer be verified at all.
    """
    for path in vl.DEFAULT_RESULTS_FILES:
        assert os.path.exists(path), f"{path} is missing; run main.py / run_optimized.py"
    return vl.merge_results([(path, _load(path)) for path in vl.DEFAULT_RESULTS_FILES])


@pytest.fixture(scope="module")
def synthetic_tex(merged):
    """The self-contained synthetic document; see :func:`synthetic_document`."""
    return synthetic_document(merged)


# ------------------------------------------------------------
# F8: the two runners no longer share one file
# ------------------------------------------------------------
def test_runners_write_separate_files_under_separate_schemas(runner_output):
    output_dir, _ = runner_output
    main_path, optimized_path = _results_paths(output_dir)
    assert os.path.basename(main_path) != os.path.basename(optimized_path)
    assert main_runner.RESULTS_SCHEMA != run_optimized.RESULTS_SCHEMA
    from_main, from_optimized = _load(main_path), _load(optimized_path)
    assert from_main["schema"] == main_runner.RESULTS_SCHEMA
    assert from_optimized["schema"] == run_optimized.RESULTS_SCHEMA
    assert from_main["runner"] == "main.py"
    assert from_optimized["runner"] == "run_optimized.py"


def test_run_optimized_does_not_destroy_the_sections_of_main(runner_output):
    """``run_optimized.py`` ran last; ``main.py``'s sections must survive."""
    output_dir, _ = runner_output
    from_main = _load(os.path.join(output_dir, main_runner.RESULTS_FILENAME))
    for key in ("chaotic_maps", "boolean_functions", "cipher_demo",
                "cipher_analysis", "dynamic_sbox_demo"):
        assert key in from_main, key
    # The tent-logistic Lyapunov exponent -- the value that used to be lost.
    assert from_main["chaotic_maps"]["tent_logistic"]["lyapunov"] > 0.0
    from_optimized = _load(os.path.join(output_dir, run_optimized.RESULTS_FILENAME))
    assert "optimizer" in from_optimized
    assert "chaotic_maps" not in from_optimized


def test_results_are_deterministic_apart_from_timings(runner_output):
    output_dir, first_main = runner_output
    path = os.path.join(output_dir, main_runner.RESULTS_FILENAME)
    with open(path, "rb") as handle:
        second_main = handle.read()
    first, second = json.loads(first_main), json.loads(second_main)
    assert first.pop("timings") != {} and second.pop("timings") != {}
    assert first == second


def test_merged_results_carry_both_runners_sections(merged):
    for key in ("chaotic_maps", "cipher_demo", "cipher_analysis", "optimizer",
                "metrics", "tables", "benchmarks", "timings"):
        assert key in merged, key
    assert vl.value_at(merged, "chaotic_maps.tent_logistic.lyapunov") is not None
    assert vl.value_at(merged, "timings.benchmarks.sbox.table_lookup.lookups_per_sec")
    assert vl.value_at(merged, vl.GATE_EQUIVALENTS_PATH)["DAC"] == 11.25
    assert vl.value_at(merged, "nope.not.here") is None


# ------------------------------------------------------------
# F13 / F14: main.py's sections are single-sourced
# ------------------------------------------------------------
def test_chaotic_section_uses_default_maps_and_one_lyapunov_value(capsys):
    record = main_runner.run_chaotic_map_analysis()
    printed = capsys.readouterr().out
    maps = default_maps()
    assert set(record) == set(maps)
    for name, chaotic_map in maps.items():
        # One computation, reused: the recorded value is the map's own exponent
        # and it is the value the banner printed.
        assert record[name]["lyapunov"] == pytest.approx(chaotic_map.lyapunov_exponent())
        assert f"{record[name]['lyapunov']:.6f}" in printed
        # The parameter banner is derived from the instance, not hand-written.
        assert main_runner.map_parameters(chaotic_map) in printed
    assert "r_logistic=3.9" in main_runner.map_parameters(maps["tent_logistic"])
    assert main_runner.map_parameters(maps["tent"]) == "r=1.8, seed=0.1"


def test_cipher_demo_prints_the_record_it_returns(capsys):
    record = main_runner.run_cipher_demo()
    printed = capsys.readouterr().out
    assert record["plaintext"] == f"0x{main_runner.DEMO_PLAINTEXT:08X}"
    assert record["round_trip_ok"] is True
    # Console and results.json cannot disagree: both come from this record.
    assert record["ciphertext"] in printed
    assert f"0x{main_runner.DEMO_PLAINTEXT:08X} -> CT1: {record['ciphertext']}" in printed
    assert str(record["avalanche_bits_lsb_flip"]) in printed


# ------------------------------------------------------------
# F11: the wins recount has a single source
# ------------------------------------------------------------
def test_improvement_wins_are_judged_by_comparison_rows():
    scoring = {key: (direction, target)
               for _, key, direction, target in COMPARISON_ROWS}
    assert vl.COMPARISON_SCORING == scoring
    # Every judged row maps onto a key the library actually scores...
    assert set(vl.IMPROVEMENT_WIN_KEYS.values()) <= set(scoring)
    # ...each quantity exactly once (no DP / DP (max DDT) double weighting).
    keys = list(vl.IMPROVEMENT_WIN_KEYS.values())
    assert len(keys) == len(set(keys))
    assert "differential_uniformity" not in keys and "max_bias" not in keys
    assert "DP (max DDT)" not in vl.IMPROVEMENT_WIN_KEYS


def test_improvement_wins_are_recounted_or_the_table_is_reported_missing(tex_content,
                                                                            merged):
    """The recount happens, or the table's absence is a loud error.

    ``paper.tex`` is being rewritten, so this asserts the *gate*, not one
    particular tally: either ``tab:improvement`` is present and its ``Wins``
    row is recounted from ``COMPARISON_ROWS``, or the table is missing and
    that is reported.
    """
    check = vl.NumericCheck(tex_content, merged)
    check.improvement()
    if "tab:improvement" in check.missing_tables:
        assert any("tab:improvement: table not found" in item for item in check.errors)
        return
    tally = [item for item in check.mismatches if "/ Wins / tally:" in item]
    unread = [item for item in check.errors if "Wins" in item]
    assert len(tally) + len(unread) <= 1
    for item in tally:
        assert vl.mismatch_key(item) == "tab:improvement/Wins/tally"
        assert "judged metric rows" in item


def test_row_values_reads_booleans_and_numbers():
    assert vl.row_values("Yes", "No", "bool") == (True, False)
    assert vl.row_values("Yes", "maybe", "bool") is None
    assert vl.row_values("0.515", "0.530", "closer") == (0.515, 0.530)
    assert vl.row_values("5/5", "5/5", "higher") is None


# ------------------------------------------------------------
# F4: the numeric check is complete and never silently skips
# ------------------------------------------------------------
def test_every_registered_check_compares_values(tex_content, merged):
    """``paper.tex`` carries every table, so every check compares cleanly.

    Run against the manuscript itself: a check whose table the paper does not
    carry -- or carries under a renamed label -- must fail here.
    """
    check = vl.NumericCheck(tex_content, merged)
    check.run()
    assert check.errors == []
    assert check.compared > 200
    assert set(vl.NumericCheck.CHECKS) >= {"lyapunov", "benchmarks", "gate_count"}
    assert set(vl.NumericCheck.CHECKS) >= set(REVISION_CHECKS)
    # Every check has a label registered for the "table not found" report.
    assert set(vl.NumericCheck.CHECK_LABELS) == set(vl.NumericCheck.CHECKS)


@pytest.mark.parametrize("name", vl.NumericCheck.CHECKS)
def test_each_check_alone_compares_at_least_one_value(name, tex_content, merged):
    check = vl.NumericCheck(tex_content, merged)
    getattr(check, name)()
    assert check.compared > 0, f"{name} compared nothing"


def test_the_synthetic_document_satisfies_the_template_checks(synthetic_tex):
    """The fixture is a valid Springer Nature manuscript skeleton.

    Guards the fixture against the template: if ``verify_latex`` gains another
    structural requirement, the synthetic document must gain it too, so that
    the revision-table gates are exercised inside a document of the same shape
    as the one they police.  ``check_references`` is deliberately excluded --
    the fixture cites nothing, and BibTeX's uncited-entry rule is about
    ``paper.tex``, not about a table fixture.
    """
    errors, warnings = [], []
    lines = synthetic_tex.split("\n")
    vl.check_braces(synthetic_tex, errors)
    vl.check_environments(lines, errors)
    vl.check_document_structure(synthetic_tex, errors)
    vl.check_packages(synthetic_tex, warnings)
    vl.check_common_errors(synthetic_tex, lines, errors, warnings)
    vl.check_template(synthetic_tex, errors, warnings)
    assert errors == []
    assert [item for item in warnings if "Missing recommended package" in item] == []


def test_the_revision_tables_pass_when_they_are_faithful(synthetic_tex, merged):
    """The synthetic tables are built from the results, so nothing may differ.

    Every cell comes out of ``merged``; a mismatch here is a bug in
    ``verify_latex``, never a disagreement about a number.
    """
    check = vl.NumericCheck(synthetic_tex, merged)
    for name in REVISION_CHECKS:
        getattr(check, name)()
    assert check.errors == []
    assert check.mismatches == []
    assert check.compared > 80


@pytest.mark.parametrize("name", REVISION_CHECKS)
def test_a_revision_table_absent_from_the_paper_is_reported_not_a_crash(name, tex_content,
                                                                       merged):
    """The gate exists before the table does: one clear 'table not found'."""
    label = vl.NumericCheck.CHECK_LABELS[name]
    check = vl.NumericCheck(tex_content.replace("\\label{%s}" % label, ""), merged)
    getattr(check, name)()          # must not raise
    assert label in check.missing_tables
    assert any(item.startswith(f"{label}: table not found") for item in check.errors)
    # ...and run() does not pile a second, generic complaint on top of it.
    full = vl.NumericCheck(tex_content.replace("\\label{%s}" % label, ""), merged)
    full.run()
    assert not any(f"numeric check {name!r} compared 0 values" in item
                   for item in full.errors)


DOCTORED_CELLS = (
    ("active_sboxes", "9 & 9 &", "8 & 9 &"),
    ("cipher_bounds", "Birthday bound (blocks) & 65536", "Birthday bound (blocks) & 65535"),
    ("test_vectors", "DAC & 00000000000000000000 & 00000000 & 31C8AEDC",
     "DAC & 00000000000000000000 & 00000000 & 31C8AEDD"),
    ("bct_spectrum", "0:438 2:247", "0:438 2:248"),
    ("algebraic_immunity", "DAC & 2 & 3 & 2.4516", "DAC & 3 & 3 & 2.4516"),
    ("sac_deviation", "0.2500 & 0.0850", "0.2500 & 0.0851"),
    ("benchmark_env", "Logical cores & 16", "Logical cores & 15"),
)


@pytest.mark.parametrize("name, original, doctored", DOCTORED_CELLS)
def test_a_doctored_revision_cell_is_a_mismatch(name, original, doctored,
                                                synthetic_tex, merged):
    """Each revision check really compares: change one cell, get a mismatch."""
    if original not in synthetic_tex:
        pytest.skip(f"this machine does not print {original!r} in the synthetic table")
    check = vl.NumericCheck(synthetic_tex.replace(original, doctored, 1), merged)
    getattr(check, name)()
    assert check.mismatches, f"{name} did not notice a doctored cell"
    assert check.errors == []


@pytest.mark.parametrize("name", REVISION_CHECKS)
def test_a_missing_revision_results_key_is_an_error(name, synthetic_tex, merged):
    """Drop the results the check reads and it must complain, never pass."""
    stripped = {key: value for key, value in merged.items()
                if key not in ("cipher_analysis", "benchmarks")}
    stripped["metrics"] = {sbox: {"name": sbox} for sbox in merged["metrics"]}
    check = vl.NumericCheck(synthetic_tex, stripped)
    getattr(check, name)()
    assert check.errors, f"{name} passed with no results to check against"


def test_the_paper_is_gated_on_every_revision_table(tex_content, merged):
    """Whatever state ``paper.tex`` is in, each revision check reports itself.

    Either the table is there and values are compared, or its absence is an
    error.  A revision table that quietly checks nothing is impossible.
    """
    check = vl.NumericCheck(tex_content, merged)
    for name in REVISION_CHECKS:
        before, missing_before = check.compared, len(check.missing_tables)
        getattr(check, name)()
        compared = check.compared > before
        reported = len(check.missing_tables) > missing_before
        assert compared or reported, f"{name} neither compared nor reported anything"


#: The two defects these checks were built to catch, as ``(check, corrected
#: text in paper.tex, the withdrawn text, the mismatch key it must produce)``.
#: The withdrawn strings are the values of the *original* submission: the
#: tent-logistic Lyapunov exponent 3.8527, which no orbit of this repository
#: produces, and the avalanche ratio 0.5023, recomputed here as 0.5014.
WITHDRAWN_VALUES = (
    ("lyapunov",
     "Tent--Logistic ($r=8$) & 0.5657", "Tent--Logistic ($r=8$) & 3.8527",
     "tab:lyapunov/tent_logistic/lyapunov"),
    ("benchmarks",
     "Cipher avalanche ratio & 0.5014", "Cipher avalanche ratio & 0.5023",
     "tab:benchmarks/software/cipher_avalanche_ratio"),
)


def test_previously_unchecked_tables_are_now_compared(tex_content, recorded_results):
    """tab:lyapunov / tab:benchmarks / tab:gate_count reach results.json.

    These three tables used to be printed without any comparison against the
    runners, which is how the withdrawn tent-logistic exponent 3.8527 and the
    stale avalanche ratio 0.5023 survived into the original submission.  The
    revision corrected both, so the assertion is now that the checks *run* --
    each compares values -- and that the corrected paper agrees with the
    recorded run everywhere except the six wall-clock rows the paper itself
    declares machine dependent.  :func:`test_a_withdrawn_value_would_still_be_caught`
    re-inserts the old values to prove the gate still bites.
    """
    check = vl.NumericCheck(tex_content, recorded_results)
    compared = {}
    for name in ("lyapunov", "benchmarks", "gate_count"):
        before = check.compared
        getattr(check, name)()
        compared[name] = check.compared - before
    assert check.errors == []
    # Deleting one of the tables, or the results key behind it, would leave a
    # check comparing nothing -- which must never read as a pass.
    for name, count in compared.items():
        assert count > 0, f"{name} compared nothing"
    keys = {vl.mismatch_key(item) for item in check.mismatches}
    # The corrected exponent (0.5657) agrees with the computed orbit...
    assert not any(key.startswith("tab:lyapunov/") for key in keys), keys
    # ...the gate-equivalent table agrees with the recorded literature values...
    assert not any(key.startswith("tab:gate_count/") for key in keys), keys
    # ...and of tab:benchmarks exactly the wall-clock rows disagree.  The
    # avalanche ratio is deterministic, so the corrected 0.5014 must match:
    # it is the one row of that table the paper does not let --allow tolerate.
    assert keys == {f"tab:benchmarks/software/{vl.slug(row)}"
                    for row in vl.BENCHMARK_TIMING_ROWS}
    assert "tab:benchmarks/software/cipher_avalanche_ratio" not in keys


@pytest.mark.parametrize("name, corrected, withdrawn, key", WITHDRAWN_VALUES)
def test_a_withdrawn_value_would_still_be_caught(name, corrected, withdrawn, key,
                                                 tex_content, recorded_results):
    """Put the original submission's number back and the check must object.

    This is the regression these gates exist for: it fails if the row label,
    the table label or the results key the check reads is ever renamed away,
    because then re-inserting the bad value would no longer be noticed.
    """
    assert tex_content.count(corrected) == 1, corrected
    check = vl.NumericCheck(tex_content.replace(corrected, withdrawn, 1),
                            recorded_results)
    getattr(check, name)()
    assert check.errors == []
    assert key in {vl.mismatch_key(item) for item in check.mismatches}


def test_renamed_table_label_is_an_error_not_a_pass(tex_content, merged):
    for label in ("tab:lyapunov", "tab:benchmarks", "tab:gate_count",
                  "tab:improvement", "tab:corrected_metrics"):
        broken = tex_content.replace("\\label{%s}" % label, "\\label{%s_renamed}" % label)
        check = vl.NumericCheck(broken, merged)
        check.run()
        assert any(item.startswith(f"{label}:") for item in check.errors), label


def test_renamed_row_label_is_an_error_not_a_pass(tex_content, merged):
    broken = tex_content.replace("Tent--Logistic ($r=8$)", "Hybrid map ($r=8$)")
    assert broken != tex_content
    check = vl.NumericCheck(broken, merged)
    check.run()
    assert any("Tent--Logistic (r=8)" in item for item in check.errors)
    # ...and the value is no longer silently absent from the comparison.
    assert not any("tent_logistic" in item for item in check.mismatches)


def test_missing_results_key_is_an_error(tex_content, merged):
    without_chaos = {key: value for key, value in merged.items() if key != "chaotic_maps"}
    check = vl.NumericCheck(tex_content, without_chaos)
    check.run()
    assert any("results / chaotic_maps" in item for item in check.errors)
    assert any("'lyapunov' compared 0 values" in item for item in check.errors)


def test_missing_sac_matrix_regex_is_an_error(tex_content, merged):
    broken = tex_content.replace("\\mathbf{P}_{\\text{DAC}}", "\\mathbf{Q}_{\\text{DAC}}")
    check = vl.NumericCheck(broken, merged)
    check.run()
    assert any(item.startswith("eq:sac_matrix:") for item in check.errors)
    assert any("'sac_matrix' compared 0 values" in item for item in check.errors)


def test_missing_metrics_section_is_an_error(tex_content, merged):
    check = vl.NumericCheck(tex_content, {"tables": merged["tables"]})
    assert any("results / metrics" in item for item in check.errors)


def test_missing_results_file_is_an_error_not_a_silent_pass(tex_content, tmp_path):
    errors, warnings = [], []
    mismatches = vl.check_numbers(tex_content, [str(tmp_path / "absent.json")],
                                  errors, warnings)
    assert mismatches == []
    assert any("verified NOTHING" in item for item in errors)
    assert warnings


def test_verify_exits_nonzero_without_results(tmp_path, capsys):
    exit_code = vl.main(["--tex", PAPER_TEX, "--results", str(tmp_path / "absent.json")])
    capsys.readouterr()
    assert exit_code == 1


# ------------------------------------------------------------
# The --allow whitelist still works per mismatch
# ------------------------------------------------------------
def test_allow_whitelists_each_mismatch_individually(runner_output, tex_content,
                                                     tmp_path, capsys):
    """``--allow`` silences exactly the mismatches it names, one by one.

    Run against a copy of ``paper.tex``: the manuscript carries every table,
    so the only reasons to exit non-zero are the mismatches themselves, and
    whitelisting all of them must reach 0.
    """
    output_dir, _ = runner_output
    tex_path = str(tmp_path / "paper_copy.tex")
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(tex_content)
    # Figures and the .bib are both resolved next to the .tex file, so place
    # empty stand-ins for the figures and copy the real database across.
    # Their absence is a structural error that has nothing to do with the
    # --allow semantics under test here.
    for figure in re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", tex_content):
        path = tmp_path / figure
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        path.write_bytes(b"")
    shutil.copy(os.path.join(PROJECT_ROOT, "paper.bib"), str(tmp_path / "paper.bib"))
    argv = ["--tex", tex_path]
    for path in _results_paths(output_dir):
        argv += ["--results", path]

    errors, _, mismatches = vl.verify(tex_path, _results_paths(output_dir))
    capsys.readouterr()
    assert errors == [], errors
    assert mismatches, "the paper currently disagrees with nothing -- check the gate"
    assert vl.main(argv) == 1
    capsys.readouterr()
    keys = [vl.mismatch_key(item) for item in mismatches]
    assert len(set(keys)) == len(keys)      # each is individually addressable
    allowed = []
    for key in keys:
        allowed += ["--allow", key]
    assert vl.main(argv + allowed) == 0
    capsys.readouterr()
    # Dropping one whitelist entry brings the failure back.
    assert vl.main(argv + allowed[2:]) == 1
    capsys.readouterr()


def test_slug_and_first_number_helpers():
    assert vl.slug("S-box lookup throughput") == "s_box_lookup_throughput"
    assert vl.slug("Encryption latency (per block)") == "encryption_latency_per_block"
    assert vl.first_number("15.45 M lookups/s") == "15.45"
    assert vl.first_number("0.5023 (ideal: 0.5)") == "0.5023"
    assert vl.first_number("Not reported") is None
    assert vl.tolerance(vl.first_number("15.45 M lookups/s")) == pytest.approx(0.005, abs=1e-8)


def test_merge_results_is_first_wins_and_recursive():
    first = {"a": 1, "shared": {"x": 1}, "only_first": True}
    second = {"a": 2, "shared": {"x": 2, "y": 3}, "only_second": True}
    merged = vl.merge_results([("first", first), ("second", second)])
    assert merged == {"a": 1, "shared": {"x": 1, "y": 3},
                      "only_first": True, "only_second": True}
    # The inputs are not mutated.
    assert first == {"a": 1, "shared": {"x": 1}, "only_first": True}
    assert copy.deepcopy(second) == second
