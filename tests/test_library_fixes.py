"""Regression tests for the library-side review findings F7, F9, F10, F12, F15.

Each test pins the *property* the fix established, not the incidental wording,
so the finding cannot come back silently:

F7  DDT/LAT heatmaps scale their colour range by the non-trivial entries and
    still show the trivial ``[0, 0]`` corner (clipped and annotated).
F9  The competitor table printed by :mod:`src.security_analysis` and the
    :data:`~src.literature.LITERATURE_REPORTED` dictionary are the same data:
    changing the dictionary changes the printed table.
F10 The base-paper summary judges only the base paper's own DAC table, and
    every row is compared in its own normalisation, so no row is stuck on
    ``CHECK``.
F12 One benchmark size table for both entry points.
F15 No caller can corrupt the frozen tables of :mod:`src.sbox_tables` by
    writing in place to a table handed out by :mod:`src.comparative_analysis`.

The last section covers the benchmark reproducibility metadata added for the
revision (reviewer 2, additional comment 4, and reviewer 4's "the
single-threaded Python throughput isn't a meaningful performance figure"):
every benchmark record must name the processor, the core counts, the OS, the
Python build, the numpy version, the number of operations timed, the amount of
data processed, the repetition count and the timing methodology, must report
both the minimum and the median, and must label itself illustrative -- without
renaming any pre-existing result key.
"""

from __future__ import annotations

import json
import os
import platform

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src import benchmarks, comparative_analysis, literature, visualization
from src.benchmarks import (BENCHMARK_REPEAT, BENCHMARK_SIZES,
                            CPU_REGISTRY_KEY, CPU_REGISTRY_VALUE,
                            ILLUSTRATIVE_NOTE, benchmark_avalanche_quality,
                            benchmark_block_cipher, benchmark_key_schedule,
                            benchmark_methodology, benchmark_sbox_lookup,
                            estimate_memory_footprint, platform_metadata,
                            run_all_benchmarks)
from src.comparative_analysis import (COMPARISON_ROWS, LITERATURE_REPORTED,
                                      all_sbox_tables, collect_benchmarks,
                                      frozen_table, judge, _judge)
from src.sbox import DAC_SBOX
from src.sbox_tables import DAC_OPT_SBOX, DAC_SBOX_REF, TABLES
from src.security_analysis import (analyze_sbox, competitor_table,
                                   compute_ddt, compute_lat,
                                   paper_summary_heading, paper_summary_table)


# ------------------------------------------------------------
# F7: heatmap colour scale
# ------------------------------------------------------------
def _axes():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(4, 4))
    FigureCanvasAgg(fig)
    return fig, fig.subplots()


def test_ddt_lat_heatmaps_scale_by_the_non_trivial_entries():
    """The trivial corner must not set the colour range (F7)."""
    ddt = compute_ddt(DAC_SBOX, 5)["table"]
    lat = abs(compute_lat(DAC_SBOX, 5)["table"])
    # The premise of the finding: the corner dwarfs everything informative.
    assert ddt[0, 0] == 32 and int(ddt[1:, 1:].max()) == 8
    assert lat[0, 0] == 16 and int(lat[1:, 1:].max()) == 8

    for table in (ddt, lat):
        fig, ax = _axes()
        image, vmax, corner = visualization._heatmap_scaled_by_nontrivial(ax, table)
        assert vmax == int(table[1:, 1:].max())
        assert corner == int(table[0, 0])
        assert image.get_clim() == (0, vmax)
        # The corner is kept, not dropped: it is outlined and annotated.
        assert any(f"= {corner}" in text.get_text() for text in ax.texts)
        assert ax.patches


def test_lat_heatmap_colorbar_agrees_with_the_lap_in_the_title(tmp_path):
    """The colorbar top must be the max bias the printed LAP is derived from."""
    lat = compute_lat(DAC_SBOX, 5)
    fig, ax = _axes()
    image, vmax, _ = visualization._heatmap_scaled_by_nontrivial(ax, abs(lat["table"]))
    assert vmax == lat["max_bias"]
    assert pytest.approx(lat["lap"]) == vmax / 32

    out = tmp_path / "lat.png"
    assert visualization.plot_lat_heatmap(DAC_SBOX, str(out), lat_result=lat) == str(out)
    assert out.stat().st_size > 0
    ddt_out = tmp_path / "ddt.png"
    assert visualization.plot_ddt_heatmap(DAC_SBOX, str(ddt_out)) == str(ddt_out)
    assert ddt_out.stat().st_size > 0


# ------------------------------------------------------------
# F9: one source for the literature values
# ------------------------------------------------------------
def test_literature_reported_is_one_object_everywhere():
    assert LITERATURE_REPORTED is literature.LITERATURE_REPORTED
    assert comparative_analysis.LITERATURE_COLUMNS is literature.LITERATURE_COLUMNS
    assert comparative_analysis.LITERATURE_SOURCE == literature.LITERATURE_SOURCE
    assert set(visualization.COMPETITOR_SAC) == set(literature.LITERATURE_REPORTED)


def test_printed_competitor_table_cannot_diverge_from_literature(monkeypatch):
    """The printed Table 8 is generated from LITERATURE_REPORTED (F9)."""
    flat = analyze_sbox(DAC_SBOX_REF, "DAC")
    text = competitor_table(flat, "DAC").get_string()
    for name, reported in literature.LITERATURE_REPORTED.items():
        assert name in text
        for key, _, fmt in literature.TABLE8_COLUMNS:
            assert format(reported[key], fmt) in text

    # Correcting a value in the single source changes the printed table.
    monkeypatch.setitem(literature.LITERATURE_REPORTED["ASCON"], "sac_avg", 0.99)
    assert "0.99" in competitor_table(flat, "DAC").get_string()


def test_paper_dac_table_matches_the_frozen_table():
    """The literature copy of paper Table 3 cannot drift from sbox_tables."""
    assert literature.PAPER_DAC_TABLE == DAC_SBOX_REF == list(DAC_SBOX)
    assert literature.is_paper_dac_table(DAC_SBOX_REF)
    assert not literature.is_paper_dac_table(DAC_OPT_SBOX)


# ------------------------------------------------------------
# F10: the paper summary is only a verdict for the paper's own S-box
# ------------------------------------------------------------
def test_paper_summary_judges_only_the_base_paper_table():
    dac = analyze_sbox(DAC_SBOX_REF, "DAC")
    table = paper_summary_table(dac, DAC_SBOX_REF, "DAC")
    assert table.field_names[-1] == "Match"
    # Every row is compared in its own normalisation, so nothing is stuck
    # on CHECK: the paper's own S-box reproduces the paper's own values.
    assert "CHECK" not in table.get_string()
    assert paper_summary_heading(DAC_SBOX_REF, "DAC").strip().startswith("SUMMARY vs BASE-PAPER")

    # run_optimized.py analyses the same table under a different name.
    assert paper_summary_heading(DAC_SBOX_REF, "DAC (Original)").strip().startswith(
        "SUMMARY vs BASE-PAPER")


def test_paper_summary_does_not_score_another_sbox_against_dac():
    opt = analyze_sbox(DAC_OPT_SBOX, "DAC-OPT")
    heading = paper_summary_heading(DAC_OPT_SBOX, "DAC-OPT")
    assert "reference column: DAC (base paper)" in heading
    assert "DAC-OPT" in heading

    table = paper_summary_table(opt, DAC_OPT_SBOX, "DAC-OPT")
    assert "Match" not in table.field_names
    assert table.field_names == ["Metric", "Computed (DAC-OPT)", "DAC (base paper)"]
    text = table.get_string()
    for verdict in ("PASS", "CHECK", "FAIL"):
        assert verdict not in text
    # The better S-box's real numbers are still shown.
    assert "0.0625" in text and "10" in text


def test_lap_row_uses_the_standard_normalisation():
    """The 'LAP = bias/2^n' row is compared against 0.25, not the paper's 0.125."""
    rows = {key: (label, value) for label, key, value, _ in literature.PAPER_DAC_SUMMARY_ROWS}
    assert rows["lap"][1] == 0.25
    assert rows["lap_half"][1] == 0.125
    assert rows["lap"][1] == 2 * rows["lap_half"][1]

    dac = analyze_sbox(DAC_SBOX_REF, "DAC")
    for _, key, value, tolerance in literature.PAPER_DAC_SUMMARY_ROWS:
        assert abs(dac[key] - value) < tolerance, key


# ------------------------------------------------------------
# F12: one benchmark size table
# ------------------------------------------------------------
def test_benchmark_sizes_have_a_single_source():
    assert comparative_analysis.BENCHMARK_SIZES is BENCHMARK_SIZES
    assert comparative_analysis.BENCHMARK_REPEAT is BENCHMARK_REPEAT
    assert set(BENCHMARK_SIZES) == {"standard", "quick"}
    quick, standard = BENCHMARK_SIZES["quick"], BENCHMARK_SIZES["standard"]
    assert len(quick) == len(standard) == 4
    assert all(q < s for q, s in zip(quick, standard))
    # The docstring's "about 10x less work" claim.
    assert standard[0] == 10 * quick[0]


def test_quick_means_the_same_work_from_both_entry_points(capsys):
    n_lookups, n_blocks, n_keys, n_avalanche = BENCHMARK_SIZES["quick"]
    sizes = collect_benchmarks(1, quick=True)["sizes"]
    assert sizes == {"n_lookups": n_lookups, "n_blocks": n_blocks,
                     "n_keys": n_keys, "n_avalanche_plaintexts": n_avalanche,
                     "repeat": BENCHMARK_REPEAT}
    capsys.readouterr()

    run_all_benchmarks(quick=True)
    printed = capsys.readouterr().out
    assert f"{n_lookups:,} random 5-bit inputs" in printed
    assert f"{n_blocks:,} random blocks" in printed
    assert f"over {n_keys:,} keys" in printed
    assert f"{n_avalanche:,} seeded random plaintexts" in printed


# ------------------------------------------------------------
# F15: the frozen tables cannot be corrupted through this module
# ------------------------------------------------------------
def test_frozen_tables_survive_in_place_writes_by_callers():
    handed_out = [frozen_table("DAC"), all_sbox_tables()["DAC"]]
    for table in handed_out:
        assert table == DAC_SBOX_REF
        assert table is not TABLES["DAC"].table
        table.sort()          # the destructive write of the finding
        table[0] = 99
    assert TABLES["DAC"].table == literature.PAPER_DAC_TABLE
    assert list(DAC_SBOX) == literature.PAPER_DAC_TABLE


def test_collect_benchmarks_never_hands_the_frozen_table_to_a_benchmark(monkeypatch):
    """Even a benchmark that sorted its argument could not corrupt TABLES."""
    seen: list[list[int]] = []

    def wrecker(table, *args, **kwargs):
        seen.append(table)
        assert table is not TABLES["DAC"].table
        table.sort()                      # in-place write on the caller's list
        return {}

    for name in ("benchmark_sbox_lookup", "benchmark_block_cipher",
                 "benchmark_key_schedule", "estimate_memory_footprint",
                 "benchmark_avalanche_quality"):
        monkeypatch.setattr(comparative_analysis, name, wrecker)

    collect_benchmarks(1, quick=True)
    assert seen and all(table == sorted(DAC_SBOX_REF) for table in seen)
    assert TABLES["DAC"].table == literature.PAPER_DAC_TABLE


# ------------------------------------------------------------
# Public names requested by the other agent's de-duplication
# ------------------------------------------------------------
def test_judge_and_comparison_rows_are_public():
    assert judge is _judge
    assert judge(8, 10, "higher", None) == ("+2", 1)
    assert judge(0.5, 0.53, "closer", 0.5)[1] == -1
    assert COMPARISON_ROWS and all(len(row) == 4 for row in COMPARISON_ROWS)


# ------------------------------------------------------------
# Benchmark reproducibility metadata (reviewer 2, additional comment 4)
# ------------------------------------------------------------
PLATFORM_KEYS = {
    "cpu_model", "cpu_model_source", "cpu_architecture", "logical_cores",
    "physical_cores", "physical_cores_source", "os", "os_release", "os_version",
    "os_platform", "python_version", "python_implementation", "python_compiler",
    "python_build", "numpy_version", "timer", "timer_resolution_ns",
    "timer_is_monotonic",
}


def test_platform_metadata_names_the_machine_and_the_stack():
    meta = platform_metadata()
    assert set(meta) == PLATFORM_KEYS
    # The processor model must be a real brand string, not platform.processor()'s
    # family/model/stepping soup, whenever the registry can be read.
    assert isinstance(meta["cpu_model"], str) and meta["cpu_model"]
    if meta["cpu_model_source"].startswith("winreg"):
        assert CPU_REGISTRY_KEY in meta["cpu_model_source"]
        assert CPU_REGISTRY_VALUE in meta["cpu_model_source"]
        assert meta["cpu_model"] != platform.processor()
    else:
        assert meta["cpu_model_source"] in {"/proc/cpuinfo", "platform.processor()",
                                            "unavailable"}
    assert meta["logical_cores"] == os.cpu_count()
    # Unknown values are None with an explicit source, never guessed.
    assert meta["physical_cores"] is None or meta["physical_cores"] >= 1
    assert (meta["physical_cores"] is None) == (meta["physical_cores_source"] == "unavailable")
    assert meta["os"] == platform.system() and meta["os_version"] == platform.version()
    assert meta["python_version"] == platform.python_version()
    assert meta["numpy_version"] == np.__version__
    assert "perf_counter" in meta["timer"] and meta["timer_resolution_ns"] > 0
    # A fresh dict every call: a caller cannot poison the cached probe.
    meta["cpu_model"] = "tampered"
    assert platform_metadata()["cpu_model"] != "tampered"


def test_benchmark_methodology_record_answers_the_reviewer_questions():
    record = benchmark_methodology("one lookup", 1234, "nanoseconds per lookup",
                                   repeat=7, bytes_processed=4936)
    assert record["n_operations_timed"] == 1234       # amount of work
    assert record["bytes_processed"] == 4936          # amount of processed data
    assert record["repetitions"] == 7                 # number of repetitions
    assert record["warmup_runs"] == 1
    assert "timeit.repeat" in record["timing_method"]  # timing methodology
    # Means or medians? Both min and median, and rates come from the minimum.
    assert "minimum" in record["statistics_reported"]
    assert "median" in record["statistics_reported"]
    assert record["rates_derived_from"] == "minimum"
    assert record["single_threaded"] and record["illustrative_only"]
    assert record["note"] == ILLUSTRATIVE_NOTE
    assert set(record["platform"]) == PLATFORM_KEYS

    # A derived, non-timed record carries no repetitions or timing claims.
    derived = benchmark_methodology("bit count", 1, "bits", timed=False)
    assert derived["timed"] is False
    assert "repetitions" not in derived and "illustrative_only" not in derived
    assert "bytes_processed" not in derived


def test_every_benchmark_record_carries_its_methodology():
    records = {
        "sbox": benchmark_sbox_lookup(DAC_SBOX_REF, n_lookups=500, repeat=2, seed=5),
        "cipher": benchmark_block_cipher(DAC_SBOX_REF, n_blocks=40, repeat=2, seed=5),
        "key_schedule": benchmark_key_schedule(DAC_SBOX_REF, n_keys=20, repeat=2, seed=5),
        "avalanche": benchmark_avalanche_quality(DAC_SBOX_REF, n_tests=3, seed=5),
        "memory": estimate_memory_footprint(DAC_SBOX_REF),
    }
    for name, record in records.items():
        methodology = record["methodology"]
        assert methodology["operation"] and methodology["unit"], name
        assert set(methodology["platform"]) == PLATFORM_KEYS, name
    # The timed records report the sample size they were actually given.
    assert records["sbox"]["methodology"]["n_operations_timed"] == 500
    assert records["sbox"]["methodology"]["repetitions"] == 2
    assert records["cipher"]["methodology"]["n_operations_timed"] == 40
    assert records["cipher"]["methodology"]["bytes_processed"] == 40 * 4
    assert records["key_schedule"]["methodology"]["repetitions"] == 2
    # ...and the derived ones say they are not timings.
    assert records["avalanche"]["methodology"]["timed"] is False
    assert records["memory"]["methodology"]["timed"] is False
    assert records["avalanche"]["methodology"]["n_operations_timed"] == 3 * 32


def test_metadata_does_not_disturb_the_existing_result_keys():
    """results.json consumers must keep working: nothing was renamed (F12)."""
    lookup = benchmark_sbox_lookup(DAC_SBOX_REF, n_lookups=500, repeat=2, seed=5)
    assert {"n_lookups", "repeat", "loop_overhead", "table_lookup", "method_call",
            "numpy_batch", "table_lookup_net_ns"} <= set(lookup)
    assert {"ns_per_lookup_min", "ns_per_lookup_median",
            "lookups_per_sec"} == set(lookup["table_lookup"])
    cipher = benchmark_block_cipher(DAC_SBOX_REF, n_blocks=40, repeat=2, seed=5)
    assert {"n_blocks", "repeat", "full_sbox_layer", "enc", "dec"} <= set(cipher)
    assert {"us_per_block_min", "us_per_block_median", "blocks_per_sec",
            "mb_per_sec"} == set(cipher["enc"])
    # The whole benchmark record still survives a JSON round trip.
    assert json.loads(json.dumps(comparative_analysis.to_jsonable(cipher))) is not None


def test_run_all_benchmarks_prints_the_methodology(capsys):
    run_all_benchmarks(quick=True)
    printed = capsys.readouterr().out
    meta = platform_metadata()
    assert meta["cpu_model"] in printed
    assert f"{meta['logical_cores']} logical" in printed
    assert meta["python_version"] in printed and meta["numpy_version"] in printed
    assert "both min and median reported" in printed
    assert "Illustrative only" in printed


def test_python_throughput_is_labelled_illustrative_in_the_docstrings():
    """Reviewer 4: the single-threaded Python throughput is not a perf figure."""
    assert "illustrative" in benchmarks.__doc__.lower()
    assert "no synthesis is performed" in benchmarks.__doc__.lower()
    for func in (benchmark_sbox_lookup, benchmark_block_cipher,
                 benchmark_key_schedule, run_all_benchmarks):
        assert "illustrative" in func.__doc__.lower(), func.__name__
