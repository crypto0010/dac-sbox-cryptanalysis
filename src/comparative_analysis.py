"""Comparative analysis of the DAC S-box family against computed competitors.

Every number in the comparison is *computed* here with
:func:`src.security_analysis.analyze_sbox` from an explicit lookup table, so
the frozen DAC family of :data:`src.sbox_tables.TABLES` (DAC, E-DAC, DAC-OPT,
DAC-OPT-GOLD) and the competitor S-boxes are compared apples-to-apples.  The
competitor values *printed in the base paper* (Table 8 of Sarmila &
Manisekaran, EIJ 2025) are kept in the separate, clearly labelled
:data:`LITERATURE_REPORTED` dictionary, and :func:`literature_discrepancies`
lists where the computed values disagree with them.

Competitor tables (5-bit permutations, ``table[x] = S(x)``)
------------------------------------------------------------
``ASCON``
    The ASCON S-box (Dobraunig, Eichlseder, Mendel and Schlaeffer; the NIST
    lightweight-cryptography standard), as tabulated in the specification.
``Keccak-chi``
    The chi step of Keccak on one 5-bit row,
    ``b_i = a_i ^ (~a_{i+1} & a_{i+2})`` with indices mod 5 and bit ``i`` of
    a value at weight ``2^i``, computed from that definition by
    :func:`keccak_chi_sbox`.
``PRIMATE``
    The S-box of the PRIMATEs authenticated-encryption family (almost bent).

The base paper's other competitors (Thakor et al., ICEPOLE) are available
only as literature values and are therefore not part of the computed
comparison.

Besides the headline metrics of :func:`summary_table`, the module reports the
quantities the referees asked for that the headline metrics do not separate:
the BCT spectrum (:func:`bct_spectrum_table`), the symmetric SAC deviations
D_max / D_mean and the algebraic immunity (:func:`discriminator_table`).

The module also hosts the helpers shared by ``main.py`` and
``run_optimized.py``: the seeded benchmark collection
(:func:`collect_benchmarks`, :func:`benchmark_table`) and the
machine-readable ``output/results.json`` (:func:`core_results`,
:func:`write_results`).  Plotting follows the lazy Agg pattern of
:mod:`src.visualization`: matplotlib is imported only when a figure is drawn
and every figure is built exactly once (no pyplot state, no stacked axes).
Nothing here prints except :func:`run_comparative_analysis`.
"""

from __future__ import annotations

import json
import os
import platform
from importlib import metadata as importlib_metadata
from typing import (TYPE_CHECKING, Any, Collection, Dict, List, Mapping,
                    Optional, Sequence, Tuple)

import numpy as np
from prettytable import PrettyTable

from . import literature
from .benchmarks import (BENCHMARK_REPEAT, BENCHMARK_SIZES,
                         benchmark_avalanche_quality, benchmark_block_cipher,
                         benchmark_key_schedule, benchmark_sbox_lookup,
                         estimate_memory_footprint, platform_metadata)
from .security_analysis import Metrics, analyze_sbox, compute_ddt, compute_lat
from .sbox_tables import TABLES

if TYPE_CHECKING:
    from matplotlib.figure import Figure

N_BITS = 5
SIZE = 1 << N_BITS
#: Resolution of every saved figure.
DPI = 150

#: Members of the DAC family drawn in the radar chart, the per-coordinate
#: nonlinearity bars and the DDT / LAT heatmaps.
FAMILY_PLOT_NAMES: Tuple[str, ...] = ("DAC", "E-DAC", "DAC-OPT")

#: Bar / line colour per S-box name (falls back to a cycle for unknown names).
_COLORS: Dict[str, str] = {
    "DAC": "dodgerblue", "E-DAC": "darkorange", "DAC-OPT": "crimson",
    "DAC-OPT-GOLD": "goldenrod", "ASCON": "seagreen", "Keccak-chi": "slategray",
    "PRIMATE": "purple",
}
_FALLBACK_COLORS: Tuple[str, ...] = ("teal", "sienna", "olive", "navy")


# ============================================================
# Competitor S-boxes (computed, not quoted)
# ============================================================
ASCON_SBOX: List[int] = [
    0x04, 0x0b, 0x1f, 0x14, 0x1a, 0x15, 0x09, 0x02, 0x1b, 0x05, 0x08, 0x12,
    0x1d, 0x03, 0x06, 0x1c, 0x1e, 0x13, 0x07, 0x0e, 0x00, 0x0d, 0x11, 0x18,
    0x10, 0x0c, 0x01, 0x19, 0x16, 0x0a, 0x0f, 0x17,
]

PRIMATE_SBOX: List[int] = [
    1, 0, 25, 26, 17, 29, 21, 27, 20, 5, 4, 23, 14, 18, 2, 28,
    15, 8, 6, 3, 13, 7, 24, 16, 30, 9, 31, 10, 22, 12, 11, 19,
]


def keccak_chi_sbox(n: int = N_BITS) -> List[int]:
    """Keccak chi on one ``n``-bit row: ``b_i = a_i ^ (~a_{i+1} & a_{i+2})``.

    Indices are taken mod ``n`` and bit ``i`` of a value has weight ``2^i``
    (a bit permutation changes none of the computed metrics except the
    ordering of the SAC matrix rows and columns).
    """
    x = np.arange(1 << n, dtype=np.int64)
    weights = 1 << np.arange(n, dtype=np.int64)
    a = (x[:, None] & weights) != 0                       # a[:, i] = bit i of x
    b = a ^ (~np.roll(a, -1, axis=1) & np.roll(a, -2, axis=1))
    return (b * weights).sum(axis=1).tolist()


COMPETITOR_SBOXES: Dict[str, List[int]] = {
    "ASCON": ASCON_SBOX,
    "Keccak-chi": keccak_chi_sbox(),
    "PRIMATE": PRIMATE_SBOX,
}

COMPETITOR_SOURCES: Dict[str, str] = {
    "ASCON": "ASCON v1.2 specification, 5-bit S-box table (NIST LWC standard)",
    "Keccak-chi": "Keccak chi on one 5-bit row, computed from "
                  "b_i = a_i ^ (~a_{i+1} & a_{i+2})",
    "PRIMATE": "PRIMATEs specification, 5-bit S-box table",
}


# ============================================================
# Literature-reported values (quoted from the base paper, NOT computed)
# ============================================================
# The values themselves live in the leaf module :mod:`src.literature` so that
# :mod:`src.security_analysis` (which this module imports) can use the very
# same objects without an import cycle.  They are re-exported here because
# this module is their documented home for importers.
LITERATURE_SOURCE = literature.LITERATURE_SOURCE
LITERATURE_REPORTED: Dict[str, Dict[str, float]] = literature.LITERATURE_REPORTED
LITERATURE_COLUMNS: Tuple[Tuple[str, str], ...] = literature.LITERATURE_COLUMNS


# ============================================================
# Tables and metrics
# ============================================================
def all_sbox_tables() -> Dict[str, List[int]]:
    """The frozen DAC family followed by the competitors, ``name -> table``.

    Every table is checked to be a permutation of ``0 .. 31``.
    """
    tables = {name: frozen_table(name) for name in TABLES}
    tables.update((name, list(table)) for name, table in COMPETITOR_SBOXES.items())
    for name, table in tables.items():
        if sorted(table) != list(range(SIZE)):
            raise ValueError(f"{name!r} is not a permutation of 0 .. {SIZE - 1}")
    return tables


def table_provenance() -> Dict[str, Dict[str, Any]]:
    """Origin of every table of :func:`all_sbox_tables` (frozen record or source)."""
    info: Dict[str, Dict[str, Any]] = {
        name: {"source": record.description, "strategy": record.strategy,
               "seed": record.seed, "iterations": record.iterations}
        for name, record in TABLES.items()
    }
    info.update((name, {"source": COMPETITOR_SOURCES[name], "strategy": None,
                        "seed": None, "iterations": None})
                for name in COMPETITOR_SBOXES)
    return info


def analyze_all(tables: Optional[Mapping[str, Sequence[int]]] = None) -> Dict[str, Metrics]:
    """:func:`~src.security_analysis.analyze_sbox` for every table (default: all)."""
    if tables is None:
        tables = all_sbox_tables()
    return {name: analyze_sbox(table, name) for name, table in tables.items()}


def frozen_table(name: str) -> List[int]:
    """A fresh, mutable copy of the frozen table ``name`` of :data:`TABLES`.

    :class:`~src.sbox_tables.FrozenSBox` is a frozen dataclass, but that
    freezes the *attribute binding*, not the list behind it: handing out
    ``TABLES[name].table`` itself lets one in-place write (a sort, an item
    assignment, a reverse) rewrite the paper's table for every other consumer
    in the process.  Every hand-out point in this module goes through this
    copy instead.
    """
    return list(TABLES[name].table)


def frozen_match(sbox: Sequence[int]) -> Optional[str]:
    """Name of the frozen table in :data:`TABLES` equal to ``sbox``, if any."""
    values = [int(v) for v in sbox]
    for name, record in TABLES.items():
        if record.table == values:
            return name
    return None


# ============================================================
# Pairwise comparison with a data-driven wins tally
# ============================================================
#: (label, analyze_sbox key, direction, target).  ``direction`` is
#: ``"higher"``, ``"lower"``, ``"bool"`` or ``"closer"`` (to ``target``).
#: Each underlying quantity appears exactly once so the wins tally is not
#: double-weighted: ``max_bias`` is represented by LAP (= max_bias / 2^n)
#: and ``differential_uniformity`` by DP (= DU / 2^n).  SAC entries are
#: ideal at 0.5, so both the average and the minimum are scored as
#: ``closer`` to 0.5 (a minimum overshooting 0.5 must not beat an exact 0.5).
COMPARISON_ROWS: Tuple[Tuple[str, str, str, Optional[float]], ...] = (
    ("Bijective", "is_bijective", "bool", None),
    ("NL(S), 31 components", "nl_sbox", "higher", None),
    ("Coordinate NL (avg)", "nl_coord_avg", "higher", None),
    ("SAC (avg)", "sac_avg", "closer", 0.5),
    ("SAC (min)", "sac_min", "closer", 0.5),
    ("SAC D_max = max |P_ij - 0.5|", "sac_max_deviation", "lower", None),
    ("SAC D_mean = mean |P_ij - 0.5|", "sac_mean_deviation", "lower", None),
    ("Hamming distance (avg)", "hd_avg", "closer", N_BITS / 2),
    ("LAP = max bias / 2^n", "lap", "lower", None),
    ("DP", "dp", "lower", None),
    ("BIC-SAC (avg)", "bic_sac_avg", "closer", 0.5),
    ("BIC-NL (avg)", "bic_nl_avg", "higher", None),
    ("Boomerang uniformity", "boomerang_uniformity", "lower", None),
    ("Mean non-trivial BCT entry", "bct_mean_nontrivial", "lower", None),
    ("Algebraic immunity AI(S)", "ai_min", "higher", None),
    ("Algebraic degree (min)", "min_degree", "higher", None),
    ("Balanced components (/31)", "balanced_components", "higher", None),
)

#: Documented alias pairs of :func:`~src.security_analysis.analyze_sbox`: the
#: two keys of a pair are the *same* number under two names (``sac_d_max`` and
#: ``sac_d_mean`` were added for the reviewer's ``D_max`` / ``D_mean``
#: notation).  At most one key of a pair may appear in
#: :data:`COMPARISON_ROWS`, or the wins tally would weight that quantity
#: twice; :func:`aliased_comparison_keys` reports any violation and the test
#: suite asserts that it stays empty.
METRIC_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("sac_max_deviation", "sac_d_max"),
    ("sac_mean_deviation", "sac_d_mean"),
)


def aliased_comparison_keys() -> List[Tuple[str, str]]:
    """Alias pairs of :data:`METRIC_ALIASES` that :data:`COMPARISON_ROWS` scores twice."""
    scored = {key for _, key, _, _ in COMPARISON_ROWS}
    return [pair for pair in METRIC_ALIASES if scored.issuperset(pair)]


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def format_value(value: Any) -> str:
    """``Yes``/``No`` for booleans, plain integers, four decimals otherwise."""
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if _is_integer(value):
        return str(int(value))
    return f"{float(value):.4f}"


def _judge(first: Any, second: Any, direction: str, target: Optional[float],
           tolerance: float = 1e-9) -> Tuple[str, int]:
    """Change text and winner sign: ``+1`` second wins, ``-1`` first, ``0`` tie."""
    if direction == "bool":
        if bool(first) == bool(second):
            return "-", 0
        return "-", (1 if second else -1)
    if direction == "closer":
        dev_first, dev_second = abs(first - target), abs(second - target)
        text = f"|dev| {dev_first:.4f} -> {dev_second:.4f}"
        gain = dev_first - dev_second
    else:
        diff = second - first
        if _is_integer(first) and _is_integer(second):
            text = f"{int(diff):+d}"
        else:
            text = f"{diff:+.4f}"
            if abs(first) > tolerance:
                text += f" ({100.0 * diff / abs(first):+.1f}%)"
        gain = diff if direction == "higher" else -diff
    sign = 0 if abs(gain) <= tolerance else (1 if gain > 0 else -1)
    return text, sign


#: Public name of :func:`_judge` (the underscore name is kept for the tests
#: and callers that already use it).  Exposed so that other modules can reuse
#: the wins logic instead of copying it.
judge = _judge


def compare_pair(metrics_first: Metrics, metrics_second: Metrics) -> Dict[str, Any]:
    """Row-by-row comparison of two :func:`analyze_sbox` dicts.

    Returns ``{"first", "second"}`` (the S-box names), ``"rows"`` (one dict
    per :data:`COMPARISON_ROWS` entry with both values, the change text and
    the winner ``"first"`` / ``"second"`` / ``"tie"``) and ``"wins"`` (the
    tally over the rows).
    """
    rows: List[Dict[str, Any]] = []
    wins = {"first": 0, "second": 0, "tie": 0}
    for label, key, direction, target in COMPARISON_ROWS:
        first, second = metrics_first[key], metrics_second[key]
        text, sign = _judge(first, second, direction, target)
        winner = {1: "second", -1: "first", 0: "tie"}[sign]
        wins[winner] += 1
        rows.append({"metric": label, "key": key, "first": first, "second": second,
                     "change": text, "winner": winner})
    return {"first": metrics_first["name"], "second": metrics_second["name"],
            "rows": rows, "wins": wins}


def comparison_table(pair: Mapping[str, Any]) -> PrettyTable:
    """Render :func:`compare_pair` output as a table with a ``Winner`` column."""
    name_first, name_second = pair["first"], pair["second"]
    table = PrettyTable(["Metric", name_first, name_second, "Change", "Winner"])
    table.align["Metric"] = "l"
    winner_name = {"first": name_first, "second": name_second, "tie": "Tie"}
    for row in pair["rows"]:
        table.add_row([row["metric"], format_value(row["first"]),
                       format_value(row["second"]), row["change"],
                       winner_name[row["winner"]]])
    return table


def wins_line(pair: Mapping[str, Any]) -> str:
    """One-line wins tally of a :func:`compare_pair` result."""
    wins = pair["wins"]
    return (f"Wins: {pair['first']} = {wins['first']}, "
            f"{pair['second']} = {wins['second']}, ties = {wins['tie']}")


# ============================================================
# Summary and literature tables
# ============================================================
SUMMARY_FOOTNOTE = (
    "NL(S) = minimum nonlinearity over all 31 nonzero components (SCV bound 12 "
    "for n = 5); 'Coord. NL' lists the five coordinate functions (MSB first).\n"
    "LAP = max |LAT bias| / 2^n (standard).  The base paper divides by 2^(n+1) "
    "instead, so its DAC value 0.125 equals 0.25 here ('lap_half' in results.json).\n"
    "DP = DU / 2^n.  Deg = algebraic degree and Bal. = balanced count over all "
    "31 components; BU = boomerang uniformity; Hd = mean Hamming distance x vs S(x).\n"
    "D_max = max |P_ij - 0.5| over the SAC matrix (0 is ideal); the mean absolute "
    "deviation D_mean is 'sac_d_mean' in results.json.\n"
    "AI = basic algebraic immunity min over the 31 nonzero components b of AI(b.S) "
    "(optimum 3 for n = 5): algebraic degree 4 does NOT imply algebraic immunity 3."
)


def summary_table(metrics: Mapping[str, Metrics]) -> PrettyTable:
    """One row per S-box with the computed metrics (see :data:`SUMMARY_FOOTNOTE`)."""
    table = PrettyTable(["S-Box", "NL(S)", "Coord. NL", "SAC", "D_max", "LAP", "DP", "DU",
                         "BIC-SAC", "BU", "AI", "Deg", "Bal./31", "Hd"])
    table.align["S-Box"] = "l"
    for name, m in metrics.items():
        degree = (str(m["min_degree"]) if m["min_degree"] == m["max_degree"]
                  else f"{m['min_degree']}-{m['max_degree']}")
        table.add_row([
            name, m["nl_sbox"], "/".join(str(v) for v in m["nl_per_bit"]),
            f"{m['sac_avg']:.4f}", f"{m['sac_d_max']:.4f}",
            f"{m['lap']:.4f}", f"{m['dp']:.4f}",
            m["differential_uniformity"], f"{m['bic_sac_avg']:.4f}",
            m["boomerang_uniformity"], m["ai_min"], degree, m["balanced_components"],
            f"{m['hd_avg']:.4f}",
        ])
    return table


def literature_table(metrics: Mapping[str, Metrics]) -> PrettyTable:
    """Literature values next to the computed ones (``-`` where no table exists)."""
    fields = ["S-Box"]
    for _, label in LITERATURE_COLUMNS:
        fields += [f"{label} lit.", f"{label} comp."]
    table = PrettyTable(fields)
    table.align["S-Box"] = "l"
    for name, reported in LITERATURE_REPORTED.items():
        computed = metrics.get(name)
        row = [name]
        for key, _ in LITERATURE_COLUMNS:
            row.append(f"{reported[key]:.4f}")
            row.append("-" if computed is None else f"{computed[key]:.4f}")
        table.add_row(row)
    return table


def literature_discrepancies(metrics: Mapping[str, Metrics],
                             tolerance: float = 0.005) -> List[Dict[str, Any]]:
    """Literature values that differ from the computed ones by more than ``tolerance``.

    Only S-boxes with a computed table are examined; each entry carries the
    S-box name, the metric key and label, the literature and computed values.
    """
    rows: List[Dict[str, Any]] = []
    for name, reported in LITERATURE_REPORTED.items():
        computed = metrics.get(name)
        if computed is None:
            continue
        for key, label in LITERATURE_COLUMNS:
            if abs(computed[key] - reported[key]) > tolerance:
                rows.append({"sbox": name, "metric": key, "label": label,
                             "literature": reported[key], "computed": computed[key]})
    return rows


# ============================================================
# Reviewer-requested discriminating quantities
# ============================================================
#: Column label -> :func:`~src.security_analysis.analyze_sbox` key of
#: :func:`discriminator_table`.  These are the quantities the referees asked
#: for that the headline metrics of :func:`summary_table` do not separate:
#: the BCT *spectrum* rather than only its maximum (reviewer 2, major 5), the
#: symmetric SAC deviations rather than only the minimum entry (major 2) and
#: the algebraic immunity rather than only the algebraic degree (additional 2).
DISCRIMINATOR_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("BU", "boomerang_uniformity"),
    ("BCT #max", "bct_max_count"),
    ("BCT mean", "bct_mean_nontrivial"),
    ("SAC avg", "sac_avg"),
    ("SAC D_max", "sac_d_max"),
    ("SAC D_mean", "sac_d_mean"),
    ("AI(S)", "ai_min"),
    ("AI avg", "ai_avg"),
    ("AI opt.", "ai_is_optimal"),
    ("Deg", "min_degree"),
)

DISCRIMINATOR_FOOTNOTE = (
    "BCT #max = how many of the 961 non-trivial BCT entries attain the boomerang "
    "uniformity BU; BCT mean = mean non-trivial entry.  A single entry at the\n"
    "maximum is a milder table than many: read #max together with BU, never on "
    "its own (496 entries at BU = 2 is the best possible 5-bit table).\n"
    "SAC D_max = max |P_ij - 0.5| and D_mean = mean |P_ij - 0.5| over the 5x5 SAC "
    "matrix; both are 0 for an ideal S-box and both treat P_ij below and\n"
    "above 0.5 symmetrically, unlike the minimum entry alone.\n"
    "AI(S) = min over the 31 nonzero components b of the algebraic immunity of "
    "b.S (optimum 3 for n = 5); 'AI opt.' marks AI(S) = 3.  Algebraic degree 4\n"
    "does NOT imply AI 3: the degree-4 DAC and E-DAC tables both have AI(S) = 2."
)


def discriminator_table(metrics: Mapping[str, Metrics]) -> PrettyTable:
    """BCT-spectrum, SAC-deviation and algebraic-immunity columns per S-box.

    One row per analysed S-box over :data:`DISCRIMINATOR_COLUMNS`; see
    :data:`DISCRIMINATOR_FOOTNOTE` for how each column must be read.
    """
    table = PrettyTable(["S-Box"] + [label for label, _ in DISCRIMINATOR_COLUMNS])
    table.align["S-Box"] = "l"
    for name, m in metrics.items():
        table.add_row([name] + [format_value(m[key]) for _, key in DISCRIMINATOR_COLUMNS])
    return table


def bct_spectrum_table(metrics: Mapping[str, Metrics]) -> PrettyTable:
    """Full frequency distribution of the non-trivial BCT entries per S-box.

    The answer to reviewer 2's major point 5: the maximum alone cannot say
    whether a rise in the boomerang uniformity is an isolated entry or a
    broader degradation, so the whole spectrum is printed.
    """
    table = PrettyTable(["S-Box", "BU", "Entries", "Mean", "Spectrum (value:count)"])
    table.align["S-Box"] = "l"
    table.align["Spectrum (value:count)"] = "l"
    for name, m in metrics.items():
        spectrum = " ".join(f"{int(value)}:{int(count)}" for value, count in m["bct_spectrum"])
        table.add_row([name, m["boomerang_uniformity"], m["bct_nontrivial_entries"],
                       f"{m['bct_mean_nontrivial']:.4f}", spectrum])
    return table


# ============================================================
# Figures (lazy Agg, one Figure per file)
# ============================================================
def _new_figure(figsize: Tuple[float, float]) -> "Figure":
    """Standalone Agg-backed figure; matplotlib is imported only here."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=DPI)
    FigureCanvasAgg(fig)
    return fig


def _save(fig: "Figure", output_path: str) -> str:
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    print(f"  Saved: {output_path}")
    return output_path


def _color(name: str, index: int) -> str:
    return _COLORS.get(name, _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)])


#: Radar-chart axes: label and the normalised value (1 = best) of a metrics
#: dict.  SAC and BIC-SAC are ideal at 0.5, so their axes plot
#: ``1 - 2 |value - 0.5|`` -- outer really means better on every axis,
#: consistent with the ``closer``-to-0.5 scoring of :data:`COMPARISON_ROWS`.
RADAR_AXES: Tuple[Tuple[str, str], ...] = (
    ("NL(S) / bound", "nl"), ("1 - 2|SAC - 0.5|", "sac"), ("1 - DP", "dp"),
    ("1 - LAP", "lap"), ("1 - 2|BIC-SAC - 0.5|", "bic"), ("min deg / max", "deg"),
)


def _radar_values(m: Metrics) -> List[float]:
    return [
        m["nl_sbox"] / m["nl_upper_bound"],
        1.0 - 2.0 * abs(m["sac_avg"] - 0.5),
        1.0 - m["dp"],
        1.0 - m["lap"],
        1.0 - 2.0 * abs(m["bic_sac_avg"] - 0.5),
        m["min_degree"] / m["max_possible_degree"],
    ]


def plot_overview(metrics: Mapping[str, Metrics], output_path: str,
                  family: Sequence[str] = FAMILY_PLOT_NAMES) -> str:
    """Radar chart (family), DP bars (every S-box) and coordinate NL bars (family)."""
    names = list(metrics)
    family = [name for name in family if name in metrics]
    fig = _new_figure((18, 6))

    ax_radar = fig.add_subplot(1, 3, 1, polar=True)
    angles = np.linspace(0.0, 2.0 * np.pi, len(RADAR_AXES), endpoint=False)
    closed = np.append(angles, angles[0])
    for index, name in enumerate(family):
        values = _radar_values(metrics[name])
        values.append(values[0])
        ax_radar.plot(closed, values, marker="o", linewidth=2, markersize=4,
                      color=_color(name, index), label=name)
        ax_radar.fill(closed, values, alpha=0.08, color=_color(name, index))
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels([label for label, _ in RADAR_AXES], fontsize=9)
    ax_radar.set_ylim(0, 1.05)
    ax_radar.set_title("Security radar (normalised, outer = better)", fontsize=11, pad=18)
    ax_radar.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.15, -0.1))

    ax_dp = fig.add_subplot(1, 3, 2)
    dp_values = [metrics[name]["dp"] for name in names]
    bars = ax_dp.bar(names, dp_values, color=[_color(n, i) for i, n in enumerate(names)],
                     alpha=0.8, edgecolor="black")
    for bar, value in zip(bars, dp_values):
        ax_dp.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                   f"{value:.4f}", ha="center", fontsize=8)
    ax_dp.set_ylabel("Differential probability (DU / 32)")
    ax_dp.set_ylim(0, max(dp_values) * 1.25)
    ax_dp.set_title("Differential probability (lower = better)", fontsize=11)
    ax_dp.tick_params(axis="x", labelrotation=30, labelsize=8)

    ax_nl = fig.add_subplot(1, 3, 3)
    coords = np.arange(N_BITS)
    width = 0.8 / max(len(family), 1)
    for index, name in enumerate(family):
        offset = (index - (len(family) - 1) / 2.0) * width
        ax_nl.bar(coords + offset, metrics[name]["nl_per_bit"], width,
                  color=_color(name, index), alpha=0.8, edgecolor="black",
                  label=f"{name} (NL(S) = {metrics[name]['nl_sbox']})")
    ax_nl.axhline(metrics[names[0]]["nl_upper_bound"], color="gray", linestyle="--",
                  linewidth=1, label="SCV bound")
    ax_nl.set_xticks(coords)
    ax_nl.set_xticklabels([f"f{i}" for i in coords])
    ax_nl.set_xlabel("Coordinate function (MSB first)")
    ax_nl.set_ylabel("Nonlinearity")
    ax_nl.set_ylim(0, 14)
    ax_nl.set_title("Coordinate nonlinearity (higher = better)", fontsize=11)
    ax_nl.legend(fontsize=8, loc="lower right")
    return _save(fig, output_path)


def _plot_heatmaps(panels: Sequence[Tuple[str, np.ndarray]], output_path: str,
                   xlabel: str, ylabel: str, colorbar_label: str) -> str:
    """One row of equally scaled heatmaps (colour range from the non-trivial entries).

    The scale, and the outlined + annotated trivial ``[0, 0]`` corner, come
    from :func:`src.visualization.heatmap_scaled_by_nontrivial`, so the
    single-panel and the side-by-side heatmaps read identically.  The import
    is function-level because :mod:`src.visualization` imports this module.
    """
    from .visualization import heatmap_scaled_by_nontrivial

    vmax = max(int(np.asarray(table)[1:, 1:].max()) for _, table in panels)
    fig = _new_figure((6.0 * len(panels), 5.8))
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for ax, (title, table) in zip(axes, panels):
        image, _, corner = heatmap_scaled_by_nontrivial(ax, table, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        label = colorbar_label
        if corner > vmax:
            label = f"{colorbar_label} (scale 0..{vmax}; trivial [0,0] = {corner} clipped)"
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=label)
    return _save(fig, output_path)


def plot_ddt_comparison(metrics: Mapping[str, Metrics],
                        tables: Mapping[str, Sequence[int]], output_path: str,
                        family: Sequence[str] = FAMILY_PLOT_NAMES) -> str:
    """Side-by-side DDT heatmaps of the family members."""
    panels = []
    for name in family:
        if name not in tables:
            continue
        m = metrics[name]
        panels.append((f"{name} DDT\nDU = {m['differential_uniformity']}, DP = {m['dp']:.4f}",
                       compute_ddt(tables[name], N_BITS)["table"]))
    return _plot_heatmaps(panels, output_path, "Output difference", "Input difference", "Count")


def plot_lat_comparison(metrics: Mapping[str, Metrics],
                        tables: Mapping[str, Sequence[int]], output_path: str,
                        family: Sequence[str] = FAMILY_PLOT_NAMES) -> str:
    """Side-by-side |LAT| heatmaps of the family members."""
    panels = []
    for name in family:
        if name not in tables:
            continue
        m = metrics[name]
        panels.append((f"{name} |LAT|\nmax bias = {m['max_bias']}, LAP = {m['lap']:.4f}",
                       np.abs(compute_lat(tables[name], N_BITS)["table"])))
    return _plot_heatmaps(panels, output_path, "Output mask", "Input mask", "|Bias|")


def plot_comparison(metrics: Mapping[str, Metrics], tables: Mapping[str, Sequence[int]],
                    output_dir: str = "output",
                    family: Sequence[str] = FAMILY_PLOT_NAMES) -> List[str]:
    """Write ``comparison_overview/ddt/lat.png`` into ``output_dir``; returns the paths."""
    os.makedirs(output_dir, exist_ok=True)
    return [
        plot_overview(metrics, os.path.join(output_dir, "comparison_overview.png"), family),
        plot_ddt_comparison(metrics, tables, os.path.join(output_dir, "comparison_ddt.png"),
                            family),
        plot_lat_comparison(metrics, tables, os.path.join(output_dir, "comparison_lat.png"),
                            family),
    ]


# ============================================================
# Report
# ============================================================
#: Pairs printed by :func:`run_comparative_analysis` (paper table, then the optimum).
REPORT_PAIRS: Tuple[Tuple[str, str], ...] = (("DAC", "E-DAC"), ("DAC", "DAC-OPT"))


def comparative_results(tables: Optional[Mapping[str, Sequence[int]]] = None
                        ) -> Dict[str, Any]:
    """Every computed comparison result, without printing.

    Returns ``{"tables", "metrics", "pairs", "literature_discrepancies"}``:
    the analysed tables (default :func:`all_sbox_tables`), their
    :func:`analyze_sbox` metrics, the :func:`compare_pair` results of
    :data:`REPORT_PAIRS` keyed ``"A vs B"`` and the
    :func:`literature_discrepancies` list.
    """
    if tables is None:
        tables = all_sbox_tables()
    tables = {name: list(table) for name, table in tables.items()}
    metrics = analyze_all(tables)
    pairs = {f"{first} vs {second}": compare_pair(metrics[first], metrics[second])
             for first, second in REPORT_PAIRS if first in metrics and second in metrics}
    return {"tables": tables, "metrics": metrics, "pairs": pairs,
            "literature_discrepancies": literature_discrepancies(metrics)}


def run_comparative_analysis(output_dir: str = "output",
                             figures: bool = True) -> Dict[str, Any]:
    """Print the comparison report and (optionally) draw the comparison figures.

    Computes every metric of the DAC family and the competitors, prints the
    pairwise tables of :data:`REPORT_PAIRS` with their wins tallies, the
    literature-vs-computed table and the discrepancy note, then writes the
    ``comparison_*`` figures.  Returns the :func:`comparative_results` dict
    plus ``"figures"`` (the written paths).
    """
    print("\n" + "=" * 72)
    print("  COMPARATIVE ANALYSIS: DAC FAMILY vs COMPUTED COMPETITORS")
    print("=" * 72)
    results = comparative_results()
    tables, metrics = results["tables"], results["metrics"]
    print(f"  S-boxes analysed: {', '.join(tables)}")

    for key, pair in results["pairs"].items():
        print(f"\n--- {key} ---")
        print(comparison_table(pair))
        print(f"  {wins_line(pair)}")
    print("\n" + SUMMARY_FOOTNOTE)

    print("\n--- Reviewer-requested discriminating metrics ---")
    print(discriminator_table(metrics))
    print(DISCRIMINATOR_FOOTNOTE)
    print("\n--- BCT spectrum (frequency of every non-trivial entry) ---")
    print(bct_spectrum_table(metrics))

    print("\n--- Literature-reported competitor values vs computed ---")
    print(f"  Literature source: {LITERATURE_SOURCE}")
    print(literature_table(metrics))
    discrepancies = results["literature_discrepancies"]
    if discrepancies:
        print("  Computed values that differ from the literature (|diff| > 0.005):")
        for row in discrepancies:
            print(f"    {row['sbox']:8s} {row['label']:8s} literature {row['literature']:.4f}"
                  f"  computed {row['computed']:.4f}")
    else:
        print("  All computed values agree with the literature.")
    print("  Thakor et al. and ICEPOLE have no table here (literature values only).")

    figure_paths: List[str] = []
    if figures:
        print("\nGenerating comparison figures...")
        figure_paths = plot_comparison(metrics, tables, output_dir)

    return dict(results, figures=figure_paths)


# ============================================================
# Seeded benchmarks (shared by main.py and run_optimized.py)
# ============================================================
#: ``(n_lookups, n_blocks, n_keys, n_avalanche_plaintexts)`` of the two
#: benchmark scales: ``standard`` (about one second) and ``quick`` (smoke
#: test).  Defined once in :mod:`src.benchmarks` -- which owns the benchmark
#: functions and its own reporting entry point
#: :func:`~src.benchmarks.run_all_benchmarks` -- and re-exported here so that
#: both entry points always mean the same thing by "quick".


def collect_benchmarks(seed: int, quick: bool = False,
                       sbox_table: Optional[Sequence[int]] = None,
                       repeat: int = BENCHMARK_REPEAT) -> Dict[str, Any]:
    """Run the benchmark suite of :mod:`src.benchmarks` with seeded inputs.

    ``seed`` drives the random lookups, plaintexts and keys, so the
    ``avalanche`` section is reproducible; ``quick`` selects the smaller
    :data:`BENCHMARK_SIZES`; ``sbox_table`` defaults to the frozen DAC
    table.  Returns ``{"scale", "sizes", "methodology", "sbox", "cipher",
    "key_schedule", "memory", "avalanche"}``: the section layout of
    :func:`~src.benchmarks.run_all_benchmarks` plus ``methodology``, the
    machine / OS / interpreter / timer description of
    :func:`~src.benchmarks.platform_metadata`.  It is recorded once at the
    top level, not only inside each individual timing record, so that the
    paper's reproducibility paragraph (reviewer 2, additional point 4) has a
    single place to quote; it is constant for a given machine and therefore
    does not make ``results.json`` non-deterministic.
    """
    scale = "quick" if quick else "standard"
    n_lookups, n_blocks, n_keys, n_avalanche = BENCHMARK_SIZES[scale]
    # Copy on both branches: never pass the frozen list object of TABLES
    # itself to a benchmark, or one in-place write would rewrite paper
    # Table 3 for every other consumer in the process.
    table = frozen_table("DAC") if sbox_table is None else list(sbox_table)
    return {
        "scale": scale,
        "sizes": {"n_lookups": n_lookups, "n_blocks": n_blocks, "n_keys": n_keys,
                  "n_avalanche_plaintexts": n_avalanche, "repeat": repeat},
        "methodology": platform_metadata(),
        "sbox": benchmark_sbox_lookup(table, n_lookups, repeat, seed=seed),
        "cipher": benchmark_block_cipher(table, n_blocks, repeat, seed=seed),
        "key_schedule": benchmark_key_schedule(table, n_keys, repeat, seed=seed),
        "memory": estimate_memory_footprint(table),
        "avalanche": benchmark_avalanche_quality(table, n_avalanche, seed=seed),
    }


def benchmark_table(benchmarks: Mapping[str, Any]) -> PrettyTable:
    """One-row-per-measurement table of a :func:`collect_benchmarks` result."""
    sbox, cipher = benchmarks["sbox"], benchmarks["cipher"]
    key_schedule, avalanche = benchmarks["key_schedule"], benchmarks["avalanche"]
    memory = benchmarks["memory"]
    table = PrettyTable(["Measurement", "Value", "Note"])
    table.align["Measurement"] = "l"
    table.align["Value"] = "r"
    table.align["Note"] = "l"
    for key, label in (("table_lookup", "S-box lookup, raw table[x]"),
                       ("method_call", "S-box lookup, DACSBox.substitute()"),
                       ("numpy_batch", "S-box lookup, numpy batch")):
        row = sbox[key]
        table.add_row([label, f"{row['ns_per_lookup_min']:.1f} ns",
                       f"median {row['ns_per_lookup_median']:.1f} ns, "
                       f"{row['lookups_per_sec']:,.0f} lookups/s"])
    for key, label in (("enc", "Block encryption"), ("dec", "Block decryption")):
        row = cipher[key]
        table.add_row([label, f"{row['us_per_block_min']:.2f} us/block",
                       f"{row['blocks_per_sec']:,.0f} blocks/s, {row['mb_per_sec']:.3f} MB/s"])
    table.add_row(["Key schedule", f"{key_schedule['us_per_key_schedule_min']:.2f} us/key",
                   f"{key_schedule['key_schedules_per_sec']:,.0f} schedules/s"])
    table.add_row(["Cipher avalanche ratio", f"{avalanche['avg_avalanche_ratio']:.4f}",
                   f"ideal 0.5; {avalanche['total_comparisons']:,} single-bit flips, "
                   f"std {avalanche['std_avalanche_ratio']:.4f}"])
    table.add_row(["S-box table storage", f"{memory['sbox_table_bits']} bits",
                   f"{memory['sbox_table_bytes']:.0f} bytes; total cipher state "
                   f"{memory['total_bits']} bits"])
    return table


# ============================================================
# results.json helpers (shared by main.py and run_optimized.py)
# ============================================================
#: Keys removed from every nested dict before writing (wall-clock fields are
#: reported separately under ``timings``).
RUNTIME_KEYS: Collection[str] = frozenset({"runtime_s"})
#: Benchmark sections whose values are timings (everything else is seeded
#: and deterministic).
BENCHMARK_TIMING_SECTIONS: Tuple[str, ...] = ("sbox", "cipher", "key_schedule")


def environment_info() -> Dict[str, str]:
    """Interpreter and package versions (no heavy imports)."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": importlib_metadata.version("matplotlib"),
        "prettytable": importlib_metadata.version("prettytable"),
        "platform": platform.platform(),
    }


def to_jsonable(obj: Any, drop_keys: Collection[str] = RUNTIME_KEYS) -> Any:
    """Recursively convert numpy scalars / arrays and tuples; drop ``drop_keys``."""
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value, drop_keys)
                for key, value in obj.items() if key not in drop_keys}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value, drop_keys) for value in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist(), drop_keys)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def partition_benchmarks(benchmarks: Mapping[str, Any]
                         ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a :func:`collect_benchmarks` result into ``(deterministic, timings)``.

    The wall-clock sections of :data:`BENCHMARK_TIMING_SECTIONS` go to
    ``timings``; the seeded avalanche statistics, the memory footprint and
    the sample sizes stay in ``deterministic``.
    """
    timings = {key: benchmarks[key] for key in BENCHMARK_TIMING_SECTIONS if key in benchmarks}
    deterministic = {key: value for key, value in benchmarks.items() if key not in timings}
    return deterministic, timings


def optimizer_record(result: Mapping[str, Any]) -> Dict[str, Any]:
    """JSON-friendly record of an :func:`~src.sbox_optimizer.optimize_dac_sbox` result.

    Wall-clock fields are excluded (see :func:`optimizer_runtimes`); the
    frozen table reproduced by the search, if any, is named in
    ``matches_frozen``.
    """
    def entry(res: Mapping[str, Any]) -> Dict[str, Any]:
        provenance = res["provenance"]
        return {
            "strategy": provenance["strategy"],
            "seed": provenance["seed"],
            "iterations": provenance["iterations"],
            "sbox": list(res["sbox"]),
            "objective": list(res["objective"]),
            "matches_frozen": frozen_match(res["sbox"]),
            "provenance": provenance,
            "metrics": res["metrics"],
        }

    record = entry(result)
    if "candidates" in result:
        record["candidates"] = {name: entry(res) for name, res in result["candidates"].items()}
    return record


def optimizer_runtimes(result: Mapping[str, Any]) -> Dict[str, float]:
    """``runtime_s`` of the winner and of every candidate strategy."""
    runtimes = {"winner": result["provenance"]["runtime_s"]}
    for name, res in result.get("candidates", {}).items():
        runtimes[name] = res["provenance"]["runtime_s"]
    return runtimes


def core_results(runner: str, arguments: Mapping[str, Any],
                 comparison: Mapping[str, Any]) -> Dict[str, Any]:
    """The part of ``results.json`` common to both runners.

    ``comparison`` is the dict returned by :func:`run_comparative_analysis`
    (or one with the same ``tables`` / ``metrics`` / ``pairs`` /
    ``literature_discrepancies`` keys).  The ``"schema"`` key is *not* set
    here: each runner writes its own file and stamps its own schema string,
    so a single library-wide constant would be dead (and misleading).
    """
    return {
        "runner": runner,
        "arguments": dict(arguments),
        "environment": environment_info(),
        "tables": comparison["tables"],
        "table_provenance": table_provenance(),
        "metrics": comparison["metrics"],
        "pairs": comparison["pairs"],
        "literature_source": LITERATURE_SOURCE,
        "literature_reported": LITERATURE_REPORTED,
        "literature_discrepancies": comparison["literature_discrepancies"],
    }


def write_results(path: str, results: Mapping[str, Any]) -> str:
    """Write ``results`` as sorted, indented JSON (LF line endings); returns ``path``.

    Two runs with the same arguments give byte-identical files except for
    the ``timings`` entry, because every wall-clock field is either kept
    there by the runner or dropped by :func:`to_jsonable`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(to_jsonable(results), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path
