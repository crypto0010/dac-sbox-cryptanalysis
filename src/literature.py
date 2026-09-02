"""Values *quoted from the literature*, kept in one leaf module.

Nothing here is computed by this code base: every number is a claim printed
in Sarmila & Manisekaran, EIJ 2025 (the base paper).  The module imports
nothing from :mod:`src`, so both :mod:`src.security_analysis` and
:mod:`src.comparative_analysis` can use it without an import cycle -- that
is the whole point of the split.  :mod:`src.comparative_analysis` re-exports
:data:`LITERATURE_SOURCE`, :data:`LITERATURE_REPORTED` and
:data:`LITERATURE_COLUMNS` so existing importers keep working.

Contents
--------
:data:`LITERATURE_REPORTED`
    The competitor rows of the base paper's Table 8.
:data:`PAPER_DAC_REPORTED` / :data:`PAPER_DAC_SUMMARY_ROWS`
    The paper's own claims about *its* DAC S-box, with the normalisation of
    each row made explicit, so a computed value is never compared against a
    differently normalised paper value.
:data:`PAPER_DAC_TABLE`
    The DAC S-box exactly as printed in the paper's Table 3, used only to
    decide whether an analysed table *is* the base paper's S-box (and hence
    whether the paper's values are a legitimate reference for it).
    ``tests/test_library_fixes.py`` pins it to
    :data:`src.sbox_tables.DAC_SBOX_REF` so the two cannot diverge.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

LITERATURE_SOURCE = ("Sarmila & Manisekaran, EIJ 2025, Table 8 "
                     "(values quoted from the paper, not computed here)")

#: Competitor metrics exactly as printed in the base paper.  ``lap`` is the
#: paper's own LAP column; its normalisation is not stated (the paper's DAC
#: value 0.125 is ``max_bias / 2^(n+1)``), which is one reason the computed
#: values differ.
LITERATURE_REPORTED: Dict[str, Dict[str, float]] = {
    "Thakor et al.": {"sac_avg": 0.51, "lap": 0.250, "dp": 0.25,
                      "bic_sac_avg": 0.53, "hd_avg": 2.675},
    "ASCON": {"sac_avg": 0.57, "lap": 0.250, "dp": 0.25,
              "bic_sac_avg": 0.58, "hd_avg": 2.500},
    "PRIMATE": {"sac_avg": 0.52, "lap": 0.375, "dp": 0.0625,
                "bic_sac_avg": 0.54, "hd_avg": 2.500},
    "ICEPOLE": {"sac_avg": 0.34, "lap": 0.250, "dp": 0.25,
                "bic_sac_avg": 0.44, "hd_avg": 1.531},
}

#: (analyze_sbox key, column label) of the literature metrics, in table order.
LITERATURE_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("sac_avg", "SAC"), ("lap", "LAP"), ("dp", "DP"),
    ("bic_sac_avg", "BIC-SAC"), ("hd_avg", "Hd"),
)

#: (key, column label, format spec) of the base paper's Table 8 as it is
#: *printed* by :func:`src.security_analysis.full_security_analysis`.  The
#: same format spec is used for the computed row so the two are comparable.
TABLE8_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("hd_avg", "Avg Hamming distance", ".3f"),
    ("sac_avg", "SAC", ".2f"),
    ("lap", "LAP", ".3f"),
    ("dp", "DP", "g"),
    ("bic_sac_avg", "BIC-SAC", ".2f"),
)


def table8_row(name: str) -> List[str]:
    """The literature row of ``name`` formatted per :data:`TABLE8_COLUMNS`."""
    reported = LITERATURE_REPORTED[name]
    return [format(reported[key], fmt) for key, _, fmt in TABLE8_COLUMNS]


# ============================================================
# The base paper's claims about its own DAC S-box
# ============================================================
#: The DAC S-box as printed in Table 3 of the base paper.  Kept here as a
#: literature quote so that :mod:`src.security_analysis` can recognise it
#: without importing :mod:`src.sbox_tables` (which would be a cycle);
#: ``tests/test_library_fixes.py`` asserts it equals ``DAC_SBOX_REF``.
PAPER_DAC_TABLE: List[int] = [
    30, 24,  9, 27, 28, 26,  4, 22,  2, 15, 29,  5, 13,  8, 25,  3,
    11, 10, 16, 31,  0, 20, 14, 21,  1, 12, 18,  6, 23, 19, 17,  7,
]

#: The paper's reported values for its own DAC S-box, keyed by the
#: :func:`~src.security_analysis.analyze_sbox` key that uses the *same*
#: normalisation.  The paper prints "LAP = 0.125"; that is
#: ``max_bias / 2^(n+1)``, i.e. ``lap_half``.  The standard
#: ``lap = max_bias / 2^n`` value implied by the same max bias is therefore
#: 0.25, and that -- not 0.125 -- is what the ``lap`` row must be compared
#: against.
PAPER_DAC_REPORTED: Dict[str, float] = {
    "sac_avg": 0.52,
    "hd_avg": 2.5625,
    "lap": 0.25,        # = 2 * the paper's printed 0.125 (re-normalised)
    "lap_half": 0.125,  # the paper's printed LAP, as printed
    "dp": 0.25,
    "bic_sac_avg": 0.52,
}

#: (row label, metrics key, printed paper value, tolerance) of the
#: "SUMMARY vs BASE-PAPER VALUES (DAC)" table.  Each row names the key whose
#: normalisation matches the paper value, so no row can be permanently
#: "CHECK" through a unit mismatch.
PAPER_DAC_SUMMARY_ROWS: Tuple[Tuple[str, str, float, float], ...] = (
    ("SAC (avg)", "sac_avg", PAPER_DAC_REPORTED["sac_avg"], 0.05),
    ("Hd (avg)", "hd_avg", PAPER_DAC_REPORTED["hd_avg"], 0.01),
    ("LAP = bias/2^n", "lap", PAPER_DAC_REPORTED["lap"], 0.01),
    ("bias/2^(n+1) (paper norm.)", "lap_half", PAPER_DAC_REPORTED["lap_half"], 0.01),
    ("DP", "dp", PAPER_DAC_REPORTED["dp"], 0.01),
    ("BIC-SAC", "bic_sac_avg", PAPER_DAC_REPORTED["bic_sac_avg"], 0.05),
)


def is_paper_dac_table(sbox) -> bool:
    """True if ``sbox`` is exactly the base paper's Table 3 DAC S-box."""
    try:
        values = [int(v) for v in sbox]
    except (TypeError, ValueError):
        return False
    return values == PAPER_DAC_TABLE
