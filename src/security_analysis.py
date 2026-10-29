"""Security analysis suite for n-bit S-boxes (default n = 5, the DAC S-box).

Metrics (paper equation numbers refer to Sarmila & Manisekaran, EIJ 2025):

1. Bijectivity / balancedness of all component functions (Eq. 14)
2. Strict Avalanche Criterion, SAC (Eq. 11)
3. Nonlinearity via the Walsh spectrum of *all* 2^n - 1 components (Eq. 8, 10)
4. Linear Approximation Table / Probability, LAT / LAP (Eq. 12)
5. Difference Distribution Table / Probability, DDT / DP (Eq. 13)
6. Bit Independence Criterion, BIC-SAC and BIC-NL
7. Boomerang Connectivity Table, BCT, and its full spectrum (not in the base paper)
8. Algebraic degree of all components (Mobius transform)
9. Hamming distance between input and output
10. Algebraic immunity: basic AI of every component and the graph algebraic
    immunity of the vectorial map (not in the base paper)

Definitions used
----------------
* Walsh spectrum ``W[a, b] = sum_x (-1)^(a.x ^ b.S(x))``.
* ``NL(f_b) = (2^n - max_a |W[a, b]|) / 2`` and ``NL(S) = min_{b != 0} NL(f_b)``.
* ``LAT[a, b] = #{x : a.x = b.S(x)} - 2^(n-1) = W[a, b] / 2``.
  ``max_bias`` is taken over ``a != 0`` **and** ``b != 0`` (the whole trivial
  row and column are excluded, not just the (0, 0) entry).
* ``lap = max_bias / 2^n`` (standard). The base paper normalises by
  ``2^(n+1)`` instead, which is reported separately as ``lap_half``.
* ``differential_uniformity = max_{dx != 0} DDT[dx, dy]``; ``dp = DU / 2^n``.
* ``BCT[a, b] = #{x : S^-1(S(x) ^ b) ^ S^-1(S(x ^ a) ^ b) = a}``;
  boomerang uniformity is the maximum over ``a, b != 0``, and the *BCT
  spectrum* is the frequency distribution ``value -> count`` of the
  ``(2^n - 1)^2`` non-trivial entries (see :func:`spectrum_of`).
* SAC deviations ``D_max = max_ij |P_ij - 1/2|`` and
  ``D_mean = (1/n^2) sum_ij |P_ij - 1/2|`` over the ``n x n`` SAC matrix.
* Algebraic immunity, in the two inequivalent senses named in
  :func:`compute_algebraic_immunity`: the *basic algebraic immunity*
  ``AI(S) = min_{b != 0} AI(b.S)`` of the component functions and the *graph
  algebraic immunity* ``AI_gr(S)`` of the set ``{(x, S(x))}``.

Conventions: 5-bit values 0..31; the SAC matrix, coordinate functions and
bit masks are MSB-first (coordinate ``i`` has mask ``1 << (n - 1 - i)``);
ANF variables ``x0..x(n-1)`` are input bits ``0..n-1``.

Everything is vectorised with numpy on cached lookup tables from
:mod:`src.boolean_function`; :func:`fast_metrics` is the optimiser-friendly
entry point (about 100 us per 5-bit S-box).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from prettytable import PrettyTable

from . import literature
from .boolean_function import (
    BitTables, as_sbox_array, bit_tables, component_degrees, component_weights,
)

Metrics = Dict[str, Any]

# Default tolerance for the element-wise SAC verdict: every |P_ij - 0.5| must
# be at most this value (4 of the 32 input pairs for n = 5).
SAC_TOLERANCE = 0.125


# ============================================================
# Cached per-n constants
# ============================================================
@dataclass(frozen=True)
class _Tables:
    """Bit tables plus the S-box specific index helpers derived from them."""

    bt: BitTables
    ddt_offsets: np.ndarray   # (size, size): dx * size, for bincount indexing
    triu: Tuple[np.ndarray, np.ndarray]  # upper-triangle (j < k) pair indices
    pairs: List[Tuple[int, int]]


@lru_cache(maxsize=None)
def _tables(n: int) -> _Tables:
    bt = bit_tables(n)
    ddt_offsets = (np.arange(bt.size, dtype=np.int64) * bt.size)[None, :]
    ddt_offsets = np.broadcast_to(ddt_offsets, (bt.size, bt.size)).copy()
    ddt_offsets.setflags(write=False)
    triu = np.triu_indices(n, 1)
    pairs = [(int(j), int(k)) for j, k in zip(*triu)]
    return _Tables(bt, ddt_offsets, triu, pairs)


# Module-level tables for the default 5-bit case (read-only).
SIGN: np.ndarray = bit_tables(5).sign       #: SIGN[a, b] = (-1)^popcount(a & b)
XOR_TABLE: np.ndarray = bit_tables(5).xor   #: XOR_TABLE[x, d] = x ^ d
POPCOUNT: np.ndarray = bit_tables(5).popcount


def _prepare(sbox: Sequence[int], n: int) -> Tuple[_Tables, np.ndarray]:
    return _tables(n), as_sbox_array(sbox, n)


def _difference_matrix(sbox: np.ndarray, t: _Tables) -> np.ndarray:
    """``D[x, dx] = S(x) ^ S(x ^ dx)``."""
    return sbox[:, None] ^ sbox[t.bt.xor]


def _avalanche_bits(sbox: np.ndarray, t: _Tables) -> np.ndarray:
    """``bits[x, i, j]`` = output bit j of ``S(x) ^ S(x ^ e_i)`` (int8)."""
    diff = _difference_matrix(sbox, t)
    return t.bt.bits[diff[:, t.bt.coord_masks]]


def _walsh(sbox: np.ndarray, t: _Tables) -> np.ndarray:
    """Walsh spectrum ``W[a, b]`` as int64 via one BLAS matmul."""
    sign = t.bt.sign
    return (sign @ sign[sbox]).astype(np.int64)


def _ddt(sbox: np.ndarray, t: _Tables) -> np.ndarray:
    size = t.bt.size
    diff = _difference_matrix(sbox, t)
    counts = np.bincount((t.ddt_offsets + diff).ravel(), minlength=size * size)
    return counts.reshape(size, size)


# ------------------------------------------------------------
# Exact GF(2) linear algebra (bit-packed Python ints; exact, no rounding)
# ------------------------------------------------------------
def pack_gf2_rows(matrix: np.ndarray) -> List[int]:
    """Pack the rows of a 0/1 matrix into Python ints, one int per row.

    Column 0 becomes the most significant bit; :func:`numpy.packbits` pads
    every row with the same number of zero bits on the right, which is a
    uniform relabelling of the columns and therefore leaves the rank and the
    kernel dimension unchanged.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    if matrix.shape[1] == 0:
        return [0] * matrix.shape[0]
    packed = np.packbits(matrix.astype(np.uint8) & 1, axis=1)
    return [int.from_bytes(row.tobytes(), "big") for row in packed]


def gf2_rank(rows: Iterable[int]) -> int:
    """Exact rank over GF(2) of bit-packed rows (Gaussian elimination on ints)."""
    basis: List[int] = []
    for row in rows:
        for pivot in basis:
            row = min(row, row ^ pivot)   # clears the pivot's leading bit
        if row:
            basis.append(row)
            basis.sort(reverse=True)
    return len(basis)


def gf2_kernel_dimension(matrix: np.ndarray) -> int:
    """Dimension of the right kernel of a 0/1 matrix over GF(2) (``cols - rank``)."""
    matrix = np.asarray(matrix)
    return int(matrix.shape[1]) - gf2_rank(pack_gf2_rows(matrix))


def spectrum_of(values: np.ndarray) -> List[List[int]]:
    """Frequency distribution of ``values`` as a sorted list of ``[value, count]``.

    A list of pairs rather than a dict so that the result survives a JSON
    round trip unchanged (JSON object keys would become strings).
    """
    unique, counts = np.unique(np.asarray(values).ravel(), return_counts=True)
    return [[int(v), int(c)] for v, c in zip(unique, counts)]


def _bic_sac_matrix(bits: np.ndarray, t: _Tables) -> np.ndarray:
    """BIC-SAC ``[j, k]`` = mean over inputs bits i and x of ``g_jk(x) ^ g_jk(x ^ e_i)``.

    Uses ``u ^ v = u + v - 2uv`` for 0/1 values so that all pairs come from one
    small Gram matrix. The diagonal is not part of BIC and is set to NaN.
    """
    n = t.bt.n
    flat = bits.reshape(-1, n).astype(np.float64)   # (size * n, n)
    ones = flat.sum(axis=0)
    both = flat.T @ flat
    matrix = (ones[:, None] + ones[None, :] - 2.0 * both) / flat.shape[0]
    np.fill_diagonal(matrix, np.nan)
    return matrix


# ============================================================
# 1. Bijective property / balancedness (Eq. 14)
# ============================================================
def check_bijective(sbox: Sequence[int], n: int = 5) -> Metrics:
    """Check that the S-box is a permutation of ``0 .. 2^n - 1``.

    Eq. 14 of the paper phrases this as "every nonzero linear combination of
    the coordinate functions has Hamming weight 2^(n-1)", which is reported as
    ``all_components_balanced``.
    """
    t, arr = _prepare(sbox, n)
    size = t.bt.size
    unique = int(np.unique(arr).size)
    weights = component_weights(arr, n)[1:]
    return {
        "is_bijective": unique == size,
        "total_hamming_weight": int(t.bt.popcount[arr].sum()),
        "unique_outputs": unique,
        "expected_unique": size,
        "all_components_balanced": bool(np.all(weights == size // 2)),
        "balanced_components": int(np.count_nonzero(weights == size // 2)),
    }


# ============================================================
# 2. Strict Avalanche Criterion (Eq. 11)
# ============================================================
def compute_sac(sbox: Sequence[int], n: int = 5,
                tolerance: float = SAC_TOLERANCE) -> Metrics:
    """Strict Avalanche Criterion matrix ``P[i, j]`` and its deviation measures.

    ``P[i, j]`` is the probability that output bit ``j`` flips when input bit
    ``i`` is flipped (both MSB-first). SAC holds when every entry is 0.5.

    Because 0.5 is a two-sided ideal (``|0.25 - 0.5| = |0.75 - 0.5|``), the
    matrix is summarised by the two symmetric deviation measures

    * ``d_max`` (alias ``max_deviation``) ``= D_max = max_ij |P_ij - 1/2|``,
    * ``d_mean`` (alias ``mean_deviation``)
      ``= D_mean = (1/n^2) sum_ij |P_ij - 1/2|``,

    both taken over all ``n * n`` entries of the matrix, in addition to the
    one-sided ``sac_min`` / ``sac_max`` and the ``average``.  Reporting only
    ``sac_min`` overstates an improvement that raises the smallest entry while
    leaving the largest deviation untouched.

    Two verdicts are returned: ``satisfies_sac`` (``D_max <= tolerance``) and
    the legacy average-only ``satisfies_sac_average`` (``average >= 0.5``).
    """
    t, arr = _prepare(sbox, n)
    bits = _avalanche_bits(arr, t)
    matrix = bits.sum(axis=0, dtype=np.int64) / t.bt.size
    deviation = np.abs(matrix - 0.5)
    average = float(matrix.mean())
    sac_min, sac_max = float(matrix.min()), float(matrix.max())
    max_dev = float(deviation.max())
    mean_dev = float(deviation.mean())
    satisfies = max_dev <= tolerance
    verdict = (
        f"average {average:.4f}; entries in [{sac_min:.3f}, {sac_max:.3f}], "
        f"D_max = max |P_ij - 0.5| = {max_dev:.3f}, "
        f"D_mean = {mean_dev:.4f}; D_max "
        f"{'<=' if satisfies else '>'} tolerance {tolerance:.3f} -> "
        f"{'SAC satisfied' if satisfies else 'SAC NOT satisfied element-wise'}"
    )
    return {
        "matrix": matrix,
        "average": average,
        "satisfies_sac": bool(satisfies),
        "satisfies_sac_average": average >= 0.5,
        "min_value": sac_min,
        "max_value": sac_max,
        "sac_min": sac_min,
        "sac_max": sac_max,
        "max_deviation": max_dev,
        "mean_deviation": mean_dev,
        "d_max": max_dev,        # D_max = max |P_ij - 1/2|
        "d_mean": mean_dev,      # D_mean = mean |P_ij - 1/2|
        "tolerance": float(tolerance),
        "sac_verdict": verdict,
    }


# ============================================================
# 3. Walsh spectrum and nonlinearity (Eq. 8, 10)
# ============================================================
def walsh_hadamard_transform(truth_table: Sequence[int], n: int = 5) -> np.ndarray:
    """Walsh-Hadamard transform ``W_f(c) = sum_m (-1)^(f(m) ^ m.c)`` (Eq. 8)."""
    t = _tables(n)
    f = np.asarray(truth_table, dtype=np.int64)
    if f.shape != (t.bt.size,):
        raise ValueError(f"truth table must have {t.bt.size} entries for n={n}")
    return (t.bt.sign @ (1.0 - 2.0 * (f & 1))).astype(np.int64)


def compute_walsh_spectrum(sbox: Sequence[int], n: int = 5) -> np.ndarray:
    """Full Walsh spectrum ``W[a, b] = sum_x (-1)^(a.x ^ b.S(x))`` as int64."""
    t, arr = _prepare(sbox, n)
    return _walsh(arr, t)


def nonlinearity_upper_bound(n: int) -> int:
    """Sidelnikov-Chabaud-Vaudenay bound ``floor(2^(n-1) - 2^((n-1)/2))`` on ``NL(S)``.

    It holds for every ``(n, n)``-function.  For odd ``n`` it is an integer
    and tight (almost bent functions attain it: 12 for ``n = 5``); for even
    ``n`` it is not attained -- the best known values are
    ``2^(n-1) - 2^(n/2)`` (4 for ``n = 4``, 112 for ``n = 8``) against bounds
    of 5 and 116.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    half = 1 << (n - 1)
    return half - (math.isqrt(half - 1) + 1)   # ceil(sqrt(2^(n-1))) exactly


def compute_nonlinearity(sbox: Sequence[int], n: int = 5) -> Metrics:
    """Nonlinearity of the coordinate functions and of the whole S-box.

    ``NL(f) = (2^n - max_c |W_f(c)|) / 2`` (Eq. 10). The legacy keys
    ``per_bit`` / ``minimum`` / ``maximum`` / ``average`` describe only the
    ``n`` coordinate functions; the S-box nonlinearity ``nl_sbox`` (alias
    ``nl_min``) is the minimum over all ``2^n - 1`` nonzero components, which
    can be strictly smaller (E-DAC: coordinates all 10, ``nl_sbox`` 8).
    ``nl_upper_bound`` is the proven bound of :func:`nonlinearity_upper_bound`.
    """
    t, arr = _prepare(sbox, n)
    size = t.bt.size
    spectrum = _walsh(arr, t)
    nl_all = (size - np.abs(spectrum).max(axis=0)) // 2   # index = output mask
    component_nl = nl_all[1:]
    per_bit = nl_all[t.bt.coord_masks].tolist()
    return {
        "per_bit": per_bit,
        "minimum": min(per_bit),
        "maximum": max(per_bit),
        "average": float(np.mean(per_bit)),
        "component_nl": component_nl.tolist(),
        "nl_sbox": int(component_nl.min()),
        "nl_min": int(component_nl.min()),
        "nl_max_components": int(component_nl.max()),
        "nl_avg_components": float(component_nl.mean()),
        "nl_upper_bound": nonlinearity_upper_bound(n),
        "walsh_spectrum": spectrum,
    }


def compute_hamming_distances(sbox: Sequence[int], n: int = 5) -> Metrics:
    """Hamming distance between each input ``x`` and its output ``S(x)`` (Table 7)."""
    t, arr = _prepare(sbox, n)
    distances = t.bt.popcount[np.arange(t.bt.size) ^ arr]
    return {
        "distances": distances.tolist(),
        "minimum": int(distances.min()),
        "maximum": int(distances.max()),
        "average": float(distances.mean()),
    }


# ============================================================
# 4. Linear Approximation Table / Probability (Eq. 12)
# ============================================================
def compute_lat(sbox: Sequence[int], n: int = 5) -> Metrics:
    """Linear Approximation Table and linear approximation probability.

    ``LAT[a, b] = #{x : a.x = b.S(x)} - 2^(n-1)``. ``max_bias`` excludes the
    trivial row ``a = 0`` and column ``b = 0``. ``lap = max_bias / 2^n`` is the
    standard LAP (Eq. 12); ``lap_half = max_bias / 2^(n+1)`` is the base
    paper's normalisation (it reports 0.125 for the DAC S-box, whose
    ``max_bias`` is 8); ``linear_probability = (2 * max_bias / 2^n)^2`` is
    Matsui's LP.
    """
    t, arr = _prepare(sbox, n)
    size = t.bt.size
    lat = _walsh(arr, t) // 2
    max_bias = int(np.abs(lat[1:, 1:]).max())
    return {
        "table": lat,
        "max_bias": max_bias,
        "lap": max_bias / size,
        "lap_half": max_bias / (2 * size),
        "linear_probability": (2 * max_bias / size) ** 2,
    }


# ============================================================
# 5. Difference Distribution Table / Probability (Eq. 13)
# ============================================================
def compute_ddt(sbox: Sequence[int], n: int = 5) -> Metrics:
    """Difference Distribution Table ``DDT[dx, dy] = #{x : S(x) ^ S(x ^ dx) = dy}``.

    ``differential_uniformity`` (alias ``max_value``) is the largest entry
    with ``dx != 0``; ``dp = differential_uniformity / 2^n`` (Eq. 13).
    """
    t, arr = _prepare(sbox, n)
    ddt = _ddt(arr, t)
    du = int(ddt[1:].max())
    return {
        "table": ddt,
        "max_value": du,
        "differential_uniformity": du,
        "dp": du / t.bt.size,
    }


# ============================================================
# 6. Bit Independence Criterion
# ============================================================
def compute_bic(sbox: Sequence[int], n: int = 5) -> Metrics:
    """Bit Independence Criterion over all output-bit pairs ``j < k``.

    For ``g_jk = f_j ^ f_k``: BIC-SAC is the mean avalanche probability of
    ``g_jk`` over all input bits and BIC-NL is its nonlinearity (which is the
    component nonlinearity for mask ``e_j | e_k``). Values are listed in
    ``pairs`` order; full symmetric matrices (NaN diagonal) are also returned.
    """
    t, arr = _prepare(sbox, n)
    size = t.bt.size
    bits = _avalanche_bits(arr, t)
    sac_matrix = _bic_sac_matrix(bits, t)
    nl_all = (size - np.abs(_walsh(arr, t)).max(axis=0)) // 2
    masks = t.bt.coord_masks
    nl_matrix = nl_all[masks[:, None] | masks[None, :]].astype(np.float64)
    np.fill_diagonal(nl_matrix, np.nan)
    sac_values = sac_matrix[t.triu].tolist()
    nl_values = [int(v) for v in nl_matrix[t.triu]]
    return {
        "bic_sac_values": sac_values,
        "bic_nl_values": nl_values,
        "avg_bic_sac": float(np.mean(sac_values)),
        "avg_bic_nl": float(np.mean(nl_values)),
        "min_bic_sac": float(min(sac_values)),
        "max_bic_sac": float(max(sac_values)),
        "min_bic_nl": min(nl_values),
        "max_bic_nl": max(nl_values),
        "pairs": list(t.pairs),
        "bic_sac_matrix": sac_matrix,
        "bic_nl_matrix": nl_matrix,
    }


# ============================================================
# 7. Boomerang Connectivity Table
# ============================================================
def compute_bct(sbox: Sequence[int], n: int = 5) -> Metrics:
    """Boomerang Connectivity Table (Cid et al., EUROCRYPT 2018) of a permutation.

    ``BCT[a, b] = #{x : S^-1(S(x) ^ b) ^ S^-1(S(x ^ a) ^ b) = a}``; the
    boomerang uniformity is the maximum over ``a != 0, b != 0``.
    Raises ``ValueError`` for a non-bijective S-box.

    The maximum alone does not describe the table, so the full **BCT
    spectrum** is returned as well: ``spectrum`` is the frequency
    distribution ``[[value, count], ...]`` of the ``(2^n - 1)^2``
    non-trivial entries (``a != 0`` and ``b != 0``), sorted by value, and
    ``max_count`` is how many entries attain the boomerang uniformity.  A
    single entry at the maximum is an isolated peak; a large ``max_count`` or
    a spectrum whose mass has moved to the right is a broader degradation.

    Returned keys: ``table``, ``boomerang_uniformity``, ``spectrum``,
    ``nontrivial_entries``, ``max_count``, ``mean_nontrivial``.
    """
    t, arr = _prepare(sbox, n)
    size = t.bt.size
    idx = np.arange(size, dtype=np.int64)
    if np.unique(arr).size != size:
        raise ValueError("BCT is only defined for bijective S-boxes")
    inv = np.empty(size, dtype=np.int64)
    inv[arr] = idx
    b = idx[None, None, :]
    lhs = inv[arr[:, None, None] ^ b] ^ inv[arr[t.bt.xor][:, :, None] ^ b]  # [x, a, b]
    bct = (lhs == idx[None, :, None]).sum(axis=0, dtype=np.int64)
    nontrivial = bct[1:, 1:]
    uniformity = int(nontrivial.max())
    return {
        "table": bct,
        "boomerang_uniformity": uniformity,
        "spectrum": spectrum_of(nontrivial),
        "nontrivial_entries": int(nontrivial.size),
        "max_count": int(np.count_nonzero(nontrivial == uniformity)),
        "mean_nontrivial": float(nontrivial.mean()),
    }


# ============================================================
# 8. Algebraic degree of all components (Mobius transform)
# ============================================================
def compute_algebraic_degree(sbox: Sequence[int], n: int = 5) -> Metrics:
    """Algebraic degree and balancedness of all ``2^n - 1`` component functions.

    ``per_bit`` lists the coordinate degrees (MSB first); ``min_degree`` /
    ``max_degree`` range over every nonzero output mask. A permutation has
    degree at most ``n - 1``.
    """
    t, arr = _prepare(sbox, n)
    size = t.bt.size
    degrees = component_degrees(arr, n)
    weights = component_weights(arr, n)
    balanced = weights[1:] == size // 2
    return {
        "per_bit": degrees[t.bt.coord_masks].tolist(),
        "component_degrees": degrees[1:].tolist(),
        "min_degree": int(degrees[1:].min()),
        "max_degree": int(degrees[1:].max()),
        "max_possible_degree": n - 1,
        "component_weights": weights[1:].tolist(),
        "coordinate_balanced": (weights[t.bt.coord_masks] == size // 2).tolist(),
        "balanced_components": int(np.count_nonzero(balanced)),
        "all_components_balanced": bool(balanced.all()),
    }


# ============================================================
# 9. Algebraic immunity (exact GF(2) linear algebra)
# ============================================================
@lru_cache(maxsize=None)
def _monomial_evaluations(n: int) -> np.ndarray:
    """``E[x, u] = 1`` iff monomial ``u`` (a variable mask) is 1 at point ``x``.

    ``prod_{i in u} x_i = 1`` exactly when ``u`` is a subset of ``x``, which
    is the incidence pattern of the cached Mobius matrix of
    :mod:`src.boolean_function`; the uint8 view is cached and read-only.
    """
    table = np.asarray(_tables(n).bt.mobius, dtype=np.uint8)
    table.setflags(write=False)
    return table


def algebraic_immunity_of_function(truth_table: Sequence[int], n: int = 5) -> int:
    """Classical algebraic immunity ``AI(f)`` of one Boolean function.

    ``AI(f)`` (Meier, Pasalic and Carlet, EUROCRYPT 2004) is the minimum
    algebraic degree of a nonzero Boolean function ``g`` with ``f * g = 0``
    (an *annihilator* of ``f``) or ``(f + 1) * g = 0`` (an annihilator of the
    complement).  It is computed exactly: for each degree ``d = 0, 1, ...``
    the ``|supp| x N_d`` evaluation matrix of the monomials of degree at most
    ``d`` restricted to the support is built and a nonzero annihilator exists
    iff its GF(2) rank is smaller than the number ``N_d`` of monomials.  The
    first ``d`` for which either ``f`` or ``f + 1`` has one is returned.

    ``AI(f) <= ceil(n / 2)`` always (that many monomials outnumber the
    ``2^(n-1)`` support points of a balanced function), so 3 is the optimum
    for ``n = 5``.  A constant function has ``AI = 0``.

    Args:
        truth_table: ``2^n`` values in ``{0, 1}``, indexed by the input ``x``.
        n: Number of input variables.
    """
    t = _tables(n)
    evaluations = _monomial_evaluations(n)
    values = np.asarray(truth_table, dtype=np.int64) & 1
    if values.shape != (t.bt.size,):
        raise ValueError(f"truth table must have {t.bt.size} entries for n={n}")
    for degree in range(n + 1):
        columns = np.flatnonzero(t.bt.popcount <= degree)
        for target in (1, 0):
            support = np.flatnonzero(values == target)
            if support.size == 0:          # constant function: 1 annihilates it
                return 0
            if gf2_kernel_dimension(evaluations[np.ix_(support, columns)]) > 0:
                return degree
    return n   # unreachable for n >= 1, kept as a total-function guard


@lru_cache(maxsize=None)
def _graph_monomial_masks(n: int, degree: int) -> np.ndarray:
    """Masks of all monomials of degree ``<= degree`` in the ``2n`` variables.

    Bit ``i`` of a mask is input variable ``x_i`` (``i < n``) and bit ``n + i``
    is output variable ``y_i``; ``x_i`` / ``y_i`` are bit ``i`` (LSB first) of
    the input and the output, the same convention the ANF of
    :mod:`src.boolean_function` uses.  Cached and read-only.
    """
    masks: List[int] = []
    for size in range(degree + 1):
        for combination in itertools.combinations(range(2 * n), size):
            mask = 0
            for variable in combination:
                mask |= 1 << variable
            masks.append(mask)
    table = np.array(masks, dtype=np.int64)
    table.setflags(write=False)
    return table


def graph_equation_counts(sbox: Sequence[int], n: int = 5,
                          max_degree: int = 3) -> List[Dict[str, int]]:
    """Independent implicit equations of degree ``<= d`` in ``(x, S(x))``.

    For every ``d = 1 .. max_degree`` the ``2^n x N_d`` matrix of all
    monomials of degree at most ``d`` in the ``2n`` variables
    ``x_0..x_(n-1), y_0..y_(n-1)`` is evaluated on the ``2^n`` points of the
    graph ``{(x, S(x))}`` and its GF(2) rank is computed exactly.  The kernel
    dimension ``N_d - rank`` is the number of linearly independent equations
    of degree ``<= d`` that hold identically over the S-box -- the system a
    Courtois-Pieprzyk style algebraic attack would solve.

    Returns one dict per degree with ``degree``, ``monomials`` (``N_d``),
    ``rank`` and ``independent_equations``.
    """
    t, arr = _prepare(sbox, n)
    size = t.bt.size
    inputs = np.arange(size, dtype=np.int64)
    counts: List[Dict[str, int]] = []
    for degree in range(1, max_degree + 1):
        masks = _graph_monomial_masks(n, degree)
        input_part = masks & (size - 1)
        output_part = masks >> n
        evaluations = (((input_part[None, :] & ~inputs[:, None]) == 0)
                       & ((output_part[None, :] & ~arr[:, None]) == 0))
        rank = gf2_rank(pack_gf2_rows(evaluations))
        counts.append({
            "degree": degree,
            "monomials": int(masks.size),
            "rank": rank,
            "independent_equations": int(masks.size) - rank,
        })
    return counts


def guaranteed_equation_degree(n: int) -> int:
    """Smallest ``d`` at which implicit equations exist for *every* ``(n, n)``-S-box.

    The monomials of degree ``<= d`` in ``2n`` variables number
    ``sum_{i <= d} C(2n, i)``; once that exceeds the ``2^n`` points of the
    graph the evaluation matrix must have a kernel, so equations of degree
    ``d`` exist unconditionally.  For ``n = 5`` this is ``d = 2``
    (``1 + 10 + 45 = 56 > 32``), which caps the graph algebraic immunity of
    every 5-bit S-box at 2.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    total = 0
    for degree in range(2 * n + 1):
        total += math.comb(2 * n, degree)
        if total > (1 << n):
            return degree
    return 2 * n   # unreachable: 2^(2n) > 2^n for n >= 1


def compute_algebraic_immunity(sbox: Sequence[int], n: int = 5,
                               max_graph_degree: Optional[int] = None) -> Metrics:
    """Algebraic immunity of an S-box, in the two senses that are standard.

    Algebraic degree is *not* a measure of resistance to algebraic attacks;
    the two quantities below are, and they are inequivalent, so both are
    computed and named explicitly.

    **(a) Basic algebraic immunity** ``AI(S) = min_{b != 0} AI(b . S)``
    (Armknecht and Krause; Carlet, *Boolean Functions for Cryptography and
    Coding Theory*, 2021), where ``AI(f)`` is the classical algebraic
    immunity of :func:`algebraic_immunity_of_function` -- the minimum degree
    of a nonzero annihilator of ``f`` or of ``f + 1``.  This is the
    ``component_ai`` / ``ai_min`` group of keys.  The bound is
    ``AI(S) <= ceil(n / 2)`` (3 for ``n = 5``), so ``ai_is_optimal`` says
    whether every one of the ``2^n - 1`` components attains it.

    **(b) Graph algebraic immunity** ``AI_gr(S)``: the minimum algebraic
    degree of a nonzero Boolean function in the ``2n`` variables
    ``(x_0..x_(n-1), y_0..y_(n-1))`` that vanishes on the graph
    ``{(x, S(x)) : x in F_2^n}``.  Equivalently, the smallest ``d`` for which
    the implicit system of :func:`graph_equation_counts` has a nontrivial
    solution.  This is the quantity that governs Courtois-Pieprzyk style
    algebraic attacks, and it is *not* the minimum over the components.

    Not computed here (different definitions found in the literature): the
    fast algebraic immunity, the algebraic immunity of the inverse map, and
    per-component annihilator *counts*.

    Both quantities are exact (GF(2) Gaussian elimination on the ``2^n``
    evaluation points, no sampling).  ``graph_ai_saturated`` records that for
    ``n = 5`` the graph algebraic immunity of *any* S-box is at most
    :func:`guaranteed_equation_degree`, so ``AI_gr`` separates S-boxes only
    through the *number* of equations, reported per degree in
    ``graph_equations``.

    Args:
        sbox: Lookup table of ``2^n`` values.
        n: Bit width.
        max_graph_degree: Highest degree at which implicit equations are
            counted (default: one above :func:`guaranteed_equation_degree`).
    """
    t, arr = _prepare(sbox, n)
    size = t.bt.size
    component_truth = t.bt.parity[arr]                       # [x, b]
    ai = [algebraic_immunity_of_function(component_truth[:, b], n)
          for b in range(1, size)]
    coordinate_ai = [ai[int(mask) - 1] for mask in t.bt.coord_masks]
    bound = -(-n // 2)                                       # ceil(n / 2)

    guaranteed = guaranteed_equation_degree(n)
    if max_graph_degree is None:
        max_graph_degree = min(2 * n, guaranteed + 1)
    equations = graph_equation_counts(sbox, n, max_graph_degree)
    graph_ai = next((row["degree"] for row in equations
                     if row["independent_equations"] > 0), None)

    return {
        "component_ai": ai,
        "coordinate_ai": coordinate_ai,
        "ai_min": min(ai),
        "ai_max": max(ai),
        "ai_avg": float(np.mean(ai)),
        "ai_sbox": min(ai),
        "ai_spectrum": spectrum_of(np.array(ai, dtype=np.int64)),
        "ai_upper_bound": bound,
        "ai_is_optimal": min(ai) == bound,
        "graph_ai": graph_ai,
        "graph_equations": equations,
        "graph_ai_guaranteed_degree": guaranteed,
        "graph_ai_saturated": graph_ai is not None and graph_ai == guaranteed,
    }


# ============================================================
# 9b. Fixed points and the automatic metric identities
# ============================================================
def compute_fixed_points(sbox: Sequence[int], n: int = 5) -> Metrics:
    """Fixed points ``S(x) = x`` and opposite fixed points ``S(x) = ~x``.

    Neither count is invariant under affine equivalence (the constants of
    ``A F(B x ^ b) ^ a`` move them freely), which is why they belong to the
    representative-dependent criteria of an affine class.
    """
    _, arr = _prepare(sbox, n)
    x = np.arange(1 << n, dtype=np.int64)
    complement = (1 << n) - 1
    return {
        "fixed_points": int(np.count_nonzero(arr == x)),
        "opposite_fixed_points": int(np.count_nonzero(arr == (x ^ complement))),
    }


def check_metric_identities(sbox: Sequence[int], n: int = 5) -> Dict[str, Any]:
    """Re-derive the normalisation identities of the paper from raw tables.

    With ``LAT[a, b] = #{x : a.x = b.S(x)} - 2^(n-1)`` and
    ``LAP(S) = max |LAT| / 2^n`` (non-trivial masks) the identities are

    * ``W_{f_b}(a) = 2 LAT[a, b]`` for every ``(a, b)``;
    * ``C_max = max |W| / 2^n = 2 LAP(S)`` (maximal absolute correlation);
    * ``NL(S) = 2^(n-1) - max |W| / 2 = 2^(n-1) - 2^n LAP(S)
      = 2^(n-1) (1 - 2 LAP(S))``;
    * ``delta(S) = 2^n DP(S)``.

    Each side is computed by a different routine (the Walsh matrix, the LAT,
    the component nonlinearities, the DDT), so a normalisation slip anywhere
    in the pipeline makes the corresponding flag false.  Runners refuse to
    write results when any flag is false.
    """
    t, arr = _prepare(sbox, n)
    size = 1 << n
    walsh = _walsh(arr, t)
    lat = compute_lat(sbox, n)
    ddt = compute_ddt(sbox, n)
    nl = compute_nonlinearity(sbox, n)
    lap = float(lat["lap"])
    max_walsh = int(np.abs(walsh[:, 1:]).max())
    correlation_max = max_walsh / size
    checks = {
        "walsh_equals_twice_lat": bool(np.array_equal(walsh, 2 * lat["table"])),
        "correlation_max_equals_twice_lap": math.isclose(correlation_max, 2.0 * lap),
        "nl_equals_half_size_minus_size_times_lap":
            nl["nl_sbox"] == size // 2 - round(size * lap),
        "nl_equals_half_size_minus_half_max_walsh":
            nl["nl_sbox"] == size // 2 - max_walsh // 2,
        "du_equals_size_times_dp":
            ddt["differential_uniformity"] == round(size * ddt["dp"]),
    }
    return {
        "checks": checks,
        "all_hold": all(checks.values()),
        "values": {
            "lap": lap,
            "correlation_max": correlation_max,
            "nl_sbox": int(nl["nl_sbox"]),
            "nl_from_lap": size / 2 - size * lap,
            "differential_uniformity": int(ddt["differential_uniformity"]),
            "du_from_dp": size * float(ddt["dp"]),
        },
    }


# ============================================================
# Fast path for optimisers
# ============================================================
def fast_metrics(sbox_array: Sequence[int], n: int = 5) -> Dict[str, float]:
    """Core metrics of one S-box with minimal overhead (no validation).

    Returns ``nl_sbox``, ``max_bias``, ``differential_uniformity``,
    ``sac_avg``, ``sac_min``, ``bic_sac_avg`` and ``min_degree`` as Python
    scalars, identical to the corresponding values from :func:`analyze_sbox`.
    """
    t = _tables(n)
    bt = t.bt
    size = bt.size
    arr = np.asarray(sbox_array, dtype=np.int64)

    abs_w = np.abs(bt.sign @ bt.sign[arr])
    nl_sbox = (size - abs_w[:, 1:].max()) // 2
    max_bias = abs_w[1:, 1:].max() // 2

    diff = arr[:, None] ^ arr[bt.xor]
    du = np.bincount((t.ddt_offsets + diff).ravel(), minlength=size * size)[size:].max()

    bits = bt.bits[diff[:, bt.coord_masks]]
    sac = bits.sum(axis=0, dtype=np.int64) / size
    bic_sac_avg = _bic_sac_matrix(bits, t)[t.triu].mean()

    truth = 0.5 - 0.5 * bt.sign[arr]                     # component truth tables
    coeffs = bt.mobius @ truth % 2                        # ANF of every component
    min_degree = (coeffs * bt.popcount[:, None]).max(axis=0)[1:].min()

    return {
        "nl_sbox": int(nl_sbox),
        "max_bias": int(max_bias),
        "differential_uniformity": int(du),
        "sac_avg": float(sac.mean()),
        "sac_min": float(sac.min()),
        "bic_sac_avg": float(bic_sac_avg),
        "min_degree": int(min_degree),
    }


def fast_metrics_batch(sboxes: np.ndarray, n: int = 5) -> Dict[str, np.ndarray]:
    """Batched :func:`fast_metrics` for a ``(K, 2^n)`` array of S-boxes.

    Returns the same keys, each an array of length ``K``.
    """
    t = _tables(n)
    bt = t.bt
    size = bt.size
    arr = np.asarray(sboxes, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != size:
        raise ValueError(f"expected an array of shape (K, {size})")
    k = arr.shape[0]

    abs_w = np.abs(bt.sign @ bt.sign[arr])                # (K, a, b)
    nl_sbox = (size - abs_w[:, :, 1:].max(axis=(1, 2))) // 2
    max_bias = abs_w[:, 1:, 1:].max(axis=(1, 2)) // 2

    diff = arr[:, :, None] ^ arr[:, bt.xor]               # (K, x, dx)
    offsets = (np.arange(k, dtype=np.int64) * size * size)[:, None, None] + t.ddt_offsets
    ddt = np.bincount((offsets + diff).ravel(), minlength=k * size * size)
    du = ddt.reshape(k, size, size)[:, 1:, :].max(axis=(1, 2))

    bits = bt.bits[diff[:, :, bt.coord_masks]]            # (K, x, i, j)
    sac = bits.sum(axis=1, dtype=np.int64) / size         # (K, i, j)
    flat = bits.reshape(k, -1, n).astype(np.float64)
    ones = flat.sum(axis=1)                               # (K, n)
    both = np.einsum("kxj,kxl->kjl", flat, flat)
    pair = (ones[:, :, None] + ones[:, None, :] - 2.0 * both) / flat.shape[1]
    bic_sac_avg = pair[:, t.triu[0], t.triu[1]].mean(axis=1)

    truth = 0.5 - 0.5 * bt.sign[arr]                      # (K, x, b)
    coeffs = bt.mobius @ truth % 2                        # (K, u, b)
    degrees = (coeffs * bt.popcount[None, :, None]).max(axis=1)
    min_degree = degrees[:, 1:].min(axis=1)

    return {
        "nl_sbox": nl_sbox.astype(np.int64),
        "max_bias": max_bias.astype(np.int64),
        "differential_uniformity": du.astype(np.int64),
        "sac_avg": sac.mean(axis=(1, 2)),
        "sac_min": sac.min(axis=(1, 2)),
        "bic_sac_avg": bic_sac_avg,
        "min_degree": min_degree.astype(np.int64),
    }


# ============================================================
# Complete analysis
# ============================================================
def _collect(sbox: Sequence[int], n: int) -> Dict[str, Optional[Metrics]]:
    """Run every metric once and return the nested result dictionaries."""
    bij = check_bijective(sbox, n)
    return {
        "bijective": bij,
        "sac": compute_sac(sbox, n),
        "nonlinearity": compute_nonlinearity(sbox, n),
        "hamming_distance": compute_hamming_distances(sbox, n),
        "lat": compute_lat(sbox, n),
        "ddt": compute_ddt(sbox, n),
        "bic": compute_bic(sbox, n),
        "bct": compute_bct(sbox, n) if bij["is_bijective"] else None,
        "degree": compute_algebraic_degree(sbox, n),
        "algebraic_immunity": compute_algebraic_immunity(sbox, n),
        "fixed_points": compute_fixed_points(sbox, n),
        "identities": check_metric_identities(sbox, n),
    }


def _flatten(nested: Dict[str, Optional[Metrics]], name: str, n: int) -> Metrics:
    """Flatten nested results into one JSON-serialisable dict (no big tables)."""
    bij, sac, nl = nested["bijective"], nested["sac"], nested["nonlinearity"]
    hd, lat, ddt = nested["hamming_distance"], nested["lat"], nested["ddt"]
    bic, bct, deg = nested["bic"], nested["bct"], nested["degree"]
    ai = nested["algebraic_immunity"]
    fp, ident = nested["fixed_points"], nested["identities"]
    return {
        "name": name,
        "n": n,
        "size": 1 << n,
        "is_bijective": bij["is_bijective"],
        "unique_outputs": bij["unique_outputs"],
        "all_components_balanced": bij["all_components_balanced"],
        "balanced_components": bij["balanced_components"],
        "sac_avg": sac["average"],
        "sac_min": sac["sac_min"],
        "sac_max": sac["sac_max"],
        "sac_max_deviation": sac["max_deviation"],
        "sac_mean_deviation": sac["mean_deviation"],
        # Explicit D_max / D_mean names for the two symmetric deviation
        # measures (same numbers as sac_max_deviation / sac_mean_deviation).
        "sac_d_max": sac["d_max"],
        "sac_d_mean": sac["d_mean"],
        "sac_tolerance": sac["tolerance"],
        "satisfies_sac": sac["satisfies_sac"],
        "satisfies_sac_average": sac["satisfies_sac_average"],
        "sac_verdict": sac["sac_verdict"],
        "sac_matrix": sac["matrix"].tolist(),
        "nl_per_bit": list(nl["per_bit"]),
        "nl_coord_min": nl["minimum"],
        "nl_coord_max": nl["maximum"],
        "nl_coord_avg": nl["average"],
        "nl_sbox": nl["nl_sbox"],
        "nl_min": nl["nl_min"],
        "nl_max_components": nl["nl_max_components"],
        "nl_avg_components": nl["nl_avg_components"],
        "nl_upper_bound": nl["nl_upper_bound"],
        "component_nl": list(nl["component_nl"]),
        "max_bias": lat["max_bias"],
        "lap": lat["lap"],
        "lap_half": lat["lap_half"],
        "linear_probability": lat["linear_probability"],
        "correlation_max": ident["values"]["correlation_max"],
        "metric_identities": dict(ident["checks"]),
        "metric_identities_hold": ident["all_hold"],
        "fixed_points": fp["fixed_points"],
        "opposite_fixed_points": fp["opposite_fixed_points"],
        "differential_uniformity": ddt["differential_uniformity"],
        "dp": ddt["dp"],
        "bic_sac_avg": bic["avg_bic_sac"],
        "bic_sac_min": bic["min_bic_sac"],
        "bic_sac_max": bic["max_bic_sac"],
        "bic_nl_avg": bic["avg_bic_nl"],
        "bic_nl_min": bic["min_bic_nl"],
        "bic_nl_max": bic["max_bic_nl"],
        "bic_sac_values": list(bic["bic_sac_values"]),
        "bic_nl_values": list(bic["bic_nl_values"]),
        "boomerang_uniformity": None if bct is None else bct["boomerang_uniformity"],
        "bct_spectrum": None if bct is None else [list(pair) for pair in bct["spectrum"]],
        "bct_nontrivial_entries": None if bct is None else bct["nontrivial_entries"],
        "bct_max_count": None if bct is None else bct["max_count"],
        "bct_mean_nontrivial": None if bct is None else bct["mean_nontrivial"],
        "component_ai": list(ai["component_ai"]),
        "coordinate_ai": list(ai["coordinate_ai"]),
        "ai_min": ai["ai_min"],
        "ai_max": ai["ai_max"],
        "ai_avg": ai["ai_avg"],
        "ai_sbox": ai["ai_sbox"],
        "ai_spectrum": [list(pair) for pair in ai["ai_spectrum"]],
        "ai_upper_bound": ai["ai_upper_bound"],
        "ai_is_optimal": ai["ai_is_optimal"],
        "graph_ai": ai["graph_ai"],
        "graph_equations": [dict(row) for row in ai["graph_equations"]],
        "graph_ai_guaranteed_degree": ai["graph_ai_guaranteed_degree"],
        "graph_ai_saturated": ai["graph_ai_saturated"],
        "degree_per_bit": list(deg["per_bit"]),
        "min_degree": deg["min_degree"],
        "max_degree": deg["max_degree"],
        "max_possible_degree": deg["max_possible_degree"],
        "component_degrees": list(deg["component_degrees"]),
        "hd_avg": hd["average"],
        "hd_min": hd["minimum"],
        "hd_max": hd["maximum"],
    }


def analyze_sbox(sbox: Sequence[int], name: str = "DAC", n: int = 5) -> Metrics:
    """All security metrics of an S-box as one flat, JSON-serialisable dict.

    No printing. Tables (LAT, DDT, BCT, Walsh spectrum) are left to the
    individual ``compute_*`` functions; everything scalar or small is here.
    """
    return _flatten(_collect(sbox, n), name, n)


def paper_summary_heading(sbox: Sequence[int], name: str = "DAC") -> str:
    """Heading of the base-paper summary block for the S-box being analysed.

    The paper's numbers describe *its own* DAC S-box (Table 3).  When some
    other table is analysed they are still a useful reference, but they are
    not a target for it, so the heading says whose values they are.
    """
    if literature.is_paper_dac_table(sbox):
        return "  SUMMARY vs BASE-PAPER VALUES (DAC)"
    return (f"  SUMMARY of {name} -- reference column: DAC (base paper)\n"
            "  The paper's values describe its own DAC S-box, not this one,\n"
            "  so they are a reference, not a target, and carry no verdict.")


def paper_summary_table(flat: Metrics, sbox: Sequence[int],
                        name: str = "DAC") -> PrettyTable:
    """Computed metrics beside the base paper's claims about its DAC S-box.

    Every row comes from :data:`src.literature.PAPER_DAC_SUMMARY_ROWS`, which
    pairs each paper value with the metrics key of the *same* normalisation
    (the paper's printed LAP 0.125 is ``max_bias / 2^(n+1)``, so the standard
    ``lap`` row is compared against 0.25 and can actually pass).  A
    ``PASS`` / ``CHECK`` verdict is given only when ``sbox`` really is the
    paper's Table 3 S-box; otherwise the paper column is a labelled
    reference and the verdict column is omitted.
    """
    is_paper_dac = literature.is_paper_dac_table(sbox)
    table = PrettyTable()
    if is_paper_dac:
        table.field_names = ["Metric", "Computed", "Paper Value", "Match"]
    else:
        table.field_names = ["Metric", f"Computed ({name})", "DAC (base paper)"]

    def add(label: str, computed: str, paper: str, verdict: str) -> None:
        table.add_row([label, computed, paper] + ([verdict] if is_paper_dac else []))

    add("Bijective", str(flat["is_bijective"]), "Yes",
        "PASS" if flat["is_bijective"] else "FAIL")
    for label, key, paper_value, tolerance in literature.PAPER_DAC_SUMMARY_ROWS:
        add(label, f"{flat[key]:.4f}", f"{paper_value:g}",
            "PASS" if abs(flat[key] - paper_value) < tolerance else "CHECK")
    add("NL(S)", f"{flat['nl_sbox']}", "n/r", "-")
    return table


def competitor_table(flat: Metrics, name: str = "DAC") -> PrettyTable:
    """The analysed S-box beside the base paper's Table 8 competitor rows.

    Columns and competitor values both come from :mod:`src.literature`, the
    single source shared with :mod:`src.comparative_analysis`.  The computed
    row uses those per-column formats except in the first column, where the
    exact average Hamming distance is printed at 4 decimals (2.5625) instead
    of the 3 the literature quotes its own values at.
    """
    labels = [label for _, label, _ in literature.TABLE8_COLUMNS]
    table = PrettyTable()
    table.field_names = ["S-Box", labels[0], "NL(S)", *labels[1:]]
    computed = [f"{flat['hd_avg']:.4f}"] + [
        format(flat[key], fmt) for key, _, fmt in literature.TABLE8_COLUMNS[1:]]
    table.add_row([f"{name} (ours)", computed[0], f"{flat['nl_sbox']}", *computed[1:]])
    for competitor in literature.LITERATURE_REPORTED:
        row = literature.table8_row(competitor)
        table.add_row([competitor, row[0], "n/r", *row[1:]])
    return table


def full_security_analysis(sbox: Sequence[int], name: str = "DAC", n: int = 5) -> Metrics:
    """Run the complete analysis, print the report and return all results.

    The returned dict keeps the legacy nested entries (``bijective``, ``sac``,
    ``nonlinearity``, ``hamming_distance``, ``lat``, ``ddt``, ``bic``,
    ``bct``) and adds ``degree``, ``algebraic_immunity`` and ``flat`` (the
    :func:`analyze_sbox` dict).
    """
    nested = _collect(sbox, n)
    flat = _flatten(nested, name, n)
    size = 1 << n
    bij, sac, nl = nested["bijective"], nested["sac"], nested["nonlinearity"]
    hd, lat_result, ddt_result = nested["hamming_distance"], nested["lat"], nested["ddt"]
    bic, bct, deg = nested["bic"], nested["bct"], nested["degree"]
    imm = nested["algebraic_immunity"]

    print("=" * 72)
    print(f"  COMPREHENSIVE SECURITY ANALYSIS: {name} S-Box ({n}-bit)")
    print("=" * 72)

    print("\n[1] BIJECTIVE PROPERTY")
    print(f"    Is bijective: {bij['is_bijective']}")
    print(f"    Unique outputs: {bij['unique_outputs']}/{bij['expected_unique']}")
    print(f"    Balanced components: {bij['balanced_components']}/{size - 1}")

    print("\n[2] STRICT AVALANCHE CRITERION (SAC)")
    print(f"    Average SAC: {sac['average']:.4f} (ideal: 0.5)")
    print(f"    Min: {sac['sac_min']:.4f}, Max: {sac['sac_max']:.4f}")
    print(f"    D_max = max |P_ij - 0.5|: {sac['d_max']:.4f}; "
          f"D_mean = mean |P_ij - 0.5|: {sac['d_mean']:.4f}")
    print(f"    Satisfies SAC (element-wise, tol {sac['tolerance']:.3f}): {sac['satisfies_sac']}")
    print(f"    Satisfies SAC (legacy average >= 0.5): {sac['satisfies_sac_average']}")
    print("    SAC Matrix (rows: input bit, cols: output bit, MSB first):")
    for i in range(n):
        row = " ".join(f"{sac['matrix'][i][j]:.3f}" for j in range(n))
        print(f"      [{row}]")

    print("\n[3] NONLINEARITY (Walsh spectrum)")
    for i, v in enumerate(nl["per_bit"]):
        print(f"    Coordinate {i}: NL = {v}")
    print(f"    Coordinate NL: min {nl['minimum']}, avg {nl['average']:.2f}")
    print(f"    NL(S) over all {size - 1} components: {nl['nl_sbox']} "
          f"(avg {nl['nl_avg_components']:.2f}, max {nl['nl_max_components']}, "
          f"SCV bound {nl['nl_upper_bound']})")

    print("\n    HAMMING DISTANCE ANALYSIS")
    print(f"    Average Hd: {hd['average']:.4f}")
    print(f"    Min Hd: {hd['minimum']}, Max Hd: {hd['maximum']}")

    print("\n[4] LINEAR CRYPTANALYSIS (LAT)")
    print(f"    Max |bias| in LAT (a, b != 0): {lat_result['max_bias']}")
    print(f"    LAP = max_bias / 2^n: {lat_result['lap']:.4f}")
    print(f"    max_bias / 2^(n+1) (base paper's normalisation): {lat_result['lap_half']:.4f}")
    print(f"    Matsui LP = (2 * max_bias / 2^n)^2: {lat_result['linear_probability']:.4f}")

    print("\n[5] DIFFERENTIAL CRYPTANALYSIS (DDT)")
    print(f"    Differential uniformity: {ddt_result['differential_uniformity']}")
    print(f"    Differential Probability: {ddt_result['dp']:.4f}")

    print("\n[6] BIT INDEPENDENCE CRITERION (BIC)")
    print(f"    BIC-SAC: avg {bic['avg_bic_sac']:.4f}, min {bic['min_bic_sac']:.4f}, "
          f"max {bic['max_bic_sac']:.4f}")
    print(f"    BIC-NL:  avg {bic['avg_bic_nl']:.2f}, min {bic['min_bic_nl']}, "
          f"max {bic['max_bic_nl']}")

    print("\n[7] BOOMERANG CONNECTIVITY TABLE (BCT)")
    if bct is None:
        print("    Not defined (S-box is not bijective)")
    else:
        print(f"    Boomerang Uniformity: {bct['boomerang_uniformity']} "
              f"(attained by {bct['max_count']} of the {bct['nontrivial_entries']} "
              f"entries with a, b != 0)")
        print("    BCT spectrum (value: count over a, b != 0): "
              + ", ".join(f"{value}: {count}" for value, count in bct["spectrum"]))
        print(f"    Mean non-trivial entry: {bct['mean_nontrivial']:.4f}")

    print("\n[8] ALGEBRAIC DEGREE (Mobius transform)")
    print(f"    Coordinate degrees: {deg['per_bit']}")
    print(f"    All {size - 1} components: min {deg['min_degree']}, max {deg['max_degree']} "
          f"(max possible for a permutation: {deg['max_possible_degree']})")
    print(f"    All components balanced: {deg['all_components_balanced']}")

    print("\n[9] ALGEBRAIC IMMUNITY (exact GF(2) linear algebra)")
    print(f"    Basic AI(S) = min over the {size - 1} components: {imm['ai_min']} "
          f"(max {imm['ai_max']}, avg {imm['ai_avg']:.4f}, "
          f"optimum ceil(n/2) = {imm['ai_upper_bound']})")
    print("    Component AI spectrum (value: count): "
          + ", ".join(f"{value}: {count}" for value, count in imm["ai_spectrum"]))
    print(f"    Every component optimal: {imm['ai_is_optimal']}")
    print(f"    Graph algebraic immunity AI_gr(S): {imm['graph_ai']} "
          f"(every {n}-bit S-box has AI_gr <= "
          f"{imm['graph_ai_guaranteed_degree']}, so this is a saturated bound)")
    for row in imm["graph_equations"]:
        print(f"      degree <= {row['degree']}: {row['monomials']} monomials in "
              f"{2 * n} variables, rank {row['rank']} -> "
              f"{row['independent_equations']} independent equations")

    print(f"\n{'=' * 72}")
    print(paper_summary_heading(sbox, name))
    print(f"{'=' * 72}")
    print(paper_summary_table(flat, sbox, name))

    print(f"\n{'=' * 72}")
    print("  COMPARISON WITH COMPETITORS (base paper Table 8)")
    print("  Note: the paper's 'Non-Linearity' column is the average Hamming")
    print("  distance between x and S(x); the true NL(S) is listed separately.")
    print(f"{'=' * 72}")
    print(competitor_table(flat, name))

    nested["flat"] = flat
    return nested
