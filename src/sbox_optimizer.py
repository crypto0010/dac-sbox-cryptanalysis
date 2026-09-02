"""S-box optimisation engine for the 5-bit DAC design.

The original DAC S-box (paper Table 3) has differential uniformity 8,
nonlinearity 8 and a maximum SAC deviation of 0.25.  For n = 5 the theoretical
optimum of a bijective S-box is differential uniformity 2 (APN) together with
either nonlinearity 12 and algebraic degree <= 3 (almost bent, e.g. the Gold
power map x^3) or nonlinearity 10 and the maximal degree 4 (e.g. the
inversion x^-1).  This module reaches those optima deterministically and
keeps the paper's narrative intact: every random decision of every strategy
is drawn from :class:`ChaoticRNG`, a generator driven by the corrected
tent-logistic map of :mod:`src.chaotic_maps`.

Strategies (:func:`optimize_dac_sbox`)
--------------------------------------
``affine_inverse`` (default)
    Chaos-driven search over the affine equivalents
    ``S(x) = A * INV(B * x ^ b) ^ a`` of the inversion ``INV(x) = x^-1`` in
    GF(2^5) (``INV(0) = 0``).  Differential uniformity (2), nonlinearity (10),
    algebraic degree (4) and boomerang uniformity (2) are invariant under
    affine equivalence, whereas the SAC matrix and the BIC-SAC values are
    not: they are read directly from the autocorrelation table of ``INV``
    (see :func:`autocorrelation_table`), so tens of thousands of candidates
    are scored per second.  The chaotic generator proposes the input matrix
    ``B``; for each proposal the output matrix ``A`` is the greedy
    minimum-deviation basis of the autocorrelation sub-table, refined by a
    chaos-driven descent; the constants ``a, b`` (which change no spectral
    metric) are then chosen to remove fixed and complement-fixed points.
``affine_gold``
    The same search for the Gold map ``x^3`` (almost bent: nonlinearity 12,
    degree 2, differential uniformity 2).
``simulated_annealing``
    The paper's swap-based simulated annealing (Algorithm 1) made sound: the
    energy :func:`sa_energy` is computed from the real DDT and Walsh
    spectrum, the initial temperature is calibrated from the median uphill
    move, the cooling schedule is geometric and the generator is seeded.
    It is kept for comparison; swap moves do not reach the APN optimum.
``all``
    Runs the three strategies and returns the best by :func:`objective_key`
    (differential uniformity, then algebraic degree, then nonlinearity, then
    the avalanche deviations), i.e. the inversion class.

:func:`enumerate_input_bases` certifies the affine strategies: it scores
every one of the 83,328 unordered input bases of GF(2)^5 (the SAC and BIC
keys depend on ``B`` only through its column set) and reports the best key
of the class, which the chaos-driven search attains for every seed tried.

Conventions: 5-bit values 0..31, coordinate ``i`` of a value is bit
``4 - i`` (MSB first), matrices over GF(2) are given as five row masks (row
``j`` is the mask of output bit ``j``, MSB first) so that
``y_j = parity(row_j & x)``.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .boolean_function import BitTables, bit_tables
from .chaotic_maps import TentLogisticMap, canonical_seed
from .security_analysis import analyze_sbox

N_BITS = 5
SIZE = 1 << N_BITS

#: Irreducible (and primitive) polynomial x^5 + x^2 + 1 defining GF(2^5).
#: Any other irreducible quintic gives a linearly equivalent field, hence the
#: same affine classes.
GF_POLY = 0b100101
INVERSION_EXPONENT = 30   #: x^30 = x^-1 in GF(2^5)* (and 0 -> 0): APN, NL 10, degree 4
GOLD_EXPONENT = 3         #: x^3, Gold / almost bent: APN, NL 12, degree 2

DEFAULT_SEED = 0.5        #: key seed of the paper's S-box generation
STRATEGIES = ("affine_inverse", "affine_gold", "simulated_annealing")
DEFAULT_ITERATIONS = {
    "affine_inverse": 4096,        # chaos-proposed input matrices B
    "affine_gold": 512,
    "simulated_annealing": 20000,  # swap moves
}

_BT: BitTables = bit_tables(N_BITS)
_X = np.arange(SIZE, dtype=np.int64)
_DDT_OFFSETS = np.broadcast_to((_X * SIZE)[None, :], (SIZE, SIZE))
_PAIRS = np.triu_indices(N_BITS, 1)
_COORD_BITS = _BT.bits.T.astype(np.int64)   # (5, 32): bit i (MSB first) of x

Result = Dict[str, Any]


# ============================================================
# Chaos-driven random source
# ============================================================
class ChaoticRNG:
    """Deterministic random source driven by the tent-logistic map.

    The generator iterates :class:`~src.chaotic_maps.TentLogisticMap`
    (paper Eqs. 3-5, parameters ``r_logistic``, ``r_tent`` and the regulating
    parameter ``r``) from the key seed, discards ``transient`` iterations and
    turns every further orbit point ``s`` into a uniform variate
    ``u = frac(2^16 * s)``, i.e. it keeps mantissa bits 17..53 of ``s``.  The
    leading bits of ``s`` follow the map's non-uniform invariant density (the
    orbit lives in about [0.18, 0.91]); the bits behind them are scrambled by
    the positive Lyapunov exponent (about 0.82 bit per iteration) and pass
    chi-square (64 cells, consecutive pairs) and lag-1 correlation tests.
    Both operations are exact in floating point, so the stream is
    reproducible bit for bit.

    Because the map satisfies ``f(1 - s) = f(s)``, the seeds ``x0`` and
    ``1 - x0`` describe the same orbit from the first step on, so the key
    space is (0, 1/2]: a seed above 1/2 is folded to ``1 - x0`` by
    :func:`~src.chaotic_maps.canonical_seed` (attribute ``canonical_seed``)
    before the map is iterated.  The fold is what makes mirror seeds produce
    identical streams bit for bit -- evaluating ``f`` on ``x0`` and on
    ``1 - x0`` rounds differently, and for a decimal seed such as 0.7 the
    double ``1 - 0.7`` is not the double 0.3 but its upper neighbour, which
    is a different (equally valid) key.  0 is a fixed point of the map and 1
    maps onto it in one step, so both endpoints are rejected.

    Raises:
        ValueError: if ``seed`` is not strictly inside (0, 1) or ``block``
            (orbit points generated per refill) is smaller than 1.
    """

    SHIFT = 16

    def __init__(self, seed: float, r: float = 8.0, r_logistic: float = 3.9,
                 r_tent: float = 1.8, transient: int = 1000,
                 block: int = 4096) -> None:
        self.seed = float(seed)
        self.canonical_seed = canonical_seed(self.seed)
        if block < 1:
            raise ValueError(f"block must be >= 1, got {block}")
        self.transient = int(transient)
        self.map = TentLogisticMap(r_logistic=r_logistic, r_tent=r_tent,
                                   seed=self.canonical_seed, r=r)
        self.draws = 0
        self._scale = float(1 << self.SHIFT)
        self._buffer = np.empty(int(block))
        self._uniform = np.empty(0)
        self._pos = 0
        state = self.canonical_seed
        step = self.map.step
        for _ in range(self.transient):
            state = step(state)
        self._state = state

    def describe(self) -> Dict[str, Any]:
        """Parameters of the generator for provenance records."""
        return {
            "map": "tent-logistic (src.chaotic_maps.TentLogisticMap)",
            "seed": self.seed,
            "canonical_seed": self.canonical_seed,
            "r_logistic": self.map.r_logistic,
            "r_tent": self.map.r_tent,
            "r": self.map.r,
            "transient": self.transient,
            "post_processing": f"u = frac(2^{self.SHIFT} * s)",
            "draws": self.draws,
        }

    def _refill(self) -> None:
        buf = self._buffer
        state = self._state
        step = self.map.step
        for i in range(buf.size):
            state = step(state)
            buf[i] = state
        if state <= 0.0 or np.any(buf[1:] == buf[:-1]):
            raise RuntimeError("the chaotic orbit collapsed onto a fixed point")
        self._state = state
        self._uniform = (buf * self._scale) % 1.0
        self._pos = 0

    def random(self, size: Optional[int] = None):
        """Uniform variates in [0, 1): a float, or an array of ``size``."""
        if size is None:
            return float(self.random(1)[0])
        out = np.empty(int(size))
        filled = 0
        while filled < out.size:
            if self._pos >= self._uniform.size:
                self._refill()
            take = min(out.size - filled, self._uniform.size - self._pos)
            out[filled:filled + take] = self._uniform[self._pos:self._pos + take]
            self._pos += take
            filled += take
        self.draws += out.size
        return out

    def integers(self, high: int, size=None):
        """Integers in ``0 .. high - 1``: a Python int, or an array of shape ``size``."""
        if high < 1:
            raise ValueError("high must be >= 1")
        if size is None:
            return int(self.random() * high)
        shape = (size,) if isinstance(size, int) else tuple(size)
        count = int(np.prod(shape))
        return (self.random(count) * high).astype(np.int64).reshape(shape)

    def pair(self, n: int) -> Tuple[int, int]:
        """Two distinct indices in ``0 .. n - 1`` from exactly two draws."""
        i = self.integers(n)
        j = self.integers(n - 1)
        return i, j + (j >= i)


# ============================================================
# GF(2^5) power maps and GF(2) linear algebra on 5-bit masks
# ============================================================
def gf_mul(a: int, b: int, poly: int = GF_POLY) -> int:
    """Multiply two elements of GF(2^5) = F_2[x] / (poly)."""
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & SIZE:
            a ^= poly
    return result


@lru_cache(maxsize=None)
def power_map(exponent: int) -> np.ndarray:
    """Lookup table of ``x -> x^exponent`` over GF(2^5) (read-only, ``0 -> 0``)."""
    table = np.empty(SIZE, dtype=np.int64)
    for x in range(SIZE):
        value = 1
        for _ in range(exponent):
            value = gf_mul(value, x)
        table[x] = value
    table.setflags(write=False)
    return table


def _linear_tables(columns: np.ndarray) -> np.ndarray:
    """Lookup tables of GF(2)-linear maps given by their columns.

    ``columns[..., i]`` is the image of input bit ``i`` (MSB first); the
    result has shape ``(..., 32)`` with ``table[..., x] = L(x)``.
    """
    cols = np.asarray(columns, dtype=np.int64)
    return np.bitwise_xor.reduce(cols[..., :, None] * _COORD_BITS, axis=-2)


def _transpose(masks: np.ndarray) -> np.ndarray:
    """Transpose 5x5 GF(2) matrices given as five masks (rows <-> columns)."""
    bits = _BT.bits[np.asarray(masks, dtype=np.int64)].astype(np.int64)  # (..., 5, 5)
    return (np.swapaxes(bits, -1, -2) * _BT.coord_masks).sum(axis=-1)


def _is_invertible(columns: np.ndarray) -> np.ndarray:
    """Element-wise invertibility of matrices given by their columns ``(..., 5)``."""
    tables = _linear_tables(columns)
    return np.all(np.sort(tables, axis=-1) == _X, axis=-1)


@lru_cache(maxsize=None)
def autocorrelation_table(exponent: int) -> np.ndarray:
    """Autocorrelation table ``AC[c, d] = sum_y (-1)^(c . (F(y) ^ F(y ^ d)))``.

    For ``F = x^exponent``.  The SAC matrix of every affine equivalent
    ``S = A o F o (B x ^ b) ^ a`` is a 5x5 sub-table of ``AC``:
    ``P[i, j] = 1/2 - AC[row_j(A), col_i(B)] / 64``, and the BIC-SAC value of
    the output pair ``(j, k)`` is the mean over ``i`` of
    ``1/2 - AC[row_j(A) ^ row_k(A), col_i(B)] / 64``.  The constants ``a``
    and ``b`` play no role.
    """
    f = power_map(exponent)
    diff = f[:, None] ^ f[_BT.xor]                       # [y, d]
    table = _BT.sign[:, diff].sum(axis=1).astype(np.int64)  # [c, d]
    table.setflags(write=False)
    return table


# ============================================================
# Objective
# ============================================================
def objective_key(metrics: Mapping[str, Any]) -> Tuple[float, ...]:
    """Lexicographic cost of an S-box (lower is better) from :func:`analyze_sbox`.

    ``(DU, -min degree, -NL(S), max |P_ij - 1/2|, |SAC - 1/2|,
    max |BIC-SAC_jk - 1/2|, |BIC-SAC - 1/2|)``: differential uniformity first,
    then the minimum algebraic degree over all 31 components, then their
    minimum nonlinearity, then the element-wise and average SAC deviations,
    then the element-wise and average BIC-SAC deviations.

    Degree is ranked above nonlinearity because for n = 5 an APN permutation
    reaches nonlinearity 12 only with degree <= 3 (almost bent), and the
    quadratic Gold class in particular is the one exposed to algebraic,
    integral and higher-order differential attacks; the maximal degree 4 with
    nonlinearity 10 (inversion class) is the paper's stated optimum.  The
    almost bent class remains available through ``strategy="affine_gold"``.
    """
    bic_dev = max(abs(v - 0.5) for v in metrics["bic_sac_values"])
    return (
        int(metrics["differential_uniformity"]),
        -int(metrics["min_degree"]),
        -int(metrics["nl_sbox"]),
        round(float(metrics["sac_max_deviation"]), 6),
        round(abs(float(metrics["sac_avg"]) - 0.5), 6),
        round(bic_dev, 6),
        round(abs(float(metrics["bic_sac_avg"]) - 0.5), 6),
    )


# ============================================================
# Affine-equivalence search
# ============================================================
@dataclass(frozen=True)
class AffineParams:
    """``S(x) = A * F(B * x ^ b) ^ a`` with ``F = x^exponent`` over GF(2^5).

    ``a_rows`` / ``b_rows`` are the five row masks (MSB first) of the
    invertible matrices ``A`` and ``B``; ``a`` and ``b`` are 5-bit constants.
    """

    exponent: int
    a_rows: Tuple[int, ...]
    a: int
    b_rows: Tuple[int, ...]
    b: int

    def sbox(self) -> np.ndarray:
        """Build the 32-entry lookup table."""
        table_a = _linear_tables(_transpose(np.array(self.a_rows)))
        table_b = _linear_tables(_transpose(np.array(self.b_rows)))
        return table_a[power_map(self.exponent)[table_b ^ self.b]] ^ self.a

    def as_dict(self) -> Dict[str, Any]:
        base = "x^-1" if self.exponent == INVERSION_EXPONENT else f"x^{self.exponent}"
        return {
            "formula": "S(x) = A * F(B * x ^ b) ^ a",
            "base_function": f"F(x) = {base} over GF(2^5) (F(0) = 0)",
            "exponent": self.exponent,
            "polynomial": "x^5 + x^2 + 1 (0b100101)",
            "matrix_convention": "row j (MSB first) is the mask of output bit j",
            "A_rows": list(self.a_rows),
            "a": self.a,
            "B_rows": list(self.b_rows),
            "b": self.b,
        }


def _sample_invertible(rng: ChaoticRNG, count: int) -> np.ndarray:
    """``count`` invertible matrices as column masks ``(count, 5)`` (rejection)."""
    cols = 1 + rng.integers(SIZE - 1, size=(count, N_BITS))
    bad = np.flatnonzero(~_is_invertible(cols))
    while bad.size:
        cols[bad] = 1 + rng.integers(SIZE - 1, size=(bad.size, N_BITS))
        bad = bad[~_is_invertible(cols[bad])]
    return cols


def _greedy_output_rows(ac: np.ndarray, b_cols: np.ndarray) -> np.ndarray:
    """Output rows ``A`` minimising the SAC deviation for each input matrix.

    For every candidate ``B`` (columns ``b_cols[k]``) each nonzero mask ``c``
    is weighted by ``(max_i |AC[c, d_i]|, sum_i |AC[c, d_i]|, c)`` packed
    into one integer, and the rows of ``A`` are the greedy minimum-weight
    basis of GF(2)^5.  Bases form a matroid, so the greedy basis is Gale
    optimal: its sorted weight vector is component-wise minimal, hence it
    minimises the largest SAC deviation of any row and, among such bases,
    the sum of the packed row weights.
    """
    count = b_cols.shape[0]
    dev = np.abs(ac)[:, b_cols]                          # (32, K, 5), [c, k, i]
    weight = (dev.max(axis=2) * 8192 + dev.sum(axis=2) * 32 + _X[:, None]).T
    weight[:, 0] = np.iinfo(np.int64).max               # never pick c = 0
    order = np.argsort(weight, axis=1, kind="stable")   # (K, 32)
    rows = np.zeros((count, N_BITS), dtype=np.int64)
    span = np.zeros((count, SIZE), dtype=bool)
    span[:, 0] = True
    filled = np.zeros(count, dtype=np.int64)
    ks = np.arange(count)
    for position in range(SIZE - 1):
        c = order[:, position]
        take = ~span[ks, c] & (filled < N_BITS)
        rows[take, filled[take]] = c[take]
        shifted = np.take_along_axis(span, _BT.xor[c], axis=1)
        span[take] |= shifted[take]
        filled[take] += 1
    return rows


def _avalanche_keys(ac: np.ndarray, a_rows: np.ndarray, b_cols: np.ndarray
                    ) -> Tuple[np.ndarray, ...]:
    """Integer SAC / BIC-SAC deviation keys of affine candidates (lower is better).

    ``(64 * max |P_ij - 1/2|, 1600 * |SAC - 1/2|,
    320 * max_jk |BIC-SAC_jk - 1/2|, 3200 * |BIC-SAC - 1/2|)`` for ``a_rows``,
    ``b_cols`` of shape ``(..., 5)``, all exact integers.
    """
    sub = ac[a_rows[..., :, None], b_cols[..., None, :]]              # [.., j, i]
    pair_masks = a_rows[..., _PAIRS[0]] ^ a_rows[..., _PAIRS[1]]      # (.., 10)
    sub_pairs = ac[pair_masks[..., :, None], b_cols[..., None, :]].sum(axis=-1)
    return (
        np.abs(sub).max(axis=(-1, -2)),
        np.abs(sub.sum(axis=(-1, -2))),
        np.abs(sub_pairs).max(axis=-1),
        np.abs(sub_pairs.sum(axis=-1)),
    )


def _refine_output_rows(ac: np.ndarray, a_rows: np.ndarray, b_cols: np.ndarray,
                        rng: ChaoticRNG, steps: int) -> Tuple[np.ndarray, int]:
    """Chaos-driven descent on the rows of ``A`` (single-row replacements).

    Improves the BIC-SAC keys (and never worsens the SAC keys) when the
    zero set of the autocorrelation sub-table offers more than five choices,
    as it does for the Gold map.  Returns the rows and the number of accepted
    moves.
    """
    rows = a_rows.copy()
    best = tuple(int(v) for v in _avalanche_keys(ac, rows, b_cols))
    accepted = 0
    for _ in range(steps):
        j, c = rng.integers(N_BITS), 1 + rng.integers(SIZE - 1)
        trial = rows.copy()
        trial[j] = c
        if not _is_invertible(_transpose(trial)):
            continue
        key = tuple(int(v) for v in _avalanche_keys(ac, trial, b_cols))
        if key < best:
            rows, best, accepted = trial, key, accepted + 1
    return rows, accepted


def _choose_constants(exponent: int, a_rows: np.ndarray, b_cols: np.ndarray,
                      rng: ChaoticRNG, count: int) -> Tuple[int, int]:
    """Chaos-proposed constants ``(a, b)`` with the fewest (complement-)fixed points.

    The constants change none of the spectral metrics (DDT, Walsh spectrum,
    SAC, BIC are translation invariant), so they are used, like the AES
    affine constant, to avoid ``S(x) = x`` and ``S(x) = ~x``.
    """
    consts = rng.integers(SIZE, size=(count, 2))
    table_a = _linear_tables(_transpose(a_rows))
    table_b = _linear_tables(b_cols)
    f = power_map(exponent)
    tables = table_a[f[table_b[None, :] ^ consts[:, 1:2]]] ^ consts[:, 0:1]  # (count, 32)
    fixed = (tables == _X).sum(axis=1)
    complemented = (tables == (_X ^ (SIZE - 1))).sum(axis=1)
    best = int(np.lexsort((complemented, fixed))[0])
    return int(consts[best, 0]), int(consts[best, 1])


def affine_search(exponent: int, rng: ChaoticRNG, candidates: int,
                  refine_steps: int = 256, constant_candidates: int = 64
                  ) -> Tuple[AffineParams, Dict[str, Any]]:
    """Chaos-driven search over the affine equivalents of ``x^exponent``.

    1. ``candidates`` invertible input matrices ``B`` are proposed by ``rng``.
    2. For each, the output matrix ``A`` is the greedy minimum-deviation
       basis (:func:`_greedy_output_rows`); all candidates are scored with
       :func:`_avalanche_keys` and the best (first in stable lexicographic
       order) is kept.
    3. The rows of ``A`` are refined by ``refine_steps`` chaos-driven
       single-row replacements (:func:`_refine_output_rows`).
    4. ``constant_candidates`` constant pairs are proposed and the one with
       the fewest fixed / complement-fixed points is chosen.

    Returns the parameters and a dict with the search statistics.
    """
    if candidates < 1:
        raise ValueError("candidates must be >= 1")
    ac = autocorrelation_table(exponent)
    b_cols = _sample_invertible(rng, candidates)
    a_rows = _greedy_output_rows(ac, b_cols)
    keys = _avalanche_keys(ac, a_rows, b_cols)
    order = np.lexsort(keys[::-1])
    best = int(order[0])
    perfect = int(np.count_nonzero(keys[0] == 0))
    rows, accepted = _refine_output_rows(ac, a_rows[best], b_cols[best], rng, refine_steps)
    a, b = _choose_constants(exponent, rows, b_cols[best], rng, constant_candidates)
    params = AffineParams(exponent, tuple(int(v) for v in rows), a,
                          tuple(int(v) for v in _transpose(b_cols[best])), b)
    stats = {
        "candidates": int(candidates),
        "best_candidate_index": best,
        "perfect_sac_candidates": perfect,
        "refine_steps": int(refine_steps),
        "refine_accepted": int(accepted),
        "constant_candidates": int(constant_candidates),
    }
    return params, stats


def enumerate_input_bases(exponent: int) -> Dict[str, Any]:
    """Score every unordered input basis of GF(2)^5 for ``F = x^exponent``.

    The SAC / BIC-SAC keys of ``S = A o F o (B x ^ b) ^ a`` depend on ``B``
    only through the set of its columns, so the 9,999,360 invertible
    matrices collapse to 83,328 unordered bases; each is scored with the
    greedy output rows of :func:`_greedy_output_rows`.  This certifies the
    optimum of the class that the chaos-driven :func:`affine_search` attains
    (about one second).

    Returns ``bases`` (count), ``best_key`` (the minimal
    :func:`_avalanche_keys` tuple), ``best_count`` (bases attaining it),
    ``perfect_sac_bases`` (bases with a zero maximal SAC deviation),
    ``max_sac_key_histogram`` (``64 * max |P_ij - 1/2|`` -> count) and
    ``zero_set_sizes`` (for the perfect-SAC bases, the number of masks ``c``
    with ``AC[c, d_i] = 0`` for every column ``d_i``; when it is 5 the rows
    of ``A`` are forced and the greedy key is exact).
    """
    ac = autocorrelation_table(exponent)
    subsets = np.array(list(itertools.combinations(range(1, SIZE), N_BITS)), dtype=np.int64)
    bases = subsets[_is_invertible(subsets)]
    rows = _greedy_output_rows(ac, bases)
    keys = np.stack(_avalanche_keys(ac, rows, bases), axis=1)      # (K, 4)
    order = np.lexsort(keys.T[::-1])
    best = keys[order[0]]
    perfect = bases[keys[:, 0] == 0]
    zero_sets = np.all(ac[:, perfect] == 0, axis=2).sum(axis=0)      # (P,)
    max_dev, counts = np.unique(keys[:, 0], return_counts=True)
    return {
        "exponent": int(exponent),
        "bases": int(bases.shape[0]),
        "best_key": tuple(int(v) for v in best),
        "best_count": int(np.count_nonzero(np.all(keys == best, axis=1))),
        "perfect_sac_bases": int(perfect.shape[0]),
        "max_sac_key_histogram": {int(k): int(v) for k, v in zip(max_dev, counts)},
        "zero_set_sizes": {int(k): int(v) for k, v in
                           zip(*np.unique(zero_sets, return_counts=True))},
    }


# ============================================================
# Simulated annealing over swap moves (paper Algorithm 1, made sound)
# ============================================================
def _spectra(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(|W[a, b]|, DDT[dx, dy])`` of a 5-bit S-box array."""
    abs_walsh = np.abs(_BT.sign @ _BT.sign[arr]).astype(np.int64)
    diff = arr[:, None] ^ arr[_BT.xor]
    ddt = np.bincount((_DDT_OFFSETS + diff).ravel(), minlength=SIZE * SIZE)
    return abs_walsh, ddt.reshape(SIZE, SIZE)


def sa_energy(sbox: Sequence[int]) -> float:
    """Annealing energy: cubic moments of the DDT and of the Walsh spectrum.

    ``E = sum_{dx != 0} DDT^3 + sum_{a != 0, b != 0} (|W| / 4)^3``.  Cubes
    penalise the largest entries super-linearly, so the energy is a smooth
    relaxation of the lexicographic ``(DU, NL)`` objective that still
    rewards reducing the number of maximal entries.  Every DDT row sums to
    32 with even entries, so by convexity the DDT term is minimised, at
    ``31 * 16 * 8 = 3968``, exactly by APN S-boxes.  The Walsh term is 3968
    for almost bent S-boxes (16 entries of ``|W| = 8`` per column) but that
    flat spectrum does not minimise the cubic moment: the inversion class
    (NL 10) spreads each column over more, smaller entries and scores 3782,
    so the annealing energy alone does not separate NL 10 from NL 12.
    """
    abs_walsh, ddt = _spectra(np.asarray(sbox, dtype=np.int64))
    d = ddt[1:]
    w = abs_walsh[1:, 1:] // 4
    return float((d * d * d).sum() + (w * w * w).sum())


def simulated_annealing(initial_sbox: Sequence[int], iterations: int, rng: ChaoticRNG,
                        calibration_moves: int = 200,
                        final_temperature_ratio: float = 1e-3
                        ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Swap-move simulated annealing on :func:`sa_energy`.

    The initial temperature ``T0`` is calibrated so that the median uphill
    move among ``calibration_moves`` random swaps of the start table is
    accepted with probability 1/2 (``T0 = median / ln 2``); the temperature
    then decays geometrically to ``T0 * final_temperature_ratio`` over
    ``iterations`` moves.  Returns the lowest-energy table visited and the
    run statistics.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    state = np.array(initial_sbox, dtype=np.int64)
    energy = sa_energy(state)
    uphill = []
    for _ in range(calibration_moves):
        i, j = rng.pair(SIZE)
        state[i], state[j] = state[j], state[i]
        delta = sa_energy(state) - energy
        state[i], state[j] = state[j], state[i]
        if delta > 0:
            uphill.append(delta)
    t0 = float(np.median(uphill)) / math.log(2.0) if uphill else 1.0
    t_end = t0 * final_temperature_ratio
    alpha = final_temperature_ratio ** (1.0 / iterations)
    temperature = t0
    best, best_energy = state.copy(), energy
    accepted = 0
    for _ in range(iterations):
        i, j = rng.pair(SIZE)
        state[i], state[j] = state[j], state[i]
        candidate = sa_energy(state)
        delta = candidate - energy
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            energy = candidate
            accepted += 1
            if energy < best_energy:
                best, best_energy = state.copy(), energy
        else:
            state[i], state[j] = state[j], state[i]
        temperature *= alpha
    stats = {
        "iterations": int(iterations),
        "initial_temperature": t0,
        "final_temperature": t_end,
        "cooling_factor": alpha,
        "accepted_moves": accepted,
        "initial_energy": sa_energy(initial_sbox),
        "best_energy": best_energy,
        "energy": "sum DDT^3 + sum (|W|/4)^3 (see sa_energy)",
    }
    return best, stats


# ============================================================
# Driver
# ============================================================
def _run_strategy(strategy: str, seed: float, iterations: int) -> Result:
    # Imported here so that src.sbox (and src.sbox_tables, which imports this
    # module) can be loaded in any order without an import cycle.
    from .sbox import DAC_SBOX

    rng = ChaoticRNG(seed)
    start = time.perf_counter()
    provenance: Dict[str, Any] = {"strategy": strategy, "seed": seed,
                                  "iterations": int(iterations)}
    if strategy in ("affine_inverse", "affine_gold"):
        exponent = INVERSION_EXPONENT if strategy == "affine_inverse" else GOLD_EXPONENT
        params, stats = affine_search(exponent, rng, iterations)
        table = params.sbox()
        provenance["affine"] = params.as_dict()
        provenance["search"] = stats
    elif strategy == "simulated_annealing":
        table, stats = simulated_annealing(DAC_SBOX, iterations, rng)
        provenance["initial_sbox"] = "DAC (paper Table 3)"
        provenance["search"] = stats
    else:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES + ('all',)}")
    sbox = [int(v) for v in table]
    if sorted(sbox) != list(range(SIZE)):
        raise RuntimeError(f"strategy {strategy} produced a non-bijective table")
    metrics = analyze_sbox(sbox, name=f"DAC-{strategy}")
    provenance["rng"] = rng.describe()
    provenance["runtime_s"] = time.perf_counter() - start
    return {
        "sbox": sbox,
        "metrics": metrics,
        "objective": objective_key(metrics),
        "provenance": provenance,
    }


def _summary(result: Result) -> str:
    m = result["metrics"]
    return (f"DU={m['differential_uniformity']} NL(S)={m['nl_sbox']} "
            f"deg={m['min_degree']}..{m['max_degree']} LAP={m['lap']:.4f} "
            f"SAC={m['sac_avg']:.4f} [{m['sac_min']:.3f}, {m['sac_max']:.3f}] "
            f"BIC-SAC={m['bic_sac_avg']:.4f} BU={m['boomerang_uniformity']} "
            f"({result['provenance']['runtime_s']:.2f} s)")


def optimize_dac_sbox(strategy: str = "affine_inverse", seed: float = DEFAULT_SEED,
                      iterations: Optional[int] = None, verbose: bool = False) -> Result:
    """Search for an optimised 5-bit S-box; deterministic in ``seed``.

    Args:
        strategy: ``"affine_inverse"`` (default), ``"affine_gold"``,
            ``"simulated_annealing"`` or ``"all"`` (run every strategy and
            return the best by :func:`objective_key`).
        seed: key seed of the :class:`ChaoticRNG` that drives the search.
        iterations: candidates (affine strategies) or swap moves (annealing);
            ``None`` selects :data:`DEFAULT_ITERATIONS`.
        verbose: print a one-line summary per strategy.

    Returns:
        ``{"sbox", "metrics", "objective", "provenance"}`` where ``metrics`` is
        the flat :func:`~src.security_analysis.analyze_sbox` dict, ``objective``
        the lexicographic key and ``provenance`` records the strategy, seed,
        generator parameters, affine parameters / annealing schedule,
        iterations and runtime.  With ``strategy="all"`` the winner also
        carries ``"candidates"`` (the result of every strategy).
    """
    names = STRATEGIES if strategy == "all" else (strategy,)
    if strategy != "all" and strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES + ('all',)}")
    results: Dict[str, Result] = {}
    for name in names:
        count = DEFAULT_ITERATIONS[name] if iterations is None else int(iterations)
        results[name] = _run_strategy(name, seed, count)
        if verbose:
            print(f"  [{name}] {_summary(results[name])}")
    best_name = min(results, key=lambda k: results[k]["objective"])
    best = results[best_name]
    if strategy == "all":
        best = dict(best, candidates=results)
        if verbose:
            print(f"  best strategy: {best_name}")
    if verbose:
        print(f"  S-box: {best['sbox']}")
    return best
