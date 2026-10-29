"""Round-level differential and linear analysis of the 32-bit demonstration SPN.

The paper claimed that halving the S-box differential probability "directly
increases the number of rounds an attacker needs".  That statement does not
follow from a single S-box metric: a cipher-level bound also needs the
diffusion layer and the minimum number of *active* S-boxes over several
rounds.  This module replaces the claim with a computation.

What is computed here
---------------------
1. :func:`describe_round_function` -- a machine-readable, re-implementable
   description of the exact round function of
   :class:`src.block_cipher.DACBlockCipher` (which state bits each of the six
   parallel S-boxes consumes, the two bits that bypass the S-layer, the bit
   permutation and the key schedule).
2. :func:`branch_numbers` -- the differential and the linear branch number of
   an S-box, read off its DDT and LAT.
3. :func:`min_active_sboxes` -- a Mouha-style mixed-integer program (Mouha,
   Wang, Gu and Preneel, Inscrypt 2011) solved with HiGHS through
   :func:`scipy.optimize.milp`, giving the minimum number of active S-boxes of
   any ``r``-round trail *of the bit-activity model*.  Both a differential and
   a linear model are built; the linear one uses the linear branch number and
   the mask permutation derived by :func:`derive_linear_mask_permutation`.
4. :func:`trail_bounds` -- the textbook single-characteristic bounds
   ``dp_max ** N_r`` and ``corr_max ** N_r`` that follow from those counts.
5. :func:`generate_test_vectors` -- deterministic (key, plaintext, ciphertext)
   triples with round-trip verification, so the cipher can be reproduced
   elsewhere.
6. :func:`birthday_bound` -- the generic 32-bit block-size limitation.

:func:`analyze_cipher` returns all of it as one JSON-serialisable dictionary.

What this is NOT
----------------
The number produced by :func:`min_active_sboxes` is a **lower bound on the
number of active S-boxes**, obtained from a *relaxation* that knows only

* which bits each S-box reads and writes,
* the branch number of the S-box, and
* the bit permutation,

and nothing else about the DDT or the LAT.  Consequently:

* the resulting probability/correlation figures bound a **single
  characteristic**, not a differential (hull) or a linear hull, and they are
  **not a proof of resistance** against differential or linear cryptanalysis;
* they say nothing about related-key, boomerang, integral, algebraic or
  meet-in-the-middle attacks, and the key schedule is not modelled at all;
* a trail attaining the bound need not exist, because the model ignores
  whether the required single-bit DDT/LAT transitions are actually possible.

The bound is nevertheless conclusive in the negative direction: where it is
weak the design *is* weak, and that is what it shows for this cipher (see
:data:`FINDINGS`).
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from .block_cipher import (BLOCK_SIZE, DACBlockCipher, KEY_SIZE, NUM_ROUNDS,
                           PERM_TABLE, permute_bits)
from .sbox_tables import TABLES
from .security_analysis import compute_ddt, compute_lat

SBOX_BITS = 5
#: Number of parallel S-boxes in the substitution layer.
NUM_SBOXES = 6

#: ``SBOX_INPUT_BITS[j]`` are the MSB-first state-bit indices read *and*
#: written by S-box ``j``.  MSB-first index ``i`` denotes state bit
#: ``BLOCK_SIZE - 1 - i``, the convention :data:`src.block_cipher.PERM_TABLE`
#: uses.  S-box ``j`` therefore covers state bits ``31 - 5j .. 27 - 5j``.
SBOX_INPUT_BITS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(range(SBOX_BITS * j, SBOX_BITS * j + SBOX_BITS)) for j in range(NUM_SBOXES))

#: MSB-first indices of the two state bits that never enter an S-box: indices
#: 30 and 31, i.e. state bits 1 and 0.  ``substitute_block32`` only replaces
#: them by their complements, which is affine and therefore invisible to a
#: difference or to a linear mask.
BYPASS_BITS: Tuple[int, ...] = (30, 31)

#: Round counts tabulated by :func:`analyze_cipher` (the cipher has 16 rounds).
REPORTED_ROUNDS = NUM_ROUNDS
#: The MILP is additionally solved up to this many rounds, so that the round
#: count at which each bound reaches the 2**-32 data limit can be reported even
#: when 16 rounds are not enough (they are not, for DAC and E-DAC).
EXTENDED_ROUNDS = 32

#: S-boxes analysed, in reporting order.
SBOX_NAMES: Tuple[str, ...] = ("DAC", "E-DAC", "DAC-OPT", "DAC-OPT-GOLD")

#: Seeds of the pseudorandom test vectors (``random.Random.getrandbits`` is
#: reproducible across platforms and Python versions).
TEST_VECTOR_SEEDS: Tuple[int, ...] = (1, 2, 3)


# ============================================================
# 1. The round function, written down exactly
# ============================================================
def round_function_text() -> str:
    """Return the prose description of the round function, precise enough to re-implement."""
    windows = "; ".join(
        f"S{j} on state bits {BLOCK_SIZE - 1 - bits[0]}..{BLOCK_SIZE - 1 - bits[-1]}"
        for j, bits in enumerate(SBOX_INPUT_BITS))
    return (
        f"State: {BLOCK_SIZE} bits b31..b0 (b31 = most significant). MSB-first "
        f"index i denotes state bit {BLOCK_SIZE - 1} - i. Key: {KEY_SIZE} bits. "
        f"Rounds: {NUM_ROUNDS}.\n"
        f"Round r = 0..{NUM_ROUNDS - 1}: state ^= RK[r]; state = S-layer(state); "
        f"state = P-layer(state), except in the last round "
        f"(r = {NUM_ROUNDS - 1}) where the P-layer is skipped; after the last "
        f"round state ^= RK[{NUM_ROUNDS}].\n"
        f"S-layer: six parallel copies of the same bijective 5-bit S-box, "
        f"{windows}. Each S-box writes its 5-bit output back to the five state "
        f"bits it read, the more significant state bit of a window carrying the "
        f"more significant bit of the S-box input and of its output. State bits 1 "
        f"and 0 (MSB-first indices {BYPASS_BITS[0]} and {BYPASS_BITS[1]}) enter "
        f"no S-box: the layer replaces them by their complements (state ^= 0x3).\n"
        f"P-layer: the bit at MSB-first index i moves to MSB-first index "
        f"PERM_TABLE[i], i.e. state bit 31 - i moves to state bit "
        f"31 - PERM_TABLE[i], with PERM_TABLE = {list(PERM_TABLE)}.\n"
        f"Key schedule: an 80-bit register K holds the master key; for "
        f"i = 0..{NUM_ROUNDS}: RK[i] = K >> {KEY_SIZE - BLOCK_SIZE}; then "
        f"K = rotate_left_80(K, 13); the top five bits of K are replaced by "
        f"S(K >> {KEY_SIZE - SBOX_BITS}); K ^= ((i + 1) mod 32) << 15.\n"
        f"Both the key addition and the complement of state bits 1 and 0 are "
        f"affine, so neither changes a difference or a linear mask: only the six "
        f"S-boxes and the bit permutation matter for the analysis below.")


def describe_round_function() -> Dict[str, Any]:
    """Return a machine-readable description of the cipher's round function.

    Every field is taken from :mod:`src.block_cipher`, and
    :func:`check_round_function_description` re-derives the same facts by
    probing the implementation, so the description cannot drift away from the
    code it documents.
    """
    return {
        "block_bits": BLOCK_SIZE,
        "key_bits": KEY_SIZE,
        "rounds": NUM_ROUNDS,
        "sbox_bits": SBOX_BITS,
        "num_sboxes": NUM_SBOXES,
        "bit_index_convention":
            f"MSB-first index i denotes state bit {BLOCK_SIZE - 1} - i "
            f"(index 0 = most significant state bit)",
        "sbox_input_bits_msb_first": [list(bits) for bits in SBOX_INPUT_BITS],
        "sbox_state_bit_windows": [
            [BLOCK_SIZE - 1 - i for i in bits] for bits in SBOX_INPUT_BITS],
        "bypass_bits_msb_first": list(BYPASS_BITS),
        "bypass_state_bits": [BLOCK_SIZE - 1 - i for i in BYPASS_BITS],
        "bypass_operation": "complement (state ^= 0x3); affine, hence transparent "
                            "to differences and to linear masks",
        "perm_table": list(PERM_TABLE),
        "perm_rule": "the bit at MSB-first index i moves to MSB-first index PERM_TABLE[i]",
        "last_round_skips_permutation": True,
        "final_whitening_key_index": NUM_ROUNDS,
        "key_schedule": "RK[i] = K >> 48; K = rotate_left_80(K, 13); "
                        "K[79:75] = S(K[79:75]); K ^= ((i + 1) mod 32) << 15",
        "text": round_function_text(),
    }


def check_round_function_description() -> Dict[str, bool]:
    """Verify the description above against :mod:`src.block_cipher` itself.

    The S-box windows and the bypass bits are re-derived by probing
    ``DACSBox.substitute_block32`` with single-bit flips, and the permutation
    is re-derived by probing :func:`src.block_cipher.permute_bits`.  Returns
    one boolean per checked claim; all of them must be True.
    """
    sbox = DACBlockCipher(0).sbox
    rng = random.Random(20260901)
    bases = [0, (1 << BLOCK_SIZE) - 1] + [rng.getrandbits(BLOCK_SIZE) for _ in range(16)]
    influence: List[set] = [set() for _ in range(BLOCK_SIZE)]
    for base in bases:
        out0 = sbox.substitute_block32(base)
        for i in range(BLOCK_SIZE):
            diff = out0 ^ sbox.substitute_block32(base ^ (1 << (BLOCK_SIZE - 1 - i)))
            for j in range(BLOCK_SIZE):
                if (diff >> (BLOCK_SIZE - 1 - j)) & 1:
                    influence[i].add(j)
    windows_ok = all(influence[i] <= set(bits)
                     for bits in SBOX_INPUT_BITS for i in bits)
    bypass_ok = all(influence[b] == {b} for b in BYPASS_BITS)
    # Restricted to the bypass bits the S-layer is x -> x ^ 1 (identity linear part).
    mask = sum(1 << (BLOCK_SIZE - 1 - b) for b in BYPASS_BITS)
    complement_ok = all(
        (sbox.substitute_block32(base) ^ base) & mask == mask for base in bases)
    perm_ok = all(
        permute_bits(1 << (BLOCK_SIZE - 1 - i)) == 1 << (BLOCK_SIZE - 1 - PERM_TABLE[i])
        for i in range(BLOCK_SIZE))
    return {
        "sbox_windows_match_implementation": windows_ok,
        "bypass_bits_touch_only_themselves": bypass_ok,
        "bypass_bits_are_complemented": complement_ok,
        "perm_table_matches_permute_bits": perm_ok,
    }


# ============================================================
# 2. Branch numbers and the mask permutation
# ============================================================
def branch_numbers(sbox: Sequence[int], n: int = SBOX_BITS) -> Dict[str, int]:
    """Differential and linear branch number of ``sbox``.

    The differential branch number is
    ``min { wt(a) + wt(b) : a != 0, b != 0, DDT[a, b] != 0 }``; the linear
    branch number is the same minimum over the nonzero entries of the LAT
    (``LAT[a, b] = #{x : a.x = b.S(x)} - 2^(n-1)``, ``a`` the input mask and
    ``b`` the output mask).  Both are at least 2 for a bijective S-box, and 2
    exactly when some single-bit input difference (resp. mask) can produce a
    single-bit output difference (resp. mask) -- the worst case for a
    bit-permutation diffusion layer.
    """
    size = 1 << n
    weights = np.array([bin(v).count("1") for v in range(size)], dtype=np.int64)
    total = weights[:, None] + weights[None, :]
    result: Dict[str, int] = {}
    for key, table in (("differential", compute_ddt(sbox, n)["table"]),
                       ("linear", np.abs(compute_lat(sbox, n)["table"]))):
        nonzero = np.asarray(table)[1:, 1:] != 0
        if not nonzero.any():          # impossible for a permutation
            raise ValueError(f"{key} table has no nonzero entry outside row/column 0")
        result[key] = int(total[1:, 1:][nonzero].min())
    return result


def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a square GF(2) matrix by Gauss-Jordan elimination.

    Raises:
        ValueError: if the matrix is singular over GF(2).
    """
    n = matrix.shape[0]
    work = np.concatenate([matrix % 2, np.eye(n, dtype=np.uint8)], axis=1)
    row = 0
    for col in range(n):
        pivot = next((k for k in range(row, n) if work[k, col]), None)
        if pivot is None:
            raise ValueError("matrix is singular over GF(2)")
        work[[row, pivot]] = work[[pivot, row]]
        for k in range(n):
            if k != row and work[k, col]:
                work[k] ^= work[row]
        row += 1
    return work[:, n:]


def derive_linear_mask_permutation(perm: Sequence[int] = PERM_TABLE) -> Tuple[int, ...]:
    """Derive how a linear mask propagates through the bit permutation ``perm``.

    For a linear layer ``y = L x`` a linear approximation satisfies
    ``alpha . x = beta . y`` with ``alpha = L^T beta``, so a mask travels
    forward through ``(L^T)^{-1}``.  That matrix is built and inverted over
    GF(2) here rather than assumed equal to ``L``, and the function raises if
    the result is not itself a permutation matrix.  (Because ``L^T = L^{-1}``
    for a permutation matrix the derivation does return ``perm`` again -- but
    it is derived, not postulated, and the caller records the outcome.)

    Raises:
        ValueError: if the derived mask propagation is not a bit permutation.
    """
    n = len(perm)
    matrix = np.zeros((n, n), dtype=np.uint8)
    for i, j in enumerate(perm):
        matrix[j, i] = 1                      # (L x)[perm[i]] = x[i]
    inverse = _gf2_inverse(matrix.T)
    if not ((inverse.sum(axis=0) == 1).all() and (inverse.sum(axis=1) == 1).all()):
        raise ValueError("mask propagation matrix is not a permutation matrix")
    return tuple(int(np.argmax(inverse[:, i])) for i in range(n))


# ============================================================
# 3. The bits that bypass the S-layer
# ============================================================
def bypass_trail_analysis(perm: Sequence[int] = PERM_TABLE) -> Dict[str, Any]:
    """How much do the two S-layer-bypassing bits weaken the cipher?

    A difference (or mask) supported only on :data:`BYPASS_BITS` crosses a
    whole round without activating a single S-box, because the S-layer only
    complements those bits.  The longest run of such free rounds is the longest
    path ``b -> perm[b] -> ...`` that stays inside :data:`BYPASS_BITS`; a run of
    two or more would let a trail skip several rounds for free and would be a
    disclosable weakness of the permutation.  The computed answer for
    :data:`src.block_cipher.PERM_TABLE` is returned in
    ``max_consecutive_free_rounds`` / ``multi_round_bypass_only_trail_exists``.
    """
    bypass = set(BYPASS_BITS)
    successors = {b: perm[b] for b in BYPASS_BITS}
    longest = 0
    for start in BYPASS_BITS:
        seen: set = set()
        current, length = start, 0
        while current in bypass and current not in seen:
            seen.add(current)
            length += 1
            current = perm[current]
        longest = max(longest, length)
    return {
        "bypass_bits_msb_first": list(BYPASS_BITS),
        "bypass_state_bits": [BLOCK_SIZE - 1 - b for b in BYPASS_BITS],
        "successor_of_each_bypass_bit_msb_first": {str(b): successors[b]
                                                   for b in BYPASS_BITS},
        "successor_is_an_sbox_bit": {str(b): successors[b] not in bypass
                                     for b in BYPASS_BITS},
        "max_consecutive_free_rounds": longest,
        "multi_round_bypass_only_trail_exists": longest >= 2,
        "note": f"a difference or mask confined to the bypass bits crosses "
                f"{longest} round(s) without activating any S-box and must then "
                f"enter an S-box, because the permutation sends both bypass bits "
                f"onto S-box bits; single free rounds can still recur, which is "
                f"what lowers the multi-round active-S-box bound",
    }


# ============================================================
# 4. Mouha-style MILP for the minimum number of active S-boxes
# ============================================================
def _build_model(rounds: int, perm: Sequence[int], branch_number: int,
                 window_bounds: Sequence[int]) -> Tuple[Any, ...]:
    """Assemble the MILP of :func:`min_active_sboxes` (documented there)."""
    n_bits = len(perm)
    n_x = (rounds + 1) * n_bits
    n_var = n_x + rounds * NUM_SBOXES

    def x(r: int, i: int) -> int:
        return r * n_bits + i

    def a(r: int, j: int) -> int:
        return n_x + r * NUM_SBOXES + j

    def y(r: int, i: int) -> int:
        # S-layer output bit i of round r *is* state bit perm[i] of the next
        # layer, so the permutation is folded in and needs no own variable.
        return x(r + 1, perm[i])

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    lower: List[float] = []
    upper: List[float] = []
    n_row = 0

    def add(coeffs: Dict[int, float], lo: float, hi: float) -> None:
        nonlocal n_row
        for index, value in coeffs.items():
            rows.append(n_row)
            cols.append(index)
            vals.append(float(value))
        lower.append(lo)
        upper.append(hi)
        n_row += 1

    for r in range(rounds):
        for j, bits in enumerate(SBOX_INPUT_BITS):
            # (i) active as soon as one input bit is active
            for i in bits:
                add({x(r, i): 1.0, a(r, j): -1.0}, -np.inf, 0.0)
            # (ii) inactive when none of them is
            coeffs: Dict[int, float] = {a(r, j): 1.0}
            for i in bits:
                coeffs[x(r, i)] = coeffs.get(x(r, i), 0.0) - 1.0
            add(coeffs, -np.inf, 0.0)
            # (iii) branch number of an active S-box
            coeffs = {a(r, j): -float(branch_number)}
            for i in bits:
                coeffs[x(r, i)] = coeffs.get(x(r, i), 0.0) + 1.0
            for i in bits:
                coeffs[y(r, i)] = coeffs.get(y(r, i), 0.0) + 1.0
            add(coeffs, 0.0, np.inf)
            # (iv) a bijective active S-box has at least one active output bit
            coeffs = {a(r, j): -1.0}
            for i in bits:
                coeffs[y(r, i)] = coeffs.get(y(r, i), 0.0) + 1.0
            add(coeffs, 0.0, np.inf)
            # (v) an inactive S-box has none
            for i in bits:
                add({y(r, i): 1.0, a(r, j): -1.0}, -np.inf, 0.0)
        # (vi) the bypass bits are the identity and belong to no S-box
        for b in BYPASS_BITS:
            add({y(r, b): 1.0, x(r, b): -1.0}, 0.0, 0.0)
        # (vii) exclude the all-zero trail
        add({x(r, i): 1.0 for i in range(n_bits)}, 1.0, np.inf)
    add({x(rounds, i): 1.0 for i in range(n_bits)}, 1.0, np.inf)

    # (viii) redundant window cuts, for speed only: any ``w`` consecutive rounds
    # of an ``r``-round trail form a valid ``w``-round trail of the same model
    # and therefore contain at least ``window_bounds[w]`` active S-boxes.
    for width, bound in enumerate(window_bounds):
        if width < 1 or width > rounds or bound <= 0:
            continue
        for start in range(rounds - width + 1):
            add({a(r, j): 1.0 for r in range(start, start + width)
                 for j in range(NUM_SBOXES)}, float(bound), np.inf)

    objective = np.zeros(n_var)
    for r in range(rounds):
        for j in range(NUM_SBOXES):
            objective[a(r, j)] = 1.0
    matrix = csc_matrix((vals, (rows, cols)), shape=(n_row, n_var))
    return objective, matrix, np.array(lower), np.array(upper), n_var, n_x


def min_active_sboxes(max_rounds: int, perm: Sequence[int] = PERM_TABLE,
                      branch_number: int = 2,
                      use_window_cuts: bool = True) -> List[int]:
    """Minimum number of active S-boxes over ``r = 1 .. max_rounds`` rounds.

    LOWER BOUND ONLY -- a Mouha-style bit-level MILP (Mouha, Wang, Gu and
    Preneel, Inscrypt 2011) that models the *activity pattern* of a trail and
    nothing else; see the module docstring for what it does not prove.  Solved
    with HiGHS via :func:`scipy.optimize.milp`.

    Variables (binary):

    * ``x[r][i]`` -- state bit ``i`` (MSB-first index) is active at the input of
      round ``r``, for ``r = 0 .. max_rounds``; the extra last layer holds the
      trail output;
    * ``a[r][j]`` -- S-box ``j`` of round ``r`` is active.

    The S-layer output bit ``i`` of round ``r`` is not a separate variable: it
    equals ``x[r + 1][perm[i]]``, which folds the bit permutation into the
    model.  Per round and per S-box ``j`` with bit set ``I_j``
    (:data:`SBOX_INPUT_BITS`):

    * ``a[r][j] >= x[r][i]`` for every ``i`` in ``I_j``, and
      ``a[r][j] <= sum_{i in I_j} x[r][i]`` -- the S-box is active iff at least
      one of its input bits is active;
    * ``sum_{i in I_j} x[r][i] + sum_{i in I_j} y[r][i] >= branch_number * a[r][j]``
      -- the branch-number constraint;
    * ``sum_{i in I_j} y[r][i] >= a[r][j]`` and ``y[r][i] <= a[r][j]`` -- a
      bijective active S-box has a nonzero output difference (mask), an
      inactive one has none.

    The two bits of :data:`BYPASS_BITS` belong to **no** S-box.  They are
    modelled exactly as the implementation treats them -- the identity,
    ``y[r][b] = x[r][b]``, since complementing a bit changes neither a
    difference nor a mask -- so an active bypass bit stays active into the next
    round while activating nothing.  This genuinely weakens the bound and is
    the main reason the counts below fall well short of one active S-box per
    round; it is modelled, not glossed over.

    ``sum_i x[r][i] >= 1`` in every layer excludes the all-zero trail.

    ``use_window_cuts`` adds the redundant constraint that any window of ``w``
    consecutive rounds holds at least as many active S-boxes as the ``w``-round
    optimum already proved.  It cannot change an optimum (a window of a trail
    is a trail); it only makes the 32-round chain solve in seconds instead of
    minutes.  ``tests/test_cipher_analysis.py`` re-solves the plain model with
    ``use_window_cuts=False`` for small ``max_rounds`` and gets the same values.

    Returns:
        ``result[r - 1]`` = minimum number of active S-boxes over ``r`` rounds.

    Raises:
        ValueError: if ``max_rounds < 1`` or a solve does not reach optimality.
    """
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
    counts: List[int] = []
    window_bounds: List[int] = [0]          # index w -> proved optimum for w rounds
    for rounds in range(1, max_rounds + 1):
        objective, matrix, lo, hi, n_var, n_x = _build_model(
            rounds, perm, branch_number, window_bounds if use_window_cuts else [0])
        integrality = np.ones(n_var)
        # The constraints and the objective force a[r][j] = max(x_i, y_i) over
        # the S-box, so it is integral automatically; declaring it continuous is
        # exact and roughly halves the number of integer variables.
        integrality[n_x:] = 0
        result = milp(c=objective, constraints=LinearConstraint(matrix, lo, hi),
                      integrality=integrality, bounds=Bounds(0, 1))
        if not result.success or result.fun is None:
            raise ValueError(f"MILP for {rounds} rounds did not solve: {result.message}")
        value = int(round(result.fun))
        counts.append(value)
        window_bounds.append(value)
    return counts


def reference_min_active_sboxes(max_rounds: int,
                                perm: Sequence[int] = PERM_TABLE) -> List[int]:
    """Brute-force reference for :func:`min_active_sboxes` when the branch number is 2.

    With branch number 2 an active S-box may turn one active input bit into one
    active output bit, and the model is *decomposable*: restricting any trail to
    the sub-trail generated by a single initially active bit is again a valid
    trail, whose per-round set of active S-boxes is a subset of the original
    one.  The optimum is therefore attained by a single-bit trail and can be
    found exactly by the dynamic program over the 32 bit positions implemented
    here, with no MILP involved.

    Only valid for ``branch_number == 2`` (which is the branch number of every
    S-box of this study); :func:`min_active_sboxes` is the general routine.

    Raises:
        ValueError: if ``max_rounds < 1``.
    """
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
    sbox_of = {i: j for j, bits in enumerate(SBOX_INPUT_BITS) for i in bits}
    best: Dict[int, int] = {i: 0 for i in range(len(perm))}
    counts: List[int] = []
    for _ in range(max_rounds):
        nxt: Dict[int, int] = {}
        for bit, cost in best.items():
            if bit in BYPASS_BITS:
                outputs: Sequence[int] = (bit,)
                added = 0
            else:
                outputs = SBOX_INPUT_BITS[sbox_of[bit]]
                added = 1
            for out in outputs:
                target = perm[out]
                if nxt.get(target, math.inf) > cost + added:
                    nxt[target] = cost + added
        best = nxt
        counts.append(min(best.values()))
    return counts


# ============================================================
# 4b. Exact trail search with the real DDT / LAT
# ============================================================
#: Absolute tolerance for comparing trail weights (sums of ``-log2`` values).
_WEIGHT_EPS = 1e-9


class _TrailModel:
    """Exact round transitions of the SPN for one S-box and one model.

    ``model = "differential"`` uses the nonzero DDT entries (``dx -> dy``)
    and the state permutation; ``model = "linear"`` uses the nonzero LAT
    entries (input mask ``a`` -> output mask ``b``) and the mask permutation
    derived by :func:`derive_linear_mask_permutation`.  The weight of one
    S-box transition is ``-log2(DDT[dx, dy] / 2^n)`` (differential
    probability) or ``-log2(|LAT[a, b]| / 2^(n-1))`` (absolute correlation),
    so the weight of a trail is ``-log2`` of its probability, resp. of its
    absolute correlation by the piling-up lemma.

    A state is a 32-bit integer whose bit ``31 - i`` is MSB-first index
    ``i``.  The key addition and the complement of the two bypass bits are
    affine and change neither a difference nor a mask, so one round maps a
    state to ``P(S-layer output)``, the bypass bits passing unchanged.
    """

    def __init__(self, sbox: Sequence[int], model: str,
                 perm: Sequence[int] = PERM_TABLE) -> None:
        size = 1 << SBOX_BITS
        if model == "differential":
            table = np.asarray(compute_ddt(sbox, SBOX_BITS)["table"])
            moves = perm
            scale = float(size)
        elif model == "linear":
            table = np.abs(np.asarray(compute_lat(sbox, SBOX_BITS)["table"]))
            moves = derive_linear_mask_permutation(perm)
            scale = float(size // 2)
        else:
            raise ValueError(f"model must be 'differential' or 'linear', got {model!r}")
        self.model = model
        self.table = table
        #: ``transitions[d]``: ``(weight, output)`` pairs with a nonzero table
        #: entry, lightest first.
        self.transitions: List[List[Tuple[float, int]]] = [[]] + [
            sorted((-math.log2(table[d, o] / scale), o)
                   for o in range(1, size) if table[d, o] != 0)
            for d in range(1, size)]
        self.min_weight = [0.0] + [self.transitions[d][0][0] for d in range(1, size)]
        self.shifts = tuple(BLOCK_SIZE - SBOX_BITS * (j + 1) for j in range(NUM_SBOXES))
        # contribution[j][o]: S-box j emitting ``o`` lands on these state bits
        # after the permutation; bypass[v]: the same for the two bypass bits.
        self.contribution = [
            [self._place(moves, bits, o) for o in range(size)] for bits in SBOX_INPUT_BITS]
        self.bypass = [self._place(moves, BYPASS_BITS, v, width=len(BYPASS_BITS))
                       for v in range(1 << len(BYPASS_BITS))]
        self.bypass_mask = (1 << len(BYPASS_BITS)) - 1

    @staticmethod
    def _place(moves: Sequence[int], bits: Sequence[int], value: int,
               width: int = SBOX_BITS) -> int:
        state = 0
        for k, i in enumerate(bits):
            if (value >> (width - 1 - k)) & 1:
                state |= 1 << (BLOCK_SIZE - 1 - moves[i])
        return state

    def active(self, state: int) -> List[Tuple[int, int]]:
        """``(window, input)`` of every active S-box of ``state``."""
        return [(j, (state >> shift) & 31) for j, shift in enumerate(self.shifts)
                if (state >> shift) & 31]

    def successors(self, state: int):
        """``(weight, next_state, outputs)`` for every one-round transition."""
        windows = self.active(state)
        base = self.bypass[state & self.bypass_mask]
        choices = [self.transitions[d] for _, d in windows]
        for combo in itertools.product(*choices):
            nxt, weight = base, 0.0
            for (j, _), (w, o) in zip(windows, combo):
                nxt ^= self.contribution[j][o]
                weight += w
            yield weight, nxt, tuple(o for _, o in combo)

    def start_states(self, max_active: int):
        """Every nonzero state with at most ``max_active`` active S-boxes."""
        for count in range(0, min(max_active, NUM_SBOXES) + 1):
            for windows in itertools.combinations(range(NUM_SBOXES), count):
                for values in itertools.product(range(1, 32), repeat=count):
                    state = 0
                    for j, v in zip(windows, values):
                        state |= v << self.shifts[j]
                    for low in range(1 << len(BYPASS_BITS)):
                        if state | low:
                            yield state | low


def exact_min_active_sboxes(sbox: Sequence[int], max_rounds: int,
                            model: str = "differential") -> Dict[str, Any]:
    """Exact minimum number of active S-boxes, using the real DDT or LAT.

    Unlike :func:`min_active_sboxes`, which knows only the branch number,
    this search admits a transition ``d -> o`` of an S-box only when the
    table entry is nonzero, so its counts are those of trails that actually
    exist.  It is a Matsui-style depth-first branch and bound:

    * the cost of ``m`` rounds from round-input state ``s`` is the number of
      active S-boxes of ``s`` plus the cost of ``m - 1`` rounds from a
      successor; the counts are found for ``r = 1, 2, ...`` in turn;
    * a budget ``B`` is tested from ``E(r - 1)`` upwards (an ``r``-round
      trail contains an ``(r - 1)``-round one); a partial trail is cut as
      soon as its active count plus ``E(m)`` for the ``m`` remaining rounds
      exceeds ``B``, and a state proved infeasible for ``(m, B)`` is
      remembered;
    * only first-round states with at most ``B - E(r - 1)`` active S-boxes
      can start a trail of cost ``B`` and all of them are enumerated.

    Every pruning rule discards only partial trails whose cost provably
    exceeds the budget, so the result is exact.  ``tests`` re-derive the
    first rounds by unpruned enumeration and re-verify every witness trail.

    Returns ``counts[r - 1]`` and, per round count, one witness trail (the
    round-input states, most significant bit = MSB-first index 0).
    """
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
    trails = _TrailModel(sbox, model)
    exact: List[int] = [0]
    infeasible: Dict[Tuple[int, int], int] = {}
    nodes = 0

    def search(state: int, rounds: int, budget: int) -> Optional[List[int]]:
        nonlocal nodes
        nodes += 1
        left = budget - len(trails.active(state))
        if left < 0:
            return None
        if rounds == 1:
            return [state]
        if left < exact[rounds - 1] or infeasible.get((state, rounds), -1) >= budget:
            return None
        for _, nxt, _ in trails.successors(state):
            tail = search(nxt, rounds - 1, left)
            if tail is not None:
                return [state] + tail
        infeasible[(state, rounds)] = budget
        return None

    witnesses: List[List[int]] = []
    for rounds in range(1, max_rounds + 1):
        budget = exact[rounds - 1]
        found: Optional[List[int]] = None
        while found is None:
            spare = budget - (exact[rounds - 1] if rounds > 1 else 0)
            for start in trails.start_states(spare):
                found = search(start, rounds, budget)
                if found is not None:
                    break
            else:
                budget += 1
        exact.append(budget)
        witnesses.append(found)
    return {
        "model": model,
        "counts": exact[1:],
        "witness_trails": [[f"{s:08X}" for s in trail] for trail in witnesses],
        "search_nodes": nodes,
    }


def exact_best_trail_weights(sbox: Sequence[int], max_rounds: int,
                             model: str = "differential",
                             step: float = 1.0) -> Dict[str, Any]:
    """Exact weight of the best single characteristic over ``r`` rounds.

    The weight is ``-log2`` of the characteristic probability (differential)
    or of its absolute correlation (linear), each S-box transition weighted as
    in :class:`_TrailModel`; it is the quantity the bounds of
    :func:`trail_bounds` only estimate from active-S-box counts.

    Depth-first branch and bound under a weight cap: ``best(s, m, cap)`` is
    the exact minimum over ``m``-round trails from ``s`` when that minimum is
    at most ``cap`` and is reported as "above the cap" otherwise.  Partial
    trails are cut when their weight plus the per-S-box minimum of the
    current round plus ``W(m - 1)`` exceeds the cap; the last round costs the
    lightest output of each active S-box.  For each ``r`` the cap is raised
    from ``W(r - 1)`` in steps of ``step`` until a trail is found, and only
    first-round states whose minimal first-round weight plus ``W(r - 1)``
    fits under the cap are started.  Within a cap the minimum is exact, so the
    result is exact.

    Returns ``weights[r - 1]`` (float), the corresponding active-S-box count
    of the best trail and one witness trail per round count.
    """
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    trails = _TrailModel(sbox, model)
    min_w = trails.min_weight
    lightest = min(min_w[1:])
    exact: List[float] = [0.0]
    memo_exact: Dict[Tuple[int, int], Tuple[float, List[int]]] = {}
    memo_above: Dict[Tuple[int, int], float] = {}

    def best(state: int, rounds: int, cap: float) -> Optional[Tuple[float, List[int]]]:
        key = (state, rounds)
        if key in memo_exact:
            value = memo_exact[key]
            return value if value[0] <= cap + _WEIGHT_EPS else None
        if memo_above.get(key, -1.0) >= cap - _WEIGHT_EPS:
            return None
        low = sum(min_w[d] for _, d in trails.active(state))
        if rounds == 1:
            result = (low, [state])
            memo_exact[key] = result
            return result if low <= cap + _WEIGHT_EPS else None
        if low + exact[rounds - 1] > cap + _WEIGHT_EPS:
            memo_above[key] = max(memo_above.get(key, -1.0), cap)
            return None
        found: Optional[Tuple[float, List[int]]] = None
        limit = cap
        for weight, nxt, _ in trails.successors(state):
            if weight + exact[rounds - 1] > limit + _WEIGHT_EPS:
                continue
            tail = best(nxt, rounds - 1, limit - weight)
            if tail is not None and (found is None
                                     or weight + tail[0] < found[0] - _WEIGHT_EPS):
                found = (weight + tail[0], [state] + tail[1])
                limit = found[0]
        if found is None:
            memo_above[key] = max(memo_above.get(key, -1.0), cap)
            return None
        memo_exact[key] = found
        return found

    weights: List[float] = []
    actives: List[int] = []
    witnesses: List[List[int]] = []
    for rounds in range(1, max_rounds + 1):
        floor = exact[rounds - 1] if rounds > 1 else 0.0
        cap = floor
        while True:
            spare = int((cap - floor + _WEIGHT_EPS) // lightest)
            winner: Optional[Tuple[float, List[int]]] = None
            for start in trails.start_states(spare):
                first = sum(min_w[d] for _, d in trails.active(start))
                bound = cap if winner is None else winner[0]
                if first + floor > bound + _WEIGHT_EPS:
                    continue
                candidate = best(start, rounds, bound)
                if candidate is not None and (winner is None
                                              or candidate[0] < winner[0] - _WEIGHT_EPS):
                    winner = candidate
            if winner is not None:
                break
            cap += step
        exact.append(winner[0])
        weights.append(winner[0])
        actives.append(sum(len(trails.active(s)) for s in winner[1]))
        witnesses.append(winner[1])
    return {
        "model": model,
        "weights": weights,
        "active_sboxes_of_best_trail": actives,
        "witness_trails": [[f"{s:08X}" for s in trail] for trail in witnesses],
    }


# ============================================================
# 5. From active S-boxes to trail bounds
# ============================================================
#: An attacker of a 32-bit block cipher has at most the full codebook, 2**32
#: blocks; a characteristic whose probability -- or whose squared correlation --
#: is below 2**-32 cannot be exploited by a single-characteristic attack.
DATA_LIMIT_LOG2 = float(BLOCK_SIZE)


def sbox_trail_parameters(sbox: Sequence[int], n: int = SBOX_BITS) -> Dict[str, float]:
    """Maximum differential probability and maximum linear correlation of ``sbox``.

    ``dp_max = differential_uniformity / 2^n``.  ``LAT[a, b]`` counts
    ``#{x : a.x = b.S(x)} - 2^(n-1)``, so the approximation holds with
    probability ``(2^(n-1) + LAT) / 2^n`` and has correlation
    ``2p - 1 = LAT / 2^(n-1)``; hence ``correlation_max = max|LAT| / 2^(n-1)``.
    """
    ddt = compute_ddt(sbox, n)
    lat = compute_lat(sbox, n)
    du = int(ddt["differential_uniformity"])
    max_bias = int(lat["max_bias"])
    dp_max = du / (1 << n)
    corr_max = max_bias / (1 << (n - 1))
    return {
        "differential_uniformity": du,
        "dp_max": dp_max,
        "log2_dp_max": math.log2(dp_max),
        "max_lat_bias": max_bias,
        "lap": max_bias / (1 << n),
        "correlation_max": corr_max,
        "log2_correlation_max": math.log2(corr_max),
    }


def trail_bounds(sbox: Sequence[int], differential_counts: Sequence[int],
                 linear_counts: Sequence[int], n: int = SBOX_BITS) -> Dict[str, Any]:
    """Single-characteristic differential and linear bounds from active-S-box counts.

    For an ``r``-round trail with ``N_r`` active S-boxes the textbook bounds are

    * differential: ``P(characteristic) <= dp_max ** N_r``;
    * linear: ``|correlation| <= corr_max ** N_r`` (piling-up lemma), so the
      data Matsui's Algorithm 2 needs for that one characteristic is of the
      order of ``correlation ** -2``.

    Both are bounds on a **single characteristic** under the standard
    independent-round heuristic, evaluated at the active-S-box **lower bound**;
    neither is a differential-hull or linear-hull statement, and neither is a
    proof of resistance.  ``rounds_to_data_limit`` is the first round count
    whose differential bound drops to or below ``2 ** -32``, resp. whose linear
    data requirement reaches ``2 ** 32``; ``None`` when the supplied counts
    never get there.
    """
    params = sbox_trail_parameters(sbox, n)
    log2_dp = params["log2_dp_max"]
    log2_corr = params["log2_correlation_max"]
    rows: List[Dict[str, Any]] = []
    diff_round: Optional[int] = None
    lin_round: Optional[int] = None
    for index, (nd, nl) in enumerate(zip(differential_counts, linear_counts)):
        rounds = index + 1
        # ``+ 0.0`` normalises the negative zero produced by ``0 * log2_dp``.
        log2_prob = nd * log2_dp + 0.0 if nd else 0.0
        log2_correlation = nl * log2_corr + 0.0 if nl else 0.0
        log2_data = -2.0 * log2_correlation if nl else 0.0
        if diff_round is None and log2_prob <= -DATA_LIMIT_LOG2:
            diff_round = rounds
        if lin_round is None and log2_data >= DATA_LIMIT_LOG2:
            lin_round = rounds
        rows.append({
            "rounds": rounds,
            "min_active_sboxes_differential": int(nd),
            "log2_max_characteristic_probability": log2_prob,
            "min_active_sboxes_linear": int(nl),
            "log2_max_characteristic_correlation": log2_correlation,
            "log2_linear_data_complexity": log2_data,
        })
    return {
        "sbox_parameters": params,
        "rows": rows,
        "rounds_to_data_limit": {
            "differential": diff_round,
            "linear": lin_round,
            "criterion": f"differential: dp_max ** N_r <= 2 ** -{int(DATA_LIMIT_LOG2)}; "
                         f"linear: corr_max ** (-2 N_r) >= 2 ** {int(DATA_LIMIT_LOG2)}",
        },
    }


# ============================================================
# 6. Generic block-size limitation
# ============================================================
def birthday_bound(block_bits: int = BLOCK_SIZE) -> Dict[str, Any]:
    """Generic birthday limitation of a ``block_bits``-bit block cipher.

    After about ``2 ** (block_bits / 2)`` blocks encrypted under one key a
    collision between two ciphertext blocks is expected, which leaks
    information in every standard mode (in ECB or CBC it already reveals a
    relation between plaintext blocks).  This depends on the block size alone
    and holds however good the S-box is, so it applies unchanged to the 32-bit
    demonstration cipher of this study.

    Raises:
        ValueError: if ``block_bits`` is not positive.
    """
    if block_bits <= 0:
        raise ValueError(f"block_bits must be positive, got {block_bits}")
    log2_blocks = block_bits / 2.0
    blocks = 2.0 ** log2_blocks
    data_bytes = blocks * block_bits / 8.0
    return {
        "block_bits": block_bits,
        "log2_birthday_blocks": log2_blocks,
        "birthday_blocks": int(blocks),
        "birthday_bytes": int(data_bytes),
        "birthday_kib": data_bytes / 1024.0,
        "full_codebook_log2_blocks": float(block_bits),
        "note": f"about 2**{log2_blocks:g} = {int(blocks)} blocks "
                f"({int(data_bytes)} bytes) may be encrypted under one key before a "
                f"ciphertext-block collision is expected; a hard limitation of the "
                f"{block_bits}-bit block size that no S-box improvement removes",
    }


# ============================================================
# 7. Test vectors
# ============================================================
def generate_test_vectors(sbox: Sequence[int], seeds: Sequence[int] = TEST_VECTOR_SEEDS
                 ) -> List[Dict[str, Any]]:
    """Deterministic (key, plaintext, ciphertext) triples with round-trip checks.

    The fixed cases are the all-zero and the all-one key and plaintext in all
    four combinations; the remaining cases draw the key and then the plaintext
    from ``random.Random(seed)`` for each seed in ``seeds``, which reproduces on
    any platform.  Every triple is verified by decrypting the ciphertext again
    (``round_trip_ok``), so a reader can check both directions of the cipher.
    """
    key_max = (1 << KEY_SIZE) - 1
    block_max = (1 << BLOCK_SIZE) - 1
    cases: List[Tuple[str, int, int]] = [
        ("all-zero key, all-zero plaintext", 0, 0),
        ("all-zero key, all-one plaintext", 0, block_max),
        ("all-one key, all-zero plaintext", key_max, 0),
        ("all-one key, all-one plaintext", key_max, block_max),
    ]
    for seed in seeds:
        rng = random.Random(seed)
        cases.append((f"random.Random({seed})",
                      rng.getrandbits(KEY_SIZE), rng.getrandbits(BLOCK_SIZE)))
    vectors: List[Dict[str, Any]] = []
    for label, key, plaintext in cases:
        cipher = DACBlockCipher(key, sbox)
        ciphertext = cipher.encrypt_block(plaintext)
        vectors.append({
            "label": label,
            "key_hex": f"{key:020X}",
            "plaintext_hex": f"{plaintext:08X}",
            "ciphertext_hex": f"{ciphertext:08X}",
            "round_trip_ok": cipher.decrypt_block(ciphertext) == plaintext,
        })
    return vectors


# ============================================================
# 8. Everything at once
# ============================================================
#: Plain-language conclusions.  They restate what the computed dictionary of
#: :func:`analyze_cipher` shows; the machine-checkable form of each is in the
#: ``findings`` entry of that dictionary.
FINDINGS: Tuple[str, ...] = (
    "Every S-box of the family has differential and linear branch number 2, the "
    "smallest possible value, so a single active bit can remain a single active "
    "bit from round to round. Unlike PRESENT, whose S-box has differential "
    "branch number 3, this cipher gets no help from its S-box in the "
    "wide-trail argument.",
    "State bits 1 and 0 bypass the S-layer, so a trail can cross rounds without "
    "activating any S-box, which is why the 16-round bound stays well below 16 "
    "active S-boxes.",
    "The branch-number bound alone therefore does NOT reach the 2**-32 data "
    "limit at 16 rounds for the DAC or the E-DAC S-box: it knows nothing about "
    "which DDT/LAT transitions exist.",
    "The exact search with the real DDT and LAT is sharper: the MILP bound is "
    "attained by DAC-OPT in both models but not by DAC or E-DAC, and the best "
    "single characteristic of every S-box of the family crosses the data limit "
    "within the 16 rounds. On the differential side the APN tables are the last "
    "to get there, because their single-bit transitions chain through the bit "
    "permutation: S-box optimality does not transfer to the cipher.",
)


def exact_trail_analysis(tables: Mapping[str, Sequence[int]], weight_rounds: int,
                         count_rounds: int,
                         milp_counts: Mapping[str, Mapping[str, Sequence[int]]]
                         ) -> Dict[str, Any]:
    """Exact per-S-box trail results next to the S-box-agnostic MILP bound.

    For every S-box and model: the exact minimum number of active S-boxes for
    ``r = 1 .. count_rounds`` (:func:`exact_min_active_sboxes`), the exact
    weight of the best single characteristic for ``r = 1 .. weight_rounds``
    (:func:`exact_best_trail_weights`), the rounds at which each reaches the
    data limit, and whether the MILP lower bound is attained.

    The data limit is ``2 ** -32`` for a characteristic probability (weight
    >= 32) and a linear data requirement ``corr ** -2 >= 2 ** 32`` (weight of
    the absolute correlation >= 16).  ``milp_counts[name][model]`` holds the
    MILP counts for the branch number of that S-box and model.
    """
    report: Dict[str, Any] = {}
    targets = {"differential": DATA_LIMIT_LOG2, "linear": DATA_LIMIT_LOG2 / 2.0}
    for name, table in tables.items():
        entry: Dict[str, Any] = {}
        params = sbox_trail_parameters(table)
        per_sbox = {"differential": -params["log2_dp_max"],
                    "linear": -params["log2_correlation_max"]}
        for model in ("differential", "linear"):
            counts = exact_min_active_sboxes(table, count_rounds, model)
            weights = exact_best_trail_weights(table, weight_rounds, model)
            milp = [int(v) for v in milp_counts[name][model][:count_rounds]]
            rounded = [round(w, 6) for w in weights["weights"]]
            target = targets[model]
            entry[model] = {
                "exact_min_active_sboxes": counts["counts"],
                "milp_min_active_sboxes": milp,
                "milp_is_tight": [int(e) == int(m) for e, m in zip(counts["counts"], milp)],
                "first_round_milp_not_tight": next(
                    (r + 1 for r, (e, m) in enumerate(zip(counts["counts"], milp)) if e != m),
                    None),
                "best_trail_weight": rounded,
                "best_trail_active_sboxes": weights["active_sboxes_of_best_trail"],
                "count_witness_trails": counts["witness_trails"],
                "weight_witness_trails": weights["witness_trails"],
                # bound evaluated at the exact count: weight >= N_exact * w_sbox
                "count_bound_weight": [round(c * per_sbox[model], 6)
                                       for c in counts["counts"][:weight_rounds]],
                "rounds_to_data_limit_exact_weight": next(
                    (r + 1 for r, w in enumerate(rounded) if w >= target - _WEIGHT_EPS), None),
                "rounds_to_data_limit_exact_count_bound": next(
                    (r + 1 for r, c in enumerate(counts["counts"])
                     if c * per_sbox[model] >= target - _WEIGHT_EPS), None),
                "weight_at_cipher_rounds": rounded[min(weight_rounds, REPORTED_ROUNDS) - 1],
                "quantities_at_cipher_rounds": trail_quantities(
                    rounded[min(weight_rounds, REPORTED_ROUNDS) - 1], model),
                "searched_rounds_weight": weight_rounds,
                "searched_rounds_count": count_rounds,
            }
        report[name] = entry
    return report


def trail_quantities(weight: float, model: str) -> Dict[str, Any]:
    """The weight of a best characteristic, spelt out in every unit it implies.

    Differential: ``w = -log2 p`` of the characteristic probability ``p``.
    Linear: ``w = -log2 |c|`` of the absolute characteristic correlation
    ``c``; the squared correlation (potential) is ``2^(-2w)`` and the data
    complexity of a single-characteristic attack is about ``c^-2 = 2^(2w)``
    known plaintexts.
    """
    if model == "differential":
        return {"weight": weight, "unit": "-log2 of the characteristic probability",
                "log2_probability": -weight}
    return {"weight": weight, "unit": "-log2 of the absolute characteristic correlation",
            "log2_abs_correlation": -weight, "log2_squared_correlation": -2.0 * weight,
            "log2_data_complexity": 2.0 * weight}


#: Pairs ``(first, second)`` compared in the text: the best characteristic of
#: ``first`` is ``2^(w_second - w_first)`` times as probable as that of ``second``.
TRAIL_COMPARISONS: Tuple[Tuple[str, str, str], ...] = (
    ("DAC-OPT", "DAC", "differential"),
    ("DAC-OPT", "E-DAC", "differential"),
    ("E-DAC", "DAC", "differential"),
    ("DAC", "DAC-OPT", "linear"),
)


def trail_comparisons(exact: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Ratios between best 16-round characteristics, from the exact weights.

    For the differential model the ratio is of probabilities,
    ``2^(w_second - w_first)``; for the linear model it is of absolute
    correlations, and the data-complexity ratio is its square.
    """
    out: Dict[str, Any] = {}
    for first, second, model in TRAIL_COMPARISONS:
        if first not in exact or second not in exact:
            continue
        w1 = float(exact[first][model]["weight_at_cipher_rounds"])
        w2 = float(exact[second][model]["weight_at_cipher_rounds"])
        diff = round(w2 - w1, 6)
        entry = {"first": first, "second": second, "model": model,
                 "weight_first": w1, "weight_second": w2,
                 "log2_ratio": diff, "ratio": round(2.0 ** diff, 2)}
        if model == "linear":
            entry["log2_data_complexity_ratio"] = round(2 * diff, 6)
            entry["data_complexity_ratio"] = round(2.0 ** (2 * diff), 2)
        out[f"{model}:{first}/{second}"] = entry
    return out


def analyze_cipher(names: Sequence[str] = SBOX_NAMES,
                   reported_rounds: int = REPORTED_ROUNDS,
                   extended_rounds: int = EXTENDED_ROUNDS) -> Dict[str, Any]:
    """Run the whole round-level analysis and return one JSON-serialisable dict.

    Deterministic: HiGHS solves every MILP to proven optimality and every random
    draw is seeded, so repeated calls return identical dictionaries.  Nothing is
    cached between calls.

    The MILP chain is solved for ``r = 1 .. extended_rounds`` once per distinct
    branch number and once per model (differential and linear are solved
    separately, even when the derived mask permutation coincides with the state
    permutation).  The extra rounds beyond the cipher's own 16 exist only so
    that the round count at which each bound reaches the ``2 ** -32`` data limit
    can be reported -- 16 rounds do not get there for every S-box.  Per-round
    rows are tabulated for ``r = 1 .. reported_rounds``.

    Raises:
        KeyError: for an unknown S-box name.
        ValueError: if a MILP does not solve or the mask derivation fails.
    """
    linear_perm = derive_linear_mask_permutation(PERM_TABLE)
    tables = {name: TABLES[name].table for name in names}
    branches = {name: branch_numbers(table) for name, table in tables.items()}

    counts: Dict[Tuple[str, int], List[int]] = {}
    for model, perm in (("differential", tuple(PERM_TABLE)), ("linear", linear_perm)):
        for branch in sorted({b[model] for b in branches.values()}):
            counts[(model, branch)] = min_active_sboxes(extended_rounds, perm, branch)

    bounds: Dict[str, Any] = {}
    for name, table in tables.items():
        diff = counts[("differential", branches[name]["differential"])]
        lin = counts[("linear", branches[name]["linear"])]
        bounds[name] = trail_bounds(table, diff[:reported_rounds], lin[:reported_rounds])
        extended = trail_bounds(table, diff, lin)
        bounds[name]["rounds_to_data_limit"] = extended["rounds_to_data_limit"]
        bounds[name]["rounds_to_data_limit"]["searched_up_to_rounds"] = extended_rounds
        bounds[name]["at_cipher_rounds"] = bounds[name]["rows"][reported_rounds - 1]

    milp_by_sbox = {name: {model: counts[(model, branches[name][model])]
                           for model in ("differential", "linear")} for name in tables}
    exact = exact_trail_analysis(tables, weight_rounds=reported_rounds,
                                 count_rounds=extended_rounds, milp_counts=milp_by_sbox)

    active = {
        "rounds": list(range(1, extended_rounds + 1)),
        "reported_rounds": reported_rounds,
        "extended_rounds": extended_rounds,
        "differential_by_branch_number": {
            str(branch): [int(v) for v in values]
            for (model, branch), values in counts.items() if model == "differential"},
        "linear_by_branch_number": {
            str(branch): [int(v) for v in values]
            for (model, branch), values in counts.items() if model == "linear"},
        "branch_number_per_sbox": branches,
        "model": "Mouha-style bit-activity MILP; LOWER BOUND, not a proof of resistance",
    }

    diff_reported = [bounds[name]["at_cipher_rounds"]["min_active_sboxes_differential"]
                     for name in tables]
    at_rounds = diff_reported[0] if diff_reported else None
    findings = {
        "all_branch_numbers_are_two": all(
            b["differential"] == 2 and b["linear"] == 2 for b in branches.values()),
        "min_active_sboxes_at_cipher_rounds": at_rounds,
        "free_rounds_at_cipher_rounds": (None if at_rounds is None
                                         else reported_rounds - at_rounds),
        "bound_reaches_data_limit_within_cipher_rounds": {
            name: {
                key: (bounds[name]["rounds_to_data_limit"][key] is not None
                      and bounds[name]["rounds_to_data_limit"][key] <= reported_rounds)
                for key in ("differential", "linear")}
            for name in tables},
        "statements": list(FINDINGS),
    }

    return {
        "round_function": describe_round_function(),
        "round_function_checks": check_round_function_description(),
        "linear_mask_permutation": {
            "table": list(linear_perm),
            "equals_state_permutation": tuple(linear_perm) == tuple(PERM_TABLE),
            "derivation": "(L^T)^{-1} over GF(2), L the matrix of the bit permutation",
        },
        "branch_numbers": branches,
        "bypass_analysis": bypass_trail_analysis(PERM_TABLE),
        "active_sboxes": active,
        "bounds": bounds,
        "exact_trails": exact,
        "exact_trail_comparisons": trail_comparisons(exact),
        "test_vectors": {name: generate_test_vectors(table) for name, table in tables.items()},
        "birthday_bound": birthday_bound(BLOCK_SIZE),
        "findings": findings,
        "caveats": [
            "The active-S-box counts are a LOWER BOUND from a branch-number "
            "relaxation that models only which bits each S-box reads and writes, "
            "the branch number, and the bit permutation.",
            "The probability and correlation figures bound a SINGLE "
            "CHARACTERISTIC under the independent-round heuristic; they are not "
            "differential-hull or linear-hull bounds and not a proof of "
            "resistance to differential or linear cryptanalysis.",
            "No related-key, boomerang, integral, algebraic or meet-in-the-middle "
            "analysis is performed, and the key schedule is not modelled.",
            "A trail attaining the bound need not exist: the model ignores "
            "whether the required single-bit DDT/LAT transitions are possible.",
        ],
    }


def format_active_sbox_table(analysis: Dict[str, Any]) -> str:
    """Render the ``r -> active S-boxes -> bound`` table of ``analysis`` as plain text."""
    names = list(analysis["bounds"])
    header = (f"{'r':>3} {'Nr(diff)':>9} {'Nr(lin)':>8} "
              + " ".join(f"{name + ' log2 P':>22}" for name in names))
    lines = [header, "-" * len(header)]
    for index in range(analysis["active_sboxes"]["reported_rounds"]):
        first = analysis["bounds"][names[0]]["rows"][index]
        cells = " ".join(
            f"{analysis['bounds'][name]['rows'][index]['log2_max_characteristic_probability']:>22.4f}"
            for name in names)
        lines.append(f"{first['rounds']:>3} "
                     f"{first['min_active_sboxes_differential']:>9} "
                     f"{first['min_active_sboxes_linear']:>8} {cells}")
    return "\n".join(lines)
