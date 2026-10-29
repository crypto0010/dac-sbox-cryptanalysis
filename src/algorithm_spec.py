"""Executable specification of the affine-class selection (paper Algorithm 2).

This module is a literal, line-by-line transcription of the pseudocode
printed in the paper (Algorithm 2 and the procedures of its appendix).  It is
written in plain Python, shares no code with :mod:`src.sbox_optimizer` (it
has its own field arithmetic, its own chaotic generator and its own GF(2)
linear algebra), and is deliberately unoptimised so that every step can be
compared with the printed text.  ``tests/test_algorithm_spec.py`` checks that
it reproduces DAC-OPT and DAC-OPT-GOLD bit for bit from seed 0.5, together
with every intermediate quantity the optimizer records (index of the kept
candidate, number of ideal-SAC candidates, accepted refinement moves,
constants and the number of generator draws).

Conventions (paper Section 6.2)
-------------------------------
* A 5-bit value ``v`` is an integer 0..31; its coordinate ``i`` is bit
  ``4 - i`` (most significant first).  ``v`` is also the element
  ``sum_k bit_k(v) X^k`` of GF(2^5) = F_2[X] / (X^5 + X^2 + 1).
* A matrix is given by five row masks ``R_0..R_4``: output coordinate ``j``
  of ``M z`` is ``parity(R_j & z)``.  Its columns are the images of the five
  unit inputs; the column for input coordinate ``i`` is the mask whose bit
  ``4 - j`` is bit ``4 - i`` of ``R_j``.
* ``S(x) = A F(B x XOR b) XOR a`` with ``F(z) = z^e`` and ``F(0) = 0``.
* All real arithmetic is IEEE 754 binary64, evaluated left to right exactly
  as written below.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

POLY = 0b100101          # X^5 + X^2 + 1
N = 5
Q = 32


# ------------------------------------------------------------------
# Field and bit helpers
# ------------------------------------------------------------------
def parity(v: int) -> int:
    return bin(v).count("1") & 1


def gf_mul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a & Q:
            a ^= POLY
    return r


def power(z: int, e: int) -> int:
    r = 1
    for _ in range(e):
        r = gf_mul(r, z)
    return r if z else 0


def apply_rows(rows: Sequence[int], z: int) -> int:
    """``M z`` for the matrix with row masks ``rows`` (row j -> bit 4 - j)."""
    out = 0
    for j, row in enumerate(rows):
        out |= parity(row & z) << (N - 1 - j)
    return out


def apply_columns(cols: Sequence[int], x: int) -> int:
    """``M x`` for the matrix with columns ``cols`` (column i <-> bit 4 - i of x)."""
    out = 0
    for i, col in enumerate(cols):
        if (x >> (N - 1 - i)) & 1:
            out ^= col
    return out


def transpose(masks: Sequence[int]) -> List[int]:
    """Rows <-> columns of a 5x5 matrix given by five masks."""
    return [sum(((masks[j] >> (N - 1 - i)) & 1) << (N - 1 - j) for j in range(N))
            for i in range(N)]


def is_invertible(vectors: Sequence[int]) -> bool:
    """True when the five masks are linearly independent over F_2."""
    basis: List[int] = []
    for v in vectors:
        for b in basis:
            v = min(v, v ^ b)
        if v == 0:
            return False
        basis.append(v)
    return True


def in_span(v: int, vectors: Sequence[int]) -> bool:
    span = {0}
    for w in vectors:
        span |= {s ^ w for s in span}
    return v in span


# ------------------------------------------------------------------
# Procedure GEN: the chaotic decision source
# ------------------------------------------------------------------
class Gen:
    """Tent-logistic generator: ``u_t = frac(2^16 * s_t)`` after 1000 steps.

    ``s_{t+1} = f1(s_t)`` if ``s_t <= 1/2`` else ``f2(s_t)``, with
    ``f1(s) = w_l r_l s (1 - s) + w_t r_t s`` and
    ``f2(s) = w_l r_l s (1 - s) + w_t r_t (1 - s)``,
    ``w_l = (9 - r) / 9``, ``w_t = r / 9``, ``r = 8``, ``r_l = 3.9``,
    ``r_t = 1.8``; the seed ``s_0`` is first folded to ``min(s_0, 1 - s_0)``.
    ``INT(h) = floor(h * u)`` for the next variate ``u``.
    """

    def __init__(self, s0: float, r: float = 8.0, r_l: float = 3.9, r_t: float = 1.8,
                 transient: int = 1000) -> None:
        self.w_l = (9.0 - r) / 9.0
        self.w_t = r / 9.0
        self.r_l = r_l
        self.r_t = r_t
        s = min(float(s0), 1.0 - float(s0))
        for _ in range(transient):
            s = self._step(s)
        self.s = s
        self.draws = 0

    def _step(self, s: float) -> float:
        logistic = self.w_l * self.r_l * s * (1.0 - s)
        if s <= 0.5:
            return logistic + self.w_t * self.r_t * s
        return logistic + self.w_t * self.r_t * (1.0 - s)

    def uniform(self) -> float:
        self.s = self._step(self.s)
        self.draws += 1
        return (self.s * 65536.0) % 1.0

    def INT(self, h: int) -> int:  # noqa: N802 - name as printed in the paper
        return int(self.uniform() * h)


# ------------------------------------------------------------------
# Autocorrelation table and the avalanche key
# ------------------------------------------------------------------
def autocorrelation(e: int) -> List[List[int]]:
    """``AC[c][d] = sum_y (-1)^(parity(c & (F(y) ^ F(y ^ d))))``."""
    f = [power(y, e) for y in range(Q)]
    return [[sum(1 - 2 * parity(c & (f[y] ^ f[y ^ d])) for y in range(Q))
             for d in range(Q)] for c in range(Q)]


def avalanche_key(ac: Sequence[Sequence[int]], a_rows: Sequence[int],
                  b_cols: Sequence[int]) -> Tuple[int, int, int, int]:
    """``kappa(A, B)``: integer form of (D_max, |SAC-1/2|, BIC max dev, BIC mean dev).

    ``P_ij = 1/2 - AC[alpha_j][beta_i] / 64`` and the BIC-SAC value of the
    output pair (j, k) is ``1/2 - sum_i AC[alpha_j ^ alpha_k][beta_i] / 320``.
    """
    entries = [ac[aj][bi] for aj in a_rows for bi in b_cols]
    pairs = [sum(ac[a_rows[j] ^ a_rows[k]][bi] for bi in b_cols)
             for j in range(N) for k in range(j + 1, N)]
    return (max(abs(v) for v in entries), abs(sum(entries)),
            max(abs(v) for v in pairs), abs(sum(pairs)))


# ------------------------------------------------------------------
# The four procedures of Algorithm 2
# ------------------------------------------------------------------
def sample_input_matrices(gen: Gen, count: int) -> List[List[int]]:
    """SAMPLE: ``count`` invertible input matrices, as column lists.

    Pass 1 draws ``1 + INT(31)`` five times per candidate, candidates in
    index order.  Then, while any candidate is singular, every singular
    candidate -- in increasing index order -- redraws all five columns.
    """
    cands = [[1 + gen.INT(31) for _ in range(N)] for _ in range(count)]
    bad = [k for k in range(count) if not is_invertible(cands[k])]
    while bad:
        for k in bad:
            cands[k] = [1 + gen.INT(31) for _ in range(N)]
        bad = [k for k in bad if not is_invertible(cands[k])]
    return cands


def greedy_output_rows(ac: Sequence[Sequence[int]], b_cols: Sequence[int]) -> List[int]:
    """GREEDY: rows of A in increasing order of (max_i |AC|, sum_i |AC|, c).

    Masks c = 1..31 are visited in that order and c is appended when it is
    not in the span of the rows already chosen, until five rows are chosen.
    """
    def weight(c: int) -> Tuple[int, int, int]:
        dev = [abs(ac[c][d]) for d in b_cols]
        return (max(dev), sum(dev), c)
    rows: List[int] = []
    for c in sorted(range(1, Q), key=weight):
        if len(rows) < N and not in_span(c, rows):
            rows.append(c)
    return rows


def refine(ac: Sequence[Sequence[int]], rows: List[int], b_cols: Sequence[int],
           gen: Gen, steps: int) -> Tuple[List[int], int]:
    """REFINE: ``steps`` proposals j = INT(5), c = 1 + INT(31), row j := c.

    A proposal leaving A singular is skipped; otherwise it is accepted iff
    its key is lexicographically smaller.  Returns the rows and the number
    of accepted proposals.
    """
    best = avalanche_key(ac, rows, b_cols)
    accepted = 0
    for _ in range(steps):
        j = gen.INT(N)
        c = 1 + gen.INT(Q - 1)
        trial = list(rows)
        trial[j] = c
        if not is_invertible(trial):
            continue
        key = avalanche_key(ac, trial, b_cols)
        if key < best:
            rows, best, accepted = trial, key, accepted + 1
    return rows, accepted


def choose_constants(e: int, a_rows: Sequence[int], b_cols: Sequence[int],
                     gen: Gen, count: int) -> Tuple[int, int]:
    """CONSTANTS: ``count`` pairs (a, b) = (INT(32), INT(32)), keep the first
    pair minimising (#fixed points, #opposite fixed points)."""
    best = None
    for m in range(count):
        a = gen.INT(Q)
        b = gen.INT(Q)
        table = [apply_rows(a_rows, power(apply_columns(b_cols, x) ^ b, e)) ^ a
                 for x in range(Q)]
        fixed = sum(1 for x in range(Q) if table[x] == x)
        opposite = sum(1 for x in range(Q) if table[x] == x ^ (Q - 1))
        key = (fixed, opposite, m)
        if best is None or key < best[0]:
            best = (key, a, b)
    return best[1], best[2]


def affine_select(e: int, s0: float, candidates: int, refine_steps: int = 256,
                  constant_pairs: int = 64) -> Dict[str, object]:
    """Algorithm 2: the affine-class selection, returning tables and statistics."""
    gen = Gen(s0)
    ac = autocorrelation(e)
    b_list = sample_input_matrices(gen, candidates)
    scored = []
    for k, b_cols in enumerate(b_list):
        a_rows = greedy_output_rows(ac, b_cols)
        scored.append((avalanche_key(ac, a_rows, b_cols), k, a_rows))
    key, best_k, a_rows = min(scored)
    b_cols = b_list[best_k]
    a_rows, accepted = refine(ac, a_rows, b_cols, gen, refine_steps)
    a, b = choose_constants(e, a_rows, b_cols, gen, constant_pairs)
    table = [apply_rows(a_rows, power(apply_columns(b_cols, x) ^ b, e)) ^ a for x in range(Q)]
    return {
        "sbox": table,
        "A_rows": list(a_rows), "a": a,
        "B_rows": transpose(b_cols), "B_columns": list(b_cols), "b": b,
        "best_candidate_index": best_k,
        "ideal_sac_candidates": sum(1 for s in scored if s[0][0] == 0),
        "refine_accepted": accepted,
        "draws": gen.draws,
    }


def worked_example(e: int, a_rows: Sequence[int], a: int, b_rows: Sequence[int], b: int,
                   x: int) -> Dict[str, int]:
    """Every intermediate value of ``S(x) = A F(B x ^ b) ^ a`` for one input."""
    bx = apply_rows(b_rows, x)
    z = bx ^ b
    y = power(z, e)
    ay = apply_rows(a_rows, y)
    return {"x": x, "Bx": bx, "z": z, "F(z)": y, "A F(z)": ay, "S(x)": ay ^ a}
