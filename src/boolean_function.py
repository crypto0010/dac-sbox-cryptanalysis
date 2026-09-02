"""Boolean-function toolkit for n-bit S-boxes.

Provides the low-level, numpy-vectorised building blocks used by
:mod:`src.security_analysis`:

* cached bit-lookup tables (popcount, parity, sign, XOR, Mobius matrices),
* truth tables of the coordinate and component functions of an S-box,
* Algebraic Normal Form (ANF) via the Mobius transform,
* algebraic degrees and balancedness of *all* 2^n - 1 component functions.

Conventions (shared with the rest of the project):

* S-box values are integers in ``0 .. 2^n - 1``.
* Coordinate ``i`` (``0 <= i < n``) is output bit ``n - 1 - i`` (MSB first),
  i.e. its mask is ``1 << (n - 1 - i)``.
* Component function ``f_b(x) = b . S(x)`` (parity of ``b & S(x)``) for a
  nonzero output mask ``b``; the coordinates are the components with a
  single-bit mask.
* ANF variables ``x0 .. x(n-1)``: ``x_i`` is input bit ``i`` (``x0`` = LSB).
  A monomial is stored as a tuple of variable indices, e.g. ``(0, 3)`` is
  ``x0 & x3``; its index in the coefficient vector is the mask ``0b1001``.

The paper states the Boolean function of the DAC S-box uses
3 NAND + 2 NOT + 3 XOR gates but does not publish the circuit; this module
therefore reverse-engineers the ANF of every component from the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Monomial = Tuple[int, ...]
ANF = Dict[Monomial, int]


# ============================================================
# Cached bit-lookup tables
# ============================================================
@dataclass(frozen=True)
class BitTables:
    """Read-only lookup tables for the n-bit Boolean space ``F_2^n``.

    Attributes:
        n: Number of bits.
        size: ``2 ** n``.
        popcount: ``(size,)`` int64 -- Hamming weight of every value.
        parity: ``(size, size)`` int8 -- ``popcount(a & b) & 1``.
        sign: ``(size, size)`` float64 -- ``(-1) ** popcount(a & b)``
            (float so that BLAS matmul can be used for Walsh spectra).
        xor: ``(size, size)`` int64 -- ``a ^ b``.
        bits: ``(size, n)`` int8 -- MSB-first bits of every value, so
            ``bits[v, i]`` is coordinate ``i`` of ``v``.
        coord_masks: ``(n,)`` int64 -- ``1 << (n - 1 - i)`` for coordinate i.
        mobius: ``(size, size)`` float64 -- ``1`` where ``x`` is a subset of
            ``u`` (``mobius[u, x]``); ``(mobius @ f) % 2`` is the ANF of ``f``.
    """

    n: int
    size: int
    popcount: np.ndarray
    parity: np.ndarray
    sign: np.ndarray
    xor: np.ndarray
    bits: np.ndarray
    coord_masks: np.ndarray
    mobius: np.ndarray


@lru_cache(maxsize=None)
def bit_tables(n: int = 5) -> BitTables:
    """Return the (cached, read-only) :class:`BitTables` for ``n`` bits."""
    if n < 1:
        raise ValueError("n must be >= 1")
    size = 1 << n
    idx = np.arange(size, dtype=np.int64)
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)  # MSB first
    bits = ((idx[:, None] >> shifts[None, :]) & 1).astype(np.int8)
    popcount = bits.sum(axis=1).astype(np.int64)
    parity = (popcount[idx[:, None] & idx[None, :]] & 1).astype(np.int8)
    sign = (1 - 2 * parity).astype(np.float64)
    xor = idx[:, None] ^ idx[None, :]
    coord_masks = (np.int64(1) << shifts).astype(np.int64)
    mobius = ((idx[None, :] & ~idx[:, None]) == 0).astype(np.float64)
    tables = BitTables(n, size, popcount, parity, sign, xor, bits,
                       coord_masks, mobius)
    for arr in (popcount, parity, sign, xor, bits, coord_masks, mobius):
        arr.setflags(write=False)
    return tables


def as_sbox_array(sbox: Sequence[int], n: int = 5) -> np.ndarray:
    """Validate an S-box and return it as an int64 numpy array.

    Raises ``ValueError`` if the table does not have ``2 ** n`` entries or
    contains values outside ``0 .. 2 ** n - 1``.
    """
    arr = np.asarray(sbox, dtype=np.int64)
    size = 1 << n
    if arr.shape != (size,):
        raise ValueError(f"S-box must have {size} entries for n={n}, "
                         f"got shape {arr.shape}")
    if arr.min() < 0 or arr.max() >= size:
        raise ValueError(f"S-box values must lie in 0..{size - 1}")
    return arr


# ============================================================
# Truth tables
# ============================================================
def coordinate_truth_tables(sbox: Sequence[int], n: int = 5) -> np.ndarray:
    """Truth tables of the ``n`` coordinate functions, MSB first.

    Returns an ``(n, 2^n)`` int8 array ``T`` with ``T[i, x]`` = output bit
    ``n - 1 - i`` of ``S(x)``.
    """
    tables = bit_tables(n)
    return tables.bits[as_sbox_array(sbox, n)].T.copy()


def component_truth_tables(sbox: Sequence[int], n: int = 5) -> np.ndarray:
    """Truth tables of all ``2^n`` component functions ``f_b(x) = b . S(x)``.

    Returns a ``(2^n, 2^n)`` int8 array ``F`` with ``F[x, b]`` = parity of
    ``b & S(x)``. Column 0 is the zero function.
    """
    tables = bit_tables(n)
    return tables.parity[as_sbox_array(sbox, n)]


# ============================================================
# Algebraic Normal Form (Mobius transform)
# ============================================================
def anf_matrix(truth_tables: np.ndarray, n: int = 5) -> np.ndarray:
    """Mobius transform of one or more truth tables.

    Args:
        truth_tables: ``(2^n,)`` or ``(2^n, m)`` array of 0/1 values, each
            column being a truth table indexed by the input ``x``.
        n: Number of input variables.

    Returns:
        int8 array of the same shape whose entry ``[u, j]`` is the ANF
        coefficient of monomial mask ``u`` in column ``j``.
    """
    tables = bit_tables(n)
    tt = np.asarray(truth_tables, dtype=np.float64)
    if tt.shape[0] != tables.size:
        raise ValueError(f"truth table must have {tables.size} rows for n={n}")
    return (tables.mobius @ tt % 2).astype(np.int8)


def compute_anf(truth_table: Sequence[int], n_vars: int = 5) -> ANF:
    """Compute the Algebraic Normal Form of a Boolean function.

    Returns a dict mapping monomial tuples to coefficient 1, e.g.
    ``{(0,): 1, (0, 1): 1}`` means ``x0 ^ (x0 & x1)``. Only monomials with
    nonzero coefficient are present; the empty tuple is the constant 1.
    """
    coeffs = anf_matrix(np.asarray(truth_table, dtype=np.int64) & 1, n_vars)
    return {
        tuple(i for i in range(n_vars) if (int(u) >> i) & 1): 1
        for u in np.flatnonzero(coeffs)
    }


def anf_to_string(monomials: ANF, var_names: Optional[Sequence[str]] = None) -> str:
    """Render an ANF dictionary as a readable XOR-of-ANDs expression."""
    terms: List[str] = []
    if monomials.get(()):
        terms.append("1")
    for monomial, coeff in sorted(monomials.items(), key=lambda kv: (len(kv[0]), kv[0])):
        if monomial == () or not coeff:
            continue
        names = [var_names[i] if var_names is not None else f"x{i}" for i in monomial]
        term = " & ".join(names)
        terms.append(f"({term})" if len(monomial) > 1 else term)
    return " ^ ".join(terms) if terms else "0"


def algebraic_degree(monomials: ANF) -> int:
    """Algebraic degree of an ANF dictionary (0 for the zero function)."""
    return max((len(m) for m, c in monomials.items() if c), default=0)


# ============================================================
# All-component degrees and weights
# ============================================================
def vectorial_anf(sbox: Sequence[int], n: int = 5) -> np.ndarray:
    """ANF coefficients of the vectorial function ``S: F_2^n -> F_2^n``.

    Returns a ``(2^n,)`` int64 array ``V`` where ``V[u]`` packs, in the same
    bit positions as the S-box output, the coefficient of monomial ``u`` in
    every coordinate. Component ``b`` has coefficient ``parity(b & V[u])``.
    """
    tables = bit_tables(n)
    coeff_bits = anf_matrix(tables.bits[as_sbox_array(sbox, n)], n)  # (2^n, n)
    return (coeff_bits.astype(np.int64) * tables.coord_masks[None, :]).sum(axis=1)


def component_anf_matrix(sbox: Sequence[int], n: int = 5) -> np.ndarray:
    """ANF coefficients of all ``2^n`` components as an ``(2^n, 2^n)`` int8 array.

    Entry ``[u, b]`` is the coefficient of monomial mask ``u`` in ``f_b``.
    """
    return anf_matrix(component_truth_tables(sbox, n), n)


def component_degrees(sbox: Sequence[int], n: int = 5) -> np.ndarray:
    """Algebraic degree of every component ``f_b``, ``b = 0 .. 2^n - 1``.

    Entry 0 (the zero function) is reported as degree 0.
    """
    tables = bit_tables(n)
    coeffs = component_anf_matrix(sbox, n)
    return (coeffs.astype(np.int64) * tables.popcount[:, None]).max(axis=0)


def component_weights(sbox: Sequence[int], n: int = 5) -> np.ndarray:
    """Hamming weight of every component truth table (``2^(n-1)`` if balanced)."""
    return component_truth_tables(sbox, n).sum(axis=0, dtype=np.int64)


# ============================================================
# Per-coordinate report used by the runners
# ============================================================
def analyze_component_functions(sbox_table: Sequence[int], n: int = 5) -> List[dict]:
    """Analyse the ``n`` coordinate Boolean functions of the S-box.

    Returns one dict per output bit (MSB first) with keys ``bit_position``,
    ``truth_table``, ``anf``, ``anf_string``, ``algebraic_degree``,
    ``hamming_weight`` and ``is_balanced``.
    """
    coords = coordinate_truth_tables(sbox_table, n)
    coeffs = anf_matrix(coords.T, n)  # (2^n, n)
    half = 1 << (n - 1)
    results = []
    for bit_pos in range(n):
        monomials: ANF = {
            tuple(i for i in range(n) if (int(u) >> i) & 1): 1
            for u in np.flatnonzero(coeffs[:, bit_pos])
        }
        weight = int(coords[bit_pos].sum())
        results.append({
            "bit_position": bit_pos,
            "truth_table": coords[bit_pos].astype(int).tolist(),
            "anf": monomials,
            "anf_string": anf_to_string(monomials),
            "algebraic_degree": algebraic_degree(monomials),
            "hamming_weight": weight,
            "is_balanced": weight == half,
        })
    return results


def verify_boolean_properties(sbox_table: Sequence[int], n: int = 5) -> List[dict]:
    """Print the ANF / degree / balancedness report and return the analysis."""
    analysis = analyze_component_functions(sbox_table, n)
    size = 1 << n
    degrees_all = component_degrees(sbox_table, n)[1:]
    weights_all = component_weights(sbox_table, n)[1:]
    n_components = size - 1
    balanced_all = int(np.count_nonzero(weights_all == size // 2))

    print("=" * 70)
    print(f"BOOLEAN FUNCTION ANALYSIS - {n}-bit S-Box")
    print("=" * 70)
    for comp in analysis:
        bit = comp["bit_position"]
        print(f"\nOutput Bit {bit} (f{bit}):")
        print(f"  ANF: {comp['anf_string']}")
        print(f"  Algebraic Degree: {comp['algebraic_degree']}")
        print(f"  Hamming Weight: {comp['hamming_weight']}/{size}")
        print(f"  Balanced: {'Yes' if comp['is_balanced'] else 'No'}")

    degrees = [c["algebraic_degree"] for c in analysis]
    balanced_count = sum(1 for c in analysis if c["is_balanced"])
    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"  Coordinate degrees: max {max(degrees)}, min {min(degrees)} "
          f"(optimal for a permutation: {n - 1})")
    print(f"  Balanced coordinates: {balanced_count}/{n}")
    print(f"  All {n_components} components: degree min {int(degrees_all.min())}, "
          f"max {int(degrees_all.max())}; balanced {balanced_all}/{n_components}")
    print(f"{'=' * 70}")
    return analysis
