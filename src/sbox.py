"""DAC (Dynamic Airy Chaotic) Substitution Box Implementation.

The 5-bit S-box maps each 5-bit input (0-31) to a unique 5-bit output (0-31).
Construction: Tent-Logistic chaotic mapping + Boolean function composition.

Block layout (paper Section 4.1 / Fig. 3): a 32-bit block is split MSB-first
into six 5-bit chunks (bit positions 31..2) that each pass through the S-box, and
the two remaining bits (positions 1 and 0) are flipped. For a general
``block_bits`` the remainder is ``block_bits % 5`` bits wide and its MSB and LSB
are flipped.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .chaotic_maps import TentLogisticMap, canonical_seed


SBOX_BITS = 5
SBOX_SIZE = 1 << SBOX_BITS

# The DAC S-Box lookup table from Table 3 of the paper
DAC_SBOX = [
    30, 24,  9, 27, 28, 26,  4, 22,  2, 15, 29,  5, 13,  8, 25,  3,  # 0-15
    11, 10, 16, 31,  0, 20, 14, 21,  1, 12, 18,  6, 23, 19, 17,  7,  # 16-31
]

_MASK32 = 0xFFFFFFFF


def validate_sbox_table(table: Sequence[int] | np.ndarray,
                        n_bits: int = SBOX_BITS) -> list[int]:
    """Return ``table`` as a list of Python ints after checking it is an n-bit permutation.

    Raises:
        TypeError: if an entry is not an integer.
        ValueError: if the length is not 2**n_bits or the entries are not a
            permutation of 0..2**n_bits-1.
    """
    size = 1 << n_bits
    values = list(table)
    if len(values) != size:
        raise ValueError(f"S-box must have {size} entries, got {len(values)}")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise TypeError(f"S-box entries must be integers, got {v!r}")
    values = [int(v) for v in values]
    if sorted(values) != list(range(size)):
        raise ValueError(f"S-box must be a permutation of 0..{size - 1}")
    return values


def invert_permutation(table: Sequence[int]) -> list[int]:
    """Return the inverse of a permutation given as a list ``table[x] = y``."""
    inverse = [0] * len(table)
    for x, y in enumerate(table):
        inverse[y] = x
    return inverse


# Inverse S-Box (for decryption)
DAC_SBOX_INV = invert_permutation(DAC_SBOX)


def _substitute_block_generic(table: Sequence[int], data: int, block_bits: int) -> int:
    """Chunk-wise substitution of a ``block_bits``-bit integer (any width >= 1).

    Full 5-bit chunks are taken MSB-first and looked up in ``table``; the
    ``block_bits % 5`` remainder bits keep their position with MSB and LSB flipped.
    """
    n_full, remainder = divmod(block_bits, SBOX_BITS)
    result = 0
    shift = block_bits
    for _ in range(n_full):
        shift -= SBOX_BITS
        result |= table[(data >> shift) & 0x1F] << shift
    if remainder:
        flip_mask = 1 | (1 << (remainder - 1))
        result |= (data & ((1 << remainder) - 1)) ^ flip_mask
    return result


def _substitute_wrap_window(table: Sequence[int], data: int) -> int:
    """Apply ``table`` to the wrap-around 5-bit window (bits 1, 0, 31, 30, 29) of ``data``.

    The block is rotated left by three so that the window becomes the low
    five bits (bit 1 of ``data`` is the window's MSB, bit 29 its LSB),
    substituted there, and rotated back.
    """
    rotated = ((data << 3) | (data >> 29)) & _MASK32
    rotated = (rotated & ~0x1F) | table[rotated & 0x1F]
    return (rotated >> 3) | ((rotated & 0x7) << 29)


class DACSBox:
    """5-bit Dynamic Airy Chaotic Substitution Box.

    Wraps any bijective 32-entry table (default: the paper's Table 3) together
    with its inverse and provides the block-level substitution layout used by the
    SPN cipher in :mod:`src.block_cipher`.

    Attributes:
        table: forward table, ``table[x] = S(x)`` (list of 32 Python ints).
        inv_table: inverse table, ``inv_table[S(x)] = x``.
    """

    def __init__(self, sbox_table: Sequence[int] | np.ndarray | None = None) -> None:
        """Initialize with a 32-entry bijective table (paper's Table 3 if None).

        Raises:
            TypeError, ValueError: see :func:`validate_sbox_table`.
        """
        self.table: list[int] = validate_sbox_table(
            DAC_SBOX if sbox_table is None else sbox_table)
        self.inv_table: list[int] = invert_permutation(self.table)
        self._np_table = np.array(self.table, dtype=np.uint8)
        self._np_inv_table = np.array(self.inv_table, dtype=np.uint8)

    @staticmethod
    def _check_value(value: int) -> None:
        if not 0 <= value < SBOX_SIZE:
            raise ValueError(f"S-box input must be 0-{SBOX_SIZE - 1}, got {value}")

    def substitute(self, value: int) -> int:
        """Forward substitution: 5-bit input -> 5-bit output."""
        self._check_value(value)
        return self.table[value]

    def inverse_substitute(self, value: int) -> int:
        """Inverse substitution for decryption."""
        self._check_value(value)
        return self.inv_table[value]

    def substitute_array(self, values: np.ndarray) -> np.ndarray:
        """Vectorised forward substitution of an integer array of 5-bit values (no validation)."""
        return self._np_table[values]

    def inverse_substitute_array(self, values: np.ndarray) -> np.ndarray:
        """Vectorised inverse substitution of an integer array of 5-bit values (no validation)."""
        return self._np_inv_table[values]

    def substitute_block32(self, data: int) -> int:
        """Unrolled 32-bit substitution layer (6 x 5-bit chunks + 2 flipped bits).

        No input validation: the caller guarantees ``0 <= data < 2**32``.
        """
        t = self.table
        return ((t[data >> 27] << 27)
                | (t[(data >> 22) & 0x1F] << 22)
                | (t[(data >> 17) & 0x1F] << 17)
                | (t[(data >> 12) & 0x1F] << 12)
                | (t[(data >> 7) & 0x1F] << 7)
                | (t[(data >> 2) & 0x1F] << 2)
                | ((data & 0x3) ^ 0x3))

    def inverse_substitute_block32(self, data: int) -> int:
        """Unrolled inverse of :meth:`substitute_block32` (no input validation)."""
        t = self.inv_table
        return ((t[data >> 27] << 27)
                | (t[(data >> 22) & 0x1F] << 22)
                | (t[(data >> 17) & 0x1F] << 17)
                | (t[(data >> 12) & 0x1F] << 12)
                | (t[(data >> 7) & 0x1F] << 7)
                | (t[(data >> 2) & 0x1F] << 2)
                | ((data & 0x3) ^ 0x3))

    def substitute_block32_full(self, data: int) -> int:
        """32-bit substitution layer in which every bit passes through the S-box.

        NOT part of the paper's design. The standard layer
        (:meth:`substitute_block32`) only flips the two low bits, so they never
        see a nonlinear operation. This variant applies the standard layer and
        then a seventh S-box to the wrap-around window formed by the two low
        bits and the three high bits (positions 1, 0, 31, 30, 29). Bits 31..29
        therefore pass through the S-box twice; the layer stays bijective.
        No input validation: the caller guarantees ``0 <= data < 2**32``.
        """
        return _substitute_wrap_window(self.table, self.substitute_block32(data))

    def inverse_substitute_block32_full(self, data: int) -> int:
        """Inverse of :meth:`substitute_block32_full` (no input validation)."""
        return self.inverse_substitute_block32(
            _substitute_wrap_window(self.inv_table, data))

    @staticmethod
    def _check_block(data: int, block_bits: int) -> None:
        if block_bits < 1:
            raise ValueError(f"block_bits must be >= 1, got {block_bits}")
        if not 0 <= data < (1 << block_bits):
            raise ValueError(f"data must be a {block_bits}-bit unsigned integer, got {data}")

    def substitute_block(self, data: int, block_bits: int = 32) -> int:
        """Apply the S-box to a ``block_bits``-bit block (Section 4.1 / Fig. 3).

        The block is split MSB-first into ``block_bits // 5`` 5-bit chunks that
        are substituted in place; the ``block_bits % 5`` remainder bits have
        their MSB and LSB flipped (both bits for a 32-bit block).

        Raises:
            ValueError: if ``block_bits < 1`` or ``data`` is not a
                ``block_bits``-bit unsigned integer.
        """
        self._check_block(data, block_bits)
        if block_bits == 32:
            return self.substitute_block32(data)
        return _substitute_block_generic(self.table, data, block_bits)

    def inverse_substitute_block(self, data: int, block_bits: int = 32) -> int:
        """Inverse of :meth:`substitute_block`."""
        self._check_block(data, block_bits)
        if block_bits == 32:
            return self.inverse_substitute_block32(data)
        return _substitute_block_generic(self.inv_table, data, block_bits)

    def is_bijective(self) -> bool:
        """True if the table is a permutation (always holds after validation)."""
        return sorted(self.table) == list(range(SBOX_SIZE))

    def get_component_functions(self) -> list[list[int]]:
        """Extract the 5 coordinate Boolean functions of the S-box.

        Component ``i`` is the truth table (32 binary values) of output bit
        ``4 - i`` (MSB first).
        """
        return [[(y >> (SBOX_BITS - 1 - bit_pos)) & 1 for y in self.table]
                for bit_pos in range(SBOX_BITS)]

    def to_hex_string(self) -> str:
        """S-box as hex string representation."""
        return " ".join(f"{v:02X}" for v in self.table)

    def to_binary_table(self) -> str:
        """S-box as a binary input -> output listing."""
        return "\n".join(f"{x:2d} ({x:05b}) -> {y:2d} ({y:05b})"
                         for x, y in enumerate(self.table))

    def __repr__(self) -> str:
        return f"DACSBox(table={self.table})"


def generate_dynamic_sbox(key_seed: float = 0.5, r_logistic: float = 3.9,
                          r_tent: float = 1.8, r: float = 8.0) -> DACSBox:
    """Generate a key-dependent dynamic S-box using the tent-logistic map.

    This is an IMPROVEMENT over the paper's static S-box.  The map is
    iterated 1000 times from the key seed (warm-up); the next 32 orbit
    points ``s`` are whitened to ``u = frac(2^16 * s)`` -- the mantissa bits
    behind the leading sixteen, the same post-processing as
    :class:`src.sbox_optimizer.ChaoticRNG`, whose first 32 variates these
    are -- and the S-box is the ranking permutation of those 32 values:
    ``S(x)`` is the rank of ``u_x``.  The same seed always gives the same
    S-box and distinct keys give distinct S-boxes (with overwhelming
    probability).

    Ranking the raw orbit points instead would couple neighbouring entries.
    Consecutive points satisfy ``s_{x+1} = f(s_x)`` with ``f`` unimodal, so
    ``S(x + 1)`` is almost a function of ``S(x)`` (the largest values are
    always followed by the smallest), and when the warm-up ends near the
    repelling fixed point of ``f2`` (about 0.6531, slope -1.73) the samples
    diverge geometrically and the table starts ``16, 15, 17, 14, 18, 13,
    ...`` with a maximum LAT bias of 12.  That structure lives in the leading
    bits of ``s``; the bits kept by the whitening are scrambled by the
    positive Lyapunov exponent, so the ranks of consecutive samples are
    decorrelated and the tables behave like random permutations.

    The map satisfies ``f(1 - s) = f(s)``, so ``key_seed`` and
    ``1 - key_seed`` are the SAME key: the effective key space is (0, 1/2]
    and a seed above 1/2 is folded to ``1 - key_seed`` by
    :func:`~src.chaotic_maps.canonical_seed`, which also rejects the
    degenerate endpoints 0 and 1.

    Raises:
        ValueError: if ``key_seed`` is not strictly inside (0, 1).
    """
    warmup, shift = 1000, 16
    tlm = TentLogisticMap(r_logistic=r_logistic, r_tent=r_tent,
                          seed=canonical_seed(key_seed), r=r)
    orbit = tlm.iterate(warmup + 1 + SBOX_SIZE)[warmup + 1:]
    samples = (orbit * float(1 << shift)) % 1.0    # exact in binary floating point
    sbox = np.empty(SBOX_SIZE, dtype=np.int64)
    sbox[np.argsort(samples, kind="stable")] = np.arange(SBOX_SIZE)
    return DACSBox(sbox_table=sbox)
