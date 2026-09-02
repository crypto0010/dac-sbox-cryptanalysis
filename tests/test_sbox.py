"""Tests for src.sbox and the SPN cipher's S-layer modes.

Covers the unrolled 32-bit S-layer against the generic chunk-wise reference,
the wrap-around window of the full S-layer (bit positions pinned), the
full-layer roundtrip inside :class:`DACBlockCipher` and the key-space fold of
the dynamic S-box generator (``key_seed`` and ``1 - key_seed`` are one key).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.block_cipher import DACBlockCipher
from src.chaotic_maps import canonical_seed
from src.sbox import (
    DAC_SBOX, DACSBox, _substitute_block_generic, _substitute_wrap_window,
    generate_dynamic_sbox,
)
from src.sbox_optimizer import ChaoticRNG
from src.security_analysis import fast_metrics

BLOCKS = np.random.default_rng(20250830).integers(0, 1 << 32, 2000, dtype=np.uint64).tolist()
BLOCKS += [0, 0xFFFFFFFF, 0x80000000, 0x00000001, 0xDEADBEEF]


def _popcount(v: int) -> int:
    return bin(v).count("1")


# ------------------------------------------------------------
# Standard S-layer (paper layout)
# ------------------------------------------------------------
def test_substitute_block32_matches_generic_reference():
    sbox = DACSBox()
    for x in BLOCKS:
        assert sbox.substitute_block32(x) == _substitute_block_generic(DAC_SBOX, x, 32)
        assert sbox.substitute_block(x, 32) == sbox.substitute_block32(x)
        assert sbox.inverse_substitute_block32(sbox.substitute_block32(x)) == x
        assert sbox.inverse_substitute_block(x, 32) == _substitute_block_generic(sbox.inv_table, x, 32)


def test_standard_layer_only_complements_low_bits():
    sbox = DACSBox()
    for x in BLOCKS:
        y = sbox.substitute_block32(x)
        assert (y & 0x3) == ((x & 0x3) ^ 0x3)
        assert (y >> 2) == _substitute_block_generic(DAC_SBOX, x >> 2, 30)


# ------------------------------------------------------------
# Wrap-around window / full S-layer
# ------------------------------------------------------------
def test_wrap_window_bit_positions():
    # Window (bits 1, 0, 31, 30, 29): window MSB = data bit 1, window LSB = data bit 29.
    for k, data_bit in enumerate([29, 30, 31, 0, 1]):
        table = [v ^ (1 << k) for v in range(32)]
        assert _substitute_wrap_window(table, 0) == 1 << data_bit
    flip_all = [v ^ 31 for v in range(32)]
    assert _substitute_wrap_window(flip_all, 0) == 0x3 | (0x7 << 29)
    identity = list(range(32))
    for x in BLOCKS:
        assert _substitute_wrap_window(identity, x) == x


def test_full_layer_roundtrip_and_inverse_consistency():
    sbox = DACSBox()
    for x in BLOCKS:
        y = sbox.substitute_block32_full(x)
        assert 0 <= y < (1 << 32)
        assert sbox.inverse_substitute_block32_full(y) == x
        # The full layer is the standard layer followed by the seventh S-box.
        assert y == _substitute_wrap_window(DAC_SBOX, sbox.substitute_block32(x))
    assert any(sbox.substitute_block32_full(x) != sbox.substitute_block32(x) for x in BLOCKS)


def test_full_layer_substitutes_the_low_bits():
    """Flipping input bit 0 or 1 changes exactly one output bit in the paper
    layout but several (through the seventh S-box) in the full layer."""
    sbox = DACSBox()
    for bit in (0, 1):
        standard = [_popcount(sbox.substitute_block32(x) ^ sbox.substitute_block32(x ^ (1 << bit)))
                    for x in BLOCKS]
        full = [_popcount(sbox.substitute_block32_full(x) ^ sbox.substitute_block32_full(x ^ (1 << bit)))
                for x in BLOCKS]
        assert set(standard) == {1}
        assert np.mean(full) > 1.5


def test_cipher_full_sbox_layer_roundtrip():
    key = 0x0123456789ABCDEF0123
    full = DACBlockCipher(key, full_sbox_layer=True)
    standard = DACBlockCipher(key)
    assert full.round_keys == standard.round_keys
    for x in BLOCKS[:300]:
        ct = full.encrypt_block(x)
        assert full.decrypt_block(ct) == x
    assert any(full.encrypt_block(x) != standard.encrypt_block(x) for x in BLOCKS[:20])
    message = b"full S-layer roundtrip"
    assert full.decrypt_bytes(full.encrypt_bytes(message)) == message


# ------------------------------------------------------------
# Dynamic S-box generation and its key space
# ------------------------------------------------------------
def test_canonical_seed_fold():
    assert canonical_seed(0.25) == 0.25
    assert canonical_seed(0.75) == 0.25
    assert canonical_seed(0.5) == 0.5
    assert canonical_seed(0.7) == 1.0 - 0.7          # the double above 0.3, a distinct key
    assert canonical_seed(0.7) != 0.3
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            canonical_seed(bad)


def test_dynamic_sbox_mirror_seeds_are_one_key():
    # Seeds whose mirror is exact in binary floating point reproduce each other.
    for seed in (0.25, 0.125, 0.375, 0.4):
        assert 1.0 - (1.0 - seed) == seed
        assert generate_dynamic_sbox(1.0 - seed).table == generate_dynamic_sbox(seed).table
    # In general a seed above 1/2 is the key 1 - seed; for 0.7 that is the
    # double just above 0.3, a distinct key, not 0.3 itself.
    for seed in (0.7, 0.95, 0.500001):
        folded = canonical_seed(seed)
        assert folded == 1.0 - seed and folded <= 0.5
        assert generate_dynamic_sbox(seed).table == generate_dynamic_sbox(folded).table
    assert generate_dynamic_sbox(0.7).table != generate_dynamic_sbox(0.3).table


def test_dynamic_sbox_distinct_keys_and_validation():
    tables = {seed: generate_dynamic_sbox(seed).table for seed in (0.1, 0.2, 0.3, 0.5, 0.49999)}
    for table in tables.values():
        assert sorted(table) == list(range(32))
    assert len({tuple(t) for t in tables.values()}) == len(tables)
    assert generate_dynamic_sbox(0.5).table == tables[0.5]
    assert generate_dynamic_sbox(0.50001).table == generate_dynamic_sbox(canonical_seed(0.50001)).table
    assert generate_dynamic_sbox(0.50001).table not in tables.values()
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            generate_dynamic_sbox(bad)


# ------------------------------------------------------------
# Whitening of the dynamic S-box generator
# ------------------------------------------------------------
def test_dynamic_sbox_ranks_the_chaotic_rng_stream():
    """The table is the ranking permutation of the first 32 ChaoticRNG
    variates (same warm-up, same frac(2^16 * s) whitening), so the dynamic
    S-box and the optimiser share one key derivation."""
    for seed in (0.3, 0.5, 0.7, 0.500001):
        uniforms = ChaoticRNG(seed).random(32)
        ranks = np.empty(32, dtype=np.int64)
        ranks[np.argsort(uniforms, kind="stable")] = np.arange(32)
        assert generate_dynamic_sbox(seed).table == ranks.tolist()


def test_dynamic_sbox_seed_near_half_is_not_near_monotone():
    """Ranking the raw consecutive orbit points made the demo seed 0.500001
    (whose warm-up ends near the repelling fixed point of f2) produce the
    near-monotone table [16, 15, 17, 14, 18, 13, 19, 12, ...] with LAT bias
    12 (LAP 0.375) and differential uniformity 12; the frac(2^16 * s)
    whitening removes that structure."""
    table = generate_dynamic_sbox(0.500001).table
    assert table[:8] != [16, 15, 17, 14, 18, 13, 19, 12]
    metrics = fast_metrics(np.array(table))
    assert metrics["max_bias"] / 32.0 <= 0.3125      # observed: bias 8, LAP 0.25
    assert metrics["differential_uniformity"] <= 10  # observed: 6
    # |S(x) - 15.5| grew almost linearly in x under the geometric divergence.
    deviation = np.abs(np.array(table, dtype=float) - 15.5)
    assert np.corrcoef(np.arange(32), deviation)[0, 1] < 0.5


def test_dynamic_sbox_consecutive_entries_are_decorrelated():
    """With raw orbit points, S(x + 1) is nearly a function of S(x) (the map
    is unimodal): the lag-1 correlation of the table was between -0.46 and
    -0.75 for every seed tried.  The whitened tables behave like random
    permutations."""
    lag1 = []
    for seed in (0.05, 0.1, 0.2, 0.25, 0.3, 0.45, 0.49999, 0.5, 0.500001, 0.7):
        table = np.array(generate_dynamic_sbox(seed).table, dtype=float)
        lag1.append(float(np.corrcoef(table[:-1], table[1:])[0, 1]))
    assert all(abs(value) < 0.45 for value in lag1)
    assert abs(sum(lag1) / len(lag1)) < 0.2
