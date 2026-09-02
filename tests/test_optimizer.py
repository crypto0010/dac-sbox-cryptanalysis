"""Tests for src.sbox_optimizer and src.sbox_tables.

Checks the chaotic generator (determinism, pinned first draws, seed fold,
argument validation), GF(2^5) arithmetic, the affine construction
``S(x) = A * F(B * x ^ b) ^ a`` against a bit-level reference, the
autocorrelation-table shortcut for the SAC matrix, annealing determinism,
the optimiser's result contract and the bit-for-bit reproduction of the
frozen optimised tables.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.chaotic_maps import canonical_seed
from src.sbox import DAC_SBOX
from src.sbox_optimizer import (
    GOLD_EXPONENT, INVERSION_EXPONENT, STRATEGIES, AffineParams, ChaoticRNG,
    _sample_invertible, _transpose, affine_search, autocorrelation_table,
    enumerate_input_bases, gf_mul, objective_key, optimize_dac_sbox, power_map,
    sa_energy, simulated_annealing,
)
from src.sbox_tables import (
    DAC_OPT_GOLD_SBOX, DAC_OPT_SBOX, TABLES, inverse_table, regenerable_tables, regenerate,
)
from src.security_analysis import analyze_sbox, compute_sac, fast_metrics


# ------------------------------------------------------------
# Chaotic generator
# ------------------------------------------------------------
def test_chaotic_rng_is_deterministic_and_pinned():
    a = ChaoticRNG(0.3).random(64)
    b = ChaoticRNG(0.3).random(64)
    assert np.array_equal(a, b)
    assert a[:4].tolist() == [0.7962840904001496, 0.3666139920533169,
                              0.9316390313470038, 0.514987742764788]
    assert np.all((a >= 0.0) & (a < 1.0))
    # The block size only changes buffering, never the stream.
    assert np.array_equal(ChaoticRNG(0.3, block=1).random(64), a)
    assert np.array_equal(ChaoticRNG(0.3, block=7).random(64), a)
    rng = ChaoticRNG(0.3)
    assert rng.random() == a[0] and rng.draws == 1
    ints = ChaoticRNG(0.3).integers(32, size=200)
    assert ints.min() >= 0 and ints.max() < 32 and ints.shape == (200,)
    i, j = ChaoticRNG(0.3).pair(32)
    assert i != j and 0 <= i < 32 and 0 <= j < 32


def test_chaotic_rng_seed_fold_and_validation():
    assert np.array_equal(ChaoticRNG(0.25).random(64), ChaoticRNG(0.75).random(64))
    assert ChaoticRNG(0.75).canonical_seed == canonical_seed(0.75) == 0.25
    assert ChaoticRNG(0.7).canonical_seed == 1.0 - 0.7
    assert not np.array_equal(ChaoticRNG(0.3).random(64), ChaoticRNG(0.7).random(64))
    assert ChaoticRNG(0.75).describe()["canonical_seed"] == 0.25
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            ChaoticRNG(bad)
    for bad_block in (0, -3):
        with pytest.raises(ValueError):
            ChaoticRNG(0.5, block=bad_block)


# ------------------------------------------------------------
# GF(2^5) and the affine construction
# ------------------------------------------------------------
def test_gf_power_maps():
    inv = power_map(INVERSION_EXPONENT)
    cube = power_map(GOLD_EXPONENT)
    assert inv[0] == 0 and cube[0] == 0
    for x in range(1, 32):
        assert gf_mul(x, int(inv[x])) == 1
    for x in range(32):
        assert cube[x] == gf_mul(x, gf_mul(x, x))
        assert gf_mul(x, 1) == x
    assert gf_mul(5, 9) == gf_mul(9, 5)
    assert sorted(inv.tolist()) == list(range(32)) and sorted(cube.tolist()) == list(range(32))
    assert not inv.flags.writeable


def _linear(rows, x):
    """``y_j = parity(row_j & x)`` with row 0 the MSB of the output."""
    return sum((bin(rows[j] & x).count("1") & 1) << (4 - j) for j in range(5))


def test_affine_params_matches_bitwise_formula():
    params = AffineParams(INVERSION_EXPONENT, (4, 11, 13, 14, 18), 4, (21, 9, 29, 22, 14), 29)
    inv = power_map(INVERSION_EXPONENT)
    expected = [_linear(params.a_rows, int(inv[_linear(params.b_rows, x) ^ params.b])) ^ params.a
                for x in range(32)]
    assert params.sbox().tolist() == expected == DAC_OPT_SBOX
    gold = AffineParams(GOLD_EXPONENT, (1, 2, 4, 8, 16), 1, (21, 5, 19, 11, 12), 1)
    assert gold.sbox().tolist() == DAC_OPT_GOLD_SBOX
    assert params.as_dict()["A_rows"] == [4, 11, 13, 14, 18]
    assert params.as_dict()["base_function"].startswith("F(x) = x^-1")


def test_autocorrelation_table_predicts_sac():
    ac = autocorrelation_table(INVERSION_EXPONENT)
    assert ac.shape == (32, 32) and ac[0, 0] == 32 and not ac.flags.writeable
    rng = ChaoticRNG(0.11)
    for _ in range(10):
        b_cols = _sample_invertible(rng, 1)[0]
        a_cols = _sample_invertible(rng, 1)[0]
        a_rows, b_rows = _transpose(a_cols), _transpose(b_cols)
        params = AffineParams(INVERSION_EXPONENT, tuple(int(v) for v in a_rows),
                              int(rng.integers(32)), tuple(int(v) for v in b_rows),
                              int(rng.integers(32)))
        table = params.sbox().tolist()
        assert sorted(table) == list(range(32))
        # P[i, j] = 1/2 - AC[row_j(A), col_i(B)] / 64
        predicted = 0.5 - ac[a_rows[:, None], b_cols[None, :]].T / 64.0
        assert np.allclose(compute_sac(table)["matrix"], predicted)
        fast = fast_metrics(np.array(table))
        assert (fast["differential_uniformity"], fast["nl_sbox"], fast["min_degree"]) == (2, 10, 4)


# ------------------------------------------------------------
# Annealing, objective and driver
# ------------------------------------------------------------
def test_simulated_annealing_is_deterministic():
    best1, stats1 = simulated_annealing(DAC_SBOX, 300, ChaoticRNG(0.5))
    best2, stats2 = simulated_annealing(DAC_SBOX, 300, ChaoticRNG(0.5))
    assert np.array_equal(best1, best2) and stats1 == stats2
    assert sorted(best1.tolist()) == list(range(32))
    assert stats1["initial_energy"] == sa_energy(DAC_SBOX) == 14378.0
    assert stats1["best_energy"] <= stats1["initial_energy"]
    # APN: DDT term 3968 for both optimised tables.  Walsh term 3968 for the
    # almost bent Gold table, 3782 for the inversion table (see sa_energy).
    assert sa_energy(DAC_OPT_GOLD_SBOX) == 3968.0 + 3968.0
    assert sa_energy(DAC_OPT_SBOX) == 3968.0 + 3782.0
    assert sa_energy(DAC_OPT_SBOX) < sa_energy(DAC_OPT_GOLD_SBOX) < sa_energy(DAC_SBOX)
    with pytest.raises(ValueError):
        simulated_annealing(DAC_SBOX, 0, ChaoticRNG(0.5))


def test_objective_key_ranks_the_classes():
    dac_opt = objective_key(analyze_sbox(DAC_OPT_SBOX))
    gold = objective_key(analyze_sbox(DAC_OPT_GOLD_SBOX))
    dac = objective_key(analyze_sbox(DAC_SBOX))
    assert dac_opt[:3] == (2, -4, -10) and gold[:3] == (2, -2, -12) and dac[:3] == (8, -4, -8)
    assert dac_opt < gold < dac


def test_optimize_dac_sbox_contract():
    r1 = optimize_dac_sbox("affine_gold", seed=0.3, iterations=16)
    r2 = optimize_dac_sbox("affine_gold", seed=0.3, iterations=16)
    assert set(r1) == {"sbox", "metrics", "objective", "provenance"}
    assert r1["sbox"] == r2["sbox"] and r1["objective"] == r2["objective"]
    assert sorted(r1["sbox"]) == list(range(32))
    assert (r1["metrics"]["nl_sbox"], r1["metrics"]["differential_uniformity"]) == (12, 2)
    assert r1["provenance"]["rng"]["canonical_seed"] == 0.3
    assert r1["provenance"]["strategy"] == "affine_gold" and r1["provenance"]["iterations"] == 16
    every = optimize_dac_sbox("all", seed=0.5, iterations=8)
    assert set(every["candidates"]) == set(STRATEGIES)
    assert every["provenance"]["strategy"] == "affine_inverse"   # degree 4 outranks NL 12
    assert every["objective"] == min(c["objective"] for c in every["candidates"].values())
    with pytest.raises(ValueError):
        optimize_dac_sbox("bogus")


# ------------------------------------------------------------
# Frozen tables
# ------------------------------------------------------------
@pytest.mark.parametrize("name", regenerable_tables())
def test_frozen_tables_regenerate_bit_for_bit(name):
    record = TABLES[name]
    result = regenerate(name)          # raises RuntimeError on any mismatch
    assert result["sbox"] == record.table
    assert result["provenance"]["strategy"] == record.strategy
    assert result["provenance"]["seed"] == record.seed
    m = result["metrics"]
    assert m["is_bijective"] and m["all_components_balanced"]
    assert m["differential_uniformity"] == 2 and m["boomerang_uniformity"] == 2
    assert m["sac_max_deviation"] == 0.0 and m["sac_avg"] == 0.5
    nl, degree, bias, bic = {"DAC-OPT": (10, 4, 6, 0.5075),
                             "DAC-OPT-GOLD": (12, 2, 4, 0.5)}[name]
    assert (m["nl_sbox"], m["min_degree"], m["max_degree"], m["max_bias"]) == (nl, degree, degree, bias)
    assert m["bic_sac_avg"] == pytest.approx(bic)
    assert record.inverse == inverse_table(record.table)
    assert all(record.table[record.inverse[y]] == y for y in range(32))


def test_static_tables_and_errors():
    assert regenerable_tables() == ["DAC-OPT", "DAC-OPT-GOLD"]
    assert TABLES["DAC"].table == DAC_SBOX
    with pytest.raises(ValueError):
        regenerate("DAC")
    with pytest.raises(KeyError):
        regenerate("no-such-table")
    with pytest.raises(ValueError):
        inverse_table([0, 0, 1])


# ------------------------------------------------------------
# Affine search internals and result consistency
# ------------------------------------------------------------
def test_optimised_tables_have_no_fixed_or_complement_fixed_points():
    # _choose_constants uses (a, b), like the AES affine constant, to remove
    # S(x) = x and S(x) = ~x; the original DAC table has S(5) = ~5.
    for table in (DAC_OPT_SBOX, DAC_OPT_GOLD_SBOX):
        assert all(table[x] != x and table[x] != x ^ 31 for x in range(32))
    assert [x for x in range(32) if DAC_SBOX[x] == x ^ 31] == [5]


def test_affine_search_contract():
    params_a, stats_a = affine_search(INVERSION_EXPONENT, ChaoticRNG(0.3), 32)
    params_b, stats_b = affine_search(INVERSION_EXPONENT, ChaoticRNG(0.3), 32)
    assert params_a == params_b and stats_a == stats_b   # chaos-driven yet deterministic
    assert stats_a["candidates"] == 32 and 0 <= stats_a["best_candidate_index"] < 32
    assert stats_a["refine_accepted"] <= stats_a["refine_steps"] == 256
    fast = fast_metrics(params_a.sbox())
    assert (fast["differential_uniformity"], fast["nl_sbox"], fast["min_degree"]) == (2, 10, 4)
    with pytest.raises(ValueError):
        affine_search(INVERSION_EXPONENT, ChaoticRNG(0.3), 0)


def test_result_metrics_are_recomputable_from_the_table():
    result = optimize_dac_sbox("simulated_annealing", seed=0.3, iterations=200)
    table = result["sbox"]
    assert result["metrics"] == analyze_sbox(table, name="DAC-simulated_annealing")
    assert result["objective"] == objective_key(result["metrics"])
    search = result["provenance"]["search"]
    assert search["best_energy"] == sa_energy(table)
    assert search["initial_energy"] == sa_energy(DAC_SBOX)
    assert search["best_energy"] <= search["initial_energy"]
    assert result["provenance"]["rng"]["draws"] > 0


def test_enumerate_input_bases_certifies_the_inversion_class():
    # About one second: every one of the 83,328 unordered input bases of
    # GF(2)^5 scored for F(x) = x^-1 with the greedy output rows.
    result = enumerate_input_bases(INVERSION_EXPONENT)
    assert result["bases"] == 83328
    assert result["best_key"][0] == 0                   # perfect SAC is attainable
    assert result["perfect_sac_bases"] == 186
    assert result["best_count"] == 155
    assert result["zero_set_sizes"] == {5: 155, 6: 31}
    assert sum(result["max_sac_key_histogram"].values()) == result["bases"]
    # The frozen DAC-OPT table attains that optimum: every SAC entry is 1/2.
    assert compute_sac(DAC_OPT_SBOX)["max_deviation"] == 0.0
