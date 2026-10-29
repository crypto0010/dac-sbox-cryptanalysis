"""Tests for the third-round additions.

* the executable specification of Algorithm 2 (``src/algorithm_spec.py``)
  reproduces both published tables and every recorded search statistic;
* the exact count of ideal-SAC affine representatives and its orbits;
* the unit-labelled trail quantities and ratios of the exact trail search;
* the Liberty function parser, table interpolation and the synthesis flow;
* the self-contained enumeration of ``verify_results.py``.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import os
import random

import numpy as np
import pytest

from src import algorithm_spec
from src.cipher_analysis import trail_comparisons, trail_quantities
from src.comparative_analysis import all_sbox_tables
from src.hardware_synthesis import Table, _interp1, compile_function
from src.search_studies import (_subspace_lattice, ideal_sac_counting, locate_basis)
from src.sbox_optimizer import (GOLD_EXPONENT, INVERSION_EXPONENT, optimize_dac_sbox)
from src.sbox_tables import TABLES
from src.security_analysis import analyze_sbox

HAVE_YOSYS = importlib.util.find_spec("yowasp_yosys") is not None


# ------------------------------------------------------------------
# Algorithm 2 as printed
# ------------------------------------------------------------------
@pytest.mark.parametrize("name,exponent,budget,strategy", [
    ("DAC-OPT", INVERSION_EXPONENT, 4096, "affine_inverse"),
    ("DAC-OPT-GOLD", GOLD_EXPONENT, 512, "affine_gold"),
])
def test_specification_reproduces_the_search(name, exponent, budget, strategy):
    spec = algorithm_spec.affine_select(exponent, 0.5, budget)
    opt = optimize_dac_sbox(strategy, 0.5, budget)
    assert spec["sbox"] == list(TABLES[name].table) == list(opt["sbox"])
    prov = opt["provenance"]
    assert spec["A_rows"] == list(prov["affine"]["A_rows"])
    assert spec["B_rows"] == list(prov["affine"]["B_rows"])
    assert (spec["a"], spec["b"]) == (prov["affine"]["a"], prov["affine"]["b"])
    assert spec["best_candidate_index"] == prov["search"]["best_candidate_index"]
    assert spec["ideal_sac_candidates"] == prov["search"]["perfect_sac_candidates"]
    assert spec["refine_accepted"] == prov["search"]["refine_accepted"]
    assert spec["draws"] == prov["rng"]["draws"]


def test_specification_uses_a_different_seed_differently():
    a = algorithm_spec.affine_select(INVERSION_EXPONENT, 0.25, 256)
    b = optimize_dac_sbox("affine_inverse", 0.25, 256)
    assert a["sbox"] == list(b["sbox"])
    assert a["sbox"] != list(TABLES["DAC-OPT"].table)


def test_worked_example_matches_the_table():
    ex = algorithm_spec.worked_example(30, [4, 11, 13, 14, 18], 4, [21, 9, 29, 22, 14], 29, 6)
    assert ex == {"x": 6, "Bx": 20, "z": 9, "F(z)": 4, "A F(z)": 22, "S(x)": 18}
    assert TABLES["DAC-OPT"].table[6] == 18
    # 9 = X^3 + 1 and 4 = X^2 are inverses modulo X^5 + X^2 + 1
    assert algorithm_spec.gf_mul(9, 4) == 1


def test_generator_folds_mirror_seeds():
    g1, g2 = algorithm_spec.Gen(0.25), algorithm_spec.Gen(0.75)
    assert [g1.uniform() for _ in range(5)] == [g2.uniform() for _ in range(5)]


# ------------------------------------------------------------------
# Exact count of ideal-SAC representatives
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def inverse_count():
    return ideal_sac_counting(INVERSION_EXPONENT)


def test_inverse_class_count(inverse_count):
    c = inverse_count
    assert c["input_bases"] == 83328
    assert c["admissible_bases"] == 186
    assert c["zero_set_sizes"] == {5: 155, 6: 31}
    assert c["output_matrices_per_admissible_basis"] == {120: 155, 720: 31}
    assert c["ordered_pairs_with_ideal_sac"] == 120 * (155 * 120 + 31 * 720) == 4910400
    assert math.isclose(c["probability_ideal_sac"], 4910400 / 9999360 ** 2)
    assert [(o["size"], o["zero_set_size"]) for o in c["orbits"]] == [(155, 5), (31, 6)]
    assert c["orbits"][1]["normal_bases"] == [[3, 5, 12, 17, 26]]
    assert c["orbits"][0]["normal_bases"] == []


def test_dac_opt_basis_is_admissible_with_forced_output_rows(inverse_count):
    where = locate_basis(inverse_count, [21, 9, 29, 22, 14])
    # |Z| = 5: the output rows of DAC-OPT are forced, up to their order
    assert where["zero_set"] == [4, 11, 13, 14, 18]
    assert where["orbit_index"] == 0


def test_gold_class_count():
    c = ideal_sac_counting(GOLD_EXPONENT)
    assert c["admissible_bases"] == c["input_bases"] == 83328
    assert c["zero_set_sizes"] == {26: 83328}
    assert 0.38 < c["probability_ideal_sac"] < 0.39


def test_moebius_count_matches_brute_force():
    member, mu = _subspace_lattice()
    assert member.shape == (374, 32)
    rng = random.Random(7)

    def rank(vs):
        basis = []
        for v in vs:
            for b in basis:
                v = min(v, v ^ b)
            if v:
                basis.append(v)
        return len(basis)

    for _ in range(25):
        size = rng.randint(5, 8)
        zero = sorted(rng.sample(range(1, 32), size))
        mask = np.zeros(32, dtype=np.int64)
        mask[zero] = 1
        moebius = int(((mask @ member.T) ** 5) @ mu)
        brute = sum(1 for rows in itertools.permutations(zero, 5) if rank(rows) == 5)
        assert moebius == brute


def test_counted_probability_matches_sampling():
    """A direct Monte Carlo of the Gold class agrees with the exact share."""
    from src.search_studies import _random_representative
    rng = np.random.Generator(np.random.PCG64(11))
    hits = sum(1 for _ in range(300)
               if analyze_sbox(_random_representative(GOLD_EXPONENT, rng))["sac_d_max"] == 0)
    exact = ideal_sac_counting(GOLD_EXPONENT)["probability_ideal_sac"]
    assert abs(hits / 300 - exact) < 4 * math.sqrt(exact * (1 - exact) / 300)


# ------------------------------------------------------------------
# Trail quantities
# ------------------------------------------------------------------
def test_trail_ratio_is_exact():
    exact = {n: {m: {"weight_at_cipher_rounds": w} for m, w in models.items()}
             for n, models in {"DAC": {"differential": 45.415037, "linear": 20.66015},
                               "E-DAC": {"differential": 45.0, "linear": 24.075187},
                               "DAC-OPT": {"differential": 36.0, "linear": 21.905262}}.items()}
    comp = trail_comparisons(exact)
    ratio = comp["differential:DAC-OPT/DAC"]
    assert ratio["log2_ratio"] == pytest.approx(11 - math.log2(3), abs=1e-6)
    assert ratio["ratio"] == pytest.approx(2048 / 3, abs=0.01)
    assert comp["differential:DAC-OPT/E-DAC"]["ratio"] == 512.0
    lin = comp["linear:DAC/DAC-OPT"]
    assert lin["data_complexity_ratio"] == pytest.approx(2 ** (2 * lin["log2_ratio"]), rel=1e-3)


def test_trail_quantities_units():
    q = trail_quantities(20.66015, "linear")
    assert q["log2_data_complexity"] == pytest.approx(41.3203)
    assert q["log2_squared_correlation"] == pytest.approx(-41.3203)
    assert trail_quantities(36.0, "differential")["log2_probability"] == -36.0


# ------------------------------------------------------------------
# Liberty helpers and synthesis
# ------------------------------------------------------------------
@pytest.mark.parametrize("expr,truth", [
    ("!(A1 & A2)", lambda a, b, c: 1 - (a & b)),
    ("(A&B) | (!C)", lambda a, b, c: (a & b) | (1 - c)),
    ("!A & B", lambda a, b, c: (1 - a) & b),
    ("A ^ B", lambda a, b, c: a ^ b),
    ("A B + C", lambda a, b, c: (a & b) | c),
    ("A' * B", lambda a, b, c: (1 - a) & b),
    ("!((A & B) | C)", lambda a, b, c: 1 - ((a & b) | c)),
])
def test_liberty_functions(expr, truth):
    names = ("A", "B", "C") if "A1" not in expr else ("A1", "A2", "C")
    func = compile_function(expr)
    for a, b, c in itertools.product((0, 1), repeat=3):
        assert func(dict(zip(names, (a, b, c)))) == truth(a, b, c)


def test_table_interpolation_and_extrapolation():
    assert _interp1([0, 1, 2], [0, 10, 30], 1.5) == 20
    assert _interp1([0, 1, 2], [0, 10, 30], 3.0) == 50
    assert _interp1([0, 1, 2], [0, 10, 30], -1.0) == -10
    t = Table(("input_net_transition", "total_output_net_capacitance"),
              ([0.0, 1.0], [0.0, 2.0]), [[0.0, 2.0], [1.0, 3.0]])
    assert t.lookup(0.5, 1.0) == pytest.approx(1.5)


@pytest.mark.skipif(not HAVE_YOSYS, reason="yowasp-yosys not installed")
def test_synthesis_is_verified_and_deterministic():
    from src.hardware_synthesis import default_cache_dir, LIBRARIES, synthesize
    lib = LIBRARIES["nangate45"]
    if not os.path.exists(os.path.join(default_cache_dir(), lib.filename)):
        pytest.skip("Liberty file not cached (run run_hardware.py once)")
    table = all_sbox_tables()["Keccak-chi"]
    first = synthesize(table, "nangate45", "test_chi")
    second = synthesize(table, "nangate45", "test_chi")
    assert first == second
    assert first["functionally_verified"] and first["area_ge"] > 0
    assert first["logic_depth"] >= 1 and first["delay_ps"] > 0 and first["energy_fj_per_evaluation"] > 0


# ------------------------------------------------------------------
# verify_results.py tier A enumeration
# ------------------------------------------------------------------
def test_independent_enumeration_agrees_with_the_engine():
    import verify_results
    for name, table in all_sbox_tables().items():
        mine = verify_results.enum_metrics(table)
        ref = analyze_sbox(list(table), name=name)
        for key in verify_results.METRIC_KEYS:
            assert mine[key] == pytest.approx(ref[key]), (name, key)
