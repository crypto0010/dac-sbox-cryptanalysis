"""Tests for the vectorised metrics engine (src.security_analysis, src.boolean_function).

Every metric is checked against a tiny brute-force pure-Python implementation
written directly from the textbook definition, on the DAC S-box, the E-DAC
S-box, a seeded random 5-bit permutation and the 4-bit PRESENT S-box (to
exercise the generic-n code path). Known reference values are asserted too.

The statistics added for the revision are covered the same way:

* the **BCT spectrum** against a hand-counted frequency table of the
  ``(2^n - 1)^2`` non-trivial entries;
* the **SAC deviations** ``D_max`` / ``D_mean`` against the brute-force SAC
  matrix, including the reviewer's observation that E-DAC raises ``sac_min``
  from 0.25 to 0.375 while ``D_max`` stays 0.25;
* the **algebraic immunity** of every component against an exhaustive search
  over all 2^16 Boolean polynomials of degree <= 2 (the definition applied
  literally, with no linear algebra), and the **implicit equation counts**
  against the known 39 quadratic equations of the AES S-box
  (Courtois-Pieprzyk) and against a brute-force count of the degree-1
  relations.
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.boolean_function import (
    algebraic_degree, analyze_component_functions, bit_tables, component_degrees,
    component_weights, compute_anf,
)
from src.security_analysis import (
    algebraic_immunity_of_function, analyze_sbox, check_bijective,
    compute_algebraic_degree, compute_algebraic_immunity, compute_bct,
    compute_bic, compute_ddt, compute_hamming_distances, compute_lat,
    compute_nonlinearity, compute_sac, compute_walsh_spectrum, fast_metrics,
    fast_metrics_batch, full_security_analysis, gf2_kernel_dimension, gf2_rank,
    graph_equation_counts, guaranteed_equation_degree, nonlinearity_upper_bound,
    pack_gf2_rows, spectrum_of, walsh_hadamard_transform,
)

DAC = [30, 24, 9, 27, 28, 26, 4, 22, 2, 15, 29, 5, 13, 8, 25, 3,
       11, 10, 16, 31, 0, 20, 14, 21, 1, 12, 18, 6, 23, 19, 17, 7]
EDAC = [27, 10, 31, 5, 14, 6, 24, 26, 13, 18, 30, 8, 20, 25, 22, 2,
        19, 1, 16, 21, 15, 28, 29, 4, 7, 11, 17, 12, 3, 0, 9, 23]
RAND = np.random.default_rng(20250827).permutation(32).tolist()
PRESENT = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]

CASES = [("DAC", DAC, 5), ("EDAC", EDAC, 5), ("RAND", RAND, 5), ("PRESENT", PRESENT, 4)]


# ------------------------------------------------------------
# Brute-force reference implementations (pure Python)
# ------------------------------------------------------------
def _parity(v):
    return bin(v).count("1") & 1


def _component(sbox, b):
    return [_parity(b & y) for y in sbox]


def _walsh(tt, n):
    size = 1 << n
    return [sum((-1) ** (tt[x] ^ _parity(x & a)) for x in range(size)) for a in range(size)]


def _nl(tt, n):
    return ((1 << n) - max(abs(w) for w in _walsh(tt, n))) // 2


def _lat(sbox, n):
    size = 1 << n
    return [[sum(1 for x in range(size) if _parity(a & x) == _parity(b & sbox[x])) - size // 2
             for b in range(size)] for a in range(size)]


def _ddt(sbox, n):
    size = 1 << n
    table = [[0] * size for _ in range(size)]
    for dx in range(size):
        for x in range(size):
            table[dx][sbox[x] ^ sbox[x ^ dx]] += 1
    return table


def _bct(sbox, n):
    size = 1 << n
    inv = [0] * size
    for x, y in enumerate(sbox):
        inv[y] = x
    return [[sum(1 for x in range(size)
                 if inv[sbox[x] ^ b] ^ inv[sbox[x ^ a] ^ b] == a)
             for b in range(size)] for a in range(size)]


def _sac(sbox, n):
    size = 1 << n
    return [[sum(((sbox[x] ^ sbox[x ^ (1 << (n - 1 - i))]) >> (n - 1 - j)) & 1
                 for x in range(size)) / size
             for j in range(n)] for i in range(n)]


def _bic(sbox, n):
    size = 1 << n
    sac_vals, nl_vals = [], []
    for j in range(n):
        for k in range(j + 1, n):
            g = [((y >> (n - 1 - j)) ^ (y >> (n - 1 - k))) & 1 for y in sbox]
            probs = [sum(g[x] ^ g[x ^ (1 << (n - 1 - i))] for x in range(size)) / size
                     for i in range(n)]
            sac_vals.append(sum(probs) / n)
            nl_vals.append(_nl(g, n))
    return sac_vals, nl_vals


def _anf(tt, n):
    coeffs = list(tt)
    for i in range(n):
        step = 1 << i
        for u in range(1 << n):
            if u & step:
                coeffs[u] ^= coeffs[u ^ step]
    return coeffs


def _degree(tt, n):
    return max((bin(u).count("1") for u, c in enumerate(_anf(tt, n)) if c), default=0)


# ------------------------------------------------------------
# Brute-force cross-checks
# ------------------------------------------------------------
@pytest.mark.parametrize("name,sbox,n", CASES)
def test_nonlinearity_all_components(name, sbox, n):
    size = 1 << n
    expected = [_nl(_component(sbox, b), n) for b in range(1, size)]
    nl = compute_nonlinearity(sbox, n)
    assert nl["component_nl"] == expected
    assert nl["nl_sbox"] == nl["nl_min"] == min(expected)
    assert nl["nl_avg_components"] == pytest.approx(sum(expected) / len(expected))
    assert nl["nl_max_components"] == max(expected)
    assert nl["per_bit"] == [expected[(1 << (n - 1 - i)) - 1] for i in range(n)]
    assert nl["minimum"] == min(nl["per_bit"]) and nl["maximum"] == max(nl["per_bit"])
    assert nl["average"] == pytest.approx(sum(nl["per_bit"]) / n)


@pytest.mark.parametrize("name,sbox,n", CASES)
def test_walsh_spectrum_and_lat(name, sbox, n):
    size = 1 << n
    spectrum = compute_walsh_spectrum(sbox, n)
    for b in range(size):
        assert spectrum[:, b].tolist() == _walsh(_component(sbox, b), n)
    lat = compute_lat(sbox, n)
    ref = _lat(sbox, n)
    assert lat["table"].tolist() == ref
    max_bias = max(abs(ref[a][b]) for a in range(1, size) for b in range(1, size))
    assert lat["max_bias"] == max_bias
    assert lat["lap"] == pytest.approx(max_bias / size)
    assert lat["lap_half"] == pytest.approx(max_bias / (2 * size))
    assert lat["linear_probability"] == pytest.approx((2 * max_bias / size) ** 2)
    # NL(S) = 2^(n-1) - max|LAT| for balanced components
    assert compute_nonlinearity(sbox, n)["nl_sbox"] == size // 2 - max_bias


@pytest.mark.parametrize("name,sbox,n", CASES)
def test_ddt(name, sbox, n):
    ddt = compute_ddt(sbox, n)
    ref = _ddt(sbox, n)
    assert ddt["table"].tolist() == ref
    du = max(max(row) for row in ref[1:])
    assert ddt["differential_uniformity"] == ddt["max_value"] == du
    assert ddt["dp"] == pytest.approx(du / (1 << n))


@pytest.mark.parametrize("name,sbox,n", CASES)
def test_bct(name, sbox, n):
    bct = compute_bct(sbox, n)
    ref = _bct(sbox, n)
    assert bct["table"].tolist() == ref
    size = 1 << n
    assert bct["boomerang_uniformity"] == max(ref[a][b] for a in range(1, size) for b in range(1, size))
    assert np.all(bct["table"] >= compute_ddt(sbox, n)["table"])


@pytest.mark.parametrize("name,sbox,n", CASES)
def test_sac(name, sbox, n):
    sac = compute_sac(sbox, n)
    ref = _sac(sbox, n)
    assert np.allclose(sac["matrix"], ref)
    flat = [v for row in ref for v in row]
    assert sac["average"] == pytest.approx(sum(flat) / len(flat))
    assert sac["sac_min"] == sac["min_value"] == min(flat)
    assert sac["sac_max"] == sac["max_value"] == max(flat)
    max_dev = max(abs(v - 0.5) for v in flat)
    assert sac["max_deviation"] == pytest.approx(max_dev)
    assert sac["satisfies_sac"] == (max_dev <= sac["tolerance"])
    assert sac["satisfies_sac_average"] == (sac["average"] >= 0.5)
    assert isinstance(sac["sac_verdict"], str)


@pytest.mark.parametrize("name,sbox,n", CASES)
def test_bic(name, sbox, n):
    bic = compute_bic(sbox, n)
    sac_ref, nl_ref = _bic(sbox, n)
    assert np.allclose(bic["bic_sac_values"], sac_ref)
    assert bic["bic_nl_values"] == nl_ref
    assert bic["avg_bic_sac"] == pytest.approx(sum(sac_ref) / len(sac_ref))
    assert bic["min_bic_sac"] == pytest.approx(min(sac_ref))
    assert bic["max_bic_sac"] == pytest.approx(max(sac_ref))
    assert bic["avg_bic_nl"] == pytest.approx(sum(nl_ref) / len(nl_ref))
    assert bic["min_bic_nl"] == min(nl_ref) and bic["max_bic_nl"] == max(nl_ref)
    assert bic["pairs"] == [(j, k) for j in range(n) for k in range(j + 1, n)]
    for (j, k), s, v in zip(bic["pairs"], sac_ref, nl_ref):
        assert bic["bic_sac_matrix"][j, k] == pytest.approx(s)
        assert bic["bic_sac_matrix"][k, j] == pytest.approx(s)
        assert bic["bic_nl_matrix"][j, k] == v


@pytest.mark.parametrize("name,sbox,n", CASES)
def test_algebraic_degree(name, sbox, n):
    size = 1 << n
    deg = compute_algebraic_degree(sbox, n)
    expected = [_degree(_component(sbox, b), n) for b in range(1, size)]
    assert deg["component_degrees"] == expected
    assert deg["min_degree"] == min(expected) and deg["max_degree"] == max(expected)
    assert deg["per_bit"] == [expected[(1 << (n - 1 - i)) - 1] for i in range(n)]
    assert deg["max_possible_degree"] == n - 1
    weights = [sum(_component(sbox, b)) for b in range(1, size)]
    assert deg["component_weights"] == weights
    assert deg["all_components_balanced"] == all(w == size // 2 for w in weights)
    assert component_degrees(sbox, n)[1:].tolist() == expected
    assert component_weights(sbox, n)[1:].tolist() == weights
    # per-coordinate ANF analysis agrees with the brute-force Mobius transform
    for comp in analyze_component_functions(sbox, n):
        tt = comp["truth_table"]
        assert comp["algebraic_degree"] == _degree(tt, n)
        assert comp["anf"] == {
            tuple(i for i in range(n) if (u >> i) & 1): 1
            for u, c in enumerate(_anf(tt, n)) if c
        }


@pytest.mark.parametrize("name,sbox,n", CASES)
def test_hamming_and_bijective(name, sbox, n):
    size = 1 << n
    hd = compute_hamming_distances(sbox, n)
    ref = [bin(x ^ sbox[x]).count("1") for x in range(size)]
    assert hd["distances"] == ref
    assert hd["average"] == pytest.approx(sum(ref) / size)
    assert (hd["minimum"], hd["maximum"]) == (min(ref), max(ref))
    bij = check_bijective(sbox, n)
    assert bij["is_bijective"] and bij["unique_outputs"] == size
    assert bij["all_components_balanced"] and bij["balanced_components"] == size - 1
    assert bij["total_hamming_weight"] == sum(bin(y).count("1") for y in sbox)


@pytest.mark.parametrize("name,sbox,n", CASES)
def test_fast_metrics_matches_full_analysis(name, sbox, n):
    flat = analyze_sbox(sbox, name, n)
    fast = fast_metrics(np.array(sbox), n)
    assert set(fast) == {"nl_sbox", "max_bias", "differential_uniformity",
                         "sac_avg", "sac_min", "bic_sac_avg", "min_degree"}
    for key, value in fast.items():
        assert value == pytest.approx(flat[key]), key


def test_fast_metrics_batch_matches_single():
    batch = np.array([DAC, EDAC, RAND])
    result = fast_metrics_batch(batch)
    for i, sbox in enumerate(batch):
        single = fast_metrics(sbox)
        for key, value in single.items():
            assert result[key][i] == pytest.approx(value), (i, key)
    with pytest.raises(ValueError):
        fast_metrics_batch(np.array(DAC))


# ------------------------------------------------------------
# Known reference values
# ------------------------------------------------------------
def test_dac_known_values():
    flat = analyze_sbox(DAC)
    assert flat["is_bijective"] and flat["all_components_balanced"]
    assert flat["nl_per_bit"] == [8, 10, 10, 8, 10]
    assert flat["nl_coord_min"] == 8 and flat["nl_coord_avg"] == pytest.approx(9.2)
    assert flat["nl_sbox"] == flat["nl_min"] == 8
    assert flat["nl_avg_components"] == pytest.approx(294 / 31)   # 9.4839
    assert flat["component_nl"].count(8) == 8 and flat["nl_max_components"] == 10
    assert flat["nl_upper_bound"] == 12
    assert flat["max_bias"] == 8
    assert flat["lap"] == pytest.approx(0.25)
    assert flat["lap_half"] == pytest.approx(0.125)
    assert flat["differential_uniformity"] == 8 and flat["dp"] == pytest.approx(0.25)
    assert flat["sac_avg"] == pytest.approx(0.515)
    assert flat["sac_min"] == pytest.approx(0.25) and flat["sac_max"] == pytest.approx(0.75)
    assert flat["sac_max_deviation"] == pytest.approx(0.25)
    assert flat["satisfies_sac"] is False and flat["satisfies_sac_average"] is True
    assert flat["bic_sac_avg"] == pytest.approx(0.5175)
    assert flat["bic_sac_min"] == pytest.approx(0.475)
    assert flat["bic_nl_avg"] == pytest.approx(9.6) and flat["bic_nl_min"] == 8
    assert flat["boomerang_uniformity"] == 14
    assert flat["degree_per_bit"] == [4] * 5
    assert flat["min_degree"] == flat["max_degree"] == 4
    assert flat["hd_avg"] == pytest.approx(2.5625)
    assert (flat["hd_min"], flat["hd_max"]) == (1, 5)


def test_edac_known_values():
    flat = analyze_sbox(EDAC, "E-DAC")
    assert flat["nl_per_bit"] == [10, 10, 10, 10, 10]
    assert flat["nl_coord_min"] == 10
    assert flat["nl_sbox"] == 8, "coordinate NL hides six components with NL 8"
    assert flat["component_nl"].count(8) == 6
    assert flat["nl_avg_components"] == pytest.approx(302 / 31)   # 9.7419
    assert flat["nl_max_components"] == 12
    assert flat["max_bias"] == 8 and flat["lap"] == pytest.approx(0.25)
    assert flat["differential_uniformity"] == 4 and flat["dp"] == pytest.approx(0.125)
    assert flat["sac_avg"] == pytest.approx(0.53)
    assert flat["sac_min"] == pytest.approx(0.375) and flat["sac_max"] == pytest.approx(0.75)
    assert flat["bic_sac_avg"] == pytest.approx(0.5125)
    assert flat["bic_nl_avg"] == pytest.approx(9.8) and flat["bic_nl_min"] == 8
    assert flat["boomerang_uniformity"] == 16
    assert flat["min_degree"] == flat["max_degree"] == 4
    assert flat["hd_avg"] == pytest.approx(2.8125)


def test_present_known_values():
    flat = analyze_sbox(PRESENT, "PRESENT", n=4)
    assert flat["nl_sbox"] == 4 and flat["max_bias"] == 4
    assert flat["differential_uniformity"] == 4
    assert flat["max_degree"] == 3 and flat["max_possible_degree"] == 3
    # SCV bound floor(8 - 2^1.5) = 5 (the best known 4-bit value is 4)
    assert flat["nl_upper_bound"] == 5


def test_nonlinearity_upper_bound():
    # Sidelnikov-Chabaud-Vaudenay: floor(2^(n-1) - 2^((n-1)/2)), tight for odd n
    assert [nonlinearity_upper_bound(n) for n in range(1, 9)] == [0, 0, 2, 5, 12, 26, 56, 116]
    for n in (3, 5, 7):
        assert nonlinearity_upper_bound(n) == (1 << (n - 1)) - (1 << ((n - 1) // 2))
    assert analyze_sbox(DAC)["nl_upper_bound"] == 12
    with pytest.raises(ValueError):
        nonlinearity_upper_bound(0)


# ------------------------------------------------------------
# API behaviour
# ------------------------------------------------------------
def test_legacy_keys_preserved():
    assert {"per_bit", "minimum", "maximum", "average"} <= set(compute_nonlinearity(DAC))
    assert {"matrix", "average", "satisfies_sac", "min_value", "max_value"} <= set(compute_sac(DAC))
    assert {"table", "max_bias", "lap"} <= set(compute_lat(DAC))
    assert {"table", "max_value", "dp"} <= set(compute_ddt(DAC))
    assert {"bic_sac_values", "bic_nl_values", "avg_bic_sac", "avg_bic_nl"} <= set(compute_bic(DAC))
    assert {"table", "boomerang_uniformity"} <= set(compute_bct(DAC))
    assert {"distances", "minimum", "maximum", "average"} <= set(compute_hamming_distances(DAC))
    assert {"is_bijective", "total_hamming_weight", "unique_outputs", "expected_unique"} <= set(check_bijective(DAC))
    result = full_security_analysis(DAC)
    assert {"bijective", "sac", "nonlinearity", "hamming_distance", "lat", "ddt", "bic", "bct",
            "degree", "flat"} <= set(result)
    assert result["flat"] == analyze_sbox(DAC)


def test_analyze_sbox_is_json_serialisable():
    import json
    text = json.dumps(analyze_sbox(DAC))
    assert '"nl_sbox": 8' in text


def test_non_bijective_sbox():
    constant = [0] * 32
    bij = check_bijective(constant)
    assert not bij["is_bijective"] and bij["balanced_components"] == 0
    nl = compute_nonlinearity(constant)
    assert nl["nl_sbox"] == 0, "constant components are affine (a = 0 column counts)"
    assert compute_lat(constant)["max_bias"] == 0, "a = 0 row and b = 0 column are excluded"
    assert compute_ddt(constant)["differential_uniformity"] == 32
    with pytest.raises(ValueError):
        compute_bct(constant)
    assert analyze_sbox(constant, "const")["boomerang_uniformity"] is None
    with pytest.raises(ValueError):
        compute_sac([0, 1, 2])
    with pytest.raises(ValueError):
        compute_sac([40] + [0] * 31)


def test_walsh_hadamard_transform_and_anf():
    n = 5
    tables = bit_tables(n)
    for a in (0, 5, 31):
        linear = [bin(x & a).count("1") & 1 for x in range(32)]
        wht = walsh_hadamard_transform(linear, n)
        assert wht[a] == 32 and np.count_nonzero(wht) == 1
    rng = np.random.default_rng(3)
    tt = rng.integers(0, 2, 32).tolist()
    assert walsh_hadamard_transform(tt, n).tolist() == _walsh(tt, n)
    anf = compute_anf(tt, n)
    for x in range(32):
        value = sum(all((x >> i) & 1 for i in mono) for mono in anf) & 1
        assert value == tt[x]
    assert algebraic_degree(anf) == _degree(tt, n)
    assert not tables.sign.flags.writeable


def test_fast_metrics_speed():
    sbox = np.array(DAC)
    fast_metrics(sbox)
    calls = 500
    start = time.perf_counter()
    for _ in range(calls):
        fast_metrics(sbox)
    per_call = (time.perf_counter() - start) / calls
    assert per_call < 1e-3, f"fast_metrics took {per_call * 1e6:.0f} us per call"


# ------------------------------------------------------------
# BCT spectrum (reviewer 2, major comment 5)
# ------------------------------------------------------------
def _bct_spectrum_bruteforce(sbox, n):
    """Frequency distribution of the non-trivial BCT entries, counted by hand."""
    size = 1 << n
    table = _bct(sbox, n)
    counts = {}
    for a in range(1, size):
        for b in range(1, size):
            counts[table[a][b]] = counts.get(table[a][b], 0) + 1
    return sorted(counts.items())


@pytest.mark.parametrize("name,sbox,n", CASES)
def test_bct_spectrum_matches_bruteforce(name, sbox, n):
    size = 1 << n
    bct = compute_bct(sbox, n)
    reference = _bct_spectrum_bruteforce(sbox, n)
    assert [tuple(pair) for pair in bct["spectrum"]] == reference
    # The spectrum accounts for every non-trivial entry, exactly once.
    assert bct["nontrivial_entries"] == (size - 1) ** 2
    assert sum(count for _, count in bct["spectrum"]) == bct["nontrivial_entries"]
    # It is consistent with the maximum it is meant to put in context.
    values = [value for value, _ in bct["spectrum"]]
    assert max(values) == bct["boomerang_uniformity"]
    assert values == sorted(values)
    assert bct["max_count"] == dict(reference)[bct["boomerang_uniformity"]]
    assert bct["mean_nontrivial"] == pytest.approx(
        sum(v * c for v, c in reference) / bct["nontrivial_entries"])
    # Every entry is even: x solves the boomerang condition iff x ^ a does.
    assert all(value % 2 == 0 for value in values)
    # analyze_sbox exposes the same spectrum.
    flat = analyze_sbox(sbox, name, n)
    assert flat["bct_spectrum"] == [list(pair) for pair in bct["spectrum"]]
    assert flat["bct_max_count"] == bct["max_count"]
    assert flat["bct_nontrivial_entries"] == bct["nontrivial_entries"]
    assert flat["bct_mean_nontrivial"] == pytest.approx(bct["mean_nontrivial"])


def test_bct_spectra_of_the_dac_family():
    """The 14 -> 16 rise from DAC to E-DAC is a single isolated entry."""
    dac, edac = compute_bct(DAC), compute_bct(EDAC)
    assert dac["spectrum"] == [[0, 438], [2, 247], [4, 178], [6, 65], [8, 26],
                               [10, 5], [12, 1], [14, 1]]
    assert edac["spectrum"] == [[0, 459], [2, 302], [4, 139], [6, 39], [8, 15],
                                [10, 5], [12, 1], [16, 1]]
    # Exactly one entry attains the maximum in both tables...
    assert dac["max_count"] == edac["max_count"] == 1
    # ...and the rest of the E-DAC distribution is lighter, not heavier:
    # more zeros, fewer large entries, a smaller mean.
    assert edac["spectrum"][0][1] > dac["spectrum"][0][1]
    assert edac["mean_nontrivial"] < dac["mean_nontrivial"]
    assert dac["mean_nontrivial"] == pytest.approx(1880 / 961)
    assert edac["mean_nontrivial"] == pytest.approx(1592 / 961)


def test_spectrum_of_helper():
    assert spectrum_of(np.array([[2, 2, 0], [4, 2, 0]])) == [[0, 2], [2, 3], [4, 1]]
    assert spectrum_of(np.zeros((2, 2), dtype=np.int64)) == [[0, 4]]


# ------------------------------------------------------------
# SAC deviation measures D_max / D_mean (reviewer 2, major comment 2)
# ------------------------------------------------------------
@pytest.mark.parametrize("name,sbox,n", CASES)
def test_sac_deviation_measures(name, sbox, n):
    sac = compute_sac(sbox, n)
    entries = [v for row in _sac(sbox, n) for v in row]
    assert len(entries) == n * n
    d_max = max(abs(v - 0.5) for v in entries)
    d_mean = sum(abs(v - 0.5) for v in entries) / len(entries)
    # Exactly D_max = max |P_ij - 0.5| and D_mean = mean |P_ij - 0.5|.
    assert sac["d_max"] == pytest.approx(d_max)
    assert sac["d_mean"] == pytest.approx(d_mean)
    assert sac["d_max"] == sac["max_deviation"]      # documented aliases
    assert sac["d_mean"] == sac["mean_deviation"]
    # The deviation is two-sided: it does not care which side of 0.5 we are on.
    assert d_max == pytest.approx(max(0.5 - sac["sac_min"], sac["sac_max"] - 0.5))
    assert d_mean <= d_max
    flat = analyze_sbox(sbox, name, n)
    assert flat["sac_d_max"] == flat["sac_max_deviation"] == pytest.approx(d_max)
    assert flat["sac_d_mean"] == flat["sac_mean_deviation"] == pytest.approx(d_mean)


def test_sac_deviation_known_values():
    """E-DAC raises the smallest SAC entry but leaves D_max unchanged."""
    dac, edac = analyze_sbox(DAC), analyze_sbox(EDAC, "E-DAC")
    assert dac["sac_min"] == pytest.approx(0.25) and edac["sac_min"] == pytest.approx(0.375)
    assert dac["sac_d_max"] == pytest.approx(0.25)
    assert edac["sac_d_max"] == pytest.approx(0.25)      # the reviewer's point
    assert dac["sac_d_mean"] == pytest.approx(0.085)
    assert edac["sac_d_mean"] == pytest.approx(0.080)
    # |0.25 - 0.5| = |0.75 - 0.5|: both tables still hold an entry at 0.75.
    assert dac["sac_max"] == edac["sac_max"] == pytest.approx(0.75)


# ------------------------------------------------------------
# GF(2) linear algebra helpers
# ------------------------------------------------------------
def _rank_gf2_reference(matrix):
    """Independent GF(2) rank by textbook row reduction on a numpy array."""
    rows = np.asarray(matrix, dtype=np.int8).copy() & 1
    rank, n_rows, n_cols = 0, rows.shape[0], rows.shape[1]
    for col in range(n_cols):
        pivot = next((r for r in range(rank, n_rows) if rows[r, col]), None)
        if pivot is None:
            continue
        rows[[rank, pivot]] = rows[[pivot, rank]]
        for r in range(n_rows):
            if r != rank and rows[r, col]:
                rows[r] ^= rows[rank]
        rank += 1
    return rank


def test_gf2_rank_and_kernel():
    rng = np.random.default_rng(11)
    for shape in [(5, 3), (8, 8), (16, 20), (32, 16)]:
        matrix = rng.integers(0, 2, shape)
        assert gf2_rank(pack_gf2_rows(matrix)) == _rank_gf2_reference(matrix)
        assert gf2_kernel_dimension(matrix) == shape[1] - _rank_gf2_reference(matrix)
    identity = np.eye(6, dtype=np.int8)
    assert gf2_rank(pack_gf2_rows(identity)) == 6
    assert gf2_kernel_dimension(identity) == 0
    # 1 + 1 = 0 over GF(2): three rows, rank 2.
    dependent = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    assert gf2_rank(pack_gf2_rows(dependent)) == 2
    assert gf2_kernel_dimension(dependent) == 1
    assert gf2_rank([]) == 0
    assert pack_gf2_rows(np.zeros((3, 0), dtype=np.int8)) == [0, 0, 0]
    with pytest.raises(ValueError):
        pack_gf2_rows(np.zeros(4, dtype=np.int8))


# ------------------------------------------------------------
# Algebraic immunity (reviewer 2, additional comment 2)
# ------------------------------------------------------------
def _monomial_masks(n, degree):
    return [u for u in range(1 << n) if bin(u).count("1") <= degree]


def _ai_bruteforce_components(sbox, n=5, max_degree=2):
    """AI of every component by exhaustive search over all low-degree polynomials.

    Enumerates *every* nonzero Boolean polynomial of degree <= ``max_degree``
    (2^16 - 1 of them for n = 5, d = 2) and checks whether it annihilates the
    component or its complement: the definition applied literally, with no
    linear algebra at all.  For n = 5, AI <= ceil(5/2) = 3, so searching
    degrees 1 and 2 determines the value exactly.
    """
    size = 1 << n
    masks = _monomial_masks(n, max_degree)
    evaluations = np.array([[1 if (u & ~x) == 0 else 0 for u in masks]
                            for x in range(size)], dtype=np.int8)
    coefficients = np.array(
        [[(c >> i) & 1 for i in range(len(masks))] for c in range(1 << len(masks))],
        dtype=np.int8)
    values = (evaluations @ coefficients.T) & 1      # values[x, c]
    degrees = np.array([max((bin(u).count("1") for u, bit in zip(masks, row) if bit),
                            default=0) for row in coefficients])
    result = []
    for b in range(1, size):
        truth = np.array(_component(sbox, b))
        best = max_degree + 1
        for target in (1, 0):
            support = np.flatnonzero(truth == target)
            annihilates = ~values[support].any(axis=0)
            annihilates[0] = False                   # the zero polynomial does not count
            if annihilates.any():
                best = min(best, int(degrees[annihilates].min()))
        result.append(best)
    return result


@pytest.mark.parametrize("name,sbox", [("DAC", DAC), ("EDAC", EDAC), ("RAND", RAND)])
def test_algebraic_immunity_matches_exhaustive_search(name, sbox):
    """The linear-algebra AI equals a literal search over all polynomials."""
    computed = compute_algebraic_immunity(sbox, 5)["component_ai"]
    assert computed == _ai_bruteforce_components(sbox, 5, 2)


def test_algebraic_immunity_of_a_single_function():
    n = 5
    # An affine function is annihilated by an affine function: AI = 1.
    affine = [(bin(x & 0b10110).count("1") ^ 1) & 1 for x in range(32)]
    assert algebraic_immunity_of_function(affine, n) == 1
    # A constant is annihilated by the constant 1: AI = 0.
    assert algebraic_immunity_of_function([0] * 32, n) == 0
    assert algebraic_immunity_of_function([1] * 32, n) == 0
    # AI(f) = AI(f + 1) by definition (both sides are searched).
    tables = bit_tables(n)
    for b in (1, 7, 19, 31):
        truth = tables.parity[np.array(DAC)][:, b]
        assert (algebraic_immunity_of_function(truth, n)
                == algebraic_immunity_of_function(1 - truth, n))
    with pytest.raises(ValueError):
        algebraic_immunity_of_function([0, 1, 0], n)


@pytest.mark.parametrize("name,sbox,n", CASES)
def test_algebraic_immunity_structure(name, sbox, n):
    size = 1 << n
    imm = compute_algebraic_immunity(sbox, n)
    assert len(imm["component_ai"]) == size - 1
    assert imm["ai_min"] == min(imm["component_ai"]) == imm["ai_sbox"]
    assert imm["ai_max"] == max(imm["component_ai"])
    assert imm["ai_avg"] == pytest.approx(sum(imm["component_ai"]) / (size - 1))
    assert imm["ai_spectrum"] == [[value, imm["component_ai"].count(value)]
                                  for value in sorted(set(imm["component_ai"]))]
    # AI <= ceil(n / 2) for every Boolean function; attained -> optimal.
    assert imm["ai_upper_bound"] == -(-n // 2)
    assert imm["ai_max"] <= imm["ai_upper_bound"]
    assert imm["ai_is_optimal"] == (imm["ai_min"] == imm["ai_upper_bound"])
    # The coordinate AIs are the components with a single-bit output mask.
    assert imm["coordinate_ai"] == [imm["component_ai"][(1 << (n - 1 - i)) - 1]
                                    for i in range(n)]
    flat = analyze_sbox(sbox, name, n)
    for key in ("component_ai", "coordinate_ai", "ai_min", "ai_max", "ai_sbox",
                "ai_upper_bound", "ai_is_optimal", "graph_ai"):
        assert flat[key] == imm[key], key
    assert flat["ai_avg"] == pytest.approx(imm["ai_avg"])


def test_algebraic_immunity_known_values():
    """AI separates DAC from E-DAC, where the algebraic degree (4) does not."""
    dac, edac = analyze_sbox(DAC), analyze_sbox(EDAC, "E-DAC")
    assert dac["min_degree"] == edac["min_degree"] == 4     # degree cannot tell them apart
    assert dac["ai_min"] == 2 and dac["ai_max"] == 3
    assert dac["ai_spectrum"] == [[2, 17], [3, 14]]
    assert dac["ai_avg"] == pytest.approx(76 / 31)
    assert dac["coordinate_ai"] == [2, 3, 2, 2, 3]
    # E-DAC keeps AI(S) = 2 but moves the distribution up (11 twos, 20 threes).
    assert edac["ai_min"] == 2 and edac["ai_spectrum"] == [[2, 11], [3, 20]]
    assert edac["ai_avg"] == pytest.approx(82 / 31)
    assert edac["coordinate_ai"] == [3, 3, 3, 3, 3]
    assert not dac["ai_is_optimal"] and not edac["ai_is_optimal"]
    assert dac["ai_upper_bound"] == edac["ai_upper_bound"] == 3


# ------------------------------------------------------------
# Implicit (graph) equations and the graph algebraic immunity
# ------------------------------------------------------------
def _graph_degree1_solutions(sbox, n):
    """Count the degree <= 1 relations a.x ^ b.S(x) ^ c = 0 by brute force."""
    size = 1 << n
    solutions = 0
    for a in range(size):
        for b in range(size):
            for c in (0, 1):
                if all(_parity(a & x) ^ _parity(b & sbox[x]) ^ c == 0
                       for x in range(size)):
                    solutions += 1
    return solutions


@pytest.mark.parametrize("name,sbox,n", CASES)
def test_graph_equation_counts(name, sbox, n):
    size = 1 << n
    rows = graph_equation_counts(sbox, n, max_degree=2)
    assert [row["degree"] for row in rows] == [1, 2]
    for row in rows:
        expected = sum(math.comb(2 * n, k) for k in range(row["degree"] + 1))
        assert row["monomials"] == expected
        assert row["rank"] <= min(size, row["monomials"])
        assert row["independent_equations"] == row["monomials"] - row["rank"]
    # Degree 1: the kernel dimension is log2 of the number of affine relations.
    assert 2 ** rows[0]["independent_equations"] == _graph_degree1_solutions(sbox, n)
    # Degree 2: more monomials than points, so equations must exist.
    assert rows[1]["monomials"] > size
    assert rows[1]["independent_equations"] >= rows[1]["monomials"] - size


def test_graph_algebraic_immunity_is_saturated_for_five_bits():
    assert guaranteed_equation_degree(5) == 2      # 1 + 10 + 45 = 56 > 32
    assert guaranteed_equation_degree(4) == 2      # 1 + 8 + 28 = 37 > 16
    assert guaranteed_equation_degree(8) == 3      # 137 < 256 < 697
    with pytest.raises(ValueError):
        guaranteed_equation_degree(0)
    for sbox in (DAC, EDAC, RAND):
        imm = compute_algebraic_immunity(sbox, 5)
        assert imm["graph_ai"] == 2 and imm["graph_ai_saturated"]
        assert imm["graph_ai_guaranteed_degree"] == 2
        assert [row["degree"] for row in imm["graph_equations"]] == [1, 2, 3]
        assert imm["graph_equations"][0]["independent_equations"] == 0
        assert imm["graph_equations"][1]["independent_equations"] >= 24


def _aes_sbox():
    """The AES S-box built from its definition (GF(2^8) inverse + affine map)."""
    exponent = [0] * 256
    log = [0] * 256
    value = 1
    for i in range(255):                       # powers of the generator 0x03
        exponent[i] = value
        log[value] = i
        value ^= (value << 1) ^ (0x11B if value & 0x80 else 0)
        value &= 0xFF
    exponent[255] = exponent[0]

    def inverse(x):
        return 0 if x == 0 else exponent[255 - log[x]]

    table = []
    for x in range(256):
        y = inverse(x)
        byte = 0
        for i in range(8):
            bit = (((y >> i) & 1) ^ ((y >> ((i + 4) % 8)) & 1)
                   ^ ((y >> ((i + 5) % 8)) & 1) ^ ((y >> ((i + 6) % 8)) & 1)
                   ^ ((y >> ((i + 7) % 8)) & 1) ^ ((0x63 >> i) & 1))
            byte |= bit << i
        table.append(byte)
    assert sorted(table) == list(range(256))
    assert (table[0], table[1], table[255]) == (0x63, 0x7C, 0x16)
    return table


def test_graph_equations_reproduce_the_aes_quadratic_system():
    """Known value: the AES S-box admits 39 independent quadratic equations.

    Courtois and Pieprzyk (ASIACRYPT 2002) count 39 linearly independent
    equations of degree <= 2 in the 16 variables (x, y) of the 8-bit AES
    S-box; the same code path used for the 5-bit tables must reproduce it.
    """
    rows = graph_equation_counts(_aes_sbox(), 8, max_degree=2)
    assert rows[0] == {"degree": 1, "monomials": 17, "rank": 17,
                       "independent_equations": 0}
    assert rows[1] == {"degree": 2, "monomials": 137, "rank": 98,
                       "independent_equations": 39}
