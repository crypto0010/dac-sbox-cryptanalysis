"""End-to-end tests of the DAC S-box code base.

Covered here:

* the frozen tables of :mod:`src.sbox_tables` (DAC, E-DAC, DAC-OPT,
  DAC-OPT-GOLD) against their known metrics, and the base paper's own
  numbers for the DAC table reconciled with the computed ones -- the
  definition-level brute-force checks of every metric live in
  ``tests/test_metrics.py`` and are not repeated here;
* the chaotic maps: Lyapunov exponents against closed forms, orbit range,
  entropy bounds and the ``s <-> 1 - s`` key fold of the tent-logistic map;
* :class:`~src.sbox.DACSBox` construction and input validation;
* the SPN block cipher round trip (integer blocks and byte strings) with
  both the paper's table and the optimised table, plus its validation and
  statistical avalanche behaviour;
* the benchmark suite (quick path, finite numbers only);
* the comparative-analysis machinery: ``_judge`` semantics, the
  data-driven ``compare_pair`` wins tallies on the frozen tables, the
  literature-discrepancy scan, the radar-chart normalisation and the
  ``write_results`` round trip / determinism contract;
* the importability of every ``src`` module.

All randomness is seeded and the whole file runs in a few seconds.
"""

import importlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.benchmarks import (
    benchmark_avalanche_quality, benchmark_block_cipher, benchmark_key_schedule,
    benchmark_sbox_lookup, estimate_memory_footprint, run_all_benchmarks,
)
from src.comparative_analysis import (
    COMPARISON_ROWS, LITERATURE_REPORTED, RADAR_AXES, _radar_values,
    compare_pair, literature_discrepancies, to_jsonable, write_results,
)

from src.comparative_analysis import (METRIC_ALIASES,  # noqa: E402
                                      aliased_comparison_keys,
                                      bct_spectrum_table, discriminator_table)
from src.comparative_analysis import judge as _judge
from src.block_cipher import BLOCK_SIZE, KEY_SIZE, NUM_ROUNDS, DACBlockCipher
from src.boolean_function import coordinate_truth_tables
from src.chaotic_maps import (
    LogisticMap, TentLogisticMap, TentMap, canonical_seed, compare_maps,
    default_maps, entropy_approximation,
)
from src.sbox import DAC_SBOX, DAC_SBOX_INV, DACSBox, invert_permutation
from src.sbox_tables import (
    DAC_OPT_GOLD_SBOX, DAC_OPT_SBOX, DAC_SBOX_REF, EDAC_SBOX, TABLES,
)
from src.security_analysis import analyze_sbox


# ------------------------------------------------------------
# Frozen tables against their known metrics
# ------------------------------------------------------------
#: Expected :func:`analyze_sbox` values per frozen table.  ``lap`` is the
#: standard ``max_bias / 2^n``; ``lap_half`` is the base paper's
#: ``max_bias / 2^(n+1)`` normalisation.
KNOWN_METRICS = {
    "DAC": {
        "nl_sbox": 8, "nl_per_bit": [8, 10, 10, 8, 10], "max_bias": 8,
        "lap": 0.25, "lap_half": 0.125, "differential_uniformity": 8,
        "dp": 0.25, "boomerang_uniformity": 14, "min_degree": 4,
        "max_degree": 4, "sac_avg": 0.515, "sac_max_deviation": 0.25,
        "bic_sac_avg": 0.5175, "hd_avg": 2.5625,
    },
    # Coordinate NL all 10, yet six of the 31 components have NL 8: the
    # printed E-DAC table's true NL(S) is 8.
    "E-DAC": {
        "nl_sbox": 8, "nl_per_bit": [10, 10, 10, 10, 10], "max_bias": 8,
        "lap": 0.25, "lap_half": 0.125, "differential_uniformity": 4,
        "dp": 0.125, "boomerang_uniformity": 16, "min_degree": 4,
        "max_degree": 4, "sac_avg": 0.53, "sac_max_deviation": 0.25,
        "bic_sac_avg": 0.5125, "hd_avg": 2.8125,
    },
    "DAC-OPT": {
        "nl_sbox": 10, "nl_per_bit": [10, 10, 10, 10, 10], "max_bias": 6,
        "lap": 0.1875, "lap_half": 0.09375, "differential_uniformity": 2,
        "dp": 0.0625, "boomerang_uniformity": 2, "min_degree": 4,
        "max_degree": 4, "sac_avg": 0.5, "sac_max_deviation": 0.0,
        "bic_sac_avg": 0.5075, "hd_avg": 2.625,
    },
    "DAC-OPT-GOLD": {
        "nl_sbox": 12, "nl_per_bit": [12, 12, 12, 12, 12], "max_bias": 4,
        "lap": 0.125, "lap_half": 0.0625, "differential_uniformity": 2,
        "dp": 0.0625, "boomerang_uniformity": 2, "min_degree": 2,
        "max_degree": 2, "sac_avg": 0.5, "sac_max_deviation": 0.0,
        "bic_sac_avg": 0.5, "hd_avg": 2.625,
    },
}


@pytest.mark.parametrize("name", sorted(KNOWN_METRICS))
def test_frozen_table_known_metrics(name):
    record = TABLES[name]
    metrics = analyze_sbox(record.table, name)
    assert metrics["is_bijective"]
    assert metrics["all_components_balanced"] and metrics["balanced_components"] == 31
    for key, expected in KNOWN_METRICS[name].items():
        assert metrics[key] == pytest.approx(expected), key
    # All 31 components share the same degree for these four tables.
    assert metrics["component_degrees"] == [KNOWN_METRICS[name]["max_degree"]] * 31
    # Sidelnikov-Chabaud-Vaudenay bound for n = 5 (attained by DAC-OPT-GOLD).
    assert metrics["nl_upper_bound"] == 12 and metrics["nl_sbox"] <= 12


def test_dac_paper_claims_reconciled():
    """The base paper's Section 5 numbers for Table 3 next to the computed ones."""
    metrics = analyze_sbox(DAC_SBOX)
    assert metrics["max_bias"] == 8                     # paper: highest LAT value 8
    assert metrics["lap_half"] == 0.125                 # paper's LAP (bias / 2^(n+1))
    assert metrics["lap"] == 0.25                       # standard LAP (bias / 2^n)
    assert metrics["dp"] == 0.25                        # paper: DP 0.25
    assert metrics["hd_avg"] == 2.5625                  # paper: average Hd 2.5625
    assert abs(metrics["sac_avg"] - 0.52) < 0.01        # paper: 0.52, computed 0.515
    assert abs(metrics["bic_sac_avg"] - 0.52) < 0.01    # paper: 0.52, computed 0.5175


def test_tables_are_single_sourced():
    assert DAC_SBOX == DAC_SBOX_REF == TABLES["DAC"].table
    assert DAC_SBOX[:4] == [30, 24, 9, 27] and DAC_SBOX[31] == 7   # Table 3 anchors
    assert DAC_SBOX_INV == invert_permutation(DAC_SBOX) == TABLES["DAC"].inverse
    assert TABLES["E-DAC"].table == EDAC_SBOX
    assert TABLES["DAC-OPT"].table == DAC_OPT_SBOX
    assert TABLES["DAC-OPT-GOLD"].table == DAC_OPT_GOLD_SBOX
    assert len({tuple(record.table) for record in TABLES.values()}) == len(TABLES)


# ------------------------------------------------------------
# Chaotic maps
# ------------------------------------------------------------
def test_logistic_map_lyapunov_exponent():
    # Fully chaotic regime r = 4: the exact exponent is ln 2.
    for seed in (0.1, 0.3):
        exponent = LogisticMap(r=4.0, seed=seed).lyapunov_exponent()
        assert abs(exponent - math.log(2.0)) < 0.05
    # Paper regime r = 3.9 is chaotic; the stable fixed point at r = 2.5 is not.
    assert LogisticMap(r=3.9, seed=0.5).lyapunov_exponent() > 0.3
    assert LogisticMap(r=2.5, seed=0.3).lyapunov_exponent() < 0.0


def test_tent_map_lyapunov_is_ln_r():
    # |f'| = r everywhere, so the estimate equals ln r exactly.
    assert TentMap(r=1.8, seed=0.1).lyapunov_exponent() == pytest.approx(math.log(1.8))
    # r = 2: the floating-point orbit collapses onto 0 after ~50 steps; the
    # estimate is taken on the non-degenerate prefix and still returns ln 2.
    assert TentMap(r=2.0, seed=0.1).lyapunov_exponent() == pytest.approx(math.log(2.0))


def test_tent_logistic_orbit_bounded_and_chaotic():
    # Maximum of both branches at s = 1/2 (see the TentLogisticMap docstring).
    bound = (9.0 - 8.0) / 9.0 * 3.9 / 4.0 + 8.0 / 9.0 * 1.8 / 2.0
    for seed in (0.5, 0.3, 0.123456, 0.999):
        tlm = TentLogisticMap(r_logistic=3.9, r_tent=1.8, seed=seed, r=8.0)
        orbit = tlm.iterate(5000)
        assert orbit.shape == (5000,) and orbit[0] == seed
        assert np.all(np.isfinite(orbit))
        # The bound holds from the first application of the map on (the seed
        # itself may lie above it, e.g. 0.999).
        assert orbit[1:].min() >= 0.0 and orbit[1:].max() <= bound + 1e-12
        assert tlm.lyapunov_exponent() > 0.3
    with pytest.raises(ValueError):
        TentLogisticMap(seed=0.3, r=9.5)
    with pytest.raises(ValueError):
        TentLogisticMap(seed=1.5)


def test_tent_logistic_limits_of_the_regulating_parameter():
    # r = 0 is the pure logistic map, r = 9 the pure tent map (paper, below Eq. 5).
    logistic = TentLogisticMap(r_logistic=3.9, r_tent=1.8, seed=0.3, r=0.0)
    assert logistic.iterate(6).tolist() == pytest.approx(
        LogisticMap(r=3.9, seed=0.3).iterate(6).tolist())
    tent = TentLogisticMap(r_logistic=3.9, r_tent=1.8, seed=0.3, r=9.0)
    assert tent.iterate(6).tolist() == pytest.approx(
        TentMap(r=1.8, seed=0.3).iterate(6).tolist())


def test_tent_logistic_key_fold():
    # f(1 - s) = f(s): mirror seeds share the orbit from the first step on.
    for seed in (0.3, 0.123, 0.4999):
        forward = TentLogisticMap(seed=seed).iterate(12)
        mirror = TentLogisticMap(seed=1.0 - seed).iterate(12)
        assert forward[1:] == pytest.approx(mirror[1:], abs=1e-9)
        assert canonical_seed(seed) == seed
        assert canonical_seed(1.0 - seed) == pytest.approx(seed)
    for seed in (0.01, 0.25, 0.5, 0.75, 0.99):
        folded = canonical_seed(seed)
        assert 0.0 < folded <= 0.5
        assert canonical_seed(folded) == folded    # idempotent


def test_entropy_approximation_bounds():
    log2_bins = math.log2(50)
    for chaotic_map in default_maps().values():
        orbit = chaotic_map.iterate(5000)
        entropy = entropy_approximation(orbit)
        assert 4.5 < entropy <= log2_bins            # all paper maps fill [0, 1] well
        assert 0.0 < entropy_approximation(orbit, normalized=True) <= 1.0
    assert entropy_approximation(np.full(100, 0.3)) == 0.0
    grid = np.arange(5000) / 5000.0 + 1e-4           # exactly 100 points per cell
    assert entropy_approximation(grid) == pytest.approx(log2_bins)
    assert entropy_approximation(grid, bins=8) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        entropy_approximation([5.0, 6.0])            # nothing inside the unit range
    results = compare_maps(n_iterations=500)
    assert set(results) == {"logistic", "tent", "tent_logistic"}
    for data in results.values():
        assert data["lyapunov"] > 0.0
        assert 0.0 < data["entropy"] <= log2_bins
        assert data["entropy_normalized"] == pytest.approx(data["entropy"] / log2_bins)


# ------------------------------------------------------------
# DACSBox construction and validation
# ------------------------------------------------------------
def test_dacsbox_rejects_invalid_tables():
    with pytest.raises(ValueError):
        DACSBox([0] * 31)                            # wrong length
    with pytest.raises(ValueError):
        DACSBox(list(range(31)) + [30])              # duplicate value
    with pytest.raises(ValueError):
        DACSBox(list(range(31)) + [32])              # value out of range
    with pytest.raises(TypeError):
        DACSBox([float(v) for v in DAC_SBOX])        # non-integer entries
    with pytest.raises(TypeError):
        DACSBox([True] + DAC_SBOX[1:])               # bools are not table entries
    with pytest.raises(ValueError):
        DACSBox(np.arange(33))                       # numpy, wrong length
    sbox = DACSBox()
    for bad in (-1, 32, 100):
        with pytest.raises(ValueError):
            sbox.substitute(bad)
        with pytest.raises(ValueError):
            sbox.inverse_substitute(bad)
    with pytest.raises(ValueError):
        sbox.substitute_block(1 << 32)
    with pytest.raises(ValueError):
        sbox.inverse_substitute_block(-1)
    with pytest.raises(ValueError):
        sbox.substitute_block(0, block_bits=0)


def test_dacsbox_accepts_any_permutation_and_numpy_tables():
    for table in (DAC_OPT_SBOX, np.array(DAC_OPT_GOLD_SBOX, dtype=np.uint8),
                  tuple(EDAC_SBOX)):
        sbox = DACSBox(table)
        assert sbox.table == [int(v) for v in table]
        assert all(type(v) is int for v in sbox.table)
        assert sbox.is_bijective()
        assert sbox.inv_table == invert_permutation(sbox.table)
        assert all(sbox.inverse_substitute(sbox.substitute(x)) == x for x in range(32))
        values = np.arange(32)
        assert sbox.substitute_array(values).tolist() == sbox.table
        assert sbox.inverse_substitute_array(sbox.substitute_array(values)).tolist() \
            == list(range(32))


def test_dacsbox_default_table_and_representations():
    sbox = DACSBox()
    assert sbox.table == DAC_SBOX and sbox.inv_table == DAC_SBOX_INV
    assert sbox.get_component_functions() == coordinate_truth_tables(DAC_SBOX).tolist()
    assert sbox.to_hex_string().split() == [f"{v:02X}" for v in DAC_SBOX]
    assert sbox.to_binary_table().splitlines()[0] == " 0 (00000) -> 30 (11110)"
    assert repr(sbox) == f"DACSBox(table={DAC_SBOX})"


def test_generic_block_widths_round_trip():
    sbox = DACSBox(DAC_OPT_SBOX)
    rng = random.Random(7)
    for bits in (1, 2, 5, 7, 12, 16, 33, 64):
        for _ in range(20):
            data = rng.getrandbits(bits)
            image = sbox.substitute_block(data, bits)
            assert 0 <= image < (1 << bits)
            assert sbox.inverse_substitute_block(image, bits) == data


# ------------------------------------------------------------
# Block cipher
# ------------------------------------------------------------
@pytest.mark.parametrize("table", [DAC_SBOX, DAC_OPT_SBOX], ids=["DAC", "DAC-OPT"])
def test_cipher_round_trip_blocks_and_bytes(table):
    rng = random.Random(20250831)
    for _ in range(8):
        cipher = DACBlockCipher(rng.getrandbits(KEY_SIZE), table)
        assert len(cipher.round_keys) == NUM_ROUNDS + 1
        assert len(set(cipher.round_keys)) == NUM_ROUNDS + 1
        assert all(0 <= rk < (1 << BLOCK_SIZE) for rk in cipher.round_keys)
        blocks = [rng.getrandbits(BLOCK_SIZE) for _ in range(40)] + [0, 0xFFFFFFFF]
        for plaintext in blocks:
            ciphertext = cipher.encrypt_block(plaintext)
            assert 0 <= ciphertext < (1 << BLOCK_SIZE)
            assert ciphertext != plaintext
            assert cipher.decrypt_block(ciphertext) == plaintext
    cipher = DACBlockCipher(0x0123456789ABCDEF0123, table)
    for length in range(1, 65):
        message = bytes(rng.getrandbits(8) for _ in range(length))
        ciphertext = cipher.encrypt_bytes(message)
        assert len(ciphertext) == 4 * (length // 4 + 1)    # PKCS#7 adds 1..4 bytes
        assert cipher.decrypt_bytes(ciphertext) == message
        assert cipher.decrypt_bytes(bytearray(ciphertext)) == message


def test_cipher_rejects_bad_inputs():
    cipher = DACBlockCipher(0x0123456789ABCDEF0123)
    with pytest.raises(ValueError):
        cipher.encrypt_bytes(b"")
    with pytest.raises(ValueError):
        cipher.decrypt_bytes(b"")
    with pytest.raises(ValueError):
        cipher.decrypt_bytes(b"12345")                     # not a multiple of 4
    other = DACBlockCipher(0xFEDCBA9876543210FEDC)
    with pytest.raises(ValueError):                        # padding check, wrong key
        other.decrypt_bytes(cipher.encrypt_bytes(b"Hello"))
    for bad_block in (1 << BLOCK_SIZE, -1):
        with pytest.raises(ValueError):
            cipher.encrypt_block(bad_block)
        with pytest.raises(ValueError):
            cipher.decrypt_block(bad_block)
    for bad_type in (1.0, "0", True):
        with pytest.raises(TypeError):
            cipher.encrypt_block(bad_type)
    with pytest.raises(ValueError):
        DACBlockCipher(1 << KEY_SIZE)
    with pytest.raises(ValueError):
        DACBlockCipher(-1)
    with pytest.raises(TypeError):
        DACBlockCipher(1.5)
    with pytest.raises(TypeError):
        DACBlockCipher(True)
    with pytest.raises(ValueError):
        DACBlockCipher(1, [0] * 32)


@pytest.mark.parametrize("table", [DAC_SBOX, DAC_OPT_SBOX], ids=["DAC", "DAC-OPT"])
def test_cipher_avalanche_statistics(table):
    result = benchmark_avalanche_quality(table, n_tests=50, seed=1)
    assert result["total_comparisons"] == 50 * 32
    assert abs(result["avg_avalanche_ratio"] - 0.5) < 0.03
    assert len(result["per_input_bit_ratio"]) == 32
    assert all(0.4 < ratio < 0.6 for ratio in result["per_input_bit_ratio"])
    assert result["deviation_from_ideal"] == pytest.approx(
        abs(result["avg_avalanche_ratio"] - 0.5))


# ------------------------------------------------------------
# Benchmarks (quick smoke: structure and finite numbers, not speed)
# ------------------------------------------------------------
def _numeric_leaves(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _numeric_leaves(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _numeric_leaves(value)
    elif isinstance(obj, bool) or isinstance(obj, str):
        return
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        yield float(obj)


def test_benchmark_functions_return_finite_numbers():
    lookup = benchmark_sbox_lookup(DAC_OPT_SBOX, n_lookups=2000, repeat=2, seed=3)
    assert lookup["n_lookups"] == 2000 and lookup["table_lookup_net_ns"] >= 0.0
    cipher = benchmark_block_cipher(DAC_OPT_SBOX, n_blocks=200, repeat=2)
    assert cipher["enc"]["blocks_per_sec"] > 0 and cipher["dec"]["mb_per_sec"] > 0
    schedule = benchmark_key_schedule(DAC_SBOX, n_keys=50, repeat=2)
    assert schedule["key_schedules_per_sec"] > 0
    memory = estimate_memory_footprint(DAC_SBOX)
    # 2 tables of 32 x 5 bits + 17 round keys of 32 bits + the 80-bit register.
    assert memory["total_bits"] == 2 * 160 + 17 * 32 + 80 == 944
    for result in (lookup, cipher, schedule, memory):
        leaves = list(_numeric_leaves(result))
        assert leaves and all(math.isfinite(value) for value in leaves)
    with pytest.raises(ValueError):
        benchmark_sbox_lookup(DAC_SBOX, n_lookups=10, repeat=0)
    with pytest.raises(ValueError):
        benchmark_avalanche_quality(DAC_SBOX, n_tests=0)


def test_run_all_benchmarks_quick_smoke(capsys):
    results = run_all_benchmarks(quick=True)
    # A superset check, not an equality: ``src.benchmarks`` may add further
    # top-level sections (a machine-description record, say) without this
    # test -- which is not that module's -- having to be edited first.  The
    # five sections the runners project into results.json must all be there.
    assert set(results) >= {"sbox", "cipher", "key_schedule", "memory", "avalanche"}
    leaves = list(_numeric_leaves(results))
    assert leaves and all(math.isfinite(value) for value in leaves)
    output = capsys.readouterr().out
    assert "SOFTWARE PERFORMANCE BENCHMARKS" in output
    assert "not synthesised" in output    # hardware numbers labelled as paper claims


# ------------------------------------------------------------
# Comparative analysis: judging, wins tallies, literature scan,
# radar normalisation and the results.json contract
# ------------------------------------------------------------
def test_judge_directions_targets_and_ties():
    """``_judge`` sign convention: +1 second wins, -1 first wins, 0 tie."""
    assert _judge(8, 10, "higher", None) == ("+2", 1)
    assert _judge(10, 8, "higher", None) == ("-2", -1)
    assert _judge(8, 8, "higher", None) == ("+0", 0)
    text, sign = _judge(0.25, 0.0625, "lower", None)
    assert sign == 1 and text.startswith("-0.1875")
    assert _judge(0.0625, 0.25, "lower", None)[1] == -1
    # bool: any change of truth decides; equal truth is a tie.
    assert _judge(True, True, "bool", None) == ("-", 0)
    assert _judge(False, True, "bool", None) == ("-", 1)
    assert _judge(True, False, "bool", None) == ("-", -1)
    # closer: judged by |value - target|, so an overshoot loses to an exact hit
    # and values on opposite sides of the target can tie.
    assert _judge(0.515, 0.5, "closer", 0.5)[1] == 1
    assert _judge(0.5, 0.53, "closer", 0.5)[1] == -1
    assert _judge(0.25, 0.55, "closer", 0.5)[1] == 1
    assert _judge(0.45, 0.55, "closer", 0.5)[1] == 0
    # the tie tolerance absorbs float noise in every direction
    assert _judge(0.5, 0.5 + 1e-12, "closer", 0.5)[1] == 0
    assert _judge(1.0, 1.0 + 1e-12, "higher", None)[1] == 0


def test_comparison_rows_score_each_quantity_once():
    """No double weighting and no wrongly-directed SAC rows in the wins tally."""
    keys = [key for _, key, _, _ in COMPARISON_ROWS]
    assert len(keys) == len(set(keys))
    # ``analyze_sbox`` publishes D_max / D_mean under two names each; scoring
    # both names of a pair would weight that quantity twice.
    assert aliased_comparison_keys() == []
    for canonical, alias in METRIC_ALIASES:
        metrics = analyze_sbox(TABLES["DAC"].table, "DAC")
        assert metrics[canonical] == metrics[alias]
    # LAP already represents max_bias and DP represents the differential
    # uniformity; the integer twins must not add a second row per quantity.
    assert "max_bias" not in keys and "differential_uniformity" not in keys
    assert "lap" in keys and "dp" in keys
    scoring = {key: (direction, target) for _, key, direction, target in COMPARISON_ROWS}
    assert scoring["sac_avg"] == ("closer", 0.5)
    assert scoring["sac_min"] == ("closer", 0.5)     # min overshooting 0.5 must not win
    assert scoring["bic_sac_avg"] == ("closer", 0.5)


def test_compare_pair_wins_tallies_on_frozen_tables():
    metrics = {name: analyze_sbox(TABLES[name].table, name)
               for name in ("DAC", "E-DAC", "DAC-OPT")}
    vs_edac = compare_pair(metrics["DAC"], metrics["E-DAC"])
    vs_opt = compare_pair(metrics["DAC"], metrics["DAC-OPT"])
    assert (vs_edac["first"], vs_edac["second"]) == ("DAC", "E-DAC")
    # E-DAC wins DP/DU, the coordinate nonlinearities, SAC (min), D_mean and
    # the mean non-trivial BCT entry, and loses SAC (avg), Hd and the
    # boomerang uniformity; LAP, D_max, NL(S) and AI(S) are ties -- which is
    # exactly the referees' point that E-DAC fixes neither LAP nor D_max.
    assert vs_edac["wins"] == {"first": 3, "second": 7, "tie": 7}
    assert vs_opt["wins"] == {"first": 1, "second": 13, "tie": 3}
    for pair in (vs_edac, vs_opt):
        assert len(pair["rows"]) == len(COMPARISON_ROWS)
        recount = {"first": 0, "second": 0, "tie": 0}
        for row in pair["rows"]:
            recount[row["winner"]] += 1
        assert recount == pair["wins"]
    # Swapping the arguments mirrors the tally.
    assert (compare_pair(metrics["E-DAC"], metrics["DAC"])["wins"]
            == {"first": 7, "second": 3, "tie": 7})
    # The referees' two "E-DAC does not fix this" points, as computed rows.
    by_key_edac = {row["key"]: row["winner"] for row in vs_edac["rows"]}
    assert by_key_edac["lap"] == "tie" and by_key_edac["sac_max_deviation"] == "tie"
    assert by_key_edac["nl_sbox"] == "tie" and by_key_edac["ai_min"] == "tie"
    assert by_key_edac["dp"] == "second"
    assert by_key_edac["boomerang_uniformity"] == "first"
    # Spot checks: DAC-OPT hits SAC 0.5 exactly and loses only Hamming distance.
    by_key = {row["key"]: row["winner"] for row in vs_opt["rows"]}
    assert by_key["sac_avg"] == "second" and by_key["hd_avg"] == "first"
    assert by_key["is_bijective"] == "tie"


def test_literature_discrepancies_flags_only_real_disagreements():
    agreeing = {name: dict(values) for name, values in LITERATURE_REPORTED.items()}
    assert literature_discrepancies(agreeing) == []
    doctored = {name: dict(values) for name, values in LITERATURE_REPORTED.items()}
    doctored["ASCON"]["sac_avg"] += 0.004       # inside the 0.005 tolerance
    doctored["PRIMATE"]["lap"] -= 0.25          # far outside
    rows = literature_discrepancies(doctored)
    assert [(row["sbox"], row["metric"]) for row in rows] == [("PRIMATE", "lap")]
    assert rows[0]["literature"] == LITERATURE_REPORTED["PRIMATE"]["lap"] == 0.375
    assert rows[0]["computed"] == pytest.approx(0.125)
    # S-boxes without a computed table are skipped, not flagged.
    assert literature_discrepancies({}) == []


def test_radar_values_reward_closeness_to_ideal():
    """Outer = better on every radar axis, consistent with the wins scoring."""
    metrics = {name: analyze_sbox(TABLES[name].table, name)
               for name in ("E-DAC", "DAC-OPT")}
    labels = [label for label, _ in RADAR_AXES]
    assert "1 - 2|SAC - 0.5|" in labels and "1 - 2|BIC-SAC - 0.5|" in labels
    for m in metrics.values():
        values = _radar_values(m)
        assert len(values) == len(RADAR_AXES)
        assert all(0.0 <= value <= 1.0 + 1e-9 for value in values)
    # DAC-OPT's exact SAC = 0.5 renders outermost, beating E-DAC's 0.53 --
    # a raw sac_avg axis would order them the other way round.
    sac_axis = labels.index("1 - 2|SAC - 0.5|")
    assert _radar_values(metrics["DAC-OPT"])[sac_axis] == pytest.approx(1.0)
    assert (_radar_values(metrics["DAC-OPT"])[sac_axis]
            > _radar_values(metrics["E-DAC"])[sac_axis])
    assert metrics["E-DAC"]["sac_avg"] > metrics["DAC-OPT"]["sac_avg"]


def test_discriminator_and_bct_spectrum_tables_report_the_new_columns():
    """The reviewer-requested columns are rendered from the computed metrics."""
    metrics = {name: analyze_sbox(TABLES[name].table, name)
               for name in ("DAC", "E-DAC", "DAC-OPT")}
    text = discriminator_table(metrics).get_string()
    for label in ("BCT #max", "BCT mean", "SAC D_max", "SAC D_mean", "AI(S)"):
        assert label in text
    # E-DAC's boomerang uniformity rises 14 -> 16 while the mean falls: the
    # answer to reviewer 2's major point 5 has to be visible in one table.
    assert metrics["E-DAC"]["boomerang_uniformity"] > \
        metrics["DAC"]["boomerang_uniformity"]
    assert metrics["E-DAC"]["bct_mean_nontrivial"] < metrics["DAC"]["bct_mean_nontrivial"]
    assert metrics["DAC"]["bct_max_count"] == metrics["E-DAC"]["bct_max_count"] == 1

    spectrum = bct_spectrum_table(metrics).get_string()
    assert "0:438 2:247" in spectrum          # DAC, computed above
    for m in metrics.values():
        assert sum(count for _, count in m["bct_spectrum"]) == m["bct_nontrivial_entries"]
        assert m["bct_nontrivial_entries"] == 961


def test_write_results_round_trip_and_determinism(tmp_path):
    results = {
        "b": {"runtime_s": 1.23, "value": np.float64(0.5), "count": np.int64(7)},
        "a": [np.arange(3), (1, 2)],
        "flag": True,
        "nested": {"deep": {"runtime_s": 9.9, "keep": [np.uint8(255)]}},
    }
    path = tmp_path / "results.json"
    assert write_results(str(path), results) == str(path)
    first_bytes = path.read_bytes()
    with open(path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    # numpy scalars/arrays and tuples become plain JSON; runtime_s is dropped
    # at every depth (wall-clock noise never reaches results.json).
    assert loaded == {
        "a": [[0, 1, 2], [1, 2]],
        "b": {"count": 7, "value": 0.5},
        "flag": True,
        "nested": {"deep": {"keep": [255]}},
    }
    assert "runtime_s" not in json.dumps(loaded)
    assert to_jsonable(results) == loaded      # the exact write-side transform
    # Byte-identical rewrite: sorted keys, LF endings, trailing newline.
    write_results(str(path), results)
    assert path.read_bytes() == first_bytes
    assert b"\r" not in first_bytes and first_bytes.endswith(b"\n")
    assert first_bytes.index(b'"a"') < first_bytes.index(b'"b"')


# ------------------------------------------------------------
# Module importability
# ------------------------------------------------------------


@pytest.mark.parametrize("module_name", [
    "src", "src.benchmarks", "src.block_cipher", "src.boolean_function",
    "src.chaotic_maps", "src.cipher_analysis", "src.comparative_analysis",
    "src.sbox", "src.sbox_optimizer", "src.sbox_tables",
    "src.security_analysis", "src.visualization",
])
def test_src_modules_import_and_are_documented(module_name):
    module = importlib.import_module(module_name)
    assert module.__doc__
