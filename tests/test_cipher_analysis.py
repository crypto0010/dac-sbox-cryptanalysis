"""Tests for :mod:`src.cipher_analysis`.

The MILP bound, the branch numbers and the bypass modelling are each checked
against an independent brute-force computation written here from the
definitions, so a mistake in the model cannot hide behind the model.
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Dict, List, Sequence, Set, Tuple

import pytest

from src import cipher_analysis as ca
from src.block_cipher import BLOCK_SIZE, DACBlockCipher, PERM_TABLE, permute_bits
from src.sbox_tables import TABLES

ALL_NAMES = ca.SBOX_NAMES


# ------------------------------------------------------------------
# Brute-force references, written from the definitions
# ------------------------------------------------------------------
def brute_force_branch_numbers(table: Sequence[int]) -> Dict[str, int]:
    """DDT/LAT branch numbers by explicit loops (no numpy, no shared helper)."""
    size = len(table)
    bits = size.bit_length() - 1
    differential = 2 * bits
    linear = 2 * bits
    for a in range(1, size):
        seen_out: Set[int] = set()
        for x in range(size):
            seen_out.add(table[x] ^ table[x ^ a])
        for b in seen_out:
            if b:
                differential = min(differential,
                                   bin(a).count("1") + bin(b).count("1"))
    for a in range(1, size):
        for b in range(1, size):
            matches = sum(
                1 for x in range(size)
                if bin(a & x).count("1") % 2 == bin(b & table[x]).count("1") % 2)
            if matches != size // 2:
                linear = min(linear, bin(a).count("1") + bin(b).count("1"))
    return {"differential": differential, "linear": linear}


def _sbox_of(bit: int) -> int:
    return bit // ca.SBOX_BITS


def brute_force_model_min_active(rounds: int, start_bits: Sequence[int],
                                 perm: Sequence[int] = PERM_TABLE) -> int:
    """Exhaustive search over the *model* trails from ``start_bits``.

    Explores every nonempty output-bit subset of every active S-box -- the model
    allows all of them when the branch number is 2 -- so it shares no assumption
    with :func:`src.cipher_analysis.reference_min_active_sboxes`, which only
    follows single-bit trails.  Iterative deepening on the cost keeps the
    enumeration finite in practice: the search for cost limit ``L`` prunes every
    partial trail that already costs more than ``L``, and the first ``L`` for
    which a trail exists is the exact optimum.
    """
    def search(limit: int) -> bool:
        def descend(active: frozenset, depth: int, cost: int) -> bool:
            if depth == rounds:
                return True
            boxes = sorted({_sbox_of(b) for b in active if b not in ca.BYPASS_BITS})
            if cost + len(boxes) > limit:
                return False
            carried = [perm[b] for b in active if b in ca.BYPASS_BITS]
            choices: List[List[Tuple[int, ...]]] = [
                [tuple(ca.SBOX_INPUT_BITS[box][k] for k in range(5) if mask >> k & 1)
                 for mask in range(1, 32)]
                for box in boxes]

            def combine(index: int, chosen: List[int]) -> bool:
                if index == len(choices):
                    nxt = frozenset(carried + [perm[b] for b in chosen])
                    return bool(nxt) and descend(nxt, depth + 1, cost + len(boxes))
                return any(combine(index + 1, chosen + list(subset))
                           for subset in choices[index])

            return combine(0, [])

        return any(descend(frozenset([bit]), 0, 0) for bit in start_bits)

    for limit in range(rounds * ca.NUM_SBOXES + 1):
        if search(limit):
            return limit
    raise AssertionError("no trail found, which is impossible")


def brute_force_real_ddt_min_active(table: Sequence[int], rounds: int,
                                    max_weight: int = 1) -> int:
    """Minimum active S-boxes over *real* differential trails of the cipher.

    Propagates 32-bit differences through the actual DDT of ``table`` (only
    transitions with a nonzero DDT entry are followed) and through the actual
    bit permutation, starting from every input difference of Hamming weight at
    most ``max_weight``.  The result is the exact minimum over that restricted
    set of starting differences, hence an upper bound on the true minimum over
    all of them, which the MILP lower bound must not exceed.  Iterative
    deepening as in :func:`brute_force_model_min_active`.
    """
    size = len(table)
    transitions = {
        dx: sorted({table[x] ^ table[x ^ dx] for x in range(size)} - {0})
        for dx in range(1, size)}
    starts = [difference
              for weight in range(1, max_weight + 1)
              for positions in combinations(range(BLOCK_SIZE), weight)
              for difference in [sum(1 << (BLOCK_SIZE - 1 - i) for i in positions)]]

    def search(limit: int) -> bool:
        def descend(state: int, depth: int, cost: int) -> bool:
            if depth == rounds:
                return True
            windows = []
            for bits in ca.SBOX_INPUT_BITS:
                value = 0
                for k, i in enumerate(bits):
                    value |= ((state >> (BLOCK_SIZE - 1 - i)) & 1) << (4 - k)
                windows.append(value)
            active = sum(1 for value in windows if value)
            if cost + active > limit:
                return False
            carried = state & 0x3          # the two bypass bits, unchanged

            def combine(index: int, out_state: int) -> bool:
                if index == len(windows):
                    nxt = out_state | carried
                    return bool(nxt) and descend(permute_bits(nxt), depth + 1,
                                                 cost + active)
                if windows[index] == 0:
                    return combine(index + 1, out_state)
                for dy in transitions[windows[index]]:
                    placed = 0
                    for k, i in enumerate(ca.SBOX_INPUT_BITS[index]):
                        placed |= ((dy >> (4 - k)) & 1) << (BLOCK_SIZE - 1 - i)
                    if combine(index + 1, out_state | placed):
                        return True
                return False

            return combine(0, 0)

        return any(descend(difference, 0, 0) for difference in starts)

    for limit in range(rounds * ca.NUM_SBOXES + 1):
        if search(limit):
            return limit
    raise AssertionError("no trail found, which is impossible")


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def analysis() -> dict:
    """The full default analysis, computed once for the whole module."""
    return ca.analyze_cipher()


@pytest.fixture(scope="module")
def milp_counts() -> List[int]:
    """MILP bound for r = 1..16 with the cipher's permutation and branch number 2."""
    return ca.min_active_sboxes(16, PERM_TABLE, 2)


# ------------------------------------------------------------------
# 1. Round function description
# ------------------------------------------------------------------
def test_round_function_description_matches_implementation():
    checks = ca.check_round_function_description()
    assert all(checks.values()), checks


def test_sbox_windows_cover_every_bit_exactly_once():
    covered = [i for bits in ca.SBOX_INPUT_BITS for i in bits] + list(ca.BYPASS_BITS)
    assert sorted(covered) == list(range(BLOCK_SIZE))


def test_bypass_bits_are_state_bits_one_and_zero():
    description = ca.describe_round_function()
    assert description["bypass_state_bits"] == [1, 0]
    assert description["sbox_state_bit_windows"][0] == [31, 30, 29, 28, 27]
    assert description["sbox_state_bit_windows"][-1] == [6, 5, 4, 3, 2]


def test_round_function_text_is_self_contained():
    text = ca.round_function_text()
    for fragment in ("PERM_TABLE", "rotate_left_80", "state ^= 0x3", "80"):
        assert fragment in text


# ------------------------------------------------------------------
# 2. Branch numbers
# ------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_NAMES)
def test_branch_numbers_match_brute_force(name):
    table = TABLES[name].table
    assert ca.branch_numbers(table) == brute_force_branch_numbers(table)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_branch_numbers_are_two(name):
    # Recorded fact, not an assumption of the model: every S-box of the family
    # has the worst possible branch number, which is what makes the bound weak.
    assert ca.branch_numbers(TABLES[name].table) == {"differential": 2, "linear": 2}


def test_branch_number_of_a_known_better_sbox():
    # The PRESENT S-box has differential branch number 3; the reference and the
    # module must agree on an S-box that is not part of the DAC family.
    present = [0xC, 5, 6, 0xB, 9, 0, 0xA, 0xD, 3, 0xE, 0xF, 8, 4, 7, 1, 2]
    assert ca.branch_numbers(present, n=4)["differential"] == 3
    assert ca.branch_numbers(present, n=4) == brute_force_branch_numbers(present)


# ------------------------------------------------------------------
# 3. Linear mask permutation
# ------------------------------------------------------------------
def test_linear_mask_permutation_is_derived_and_is_a_permutation():
    derived = ca.derive_linear_mask_permutation(PERM_TABLE)
    assert sorted(derived) == list(range(BLOCK_SIZE))
    # For a bit permutation (L^T)^-1 = L, so the derivation reproduces the
    # state permutation.  Checked here, not assumed by the module.
    assert derived == tuple(PERM_TABLE)


def test_linear_mask_permutation_of_a_non_involutive_example():
    perm = tuple((3 * i + 1) % 8 for i in range(8))
    derived = ca.derive_linear_mask_permutation(perm)
    assert sorted(derived) == list(range(8))
    assert derived == perm


def test_mask_relation_holds_bitwise():
    # alpha . x == beta . permute(x) for beta = derived mask permutation image.
    derived = ca.derive_linear_mask_permutation(PERM_TABLE)
    for i in range(BLOCK_SIZE):
        alpha = 1 << (BLOCK_SIZE - 1 - i)
        beta = 1 << (BLOCK_SIZE - 1 - derived[i])
        for x in (0, 0xFFFFFFFF, 0xDEADBEEF, 0x12345678, 1 << (BLOCK_SIZE - 1 - i)):
            assert bin(alpha & x).count("1") % 2 == \
                bin(beta & permute_bits(x)).count("1") % 2


# ------------------------------------------------------------------
# 4. The MILP bound
# ------------------------------------------------------------------
def test_milp_matches_single_bit_reference(milp_counts):
    assert milp_counts == ca.reference_min_active_sboxes(16, PERM_TABLE)


def test_milp_window_cuts_change_nothing():
    plain = ca.min_active_sboxes(5, PERM_TABLE, 2, use_window_cuts=False)
    assert plain == ca.min_active_sboxes(5, PERM_TABLE, 2, use_window_cuts=True)


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_milp_matches_exhaustive_model_search(rounds, milp_counts):
    exhaustive = brute_force_model_min_active(rounds, range(BLOCK_SIZE))
    assert milp_counts[rounds - 1] == exhaustive


@pytest.mark.parametrize("name", ["DAC", "E-DAC", "DAC-OPT"])
@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_milp_is_a_lower_bound_on_real_ddt_trails(name, rounds, milp_counts):
    real = brute_force_real_ddt_min_active(TABLES[name].table, rounds, max_weight=1)
    assert milp_counts[rounds - 1] <= real


def test_milp_counts_are_monotone_and_grow(milp_counts):
    assert all(b >= a for a, b in zip(milp_counts, milp_counts[1:]))
    assert milp_counts[0] == 0          # one round can be crossed for free
    assert milp_counts[-1] == 9         # 16 rounds: 9 active S-boxes, not 16


def test_milp_rejects_bad_round_count():
    with pytest.raises(ValueError):
        ca.min_active_sboxes(0)
    with pytest.raises(ValueError):
        ca.reference_min_active_sboxes(0)


def test_higher_branch_number_gives_a_stronger_bound():
    # Sanity check on the branch-number constraint itself: raising it must not
    # lower the bound, and for this permutation it strictly raises it.
    weak = ca.min_active_sboxes(5, PERM_TABLE, 2)
    strong = ca.min_active_sboxes(5, PERM_TABLE, 3)
    assert all(s >= w for s, w in zip(strong, weak))
    assert strong[-1] > weak[-1]


# ------------------------------------------------------------------
# 5. Bypass-bit modelling
# ------------------------------------------------------------------
def test_single_round_bypass_trail_activates_no_sbox(milp_counts):
    assert milp_counts[0] == 0
    assert brute_force_model_min_active(1, ca.BYPASS_BITS) == 0


def test_no_multi_round_bypass_only_trail_exists():
    """The interesting question: can a trail live in bits 1 and 0 for ever?

    It cannot.  ``PERM_TABLE`` sends MSB-first index 30 to 21 and index 31 to
    27, both S-box bits, so a bypass-only difference survives exactly one free
    round.  Single free rounds do recur, though, which is what pulls the
    16-round bound down from 16 to 9 active S-boxes.
    """
    report = ca.bypass_trail_analysis(PERM_TABLE)
    assert report["max_consecutive_free_rounds"] == 1
    assert report["multi_round_bypass_only_trail_exists"] is False
    assert all(report["successor_is_an_sbox_bit"].values())
    # Independent check straight off the permutation table.
    assert all(PERM_TABLE[b] not in ca.BYPASS_BITS for b in ca.BYPASS_BITS)
    # A two-round trail therefore has to pay for at least one S-box.
    assert brute_force_model_min_active(2, ca.BYPASS_BITS) >= 1


def test_bypass_bits_really_bypass_the_sbox_layer():
    sbox = DACBlockCipher(0).sbox
    for base in (0, 0xFFFFFFFF, 0xA5A5A5A5):
        for delta in (0b01, 0b10, 0b11):
            out = sbox.substitute_block32(base) ^ sbox.substitute_block32(base ^ delta)
            assert out == delta     # difference passes through unchanged


# ------------------------------------------------------------------
# 6. Trail bounds
# ------------------------------------------------------------------
@pytest.mark.parametrize("name,dp,corr", [
    ("DAC", 0.25, 0.5),
    ("E-DAC", 0.125, 0.5),
    ("DAC-OPT", 0.0625, 0.375),
    ("DAC-OPT-GOLD", 0.0625, 0.25),
])
def test_sbox_trail_parameters(name, dp, corr):
    params = ca.sbox_trail_parameters(TABLES[name].table)
    assert params["dp_max"] == pytest.approx(dp)
    assert params["correlation_max"] == pytest.approx(corr)


def test_trail_bounds_arithmetic():
    bounds = ca.trail_bounds(TABLES["E-DAC"].table, [0, 1, 4], [0, 1, 4])
    rows = bounds["rows"]
    assert rows[0]["log2_max_characteristic_probability"] == 0.0
    assert rows[1]["log2_max_characteristic_probability"] == pytest.approx(-3.0)
    assert rows[2]["log2_max_characteristic_probability"] == pytest.approx(-12.0)
    assert rows[2]["log2_linear_data_complexity"] == pytest.approx(8.0)
    assert bounds["rounds_to_data_limit"]["differential"] is None


def test_bounds_at_sixteen_rounds(analysis):
    expected = {"DAC": -18.0, "E-DAC": -27.0, "DAC-OPT": -36.0, "DAC-OPT-GOLD": -36.0}
    for name, value in expected.items():
        row = analysis["bounds"][name]["at_cipher_rounds"]
        assert row["rounds"] == 16
        assert row["min_active_sboxes_differential"] == 9
        assert row["log2_max_characteristic_probability"] == pytest.approx(value)


def test_sixteen_rounds_do_not_reach_the_data_limit_for_dac(analysis):
    reached = analysis["findings"]["bound_reaches_data_limit_within_cipher_rounds"]
    assert reached["DAC"] == {"differential": False, "linear": False}
    assert reached["E-DAC"] == {"differential": False, "linear": False}
    assert reached["DAC-OPT"]["differential"] is True
    assert analysis["bounds"]["DAC-OPT"]["rounds_to_data_limit"]["differential"] == 14
    assert analysis["bounds"]["E-DAC"]["rounds_to_data_limit"]["differential"] == 19
    assert analysis["bounds"]["DAC"]["rounds_to_data_limit"]["differential"] == 27


# ------------------------------------------------------------------
# 7. Test vectors
# ------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_NAMES)
def test_generated_vectors_round_trip(name):
    table = TABLES[name].table
    vectors = ca.generate_test_vectors(table)
    assert len(vectors) == 4 + len(ca.TEST_VECTOR_SEEDS)
    for vector in vectors:
        assert vector["round_trip_ok"] is True
        key = int(vector["key_hex"], 16)
        plaintext = int(vector["plaintext_hex"], 16)
        cipher = DACBlockCipher(key, table)
        ciphertext = cipher.encrypt_block(plaintext)
        assert f"{ciphertext:08X}" == vector["ciphertext_hex"]
        assert cipher.decrypt_block(ciphertext) == plaintext


def test_generated_vectors_are_deterministic():
    assert ca.generate_test_vectors(TABLES["DAC"].table) == \
        ca.generate_test_vectors(TABLES["DAC"].table)


def test_generated_vectors_include_the_extreme_cases():
    vectors = ca.generate_test_vectors(TABLES["DAC"].table)
    pairs = {(v["key_hex"], v["plaintext_hex"]) for v in vectors[:4]}
    assert pairs == {("0" * 20, "0" * 8), ("0" * 20, "F" * 8),
                     ("F" * 20, "0" * 8), ("F" * 20, "F" * 8)}


def test_different_sboxes_give_different_ciphertexts():
    dac = ca.generate_test_vectors(TABLES["DAC"].table)
    opt = ca.generate_test_vectors(TABLES["DAC-OPT"].table)
    assert any(a["ciphertext_hex"] != b["ciphertext_hex"] for a, b in zip(dac, opt))


# ------------------------------------------------------------------
# 8. Birthday bound
# ------------------------------------------------------------------
def test_birthday_bound_for_the_cipher():
    bound = ca.birthday_bound()
    assert bound["block_bits"] == 32
    assert bound["log2_birthday_blocks"] == 16.0
    assert bound["birthday_blocks"] == 65536
    assert bound["birthday_bytes"] == 65536 * 4
    assert bound["birthday_kib"] == pytest.approx(256.0)


def test_birthday_bound_scales_and_validates():
    assert ca.birthday_bound(64)["birthday_blocks"] == 2 ** 32
    with pytest.raises(ValueError):
        ca.birthday_bound(0)


# ------------------------------------------------------------------
# 9. Public API
# ------------------------------------------------------------------
def test_analysis_is_json_serialisable(analysis):
    text = json.dumps(analysis)
    assert json.loads(text) == analysis


def test_analysis_is_deterministic():
    kwargs = dict(names=("DAC", "E-DAC"), reported_rounds=2, extended_rounds=3)
    assert ca.analyze_cipher(**kwargs) == ca.analyze_cipher(**kwargs)


def test_analysis_has_every_promised_section(analysis):
    for key in ("round_function", "round_function_checks", "linear_mask_permutation",
                "branch_numbers", "bypass_analysis", "active_sboxes", "bounds",
                "test_vectors", "birthday_bound", "findings", "caveats"):
        assert key in analysis
    assert set(analysis["bounds"]) == set(ALL_NAMES)
    assert set(analysis["test_vectors"]) == set(ALL_NAMES)
    assert len(analysis["bounds"]["DAC"]["rows"]) == 16
    assert analysis["caveats"] and all(isinstance(c, str) for c in analysis["caveats"])


def test_analysis_reports_the_active_sbox_table(analysis, milp_counts):
    counts = analysis["active_sboxes"]["differential_by_branch_number"]["2"]
    assert counts[:16] == milp_counts
    assert analysis["active_sboxes"]["linear_by_branch_number"]["2"][:16] == milp_counts
    assert analysis["findings"]["min_active_sboxes_at_cipher_rounds"] == 9
    assert analysis["findings"]["free_rounds_at_cipher_rounds"] == 7


def test_analysis_rejects_unknown_sbox():
    with pytest.raises(KeyError):
        ca.analyze_cipher(names=("no-such-sbox",), reported_rounds=2, extended_rounds=2)


def test_formatted_table_lists_every_round(analysis):
    text = ca.format_active_sbox_table(analysis)
    lines = text.splitlines()
    assert len(lines) == 2 + 16
    assert "Nr(diff)" in lines[0] and "Nr(lin)" in lines[0]
    assert lines[-1].split()[0] == "16"
