"""Tests for the second revision: exact trails, identities, parameter domains, studies.

The exact trail search is checked against an enumerator written here from the
definitions -- S-box transitions read off explicit DDT/LAT loops and the
diffusion layer taken from :func:`src.block_cipher.permute_bits`, the cipher's
own reference permutation -- so the search cannot hide a placement error
behind its own bookkeeping.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

from src import cipher_analysis as ca
from src.block_cipher import BLOCK_SIZE, PERM_TABLE, permute_bits
from src.comparative_analysis import all_sbox_tables, require_metric_identities
from src.sbox import DAC_SBOX
from src.sbox_optimizer import ChaoticRNG, sa_energy, simulated_annealing
from src.sbox_tables import TABLES
from src.search_studies import (PCG64Source, WeylSource, affine_class_statistics,
                                lexicographic_candidates_to_perfect_sac, make_source,
                                variable_criteria)
from src.security_analysis import (analyze_sbox, check_metric_identities,
                                   compute_fixed_points)

NAMES = ("DAC", "E-DAC", "DAC-OPT", "DAC-OPT-GOLD")
WINDOW_SHIFTS = tuple(BLOCK_SIZE - 5 * (j + 1) for j in range(6))


# ------------------------------------------------------------------
# Reference trail enumerator, written from the definitions
# ------------------------------------------------------------------
def reference_transitions(table: Sequence[int], model: str) -> Dict[int, List[Tuple[float, int]]]:
    """``d -> [(weight, o)]`` by explicit loops over the 32 inputs."""
    out: Dict[int, List[Tuple[float, int]]] = {}
    for d in range(1, 32):
        entries = []
        for o in range(1, 32):
            if model == "differential":
                count = sum(1 for x in range(32) if table[x] ^ table[x ^ d] == o)
                if count:
                    entries.append((-math.log2(count / 32), o))
            else:
                agree = sum(1 for x in range(32)
                            if bin(d & x).count("1") % 2 == bin(o & table[x]).count("1") % 2)
                bias = agree - 16
                if bias:
                    entries.append((-math.log2(abs(bias) / 16), o))
        out[d] = sorted(entries)          # lightest first
    return out


def reference_round(state: int, outputs: Dict[int, int]) -> int:
    """S-layer output with the given per-window outputs, then the permutation."""
    y = state & 0x3                      # bypass bits pass unchanged
    for j, shift in enumerate(WINDOW_SHIFTS):
        y |= outputs.get(j, 0) << shift
    return permute_bits(y, PERM_TABLE)


def enumerate_trails(table: Sequence[int], model: str, rounds: int, cap: float,
                     cost: str) -> List[float]:
    """Costs of every ``rounds``-round trail with cost <= ``cap`` (no pruning
    beyond the cap itself).  ``cost`` is ``"active"`` or ``"weight"``."""
    trans = reference_transitions(table, model)
    lightest = min(w for entries in trans.values() for w, _ in entries)
    per_active = 1.0 if cost == "active" else lightest
    found: List[float] = []

    def walk(state: int, left: int, spent: float) -> None:
        windows = [(j, (state >> s) & 31) for j, s in enumerate(WINDOW_SHIFTS)
                   if (state >> s) & 31]
        floor = (len(windows) if cost == "active"
                 else sum(trans[d][0][0] for _, d in windows))
        if spent + floor > cap + 1e-9:         # this round alone exceeds the cap
            return
        if left == 1:
            last = (len(windows) if cost == "active"
                    else sum(trans[d][0][0] for _, d in windows))
            if spent + last <= cap + 1e-9:
                found.append(spent + last)
            return

        def choose(index: int, outputs: Dict[int, int], weight: float) -> None:
            if index == len(windows):
                add = len(windows) if cost == "active" else weight
                if spent + add <= cap + 1e-9:
                    walk(reference_round(state, outputs), left - 1, spent + add)
                return
            j, d = windows[index]
            for w, o in trans[d]:
                choose(index + 1, {**outputs, j: o}, weight + w)

        choose(0, {}, 0.0)

    max_active = int(cap // per_active + 1e-9)
    for start in ca._TrailModel(table, model).start_states(max_active):
        walk(start, rounds, 0.0)
    return found


# ------------------------------------------------------------------
# Exact trail search
# ------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("model", ("differential", "linear"))
def test_exact_counts_match_unpruned_enumeration(name: str, model: str) -> None:
    table = TABLES[name].table
    counts = ca.exact_min_active_sboxes(table, 4, model)["counts"]
    for rounds in range(1, 5):
        exact = counts[rounds - 1]
        if exact - 1 > 1:          # enumeration below the optimum stays cheap
            continue
        if exact > 0:
            assert enumerate_trails(table, model, rounds, exact - 1, "active") == [], \
                f"{name} {model}: a {rounds}-round trail with {exact - 1} active S-boxes exists"
        # existence at the optimum is proved by test_witness_trails_are_valid


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("model", ("differential", "linear"))
def test_exact_weights_match_unpruned_enumeration(name: str, model: str) -> None:
    table = TABLES[name].table
    weights = ca.exact_best_trail_weights(table, 2, model)["weights"]
    for rounds in (1, 2):
        best = enumerate_trails(table, model, rounds, weights[rounds - 1], "weight")
        assert best and min(best) == pytest.approx(weights[rounds - 1])


def _check_witness(table: Sequence[int], model: str, trail: Sequence[str]) -> Tuple[int, float]:
    """Re-derive a witness trail transition by transition; returns (active, weight)."""
    trans = reference_transitions(table, model)
    states = [int(s, 16) for s in trail]
    active, weight = 0, 0.0
    for index, state in enumerate(states):
        windows = [(j, (state >> s) & 31) for j, s in enumerate(WINDOW_SHIFTS)
                   if (state >> s) & 31]
        active += len(windows)
        if index + 1 == len(states):
            weight += sum(trans[d][0][0] for _, d in windows)
            break
        # some choice of per-window outputs must produce the next state exactly
        nxt = states[index + 1]
        pre = 0
        for i, j in enumerate(PERM_TABLE):            # undo the permutation
            pre |= ((nxt >> (BLOCK_SIZE - 1 - j)) & 1) << (BLOCK_SIZE - 1 - i)
        assert pre & 0x3 == state & 0x3, "bypass bits changed"
        for j, d in windows:
            o = (pre >> WINDOW_SHIFTS[j]) & 31
            match = [w for w, out in trans[d] if out == o]
            assert match, f"transition {d} -> {o} is impossible"
            weight += match[0]
        for j, shift in enumerate(WINDOW_SHIFTS):
            if (state >> shift) & 31 == 0:
                assert (pre >> shift) & 31 == 0, "inactive S-box produced a difference"
    return active, weight


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("model", ("differential", "linear"))
def test_witness_trails_are_valid(name: str, model: str) -> None:
    table = TABLES[name].table
    counts = ca.exact_min_active_sboxes(table, 10, model)
    for r, trail in enumerate(counts["witness_trails"], start=1):
        assert len(trail) == r
        assert _check_witness(table, model, trail)[0] == counts["counts"][r - 1]
    weights = ca.exact_best_trail_weights(table, 8, model)
    for r, trail in enumerate(weights["witness_trails"], start=1):
        _, weight = _check_witness(table, model, trail)
        assert weight == pytest.approx(weights["weights"][r - 1])


@pytest.mark.parametrize("name", NAMES)
def test_exact_counts_never_below_milp(name: str) -> None:
    milp = ca.min_active_sboxes(16)
    for model in ("differential", "linear"):
        exact = ca.exact_min_active_sboxes(TABLES[name].table, 16, model)["counts"]
        assert all(e >= m for e, m in zip(exact, milp))


def test_milp_is_tight_for_dac_opt_and_not_for_dac() -> None:
    milp = ca.min_active_sboxes(16)
    for model in ("differential", "linear"):
        assert ca.exact_min_active_sboxes(TABLES["DAC-OPT"].table, 16, model)["counts"] == milp
    dac = ca.exact_min_active_sboxes(TABLES["DAC"].table, 16, "differential")["counts"]
    assert dac[2] == 2 and milp[2] == 1     # first gap at 3 rounds


def test_apn_best_trail_weight_is_four_per_active_sbox() -> None:
    result = ca.exact_best_trail_weights(TABLES["DAC-OPT"].table, 10, "differential")
    assert result["weights"] == [4.0 * a for a in result["active_sboxes_of_best_trail"]]


def test_exact_search_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        ca.exact_min_active_sboxes(TABLES["DAC"].table, 0)
    with pytest.raises(ValueError):
        ca.exact_best_trail_weights(TABLES["DAC"].table, 2, step=0)
    with pytest.raises(ValueError):
        ca.exact_min_active_sboxes(TABLES["DAC"].table, 2, model="boomerang")


# ------------------------------------------------------------------
# Metric identities and fixed points
# ------------------------------------------------------------------
@pytest.mark.parametrize("name", list(all_sbox_tables()))
def test_metric_identities_hold(name: str) -> None:
    table = all_sbox_tables()[name]
    report = check_metric_identities(table)
    assert report["all_hold"], report
    v = report["values"]
    assert v["nl_sbox"] == 16 - 32 * v["lap"] == 16 * (1 - 2 * v["lap"])
    assert v["correlation_max"] == 2 * v["lap"]


def test_reviewer_lap_nl_correspondence() -> None:
    """LAP 0.25 <-> NL 8, 0.1875 <-> 10, 0.125 <-> 12 (reviewer 2, point 1)."""
    pairs = {(analyze_sbox(t, n)["lap"], analyze_sbox(t, n)["nl_sbox"])
             for n, t in all_sbox_tables().items()}
    assert pairs <= {(0.25, 8), (0.1875, 10), (0.125, 12)}


def test_require_metric_identities_rejects_violations() -> None:
    good = {"X": {"metric_identities": {"a": True}}}
    require_metric_identities(good)
    with pytest.raises(RuntimeError):
        require_metric_identities({"X": {"metric_identities": {"a": False}}})
    with pytest.raises(RuntimeError):
        require_metric_identities({"X": {}})


def test_fixed_points() -> None:
    identity = list(range(32))
    assert compute_fixed_points(identity) == {"fixed_points": 32, "opposite_fixed_points": 0}
    complement = [31 - x for x in range(32)]
    assert compute_fixed_points(complement) == {"fixed_points": 0, "opposite_fixed_points": 32}
    for name in ("DAC-OPT", "DAC-OPT-GOLD"):
        assert compute_fixed_points(TABLES[name].table) == {"fixed_points": 0,
                                                            "opposite_fixed_points": 0}


# ------------------------------------------------------------------
# Annealing parameter domains (reviewer 5, points 3 and 4)
# ------------------------------------------------------------------
@pytest.mark.parametrize("bad", (0, -5, 1.5, True))
def test_annealing_rejects_invalid_iteration_counts(bad) -> None:
    with pytest.raises(ValueError):
        simulated_annealing(DAC_SBOX, bad, ChaoticRNG(0.5))


@pytest.mark.parametrize("bad", (-1.0, float("inf"), float("nan")))
def test_annealing_rejects_invalid_temperatures(bad: float) -> None:
    with pytest.raises(ValueError):
        simulated_annealing(DAC_SBOX, 10, ChaoticRNG(0.5), initial_temperature=bad)


@pytest.mark.parametrize("bad", (0.0, -0.1, 1.5))
def test_annealing_rejects_invalid_cooling_ratio(bad: float) -> None:
    with pytest.raises(ValueError):
        simulated_annealing(DAC_SBOX, 10, ChaoticRNG(0.5), final_temperature_ratio=bad)


def test_single_iteration_returns_start_or_one_swap() -> None:
    table, stats = simulated_annealing(DAC_SBOX, 1, ChaoticRNG(0.5))
    assert stats["iterations"] == 1
    differing = sum(1 for a, b in zip(table, DAC_SBOX) if a != b)
    assert differing in (0, 2)
    assert stats["best_energy"] <= sa_energy(DAC_SBOX)


def test_zero_temperature_is_greedy_descent() -> None:
    """T0 = 0 accepts exactly the non-increasing moves, drawing no extra numbers."""
    n = 300
    table, stats = simulated_annealing(DAC_SBOX, n, ChaoticRNG(0.25), initial_temperature=0.0)
    assert stats["acceptance_rule"] == "greedy (T0 = 0)"
    rng = ChaoticRNG(0.25)
    state = np.array(DAC_SBOX)
    energy = sa_energy(state)
    accepted = 0
    for _ in range(n):
        i, j = rng.pair(32)
        state[i], state[j] = state[j], state[i]
        candidate = sa_energy(state)
        if candidate <= energy:
            energy, accepted = candidate, accepted + 1
        else:
            state[i], state[j] = state[j], state[i]
    assert stats["accepted_moves"] == accepted
    assert stats["best_energy"] == energy
    # the annealer returns the first table that reached the best energy; greedy
    # descent may later accept equal-energy (sideways) moves, so compare energies
    assert sa_energy(table) == energy


def test_calibrated_run_is_unchanged_by_the_domain_checks() -> None:
    """The frozen 20,000-move run of the paper still gives its recorded energy."""
    _, stats = simulated_annealing(DAC_SBOX, 20000, ChaoticRNG(0.5))
    assert stats["best_energy"] == 9880.0
    assert stats["accepted_moves"] == 1864
    assert stats["initial_temperature_calibrated"] is True


# ------------------------------------------------------------------
# Decision sources and studies (reviewer 2, points 4 and 5)
# ------------------------------------------------------------------
def test_sources_share_the_interface_and_are_reproducible() -> None:
    for make in (lambda: PCG64Source(7), lambda: WeylSource(0.25), lambda: ChaoticRNG(0.3)):
        a, b = make(), make()
        assert np.array_equal(a.random(50), b.random(50))
        i, j = a.pair(32)
        assert i != j and 0 <= i < 32 and 0 <= j < 32
        values = a.integers(31, size=(4, 5))
        assert values.shape == (4, 5) and values.min() >= 0 and values.max() < 31


def test_weyl_sequence_is_deterministic_and_equidistributed() -> None:
    u = WeylSource(0.0).random(10000)
    counts, _ = np.histogram(u, bins=10, range=(0.0, 1.0))
    assert counts.min() >= 990 and counts.max() <= 1010
    with pytest.raises(ValueError):
        WeylSource(1.0)


def test_chaotic_runs_include_the_paper_seed() -> None:
    assert make_source("chaotic", 31).seed == 0.5
    with pytest.raises(ValueError):
        make_source("mersenne", 0)


def test_lexicographic_baseline() -> None:
    assert lexicographic_candidates_to_perfect_sac(3)["candidates_to_perfect_sac"] == 1
    inverse = lexicographic_candidates_to_perfect_sac(30)
    assert inverse["bases"] == 83328 and inverse["candidates_to_perfect_sac"] > 1


def test_affine_class_invariants_and_reference() -> None:
    report = affine_class_statistics(
        samples=20, reference={"inverse": TABLES["DAC-OPT"].table,
                               "gold": TABLES["DAC-OPT-GOLD"].table})
    inverse = report["classes"]["inverse"]
    assert inverse["expected_invariants_that_varied"] == []
    assert inverse["invariant"]["differential_uniformity"] == 2
    assert inverse["invariant"]["nl_sbox"] == 10
    assert inverse["invariant"]["ai_sbox"] == 3
    assert inverse["reference_invariants_match_class"] is True
    assert report["classes"]["gold"]["invariant"]["nl_sbox"] == 12


def _paper() -> str:
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "paper.tex"), encoding="utf-8") as handle:
        return handle.read()


def test_editorial_gate_passes_on_the_paper() -> None:
    import verify_latex as vl
    errors: List[str] = []
    vl.check_editorial(_paper(), errors)
    assert errors == []
    short, full = vl.title_texts(_paper())
    assert len(full.split()) <= 20 and ":" not in full and "-" not in full


def test_editorial_gate_rejects_the_old_title_and_formulas() -> None:
    import verify_latex as vl
    paper = _paper()
    _, full = vl.title_texts(paper)
    old_title = ("Corrected and Optimized Chaotic Substitution Boxes for Lightweight IoT "
                 "Cryptography: A Critical Reappraisal of the DAC S-Box and a Provably "
                 "Optimal Replacement")
    errors: List[str] = []
    vl.check_editorial(paper.replace(full, old_title), errors)
    assert any("words" in e for e in errors) and any("contains" in e for e in errors)
    broken = paper.replace(r"2^{n-1}\bigl(1 - 2\,\mathrm{LAP}(S)\bigr)",
                           r"2^{n-1}\bigl(1 - \mathrm{LAP}(S)\bigr)")
    broken = broken.replace(r"\bigl(2\,\mathrm{LAP}(S)\bigr)^{N_l(r)}",
                            r"\bigl(\mathrm{LAP}(S)\bigr)^{N_l(r)}")
    errors = []
    vl.check_editorial(broken, errors)
    assert any("eq:lap_nl" in e for e in errors)
    assert any("eq:trail_lin" in e for e in errors)


def test_variable_criteria_of_dac_opt() -> None:
    crit = variable_criteria(TABLES["DAC-OPT"].table)
    assert crit["sac_d_max"] == 0.0 and crit["sac_d_mean"] == 0.0
    assert crit["fixed_points"] == 0 and crit["opposite_fixed_points"] == 0
