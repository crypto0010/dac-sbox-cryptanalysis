"""Controlled studies of the two search procedures (second revision).

Three studies answer the second round of review:

* :func:`generator_ablation` -- the affine-equivalence search of
  :func:`src.sbox_optimizer.affine_search` run under the same budget with
  three interchangeable sources of decisions: the chaotic tent-logistic
  generator of the paper (:class:`~src.sbox_optimizer.ChaoticRNG`), a
  conventional pseudorandom generator (NumPy's PCG64, :class:`PCG64Source`)
  and a deterministic, seed-free low-discrepancy sequence
  (:class:`WeylSource`).  It reports the representative-dependent criteria
  (SAC and BIC deviations, fixed and opposite fixed points) and the number of
  candidates each run needed to reach its result.
* :func:`affine_class_statistics` -- random representatives
  ``A F(B x ^ b) ^ a`` of the inversion and the Gold class, analysed in full,
  to separate the criteria that the class fixes (differential uniformity,
  nonlinearity, degree, boomerang uniformity, algebraic immunity and the
  spectra behind them) from those that vary with the representative and that
  the search therefore actually selects.
* :func:`annealing_parameter_study` -- the swap-move annealing of
  :func:`src.sbox_optimizer.simulated_annealing` as a function of the move
  budget ``N`` and of the initial temperature ``T0``, including ``N = 1``
  and ``T0 = 0`` (greedy descent).

Every study is deterministic: each run is driven by an explicitly seeded
source, and the results contain no timings.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .sbox_optimizer import (GOLD_EXPONENT, INVERSION_EXPONENT, SIZE, AffineParams,
                             ChaoticRNG, _avalanche_keys, _is_invertible,
                             _transpose, affine_search, autocorrelation_table,
                             enumerate_input_bases, sa_energy, simulated_annealing)
from .security_analysis import analyze_sbox

#: Golden-ratio conjugate; the Weyl sequence ``frac(u0 + k * PHI)`` is the
#: classical one-dimensional low-discrepancy sequence.
PHI = (math.sqrt(5.0) - 1.0) / 2.0

#: Search budgets of the paper's two affine searches (candidate matrices B).
AFFINE_BUDGETS: Dict[str, int] = {"inverse": 4096, "gold": 512}
EXPONENTS: Dict[str, int] = {"inverse": INVERSION_EXPONENT, "gold": GOLD_EXPONENT}

#: Runs per generator in the ablation.
ABLATION_RUNS = 32
#: Representatives sampled per affine class.
CLASS_SAMPLES = 1000
#: Move budgets N and initial-temperature settings of the annealing study.
SA_BUDGETS: Tuple[int, ...] = (1, 10, 100, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
SA_TEMPERATURE_FACTORS: Tuple[Optional[float], ...] = (0.0, 0.1, None, 10.0)
SA_STUDY_BUDGET = 20000
#: Chaotic seeds of the annealing study (the key space is (0, 1/2]).
SA_SEEDS: Tuple[float, ...] = tuple(round(k / 20, 2) for k in range(1, 11))


# ============================================================
# Interchangeable decision sources
# ============================================================
class _UniformSource:
    """Shared ``integers`` / ``pair`` on top of a ``_uniform(count)`` stream.

    The interface is the one :class:`~src.sbox_optimizer.ChaoticRNG` offers
    (``random``, ``integers``, ``pair``, ``describe``, ``draws``), so the
    search routines run unchanged on any source.
    """

    def __init__(self) -> None:
        self.draws = 0

    def _uniform(self, count: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def random(self, size: Optional[int] = None):
        if size is None:
            return float(self.random(1)[0])
        out = self._uniform(int(size))
        self.draws += out.size
        return out

    def integers(self, high: int, size=None):
        if high < 1:
            raise ValueError("high must be >= 1")
        if size is None:
            return int(self.random() * high)
        shape = (size,) if isinstance(size, int) else tuple(size)
        count = int(np.prod(shape))
        return (self.random(count) * high).astype(np.int64).reshape(shape)

    def pair(self, n: int) -> Tuple[int, int]:
        i = self.integers(n)
        j = self.integers(n - 1)
        return i, j + (j >= i)


class PCG64Source(_UniformSource):
    """NumPy's PCG64 generator (a conventional, well-tested PRNG)."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.seed = int(seed)
        self._generator = np.random.Generator(np.random.PCG64(self.seed))

    def _uniform(self, count: int) -> np.ndarray:
        return self._generator.random(count)

    def describe(self) -> Dict[str, Any]:
        return {"generator": "numpy.random.PCG64", "seed": self.seed, "draws": self.draws}


class WeylSource(_UniformSource):
    """Deterministic additive recurrence ``u_k = frac(u0 + k * PHI)``.

    No randomness at all: a fixed, equidistributed sequence of decisions.
    ``offset = 0`` is the canonical seed-free procedure; other offsets only
    shift the sequence.
    """

    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= offset < 1.0:
            raise ValueError(f"offset must lie in [0, 1), got {offset!r}")
        self.offset = float(offset)
        self._k = 0

    def _uniform(self, count: int) -> np.ndarray:
        k = np.arange(self._k + 1, self._k + count + 1, dtype=np.float64)
        self._k += count
        return np.mod(self.offset + k * PHI, 1.0)

    def describe(self) -> Dict[str, Any]:
        return {"generator": "Weyl sequence frac(u0 + k (sqrt(5) - 1) / 2)",
                "offset": self.offset, "draws": self.draws}


def make_source(kind: str, index: int):
    """Run ``index`` (0-based) of source ``kind``: chaotic, pcg64 or weyl."""
    if kind == "chaotic":
        # seeds k / 64 for k = 1 .. 32 cover the key space (0, 1/2] evenly
        # and include the paper's seed 0.5 as the last run
        return ChaoticRNG((index + 1) / 64.0)
    if kind == "pcg64":
        return PCG64Source(index)
    if kind == "weyl":
        return WeylSource(index / ABLATION_RUNS)
    raise ValueError(f"unknown source {kind!r}")


def source_label(kind: str, index: int) -> str:
    """Human-readable seed of run ``index`` of ``kind``."""
    if kind == "chaotic":
        return f"x0 = {index + 1}/64"
    if kind == "pcg64":
        return f"seed = {index}"
    return f"u0 = {index}/{ABLATION_RUNS}"


GENERATORS: Tuple[str, ...] = ("chaotic", "pcg64", "weyl")


# ============================================================
# Representative-dependent criteria
# ============================================================
def variable_criteria(sbox: Sequence[int]) -> Dict[str, Any]:
    """The criteria that differ between affine representatives.

    ``D_max`` / ``D_mean`` of the SAC matrix, the largest deviation of a
    BIC-SAC pair value from 1/2 and the deviation of their mean, the average
    BIC nonlinearity, and the fixed / opposite fixed point counts.
    """
    m = analyze_sbox(list(sbox), name="candidate")
    return {
        "sac_d_max": round(float(m["sac_d_max"]), 6),
        "sac_d_mean": round(float(m["sac_d_mean"]), 6),
        "bic_sac_d_max": round(float(max(abs(v - 0.5) for v in m["bic_sac_values"])), 6),
        "bic_sac_d_mean": round(float(abs(m["bic_sac_avg"] - 0.5)), 6),
        "bic_nl_avg": round(float(m["bic_nl_avg"]), 6),
        "fixed_points": int(m["fixed_points"]),
        "opposite_fixed_points": int(m["opposite_fixed_points"]),
    }


def invariant_criteria(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """The criteria fixed by the affine class, read from :func:`analyze_sbox`."""
    return {
        "differential_uniformity": int(metrics["differential_uniformity"]),
        "nl_sbox": int(metrics["nl_sbox"]),
        "lap": float(metrics["lap"]),
        "min_degree": int(metrics["min_degree"]),
        "max_degree": int(metrics["max_degree"]),
        "boomerang_uniformity": int(metrics["boomerang_uniformity"]),
        "ai_sbox": int(metrics["ai_sbox"]),
        "component_nl_spectrum": sorted(set(int(v) for v in metrics["component_nl"])),
        "bct_spectrum": [list(map(int, pair)) for pair in metrics["bct_spectrum"]],
    }


def _first_index(mask: np.ndarray) -> Optional[int]:
    hits = np.flatnonzero(mask)
    return int(hits[0]) + 1 if hits.size else None


# ============================================================
# 1. Generator ablation
# ============================================================
def _ablation_run(exponent: int, budget: int, source) -> Dict[str, Any]:
    params, stats = affine_search(exponent, source, budget)
    sbox = [int(v) for v in params.sbox()]
    metrics = analyze_sbox(sbox, name="candidate")
    return {
        "sbox": sbox,
        "affine": params.as_dict(),
        "invariants": invariant_criteria(metrics),
        "criteria": variable_criteria(sbox),
        # index (1-based) of the candidate B that the search kept, i.e. the
        # number of candidates it needed before reaching its final B
        "candidates_to_result": int(stats["best_candidate_index"]) + 1,
        "perfect_sac_candidates": int(stats["perfect_sac_candidates"]),
    }


def _candidates_to_perfect_sac(exponent: int, budget: int, kind: str,
                               index: int) -> Optional[int]:
    """First candidate (1-based) whose greedy output rows give an ideal SAC matrix.

    Re-draws the candidate matrices exactly as :func:`affine_search` does, from
    a fresh source with the same seed, so the count refers to the same stream.
    """
    from .sbox_optimizer import _greedy_output_rows, _sample_invertible
    source = make_source(kind, index)
    ac = autocorrelation_table(exponent)
    b_cols = _sample_invertible(source, budget)
    keys = _avalanche_keys(ac, _greedy_output_rows(ac, b_cols), b_cols)
    return _first_index(keys[0] == 0)


def lexicographic_candidates_to_perfect_sac(exponent: int) -> Dict[str, Any]:
    """Deterministic baseline: input bases taken in lexicographic order.

    The unordered input bases ``{d_0 < ... < d_4}`` of GF(2)^5 are visited in
    lexicographic order (the order of :func:`enumerate_input_bases`) and each
    is completed with the greedy output rows; the result is the 1-based
    position of the first basis giving an ideal SAC matrix.
    """
    import itertools
    from .sbox_optimizer import _greedy_output_rows
    ac = autocorrelation_table(exponent)
    subsets = np.array(list(itertools.combinations(range(1, SIZE), 5)), dtype=np.int64)
    bases = subsets[_is_invertible(subsets)]
    keys = _avalanche_keys(ac, _greedy_output_rows(ac, bases), bases)
    return {"order": "lexicographic over unordered input bases",
            "bases": int(bases.shape[0]),
            "candidates_to_perfect_sac": _first_index(keys[0] == 0)}


def _summary(values: Sequence[float]) -> Dict[str, float]:
    data = [float(v) for v in values]
    return {"min": min(data), "median": float(median(data)), "max": max(data),
            "mean": round(float(np.mean(data)), 6)}


def generator_ablation(runs: int = ABLATION_RUNS) -> Dict[str, Any]:
    """Run the affine search under every generator with the paper's budgets.

    For each class (inverse, Gold) and generator, ``runs`` seeded runs are
    made; per run the invariant criteria are checked against the class and
    the variable criteria and candidate counts are recorded.  ``summary``
    aggregates them and ``class_optimum`` is the best avalanche key of the
    whole class by exhaustive enumeration of the 83,328 input bases, so that
    "reached the optimum of the class" can be stated per run.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    report: Dict[str, Any] = {"runs_per_generator": runs, "budgets": dict(AFFINE_BUDGETS),
                              "refine_steps": 256, "constant_candidates": 64,
                              "classes": {}}
    for cls, exponent in EXPONENTS.items():
        budget = AFFINE_BUDGETS[cls]
        optimum = enumerate_input_bases(exponent)
        per_generator: Dict[str, Any] = {}
        for kind in GENERATORS:
            records = []
            for index in range(runs):
                record = _ablation_run(exponent, budget, make_source(kind, index))
                record["seed"] = source_label(kind, index)
                record["candidates_to_perfect_sac"] = _candidates_to_perfect_sac(
                    exponent, budget, kind, index)
                records.append(record)
            invariants = {json_key(r["invariants"]) for r in records}
            crit = [r["criteria"] for r in records]
            reach = [r["candidates_to_perfect_sac"] for r in records]
            per_generator[kind] = {
                "runs": records,
                "summary": {
                    **{key: _summary([c[key] for c in crit]) for key in crit[0]},
                    "candidates_to_result": _summary([r["candidates_to_result"]
                                                      for r in records]),
                    "runs_reaching_perfect_sac":
                        sum(1 for c in crit if c["sac_d_max"] == 0.0),
                    "runs_with_no_fixed_or_opposite_fixed_point":
                        sum(1 for c in crit if c["fixed_points"] == 0
                            and c["opposite_fixed_points"] == 0),
                    "candidates_to_perfect_sac": (
                        _summary([v for v in reach if v is not None])
                        if any(v is not None for v in reach) else None),
                    "runs_without_perfect_sac_candidate":
                        sum(1 for v in reach if v is None),
                    "invariants_identical_across_runs": len(invariants) == 1,
                },
            }
        report["classes"][cls] = {
            "exponent": exponent,
            "budget": budget,
            "lexicographic": lexicographic_candidates_to_perfect_sac(exponent),
            "class_optimum_key": list(optimum["best_key"]),
            "class_perfect_sac_bases": optimum["perfect_sac_bases"],
            "class_bases": optimum["bases"],
            "generators": per_generator,
        }
    return report


def json_key(value: Any) -> str:
    """Canonical string of a JSON-like value (for equality across runs)."""
    import json
    return json.dumps(value, sort_keys=True)


# ============================================================
# 2. Invariant versus representative-dependent criteria
# ============================================================
def _random_representative(exponent: int, rng: np.random.Generator) -> List[int]:
    while True:
        cols = rng.integers(1, SIZE, size=(2, 5))
        if bool(_is_invertible(cols[0])) and bool(_is_invertible(cols[1])):
            break
    a_rows = tuple(int(v) for v in _transpose(cols[0]))
    b_rows = tuple(int(v) for v in _transpose(cols[1]))
    a, b = (int(v) for v in rng.integers(0, SIZE, size=2))
    return [int(v) for v in AffineParams(exponent, a_rows, a, b_rows, b).sbox()]


def affine_class_statistics(samples: int = CLASS_SAMPLES, seed: int = 2026,
                            reference: Optional[Mapping[str, Sequence[int]]] = None
                            ) -> Dict[str, Any]:
    """Distribution of every criterion over random representatives of each class.

    ``samples`` representatives ``A F(B x ^ b) ^ a`` with uniformly random
    invertible ``A, B`` and constants ``a, b`` (PCG64, ``seed``) are analysed
    with the full metric engine.  A criterion is reported as *invariant* when
    it takes one value over all samples (and that value is then a property of
    ``F``), and as *variable* otherwise, with its range, median and the share
    of samples attaining the ideal value.  ``reference`` maps a class to the
    table produced by the search (DAC-OPT, DAC-OPT-GOLD), whose criteria are
    placed in the sampled distribution.
    """
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")
    rng = np.random.Generator(np.random.PCG64(seed))
    report: Dict[str, Any] = {"samples_per_class": samples, "seed": seed,
                              "generator": "numpy.random.PCG64", "classes": {}}
    for cls, exponent in EXPONENTS.items():
        invariant_values: Dict[str, set] = {}
        variable: Dict[str, List[float]] = {}
        for _ in range(samples):
            sbox = _random_representative(exponent, rng)
            metrics = analyze_sbox(sbox, name="sample")
            for key, value in invariant_criteria(metrics).items():
                invariant_values.setdefault(key, set()).add(json_key(value))
            for key, value in variable_criteria(sbox).items():
                variable.setdefault(key, []).append(value)
        import json
        invariant = {key: json.loads(next(iter(vals)))
                     for key, vals in invariant_values.items() if len(vals) == 1}
        not_invariant = sorted(key for key, vals in invariant_values.items() if len(vals) > 1)
        variable_summary: Dict[str, Any] = {}
        for key, values in variable.items():
            arr = np.asarray(values, dtype=np.float64)
            entry = {"min": float(arr.min()), "median": float(np.median(arr)),
                     "max": float(arr.max()), "mean": round(float(arr.mean()), 6),
                     "distinct_values": int(np.unique(arr).size)}
            if key in ("sac_d_max", "sac_d_mean", "bic_sac_d_max", "bic_sac_d_mean",
                       "fixed_points", "opposite_fixed_points"):
                entry["share_at_zero"] = float(np.mean(arr == 0.0))
            variable_summary[key] = entry
        record: Dict[str, Any] = {
            "exponent": exponent,
            "invariant": invariant,
            "expected_invariants_that_varied": not_invariant,
            "variable": variable_summary,
        }
        if reference and cls in reference:
            ref = variable_criteria(reference[cls])
            record["reference"] = {
                key: {"value": value,
                      "share_of_samples_at_least_as_good":
                          float(np.mean(np.asarray(variable[key]) <= value))
                          if key != "bic_nl_avg" else
                          float(np.mean(np.asarray(variable[key]) >= value))}
                for key, value in ref.items()}
            ref_metrics = analyze_sbox(list(reference[cls]), name="reference")
            record["reference_invariants_match_class"] = (
                {key: value for key, value in invariant_criteria(ref_metrics).items()
                 if key in invariant} == invariant)
        report["classes"][cls] = record
    return report


# ============================================================
# 2b. Exact count of the representatives with an ideal SAC matrix
# ============================================================
#: Order of GL(5, 2): the number of invertible 5x5 matrices over F_2.
GL5_ORDER = 9_999_360
#: Unordered bases of F_2^5: GL5_ORDER / 5!.
UNORDERED_BASES = GL5_ORDER // 120


def _subspace_lattice() -> Tuple[np.ndarray, np.ndarray]:
    """Membership matrix ``(374, 32)`` of every subspace of F_2^5 and the
    Moebius values ``mu(U, F_2^5) = (-1)^k 2^(k(k-1)/2)``, ``k = codim U``."""
    import itertools
    seen = set()
    for k in range(6):
        for vectors in itertools.combinations(range(1, SIZE), k):
            span = {0}
            for v in vectors:
                if v not in span:
                    span |= {u ^ v for u in span}
            seen.add(frozenset(span))
    spaces = sorted(seen, key=lambda s: (len(s), sorted(s)))
    member = np.zeros((len(spaces), SIZE), dtype=np.int64)
    codim = np.zeros(len(spaces), dtype=np.int64)
    for index, space in enumerate(spaces):
        member[index, list(space)] = 1
        codim[index] = 5 - (len(space).bit_length() - 1)
    mu = ((-1) ** codim) * (2 ** (codim * (codim - 1) // 2))
    return member, mu


def _orbit(basis: frozenset, gf_mul) -> frozenset:
    """Orbit of an unordered basis under d -> lambda * d and d -> d^2."""
    out = set()
    for lam in range(1, SIZE):
        current = basis
        for _ in range(5):
            out.add(frozenset(gf_mul(lam, d) for d in current))
            current = frozenset(gf_mul(d, d) for d in current)
    return frozenset(out)


def ideal_sac_counting(exponent: int) -> Dict[str, Any]:
    """Exact count of the affine representatives with an ideal SAC matrix.

    For ``S = A F(B x ^ b) ^ a`` the SAC entry is
    ``P_ij = 1/2 - AC[alpha_j, beta_i] / 64`` with ``beta_i`` column ``i`` of
    ``B`` and ``alpha_j`` row ``j`` of ``A``, so ``S`` has an ideal SAC
    matrix iff every row of ``A`` lies in the zero set
    ``Z(beta) = {alpha != 0 : AC[alpha, beta_i] = 0 for every i}``.

    1. Input bases are the unordered sets ``{beta_0..beta_4}`` of linearly
       independent nonzero vectors (column sets of ``B``; the column order
       only permutes the rows of the SAC matrix): 83,328 = |GL(5,2)| / 5!.
    2. For each, ``Z(beta)`` is read from the autocorrelation table.
    3. The basis is *admissible* iff ``Z(beta)`` spans F_2^5; the number of
       invertible ``A`` with all rows in ``Z(beta)`` -- ordered bases drawn
       from ``Z(beta)`` -- is ``sum_U mu(U, F_2^5) |Z(beta) n U|^5`` over the
       374 subspaces ``U`` (Moebius inversion on the subspace lattice).
    4. Over uniformly random invertible ``(A, B)`` -- the constants ``a, b``
       do not affect the SAC matrix -- the probability of an ideal SAC matrix
       is ``120 * sum_beta N(beta) / |GL(5,2)|^2``.

    The admissible bases are also split into orbits under the group of
    order 155 generated by ``d -> lambda d`` (``lambda`` in GF(2^5)*) and
    ``d -> d^2``, which maps admissible bases to admissible bases for any
    power map.
    """
    import itertools
    from .sbox_optimizer import gf_mul
    ac = autocorrelation_table(exponent)
    subsets = np.array(list(itertools.combinations(range(1, SIZE), 5)), dtype=np.int64)
    bases = subsets[_is_invertible(subsets)]
    zero = (ac == 0)[:, bases].all(axis=2).T.astype(np.int64)    # (K, 32): alpha in Z
    zero[:, 0] = 0
    member, mu = _subspace_lattice()
    ordered = (zero @ member.T) ** 5 @ mu                          # (K,): #A per basis
    admissible = np.flatnonzero(ordered > 0)
    sizes = zero[admissible].sum(axis=1)
    total_a = int(ordered.sum())
    pairs = 120 * total_a
    probability = pairs / GL5_ORDER ** 2
    adm_sets = {frozenset(int(v) for v in bases[k]): k for k in admissible}
    normal_bases = set()
    for alpha in range(1, SIZE):
        conj = [alpha]
        for _ in range(4):
            conj.append(gf_mul(conj[-1], conj[-1]))
        if len(set(conj)) == 5 and bool(_is_invertible(np.array(conj))):
            normal_bases.add(frozenset(conj))
    orbits: List[Dict[str, Any]] = []
    if len(adm_sets) <= 2000:     # the inverse class; the Gold class admits every basis
        seen: set = set()
        for basis in sorted(adm_sets, key=lambda s: sorted(s)):
            if basis in seen:
                continue
            orbit = _orbit(basis, gf_mul)
            if not orbit <= set(adm_sets):
                raise RuntimeError("an orbit left the admissible set")
            seen |= orbit
            k = adm_sets[basis]
            orbits.append({"representative": sorted(basis), "size": len(orbit),
                           "zero_set_size": int(zero[k].sum()),
                           "output_matrices": int(ordered[k]),
                           "normal_bases": sorted(sorted(n) for n in normal_bases & orbit)})
    return {
        "exponent": int(exponent),
        "input_bases": int(bases.shape[0]),
        "admissible_bases": int(admissible.size),
        "admissible_share": float(admissible.size / bases.shape[0]),
        "zero_set_sizes": {int(k): int(v) for k, v in
                           zip(*np.unique(sizes, return_counts=True))},
        "output_matrices_per_admissible_basis": {
            int(k): int(v) for k, v in
            zip(*np.unique(ordered[admissible], return_counts=True))},
        "ordered_pairs_with_ideal_sac": pairs,
        "gl5_order": GL5_ORDER,
        "probability_ideal_sac": probability,
        "one_in": (1.0 / probability) if probability else None,
        "expected_in_class_samples": probability * CLASS_SAMPLES,
        "normal_bases": sorted(sorted(n) for n in normal_bases),
        "orbits": orbits,
    }


def locate_basis(counting: Mapping[str, Any], b_rows: Sequence[int]) -> Dict[str, Any]:
    """Orbit and zero set of the input basis of a published table (``B`` rows)."""
    from .sbox_optimizer import gf_mul
    cols = frozenset(int(v) for v in _transpose(np.array(b_rows)))
    ac = autocorrelation_table(int(counting["exponent"]))
    zero_set = [c for c in range(1, SIZE) if all(ac[c, d] == 0 for d in cols)]
    orbit = _orbit(cols, gf_mul)
    index = next((i for i, o in enumerate(counting["orbits"])
                  if frozenset(o["representative"]) in orbit), None)
    return {"columns": sorted(cols), "zero_set": zero_set, "orbit_index": index}


# ============================================================
# 3. Annealing budget and initial temperature
# ============================================================
def _sa_record(table: np.ndarray, stats: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = analyze_sbox([int(v) for v in table], name="sa")
    return {
        "best_energy": float(stats["best_energy"]),
        "accepted_moves": int(stats["accepted_moves"]),
        "initial_temperature": float(stats["initial_temperature"]),
        "differential_uniformity": int(metrics["differential_uniformity"]),
        "nl_sbox": int(metrics["nl_sbox"]),
        "boomerang_uniformity": int(metrics["boomerang_uniformity"]),
        "min_degree": int(metrics["min_degree"]),
    }


def _aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    energies = [r["best_energy"] for r in records]
    du = [r["differential_uniformity"] for r in records]
    return {
        "best_energy": _summary(energies),
        "accepted_moves": _summary([r["accepted_moves"] for r in records]),
        "differential_uniformity": {"min": min(du), "max": max(du),
                                    "runs_at_min": du.count(min(du))},
        "nl_sbox": {"min": min(r["nl_sbox"] for r in records),
                    "max": max(r["nl_sbox"] for r in records)},
        "boomerang_uniformity": {"min": min(r["boomerang_uniformity"] for r in records),
                                 "max": max(r["boomerang_uniformity"] for r in records)},
        "runs": len(records),
    }


def annealing_parameter_study(start: Sequence[int],
                              budgets: Sequence[int] = SA_BUDGETS,
                              seeds: Sequence[float] = SA_SEEDS,
                              factors: Sequence[Optional[float]] = SA_TEMPERATURE_FACTORS,
                              study_budget: int = SA_STUDY_BUDGET) -> Dict[str, Any]:
    """Annealing quality as a function of ``N`` and of ``T0``.

    *Budget sweep*: for every ``N`` in ``budgets`` and every chaotic seed in
    ``seeds`` the annealer runs with the calibrated ``T0`` from ``start``.
    *Temperature sweep*: at ``N = study_budget`` the same seeds run with
    ``T0 = factor * T0_cal`` for each factor (``0`` is greedy descent,
    ``None`` the calibrated value itself).  ``T0_cal`` is computed per seed by
    the same calibration the annealer uses, so the factors are relative to
    the value the default run would have used.
    """
    if not budgets or not seeds:
        raise ValueError("budgets and seeds must be non-empty")
    start_energy = sa_energy(start)
    sweep: List[Dict[str, Any]] = []
    for n in budgets:
        records = [_sa_record(*simulated_annealing(start, n, ChaoticRNG(s)))
                   for s in seeds]
        sweep.append({"iterations": int(n), **_aggregate(records)})
    temperature: List[Dict[str, Any]] = []
    for factor in factors:
        records = []
        for s in seeds:
            if factor is None:
                table, stats = simulated_annealing(start, study_budget, ChaoticRNG(s))
            else:
                _, calib = simulated_annealing(start, 1, ChaoticRNG(s))
                t0 = factor * calib["initial_temperature"]
                table, stats = simulated_annealing(start, study_budget, ChaoticRNG(s),
                                                   initial_temperature=t0)
            records.append(_sa_record(table, stats))
        temperature.append({"factor": "calibrated" if factor is None else float(factor),
                            "iterations": int(study_budget), **_aggregate(records)})
    # The first budget at which the median best energy is within 1% of the
    # best median of the sweep -- a reproducible "plateau" criterion.
    medians = [row["best_energy"]["median"] for row in sweep]
    floor = min(medians)
    plateau = next(row["iterations"] for row, med in zip(sweep, medians)
                   if med <= floor * 1.01)
    return {
        "start_energy": float(start_energy),
        "seeds": [float(s) for s in seeds],
        "budget_sweep": sweep,
        "temperature_sweep": temperature,
        "plateau_iterations_within_1pct": int(plateau),
        "study_budget": int(study_budget),
    }
