"""Chaotic maps used in the DAC S-box construction.

Implements the one-dimensional maps of Sarmila & Manisekaran (EIJ 2025):

* :class:`LogisticMap`      -- logistic map, Eq. (1)
* :class:`TentMap`          -- classical tent map
* :class:`TentLogisticMap`  -- tent-logistic hybrid map, Eqs. (3)-(5)

together with the Lyapunov exponent of Eq. (2), computed on the orbit for
every map, and the Shannon entropy of the binned orbit used for Fig. 5.

All maps act on the unit interval [0, 1].  Orbits are generated in double
precision exactly as the equations are written: no wrap-around, clipping or
other normalisation is applied.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _non_degenerate_prefix(orbit: FloatArray) -> FloatArray:
    """Truncate ``orbit`` at the first exact fixed point.

    In binary floating point an orbit can collapse onto a fixed point that is
    repelling for the exact map.  The tent map with ``r = 2`` is the classic
    example: every step shifts one mantissa bit out, so after roughly 50
    iterations the orbit reaches ``0.5 -> 1.0 -> 0.0`` and stays at 0.  Once
    ``s[n+1] == s[n]`` the remaining points carry no dynamical information,
    so they are dropped; the first point of the absorbing state is kept.
    """
    stuck = np.flatnonzero(orbit[1:] == orbit[:-1])
    if stuck.size == 0:
        return orbit
    return orbit[: stuck[0] + 1]


class ChaoticMap1D:
    """Base class for one-dimensional maps on the unit interval.

    Subclasses implement :meth:`step` (one iteration on a Python float) and
    :meth:`derivative` (vectorised ``f'(s)``).  Iteration and the Lyapunov
    estimate are shared.
    """

    def __init__(self, seed: float) -> None:
        if not 0.0 <= seed <= 1.0:
            raise ValueError(f"seed must lie in [0, 1], got {seed}")
        self.seed = float(seed)

    def step(self, s: float) -> float:
        """Apply the map once: ``s(n+1) = f(s(n))``."""
        raise NotImplementedError

    def derivative(self, s: ArrayLike) -> FloatArray:
        """Return ``f'(s)`` element-wise."""
        raise NotImplementedError

    def iterate(self, n_iterations: int) -> FloatArray:
        """Return the orbit ``s(0) = seed, s(1), ..., s(n_iterations - 1)``."""
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        orbit = np.empty(n_iterations)
        s = self.seed
        orbit[0] = s
        step = self.step
        for i in range(1, n_iterations):
            s = step(s)
            orbit[i] = s
        return orbit

    def lyapunov_exponent(self, n_iterations: int = 10000,
                          transient: int = 100) -> float:
        """Lyapunov exponent ``lambda = (1/N) sum_n ln|f'(s(n))|`` (Eq. 2).

        The estimate is taken on the orbit itself:

        * the first ``transient`` points are discarded so the average is over
          the attractor rather than the approach to it;
        * the orbit is truncated where it collapses onto an exact fixed
          point (see :func:`_non_degenerate_prefix`); if fewer than
          ``2 * transient`` non-degenerate points remain, only the first half
          of them is discarded so that a short valid orbit still yields an
          estimate;
        * points with a vanishing derivative (the logistic map at
          ``s = 1/2``), whose logarithm is ``-inf`` but which have measure
          zero, are skipped.

        Returns ``-inf`` if no usable point remains.
        """
        orbit = _non_degenerate_prefix(self.iterate(n_iterations))
        skip = min(transient, orbit.size // 2)
        slopes = np.abs(self.derivative(orbit[skip:]))
        slopes = slopes[slopes > 0.0]
        if slopes.size == 0:
            return -np.inf
        return float(np.mean(np.log(slopes)))


class LogisticMap(ChaoticMap1D):
    """Logistic map ``s(n+1) = r * s(n) * (1 - s(n))``, ``r`` in [0, 4] (Eq. 1)."""

    def __init__(self, r: float = 3.9, seed: float = 0.5) -> None:
        super().__init__(seed)
        self.r = float(r)

    def step(self, s: float) -> float:
        return self.r * s * (1.0 - s)

    def derivative(self, s: ArrayLike) -> FloatArray:
        """``f'(s) = r * (1 - 2s)``."""
        return self.r * (1.0 - 2.0 * np.asarray(s, dtype=float))

    def bifurcation_data(self, r_range: tuple[float, float] = (0.0, 4.0),
                         r_steps: int = 1000, n_iterations: int = 1000,
                         n_last: int = 300) -> tuple[FloatArray, FloatArray]:
        """Bifurcation-diagram data (Fig. 2).

        For each of ``r_steps`` control parameters evenly spaced over
        ``r_range`` the map is iterated ``n_iterations`` times from
        ``self.seed`` and the last ``n_last`` states are kept.  All parameter
        values are advanced simultaneously as one NumPy vector.

        Returns ``(r_values, states)``, two arrays of length
        ``r_steps * n_last`` suitable for a scatter plot.
        """
        if not 0 < n_last <= n_iterations:
            raise ValueError("require 0 < n_last <= n_iterations")
        r = np.linspace(r_range[0], r_range[1], r_steps)
        x = np.full(r_steps, self.seed)
        for _ in range(n_iterations - n_last):
            x = r * x * (1.0 - x)
        states = np.empty((n_last, r_steps))
        for i in range(n_last):
            x = r * x * (1.0 - x)
            states[i] = x
        return np.tile(r, n_last), states.ravel()


class TentMap(ChaoticMap1D):
    """Tent map ``s(n+1) = r * s(n)`` for ``s(n) < 1/2``, ``r * (1 - s(n))``
    otherwise, with ``r`` in [0, 2].

    ``|f'(s)| = r`` everywhere, so the orbit-based Lyapunov estimate equals
    ``ln r`` whenever at least one non-degenerate orbit point exists.  For
    ``r = 2`` the floating-point orbit collapses to 0 after about 50 steps
    (see :func:`_non_degenerate_prefix`); the estimate is then taken on the
    non-degenerate prefix and still returns ``ln 2``.
    """

    def __init__(self, r: float = 1.8, seed: float = 0.1) -> None:
        super().__init__(seed)
        self.r = float(r)

    def step(self, s: float) -> float:
        return self.r * s if s < 0.5 else self.r * (1.0 - s)

    def derivative(self, s: ArrayLike) -> FloatArray:
        """Piecewise slope ``+r`` / ``-r``; the kink at ``s = 1/2`` takes the
        right-hand slope, matching the branch used by :meth:`step`."""
        s = np.asarray(s, dtype=float)
        return np.where(s < 0.5, self.r, -self.r)


class TentLogisticMap(ChaoticMap1D):
    """Tent-logistic hybrid map of Sarmila & Manisekaran, Eqs. (3)-(5)::

        s(n+1) = f1(s(n))  if s(n) <= 1/2                               (3)
                 f2(s(n))  if s(n) >  1/2
        f1(s) = r_logistic * (9 - r)/9 * s * (1 - s) + r_tent * r/9 * s       (4)
        f2(s) = r_logistic * (9 - r)/9 * s * (1 - s) + r_tent * r/9 * (1 - s) (5)

    The regulating parameter ``r`` in [0, 9] blends the two component maps
    with weights ``(9 - r)/9`` and ``r/9``: ``r = 0`` is the pure logistic
    map and ``r = 9`` the pure tent map, exactly as stated below Eq. (5) of
    the paper.  (The typeset fractions are easy to misread as ``9 - 9r`` and
    ``9r``; that reading has no logistic/tent limit and diverges to infinity
    within a handful of iterations.)

    For ``r_logistic <= 4`` and ``r_tent <= 2`` both branches map [0, 1] into
    [0, 1]: on [0, 1/2] ``f1`` is increasing with maximum
    ``(9 - r)/9 * r_logistic/4 + r/9 * r_tent/2 <= 1`` at ``s = 1/2``, and on
    (1/2, 1] ``f2`` decreases from that same value to 0.  The orbit therefore
    stays in the unit interval by construction and no modular reduction is
    part of the definition or of this implementation.

    Both branches satisfy ``f(1 - s) = f(s)``, so the seeds ``s`` and
    ``1 - s`` generate the same orbit from the first step on; the effective
    key space is (0, 1/2] (see :func:`canonical_seed`).
    """

    def __init__(self, r_logistic: float = 3.9, r_tent: float = 1.8,
                 seed: float = 0.5, r: float = 8.0) -> None:
        super().__init__(seed)
        if not 0.0 <= r <= 9.0:
            raise ValueError(f"regulating parameter r must lie in [0, 9], got {r}")
        self.r_logistic = float(r_logistic)
        self.r_tent = float(r_tent)
        self.r = float(r)
        self.w_logistic = (9.0 - self.r) / 9.0
        self.w_tent = self.r / 9.0

    def _logistic_term(self, s: float) -> float:
        return self.w_logistic * self.r_logistic * s * (1.0 - s)

    def f1(self, s: float) -> float:
        """Branch for ``s <= 1/2`` (Eq. 4)."""
        return self._logistic_term(s) + self.w_tent * self.r_tent * s

    def f2(self, s: float) -> float:
        """Branch for ``s > 1/2`` (Eq. 5)."""
        return self._logistic_term(s) + self.w_tent * self.r_tent * (1.0 - s)

    def step(self, s: float) -> float:
        return self.f1(s) if s <= 0.5 else self.f2(s)

    def derivative(self, s: ArrayLike) -> FloatArray:
        """``f1'(s) = w_l r_logistic (1 - 2s) + w_t r_tent`` for ``s <= 1/2``,
        ``f2'(s) = w_l r_logistic (1 - 2s) - w_t r_tent`` otherwise."""
        s = np.asarray(s, dtype=float)
        logistic_slope = self.w_logistic * self.r_logistic * (1.0 - 2.0 * s)
        tent_slope = self.w_tent * self.r_tent
        return np.where(s <= 0.5, logistic_slope + tent_slope,
                        logistic_slope - tent_slope)

    def generate_sbox_sequence(self, n_values: int = 32,
                               warmup: int = 1000) -> NDArray[np.int_]:
        """Chaotic sequence quantised to 5-bit integers in [0, 31].

        Runs ``warmup`` iterations first, then maps the next ``n_values``
        orbit points ``s`` to ``floor(32 s)`` (the endpoint ``s = 1`` is
        clipped to 31).
        """
        orbit = self.iterate(warmup + n_values)
        return np.clip(np.floor(orbit[warmup:] * 32).astype(int), 0, 31)


def canonical_seed(seed: float) -> float:
    """Fold a tent-logistic key seed into the effective key space (0, 1/2].

    Both branches of :class:`TentLogisticMap` satisfy ``f(1 - s) = f(s)``
    (``f2(1 - s) = w_l r_l (1 - s) s + w_t r_t s = f1(s)``), so the seeds
    ``s`` and ``1 - s`` describe the same orbit from the first step on and
    only seeds in (0, 1/2] are distinct keys.  A seed above 1/2 is replaced
    by ``1 - seed``.  Note that for a decimal seed such as 0.7 the double
    ``1 - 0.7`` is the upper neighbour of the double 0.3, i.e. a different
    (equally valid) key; the fold is what makes mirror seeds reproduce each
    other bit for bit, because evaluating ``f`` on ``s`` and on ``1 - s``
    rounds differently.

    The endpoints are rejected: 0 is a fixed point of the map and 1 maps
    onto it in one step, so neither yields a chaotic orbit.

    Raises:
        ValueError: if ``seed`` is not strictly inside (0, 1).
    """
    if not 0.0 < seed < 1.0:
        raise ValueError(f"seed must lie strictly inside (0, 1), got {seed}")
    return min(float(seed), 1.0 - float(seed))


def default_maps() -> dict[str, ChaoticMap1D]:
    """The three maps with the parameters used throughout the paper.

    Logistic ``r = 3.9, seed = 0.5``; tent ``r = 1.8, seed = 0.1``;
    tent-logistic ``r_logistic = 3.9, r_tent = 1.8, seed = 0.5, r = 8``.
    """
    return {
        "logistic": LogisticMap(r=3.9, seed=0.5),
        "tent": TentMap(r=1.8, seed=0.1),
        "tent_logistic": TentLogisticMap(r_logistic=3.9, r_tent=1.8,
                                         seed=0.5, r=8.0),
    }


def entropy_approximation(sequence: ArrayLike, bins: int = 50,
                          normalized: bool = False,
                          value_range: tuple[float, float] | None = (0.0, 1.0)
                          ) -> float:
    """Shannon entropy in bits of the orbit binned into ``bins`` equal cells.

    ``H = -sum_k p_k log2 p_k`` with ``p_k`` the fraction of points in cell
    ``k``.  ``H`` ranges from 0 (all points in one cell) to ``log2(bins)``
    (uniform occupation; 5.64 bits for 50 cells).  With ``normalized=True``
    the ratio ``H / log2(bins)`` in [0, 1] is returned instead.

    The cells span ``value_range`` -- by default the unit interval on which
    every map in this module acts -- so values are comparable across maps
    and sequence lengths (Fig. 5); points outside it are ignored.  Pass
    ``value_range=None`` to bin the observed ``[min, max]`` of the sequence
    instead, which measures only the shape of the invariant density.
    """
    counts, _ = np.histogram(np.asarray(sequence, dtype=float), bins=bins,
                             range=value_range)
    total = counts.sum()
    if total == 0:
        raise ValueError("sequence contains no samples inside value_range")
    p = counts[counts > 0] / total
    entropy = float(-np.sum(p * np.log2(p)))
    return entropy / float(np.log2(bins)) if normalized else entropy


def compare_maps(n_iterations: int = 1000, bins: int = 50) -> dict[str, dict]:
    """Entropy and Lyapunov exponent of the three paper maps.

    Returns ``{name: {"entropy", "entropy_normalized", "lyapunov",
    "sequence"}}`` for ``logistic``, ``tent`` and ``tent_logistic``.
    """
    results: dict[str, dict] = {}
    for name, chaotic_map in default_maps().items():
        sequence = chaotic_map.iterate(n_iterations)
        # Bin the orbit once: the normalised value is that same histogram
        # divided by log2(bins) (see :func:`entropy_approximation`).
        entropy = entropy_approximation(sequence, bins)
        results[name] = {
            "entropy": entropy,
            "entropy_normalized": entropy / float(np.log2(bins)),
            "lyapunov": chaotic_map.lyapunov_exponent(),
            "sequence": sequence,
        }
    return results
