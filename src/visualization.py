"""Figure generation for the DAC S-box analysis.

Reproduces the figures of the paper -- Fig. 2 (logistic-map bifurcation
diagram), Fig. 4 (chaotic sequences), Fig. 5 (entropy vs. sequence length)
and Fig. 6 (SAC comparison) -- plus DDT/LAT heatmaps and an S-box mapping
plot.

Matplotlib is imported lazily so that importing this module stays cheap.
Every plot builds exactly one Agg-backed :class:`~matplotlib.figure.Figure`
(no GUI, no pyplot state) and writes it as PNG at :data:`DPI`.  Security
metrics (SAC, DDT, LAT) can be passed in precomputed so that plotting never
recalculates them.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .chaotic_maps import (LogisticMap, TentLogisticMap, TentMap, default_maps,
                           entropy_approximation)
from .comparative_analysis import LITERATURE_REPORTED
from .security_analysis import compute_ddt, compute_lat, compute_sac
from .sbox import DAC_SBOX

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Resolution of every saved figure.
DPI = 150
SBOX_BITS = 5

#: Average SAC of the competitor S-boxes, taken from the single source of
#: the base paper's Table 8 quotes in :mod:`src.comparative_analysis`.
COMPETITOR_SAC: dict[str, float] = {
    name: values["sac_avg"] for name, values in LITERATURE_REPORTED.items()
}

#: Line style and legend label per map key of :func:`default_maps`.
_MAP_STYLES: dict[str, tuple[str, str]] = {
    "logistic": ("b-o", "Logistic Map"),
    "tent": ("r-s", "Tent Map"),
    "tent_logistic": ("g-^", "Tent-Logistic Map"),
}


def _new_figure(figsize: tuple[float, float]) -> "Figure":
    """Create a standalone Figure rendered by the Agg backend.

    Matplotlib is imported here, not at module level, so that importing
    :mod:`src.visualization` costs nothing until a figure is drawn.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=DPI)
    FigureCanvasAgg(fig)
    return fig


def _save(fig: "Figure", output_path: str) -> str:
    """Apply tight layout, write ``fig`` to ``output_path`` and report it."""
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    print(f"  Saved: {output_path}")
    return output_path


def plot_bifurcation_diagram(output_path: str = "output/fig2_bifurcation.png",
                             r_steps: int = 2000, n_iterations: int = 10000,
                             n_last: int = 200) -> str:
    """Fig. 2: bifurcation diagram of the logistic map for ``r`` in [0, 4].

    Each of ``r_steps`` parameter values is iterated ``n_iterations`` times
    from seed 0.5 and the last ``n_last`` states are drawn as single pixels.
    """
    r_vals, x_vals = LogisticMap(seed=0.5).bifurcation_data(
        r_range=(0.0, 4.0), r_steps=r_steps, n_iterations=n_iterations,
        n_last=n_last)

    fig = _new_figure((12, 7))
    ax = fig.subplots()
    ax.plot(r_vals, x_vals, ',', color='black', alpha=0.3)
    ax.set_xlabel('Control Parameter r', fontsize=12)
    ax.set_ylabel('State x', fontsize=12)
    ax.set_title(f'Logistic Map - Bifurcation Diagram ($r \\in [0, 4]$, '
                 f'{n_iterations} iterations)', fontsize=14)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    return _save(fig, output_path)


def plot_chaotic_sequences(output_path: str = "output/fig4_chaotic_sequences.png",
                           n_iter: int = 50,
                           perturbed_seed: float = 0.50001) -> str:
    """Fig. 4: the first ``n_iter`` iterates of each map.

    (a) logistic map ``r = 3.9``, seed 0.5; (b) tent map ``r = 1.8``, seed
    0.1; (c) tent-logistic map ``r = 8`` from seed 0.5 and from
    ``perturbed_seed`` to show sensitivity to initial conditions.
    """
    logistic = LogisticMap(r=3.9, seed=0.5)
    tent = TentMap(r=1.8, seed=0.1)
    hybrid = TentLogisticMap(r_logistic=3.9, r_tent=1.8, seed=0.5, r=8.0)
    hybrid_perturbed = TentLogisticMap(r_logistic=3.9, r_tent=1.8,
                                       seed=perturbed_seed, r=8.0)
    steps = np.arange(n_iter)

    fig = _new_figure((14, 12))
    axes = fig.subplots(3, 1)

    axes[0].plot(steps, logistic.iterate(n_iter), 'b-o', markersize=3, linewidth=1)
    axes[0].set_title(f'(a) Logistic Map: r = {logistic.r:g}, '
                      f'seed = {logistic.seed:g}', fontsize=12)

    axes[1].plot(steps, tent.iterate(n_iter), 'r-o', markersize=3, linewidth=1)
    axes[1].set_title(f'(b) Tent Map: r = {tent.r:g}, seed = {tent.seed:g}',
                      fontsize=12)

    axes[2].plot(steps, hybrid.iterate(n_iter), 'g-o', markersize=3,
                 linewidth=1, label=f'seed = {hybrid.seed:g}')
    axes[2].plot(steps, hybrid_perturbed.iterate(n_iter), 'm-s', markersize=3,
                 linewidth=1, label=f'seed = {perturbed_seed:g}')
    axes[2].set_title(f'(c) Tent-Logistic Map: r = {hybrid.r:g}, '
                      'sensitivity comparison', fontsize=12)
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_entropy_comparison(output_path: str = "output/fig5_entropy.png",
                            bins: int = 50,
                            lengths: Sequence[int] = (100, 200, 500, 1000,
                                                     2000, 5000, 10000)) -> str:
    """Fig. 5: Shannon entropy (bits) of each map's orbit vs. sequence length.

    Each map is iterated once up to ``max(lengths)``; the entropy of every
    prefix is taken from that single orbit, which is identical to iterating
    separately per length because the maps are deterministic.  The dashed
    line marks the maximum ``log2(bins)`` reachable with ``bins`` cells.
    """
    lengths = sorted(lengths)
    orbits = {name: chaotic_map.iterate(lengths[-1])
              for name, chaotic_map in default_maps().items()}
    entropies = {name: [entropy_approximation(orbit[:n], bins) for n in lengths]
                 for name, orbit in orbits.items()}
    h_max = float(np.log2(bins))

    fig = _new_figure((10, 6))
    ax = fig.subplots()
    for name, (style, label) in _MAP_STYLES.items():
        ax.plot(lengths, entropies[name], style, label=label, linewidth=2)
    ax.axhline(h_max, color='gray', linestyle='--', linewidth=1.5,
               label=f'Maximum $\\log_2({bins})$ = {h_max:.2f} bits')
    ax.set_xscale('log')
    ax.set_xlabel('Number of Iterations', fontsize=12)
    ax.set_ylabel(f'Shannon Entropy (bits, {bins} bins)', fontsize=12)
    ax.set_title('Shannon Entropy of Chaotic Map Sequences vs. Sequence Length',
                 fontsize=14)
    lowest = min(min(values) for values in entropies.values())
    ax.set_ylim(max(0.0, np.floor(2.0 * (lowest - 0.5)) / 2.0), h_max + 0.25)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    return _save(fig, output_path)


def plot_sac_comparison(sbox: Sequence[int],
                        output_path: str = "output/fig6_sac_comparison.png",
                        sac_result: dict[str, Any] | None = None) -> str:
    """Fig. 6: SAC matrix heatmap and average-SAC comparison with competitors.

    ``sac_result`` is the dict returned by
    :func:`~src.security_analysis.compute_sac`; it is computed here if omitted.
    """
    if sac_result is None:
        sac_result = compute_sac(sbox, SBOX_BITS)
    matrix = np.asarray(sac_result['matrix'])

    fig = _new_figure((14, 5))
    ax_matrix, ax_bars = fig.subplots(1, 2)

    im = ax_matrix.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
    ax_matrix.set_title('DAC S-Box SAC Matrix', fontsize=12)
    ax_matrix.set_xlabel('Output Bit')
    ax_matrix.set_ylabel('Input Bit')
    for (i, j), value in np.ndenumerate(matrix):
        ax_matrix.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=10)
    fig.colorbar(im, ax=ax_matrix)

    names = ['DAC', *COMPETITOR_SAC]
    values = [float(sac_result['average']), *COMPETITOR_SAC.values()]
    colors = ['green', 'blue', 'orange', 'purple', 'red']
    bars = ax_bars.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
    ax_bars.axhline(y=0.5, color='red', linestyle='--', label='SAC threshold (0.5)')
    ax_bars.set_ylabel('Average SAC Value', fontsize=12)
    ax_bars.set_title('SAC Comparison with Competitors', fontsize=12)
    ax_bars.legend()
    ax_bars.set_ylim(0, 0.7)
    for bar, value in zip(bars, values):
        ax_bars.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', fontsize=10)
    return _save(fig, output_path)


def heatmap_scaled_by_nontrivial(ax, table: np.ndarray, cmap: str = 'YlOrRd',
                                 vmax: int | None = None):
    """Draw ``table`` with the colour scale taken from its non-trivial part.

    ``DDT[0, 0] = 2^n`` and ``|LAT[0, 0]| = 2^(n-1)`` are structural constants
    of every S-box (the zero difference / the empty mask), several times
    larger than any entry that carries information.  Letting them set the
    colour range flattens the whole informative range 0 .. ``vmax`` into the
    bottom quarter of the colormap, which is what made these figures look
    blank.  The scale therefore comes from ``table[1:, 1:]`` -- the same rule
    :func:`src.comparative_analysis._plot_heatmaps` already uses -- and the
    trivial corner is *not* dropped: it is clipped to the top colour, outlined
    and labelled with its true value.  Passing ``vmax`` overrides the
    per-table scale so that several panels can share one colour range.
    Returns ``(image, vmax, corner)``.
    """
    from matplotlib.patches import Rectangle

    array = np.asarray(table)
    if vmax is None:
        vmax = int(array[1:, 1:].max())
    image = ax.imshow(table, cmap=cmap, vmin=0, vmax=max(vmax, 1), aspect='equal')
    corner = int(array[0, 0])
    if corner > vmax:
        ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor='black',
                               linewidth=1.2))
        ax.annotate(f'[0,0] = {corner}', xy=(0, 0), xytext=(4.5, 3.0),
                    fontsize=9, color='black',
                    arrowprops=dict(arrowstyle='->', color='black', linewidth=1.0))
    return image, vmax, corner


#: Pre-rename private spelling, kept for existing callers and tests.
_heatmap_scaled_by_nontrivial = heatmap_scaled_by_nontrivial


def plot_ddt_heatmap(sbox: Sequence[int],
                     output_path: str = "output/ddt_heatmap.png",
                     ddt_result: dict[str, Any] | None = None) -> str:
    """Difference distribution table as a heatmap.

    ``ddt_result`` is the dict returned by
    :func:`~src.security_analysis.compute_ddt`; it is computed here if omitted.
    """
    if ddt_result is None:
        ddt_result = compute_ddt(sbox, SBOX_BITS)

    fig = _new_figure((10, 8))
    ax = fig.subplots()
    im, vmax, corner = _heatmap_scaled_by_nontrivial(ax, ddt_result['table'])
    ax.set_title("Difference Distribution Table (DDT)\n"
                 f"differential uniformity = {vmax}, "
                 f"max DP = {ddt_result['dp']:.4f}", fontsize=14)
    ax.set_xlabel(r'Output Difference ($\Delta y$)', fontsize=12)
    ax.set_ylabel(r'Input Difference ($\Delta x$)', fontsize=12)
    fig.colorbar(im, ax=ax,
                 label=rf'Count (scale 0..{vmax} over $\Delta x \neq 0$; '
                       f'trivial [0,0] = {corner} clipped)')
    return _save(fig, output_path)


def plot_lat_heatmap(sbox: Sequence[int],
                     output_path: str = "output/lat_heatmap.png",
                     lat_result: dict[str, Any] | None = None) -> str:
    """Linear approximation table (absolute bias) as a heatmap.

    ``lat_result`` is the dict returned by
    :func:`~src.security_analysis.compute_lat`; it is computed here if omitted.
    """
    if lat_result is None:
        lat_result = compute_lat(sbox, SBOX_BITS)

    fig = _new_figure((10, 8))
    ax = fig.subplots()
    im, vmax, corner = _heatmap_scaled_by_nontrivial(ax, np.abs(lat_result['table']))
    ax.set_title("Linear Approximation Table (LAT)\n"
                 f"max |bias| = {vmax}, LAP = max |bias| / 2^n "
                 f"= {lat_result['lap']:.4f}", fontsize=14)
    ax.set_xlabel('Output Mask (b)', fontsize=12)
    ax.set_ylabel('Input Mask (a)', fontsize=12)
    fig.colorbar(im, ax=ax,
                 label=rf'|Bias| (scale 0..{vmax} over $a, b \neq 0$; '
                       f'trivial [0,0] = {corner} clipped)')
    return _save(fig, output_path)


def plot_sbox_mapping(sbox: Sequence[int],
                      output_path: str = "output/sbox_mapping.png") -> str:
    """S-box input/output mapping and per-input Hamming distance."""
    inputs = np.arange(len(sbox))
    hamming = np.array([bin(x ^ y).count('1') for x, y in zip(inputs, sbox)])

    fig = _new_figure((14, 5))
    ax_map, ax_hd = fig.subplots(1, 2)

    ax_map.plot(inputs, sbox, 'b-o', markersize=4)
    ax_map.plot(inputs, inputs, 'r--', alpha=0.3, label='Identity')
    ax_map.set_xlabel('Input', fontsize=12)
    ax_map.set_ylabel('Output', fontsize=12)
    ax_map.set_title('DAC S-Box Mapping', fontsize=14)
    ax_map.legend()
    ax_map.grid(True, alpha=0.3)

    ax_hd.bar(inputs, hamming, color='teal', alpha=0.7, edgecolor='black')
    ax_hd.axhline(y=hamming.mean(), color='red', linestyle='--',
                  label=f'Avg Hd = {hamming.mean():.4f}')
    ax_hd.set_xlabel('Input', fontsize=12)
    ax_hd.set_ylabel('Hamming Distance', fontsize=12)
    ax_hd.set_title('Hamming Distance per Input', fontsize=14)
    ax_hd.legend()
    ax_hd.grid(True, alpha=0.3)
    return _save(fig, output_path)


def generate_all_figures(sbox: Sequence[int] | None = None,
                         output_dir: str = "output",
                         security_results: dict[str, Any] | None = None) -> list[str]:
    """Generate every figure into ``output_dir`` (created if missing).

    ``security_results`` may be the dict returned by
    :func:`~src.security_analysis.full_security_analysis`; its ``"sac"``,
    ``"ddt"`` and ``"lat"`` entries are reused so the metrics are not
    recomputed for plotting.  Missing entries are computed on demand.
    Returns the list of written figure paths so runners can record them.
    """
    if sbox is None:
        sbox = DAC_SBOX
    precomputed = security_results or {}
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)

    paths = [
        plot_bifurcation_diagram(os.path.join(output_dir, "fig2_bifurcation.png")),
        plot_chaotic_sequences(os.path.join(output_dir, "fig4_chaotic_sequences.png")),
        plot_entropy_comparison(os.path.join(output_dir, "fig5_entropy.png")),
        plot_sac_comparison(sbox, os.path.join(output_dir, "fig6_sac_comparison.png"),
                            sac_result=precomputed.get("sac")),
        plot_ddt_heatmap(sbox, os.path.join(output_dir, "ddt_heatmap.png"),
                         ddt_result=precomputed.get("ddt")),
        plot_lat_heatmap(sbox, os.path.join(output_dir, "lat_heatmap.png"),
                         lat_result=precomputed.get("lat")),
        plot_sbox_mapping(sbox, os.path.join(output_dir, "sbox_mapping.png")),
    ]

    print("\nAll figures generated successfully.")
    return paths
