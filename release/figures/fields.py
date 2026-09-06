#!/usr/bin/env python3
"""Per-PDE field figures: rows = weighting schemes, columns = exact solution,
prediction before / after the SGR readout, and pointwise absolute error before /
after.  The three solution panels of a row share one colour scale; the two error
panels share a second.  Reads results/fields/{pde}_{method}_seed1.npz; writes
figures/fields_{pde}.{pdf,png}."""
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # release/
sys.path.insert(0, os.path.join(_R, 'src'))
os.chdir(os.path.join(_R, '..'))                                          # repository root
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 1
PDES = ['kg', 'helmholtz', 'heat', 'convection_diffusion', 'burgers', 'advection', 'allen_cahn']
SCH = [('vanilla', 'Vanilla'), ('ntk_trace', 'NTK weight'), ('lr_anneal', 'LR-annealing'),
       ('sa_pinn', 'SA-PINN'), ('relobralo', 'ReLoBRaLo')]
COLHEAD = [r'Exact $u^\star$', 'Prediction\n(before SGR)', 'Prediction\n(after SGR)',
           'Error\n(before SGR)', 'Error\n(after SGR)']

plt.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
    "font.size": 9.8, "axes.titlesize": 11.0, "axes.labelsize": 10.4, 'axes.linewidth': 0.8,
    'axes.edgecolor': 'black', 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.top': False, 'ytick.right': False, 'axes.grid': False,
    'savefig.dpi': 200, 'savefig.bbox': 'tight', 'image.cmap': 'jet',
})


def show(ax, TT, XX, Z, vmin, vmax):
    return ax.pcolormesh(TT, XX, Z, shading='gouraud', rasterized=True, vmin=vmin, vmax=vmax)


for pde in PDES:
    # the Helmholtz benchmark is steady: its coordinate pair is (x, y)
    xlab, ylab = (r'$x$', r'$y$') if pde == 'helmholtz' else (r'$t$', r'$x$')
    fig, axes = plt.subplots(len(SCH), 5, figsize=(7.4, 1.28 * len(SCH) + 0.35),
                             constrained_layout=True)
    for i, (m, mlab) in enumerate(SCH):
        d = np.load(f'results/fields/{pde}_{m}_seed{SEED}.npz')
        TT, XX = d['TT'], d['XX']; shape = TT.shape
        u_ex = d['u_exact'].reshape(shape)
        u_pre = d['u_pred'].reshape(shape)
        u_post = d['u_post'].reshape(shape)
        err_pre, err_post = np.abs(u_pre - u_ex), np.abs(u_post - u_ex)
        vmin, vmax = float(u_ex.min()), float(u_ex.max())
        err_max = float(max(err_pre.max(), err_post.max()))
        im_u = show(axes[i, 0], TT, XX, u_ex, vmin, vmax)
        show(axes[i, 1], TT, XX, u_pre, vmin, vmax)
        show(axes[i, 2], TT, XX, u_post, vmin, vmax)
        im_e = show(axes[i, 3], TT, XX, err_pre, 0.0, err_max)
        show(axes[i, 4], TT, XX, err_post, 0.0, err_max)
        fig.colorbar(im_u, ax=axes[i, 2], fraction=0.05, pad=0.02)
        fig.colorbar(im_e, ax=axes[i, 4], fraction=0.05, pad=0.02)
        for k, v in ((3, float(d['rel_l2'])), (4, float(d['rel_l2_post']))):
            axes[i, k].text(0.03, 0.06, rf'{v:.1e}', transform=axes[i, k].transAxes,
                            fontsize=8.7, color='white',
                            bbox=dict(boxstyle='round,pad=0.15', fc='0.15', ec='none', alpha=0.6))
        axes[i, 0].set_ylabel(mlab + r',\ \ ' + ylab)
        for k in range(5):
            axes[i, k].set_xticks([0, 1]); axes[i, k].set_yticks([0, 1])
            if i == 0:
                axes[i, k].set_title(COLHEAD[k], fontsize=9.8)
            if i < len(SCH) - 1:
                axes[i, k].set_xticklabels([])
    for ax in axes[-1]:
        ax.set_xlabel(xlab)
    for ext in ('pdf', 'png'):
        fig.savefig(f'release/figures/fields_{pde}.{ext}')
    plt.close(fig)
    print(f'wrote release/figures/fields_{pde}.pdf')
