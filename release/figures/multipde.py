#!/usr/bin/env python3
"""Main comparison figure: interior relative L2 error of every weighting scheme
on the seven benchmarks, before (light) and after (dark) the spectral-gap
readout, mean over five seeds.  Reads results/runs/{pde}_{scheme}_seed*.json;
writes figures/multipde.{pdf,png}."""
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # release/
sys.path.insert(0, os.path.join(_R, 'src'))
os.chdir(os.path.join(_R, '..'))                                          # repository root
import glob
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

PDES = ['kg', 'helmholtz', 'heat', 'convection_diffusion', 'burgers', 'advection', 'allen_cahn']
PDE_LAB = ['Klein--Gordon', 'Helmholtz', 'Heat', 'Conv.--diff.', 'Burgers', 'Advection', 'Allen--Cahn']
SCHEMES = ['vanilla', 'ntk_trace', 'lr_anneal', 'sa_pinn', 'relobralo']
SCH_SHORT = ['Van.', 'NTK', 'LR.', 'SA.', 'ReLo.']
C_PRE, C_POST = '#bdbdbd', '#08519c'


def mean_over_seeds(pde, sch, key):
    xs = []
    for f in glob.glob(f'results/runs/{pde}_{sch}_seed*.json'):
        v = json.load(open(f))[key]
        if np.isfinite(v):                       # a diverged run records nan
            xs.append(v)
    return float(np.mean(xs)) if xs else np.nan


plt.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
    "font.size": 7.5, "axes.titlesize": 8.4, "axes.labelsize": 8.0, 'axes.linewidth': 1.2,
    'axes.edgecolor': 'black', 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.top': False, 'ytick.right': False, 'xtick.major.size': 4, 'ytick.major.size': 4,
    'axes.grid': False, 'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

nS, nP = len(SCHEMES), len(PDES)
stride = nS + 1.6
fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.set_yscale('log')
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
vals = []
for i, p in enumerate(PDES):
    for j, sc in enumerate(SCHEMES):
        pre, post = mean_over_seeds(p, sc, 'pre_L2'), mean_over_seeds(p, sc, 'post_L2')
        pos = i * stride + j
        if np.isfinite(pre):
            ax.bar(pos, pre, 0.80, color=C_PRE, zorder=2); vals.append(pre)
        if np.isfinite(post):
            ax.bar(pos, post, 0.80, color=C_POST, zorder=3); vals.append(post)
        ax.text(pos, 0.02, SCH_SHORT[j], rotation=90, fontsize=6.6,
                ha='center', va='bottom', color='white', transform=trans, zorder=6)
ax.set_xticks([])
ax.set_xlim(-0.9, (nP - 1) * stride + nS - 0.1)
ax.set_ylim(min(vals) / 2.5, max(vals) * 8)
ax.set_ylabel(r'Interior rel.\ $L^2$ error')
ax.spines[['top', 'right']].set_visible(False)
for i, lab in enumerate(PDE_LAB):
    ax.text(i * stride + (nS - 1) / 2, -0.10, lab, transform=trans,
            ha='center', va='top', fontsize=7.5)
for i in range(1, nP):
    x = i * stride - 1.3
    ax.plot([x, x], [0.0, 1.0], transform=trans, color='0.7', lw=0.7,
            clip_on=False, zorder=1)
ax.legend(handles=[Patch(fc=C_PRE, label='Trained model (before SGR)'),
                   Patch(fc=C_POST, label='After SGR readout')],
          ncol=2, loc='upper center', fontsize=7.5,
          frameon=True, facecolor='white', edgecolor='0.7', framealpha=1.0)
fig.subplots_adjust(bottom=0.16)
for ext in ('pdf', 'png'):
    fig.savefig(f'release/figures/multipde.{ext}')
print('wrote release/figures/multipde.pdf')
