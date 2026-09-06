"""Benchmark definitions: seven PDEs with manufactured exact solutions.

Each benchmark is posed on the unit square with a manufactured solution u*,
and the forcing f = L[u*] comes from automatic differentiation (model.py),
so the residual of u* vanishes identically.  The condition targets are the
true traces of u* on the corresponding parts of the boundary.

For the six evolution equations the coordinates are (t, x) in [0,1]^2 and the
two condition groups are the initial slice t = 0 and the spatial boundary
x in {0, 1}.  The steady Helmholtz benchmark runs through the same code path
with the coordinate pair read as (x, y); its two groups are the edge pairs
x in {0, 1} and y in {0, 1}, i.e. the full Dirichlet boundary.

Coefficients follow the reference implementations: Helmholtz modes (1, 4) with
lambda = 1 [Wang et al. 2021; McClenny & Braga-Neto 2023], Burgers viscosity
0.01/pi [Raissi et al. 2019], Allen-Cahn 1e-4 u_xx and 5(u^3 - u) [McClenny &
Braga-Neto 2023].
"""
import math

import numpy as onp
import jax.numpy as jnp
from jax import random

ALIAS = {'kg': 'klein_gordon', 'cd': 'convection_diffusion', 'ac': 'allen_cahn'}
PDES = ['klein_gordon', 'helmholtz', 'heat', 'convection_diffusion',
        'burgers', 'advection', 'allen_cahn']

# second order in time, so the initial data additionally pins u_t(0, x) = 0
HAS_UT_IC = {'klein_gordon'}

# the one free coefficient of each equation: Klein-Gordon cubic strength, Helmholtz
# second mode a_2, heat diffusivity, convection and advection speed, Burgers
# viscosity, Allen-Cahn interface width
DEFAULT_COEF = {'klein_gordon': 1.0, 'helmholtz': 4.0, 'heat': 0.1,
                'convection_diffusion': 1.5, 'burgers': 0.01 / math.pi,
                'advection': 1.0, 'allen_cahn': 1e-4}

_NU_CD = 0.05      # convection-diffusion viscosity (`coef` is the convection speed)
_LAM_HELM = 1.0    # Helmholtz lambda (`coef` is the second mode number a_2)


def canon(pde):
    return ALIAS.get(pde, pde)


def short(pde):
    pde = canon(pde)
    return 'kg' if pde == 'klein_gordon' else pde


def exact(pde, t, x, coef):
    """Manufactured solution u*; elementwise, and differentiable in (t, x)."""
    pde = canon(pde)
    if pde == 'klein_gordon':
        return x * jnp.cos(5 * jnp.pi * t) + (t * x) ** 3
    if pde == 'helmholtz':
        # steady benchmark: (t, x) is the spatial pair (x, y); modes (1, coef)
        return jnp.sin(jnp.pi * t) * jnp.sin(coef * jnp.pi * x)
    if pde == 'heat':
        return jnp.exp(-t) * (jnp.sin(jnp.pi * x) + 0.3 * jnp.sin(6 * jnp.pi * x))
    if pde == 'convection_diffusion':
        return (jnp.exp(-0.5 * t) * jnp.sin(2 * jnp.pi * x)
                + 0.3 * jnp.sin(5 * jnp.pi * x) * jnp.cos(jnp.pi * t))
    if pde == 'burgers':
        return jnp.exp(-t) * (jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.sin(jnp.pi * x))
    if pde == 'advection':
        return (jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * t)
                + 0.3 * jnp.exp(-t) * jnp.sin(4 * jnp.pi * x))
    if pde == 'allen_cahn':
        return jnp.exp(-t) * jnp.sin(jnp.pi * x)
    raise ValueError(pde)


def operator(pde, u, u_t, u_x, u_tt, u_xx, coef):
    """PDE operator L[u]; the residual is L[u] - f with f = L[u*]."""
    pde = canon(pde)
    if pde == 'klein_gordon':
        return u_tt - u_xx + coef * u ** 3
    if pde == 'helmholtz':                          # Delta u + lam u, with (t, x) = (x, y)
        return u_tt + u_xx + _LAM_HELM * u
    if pde == 'heat':
        return u_t - coef * u_xx
    if pde == 'convection_diffusion':
        return u_t + coef * u_x - _NU_CD * u_xx
    if pde == 'burgers':
        return u_t + u * u_x - coef * u_xx
    if pde == 'advection':
        return u_t + coef * u_x
    if pde == 'allen_cahn':
        return u_t - coef * u_xx + 5.0 * (u ** 3 - u)
    raise ValueError(pde)


def sample_batch(key, pde, coef, N_r=4096, N_ic=256, N_bc=256):
    """One training batch, ordered (t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_r, x_r)."""
    pde = canon(pde)
    k1, k2, k3, k4 = random.split(key, 4)
    # group 1: initial slice t = 0 (Helmholtz: the two edges t in {0, 1})
    x_ic = random.uniform(k1, (N_ic,))
    if pde == 'helmholtz':
        n = N_ic // 2
        t_ic = jnp.concatenate([jnp.zeros(n), jnp.ones(N_ic - n)])
    else:
        t_ic = jnp.zeros(N_ic)
    u_ic = exact(pde, t_ic, x_ic, coef)
    # group 2: spatial boundary x in {0, 1}
    n = N_bc // 2
    t_bc = jnp.concatenate([random.uniform(k2, (n,)), random.uniform(k3, (N_bc - n,))])
    x_bc = jnp.concatenate([jnp.zeros(n), jnp.ones(N_bc - n)])
    u_bc = exact(pde, t_bc, x_bc, coef)
    # interior residual points
    tx = random.uniform(k4, (N_r, 2))
    return t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, tx[:, 0], tx[:, 1]


def eval_grid(pde, coef, n=100):
    # regular n x n grid; used for scoring only, never sampled during training
    g = onp.linspace(0.0, 1.0, n)
    TT, XX = onp.meshgrid(g, g)
    return TT, XX, onp.asarray(exact(canon(pde), TT, XX, coef))
