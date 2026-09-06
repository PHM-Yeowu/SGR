"""Backbone MLP and the differentiable PINN fields.

The PINN output is u = Phi_w(t, x) . theta: a tanh MLP backbone with feature
map Phi (parameters w) and a linear last layer theta in R^p (p = 257 with the
bias).  The readout in sgr.py freezes Phi_w and recomputes theta alone.
"""
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax import grad, vmap, random

import pdes


def make_mlp(layers=(2, 256, 256, 256, 256, 1), seed=0):
    """tanh MLP; params is a list of (W, b) whose last entry is the readout theta."""
    keys = random.split(random.PRNGKey(seed), len(layers) - 1)
    params = []
    for k, d_in, d_out in zip(keys, layers[:-1], layers[1:]):
        std = 1.0 / jnp.sqrt((d_in + d_out) / 2.0)      # Glorot normal
        params.append((std * random.normal(k, (d_in, d_out)), jnp.zeros(d_out)))

    def apply_fn(params, x):
        h = x
        for W, b in params[:-1]:
            h = jnp.tanh(jnp.dot(h, W) + b)
        W, b = params[-1]
        return jnp.dot(h, W) + b

    return params, apply_fn


@dataclass
class Fields:
    """Vectorized differentiable fields for one (pde, coef)."""
    u_net: Callable          # u(params, t, x)               (scalar)
    u_t_net: Callable        # d u / d t                     (scalar)
    residual_net: Callable   # L[u] - f                      (scalar)
    u_pred: Callable         # vmapped u_net
    u_t_pred: Callable       # vmapped u_t_net
    r_pred: Callable         # vmapped residual_net
    has_ut_ic: bool


def build_fields(apply_fn, pde, coef):
    pde = pdes.canon(pde)

    def u_net(params, t, x):
        return apply_fn(params, jnp.array([t, x]))[0]

    def u_t_net(params, t, x):
        return grad(u_net, argnums=1)(params, t, x)

    def residual_net(params, t, x):
        u = u_net(params, t, x)
        u_t = grad(u_net, 1)(params, t, x)
        u_x = grad(u_net, 2)(params, t, x)
        u_tt = grad(grad(u_net, 1), 1)(params, t, x)
        u_xx = grad(grad(u_net, 2), 2)(params, t, x)
        # forcing f = L[u*] by autodiff of the manufactured solution, so residual(u*) = 0
        ue_fn = lambda ti, xi: pdes.exact(pde, ti, xi, coef)
        ue, ue_t, ue_x = ue_fn(t, x), grad(ue_fn, 0)(t, x), grad(ue_fn, 1)(t, x)
        ue_tt = grad(grad(ue_fn, 0), 0)(t, x)
        ue_xx = grad(grad(ue_fn, 1), 1)(t, x)
        f = pdes.operator(pde, ue, ue_t, ue_x, ue_tt, ue_xx, coef)
        return pdes.operator(pde, u, u_t, u_x, u_tt, u_xx, coef) - f

    return Fields(
        u_net=u_net, u_t_net=u_t_net, residual_net=residual_net,
        u_pred=vmap(u_net, (None, 0, 0)),
        u_t_pred=vmap(u_t_net, (None, 0, 0)),
        r_pred=vmap(residual_net, (None, 0, 0)),
        has_ut_ic=pde in pdes.HAS_UT_IC)
