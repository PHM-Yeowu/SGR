"""SGR, the spectral-gap readout.

A post-hoc last-layer readout: on the frozen backbone we assemble the
last-layer Gauss-Newton system, select the readout weight lambda at the
velocity dip, recompute the head there, and replace the last layer.  No
retraining.  The one weight introduced is the readout lambda; the training
weights stay the baseline's business.  Everything here is solution-free:
A_r / A_c / A_test come from the operator and the network, b_r from the
residual, b_c from the given condition data.
"""
from dataclasses import dataclass
from typing import Callable
import math

import numpy as onp
import jax.numpy as jnp
from jax import jacrev, vmap
from jax.flatten_util import ravel_pytree

_EPS = 1.5e-12          # relative ridge floor, in units of tr(M)/p
# sqrt(machine eps) ~ 1.5e-8 lands above 40-90% of the spectrum of M and filters
# the solution rather than stabilizing it; this floor stays below the informative
# modes on every benchmark and still holds off the rank deficiency an
# unregularized solve hits.
_H_LN = 0.1                                     # symmetric step in natural-log lambda


@dataclass
class Sys:
    """Frozen last-layer system (host float64): normal-equation blocks + output map."""
    ArtAr: onp.ndarray; ActAc: onp.ndarray; Art_br: onp.ndarray; Act_bc: onp.ndarray
    A_test: onp.ndarray; unravel: Callable; p: int


def features(params, t, x):
    """Backbone features of the last layer, the rows of u = Phi theta.

    The last layer is linear, so d u / d theta is the feature vector itself with
    a 1 appended for the bias; the cost is one backbone evaluation, with nothing
    to differentiate."""
    head = params[:-1]

    def row(t_i, x_i):
        h = jnp.array([t_i, x_i])
        for W, b in head:
            h = jnp.tanh(jnp.dot(h, W) + b)
        return jnp.append(h, 1.0)

    return vmap(row)(t, x)


def rows(field_fn, params, t, x):
    """Rows A_i = d field(t_i, x_i) / d theta with the backbone frozen.

    Only the residual and the u_t condition need this: they carry derivatives in
    (t, x), so the row is the Gauss-Newton row rather than the feature vector.
    The plain u rows are features()."""
    head, theta0 = params[:-1], params[-1]

    def row(t_i, x_i):
        g = jacrev(lambda th: field_fn(head + [th], t_i, x_i))(theta0)
        flat, _ = ravel_pytree(g)
        return flat

    return vmap(row)(t, x)


def build_system(fields, params, batch):
    """Assemble Sys on the trained features.  A_test is the output map on the
    interior points, and b_r = A_r theta0 - r0 is the GN target that drives the
    residual to zero."""
    t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_r, x_r = batch
    theta0_flat, unravel = ravel_pytree(params[-1])
    theta0 = onp.asarray(theta0_flat, onp.float64)
    p = theta0.shape[0]
    phi = lambda t, x: onp.asarray(features(params, t, x), onp.float64)
    A_r = onp.asarray(rows(fields.residual_net, params, t_r, x_r), onp.float64)
    A_test = phi(t_r, x_r)
    A_c = [phi(t_ic, x_ic)]
    b_c = [onp.asarray(u_ic, onp.float64)]
    if fields.has_ut_ic:                        # second order in time: u_t(0,x) = 0
        A_c.append(onp.asarray(rows(fields.u_t_net, params, t_ic, x_ic), onp.float64))
        b_c.append(onp.zeros(len(t_ic)))
    A_c.append(phi(t_bc, x_bc))
    b_c.append(onp.asarray(u_bc, onp.float64))
    A_c = onp.concatenate(A_c, 0); b_c = onp.concatenate(b_c)
    r0 = onp.asarray(fields.r_pred(params, t_r, x_r), onp.float64)
    b_r = A_r @ theta0 - r0
    return Sys(A_r.T @ A_r, A_c.T @ A_c, A_r.T @ b_r, A_c.T @ b_c, A_test, unravel, p)


def ridge_solve(s, lam):
    M = s.ArtAr + lam * s.ActAc
    M = M + _EPS * (onp.trace(M) / s.p) * onp.eye(s.p)
    return onp.linalg.solve(M, s.Art_br + lam * s.Act_bc)


def velocity(s, t):
    """g(t) = ||A_test dc/dln(lam)|| at lam = 10^t (symmetric log step)."""
    lam = 10.0 ** t
    dc = ridge_solve(s, lam * math.exp(_H_LN)) - ridge_solve(s, lam * math.exp(-_H_LN))
    return float(onp.linalg.norm(s.A_test @ dc) / (2.0 * _H_LN))


def select_lambda(s, lo=0.0, hi=8.0, n_coarse=20, tol=1e-3):
    """Readout weight at the velocity dip, by coarse bracketing then golden section.

    g -> 0 at both ends, where the solve saturates to interior-only and to
    constraint-only, so the tail zeros are spurious and the quasi-optimal lambda
    is the interior valley floor.  The coarse pass brackets an interior local
    minimum over n_coarse nodes, endpoints excluded so the monotone tails cannot
    be picked; golden section then refines it on the smooth g(t)."""
    ts = onp.linspace(lo, hi, n_coarse)
    g = onp.array([velocity(s, t) for t in ts])
    interior = [j for j in range(1, n_coarse - 1) if g[j] <= g[j - 1] and g[j] <= g[j + 1]]
    j_star = (min(interior, key=lambda j: g[j]) if interior
              else 1 + int(onp.argmin(g[1:-1])))          # fallback still excludes the tails
    a, b = ts[j_star - 1], ts[j_star + 1]
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c1, c2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2 = velocity(s, c1), velocity(s, c2)
    while b - a > tol:
        if f1 < f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - phi * (b - a); f1 = velocity(s, c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + phi * (b - a); f2 = velocity(s, c2)
    return 10.0 ** (0.5 * (a + b))


def dip_quality(s, lam_hat, lo=0.0, hi=8.0, n=12):
    """Contrast (median off-valley velocity / valley velocity) and leakage
    (valley / peak; small = clean dip)."""
    g = onp.array([velocity(s, t) for t in onp.linspace(lo, hi, n)])
    g_hat = velocity(s, math.log10(lam_hat))
    return float(onp.median(g) / (g_hat + 1e-30)), float(g_hat / (g.max() + 1e-30))


def sgr_readout(fields, params, batch):
    s = build_system(fields, params, batch)
    lam_hat = select_lambda(s)
    contrast, leak = dip_quality(s, lam_hat)
    c = ridge_solve(s, lam_hat)
    new_params = params[:-1] + [s.unravel(jnp.asarray(c, jnp.float32))]  # backbone untouched
    return new_params, {'lam_hat': lam_hat, 'valley_contrast': contrast, 'leakage': leak}
