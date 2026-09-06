"""Train one baseline scheme and apply the spectral-gap readout in the same run.

The exact solution enters only at the scoring fence at the bottom (pre/post
errors and the span diagnostic); the readout in sgr.py never sees it. Each
weight rule keeps the update ordering of its own reference implementation (see
weighting.py): ntk_trace and lr_anneal refresh periodically after the Adam
step, ReLoBRaLo rebalances from the current losses so the step already uses
the new weights.
"""
import time

import numpy as onp
import jax
import jax.numpy as jnp
import optax
from jax import jit, grad, random

import pdes
import sgr
from model import make_mlp, build_fields
from weighting import ntk_weights, make_lr_anneal, ReLoBRaLo, SelfAdaptive

N_R, N_IC, N_BC = 4096, 256, 256
NTK_EVERY = 100          # ntk_trace refresh cadence, as in its reference
ANNEAL_EVERY = 10        # lr_anneal refresh cadence, as in its reference
ANNEAL_BETA = 0.9        # lr_anneal EMA memory on the old value


def _rel_l2(u_pred, u_ref):
    return float(onp.linalg.norm(u_pred - u_ref) / (onp.linalg.norm(u_ref) + 1e-30))


def run(pde, method, seed=0, iters=30000, coef=None, weights=None,
        dump_field=None, return_state=False):
    """Train `method` on `pde` for `iters` Adam steps, then apply the SGR readout.

    `weights` pins the loss weights to a constant (lambda_ic, lambda_bc) in place
    of the rule, which is what the constant-weight sweeps train on.  Under
    `return_state` the trained state comes back too, keyed fields, params,
    post_params, batch and grid.
    """
    pde = pdes.canon(pde)
    name = pdes.short(pde)
    coef = pdes.DEFAULT_COEF[pde] if coef is None else float(coef)
    start_time = time.time()

    params, apply_fn = make_mlp(seed=seed)
    fields = build_fields(apply_fn, pde, coef)
    lr = optax.exponential_decay(1e-3, transition_steps=2000, decay_rate=0.9)
    tx = optax.adam(lr)
    opt_state = tx.init(params)

    def losses(params, batch):
        t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_r, x_r = batch
        L_r = jnp.mean(fields.r_pred(params, t_r, x_r) ** 2)
        L_ic = jnp.mean((fields.u_pred(params, t_ic, x_ic) - u_ic) ** 2)
        if fields.has_ut_ic:                    # second order in time: u_t(0, x) = 0
            L_ic += jnp.mean(fields.u_t_pred(params, t_ic, x_ic) ** 2)
        L_bc = jnp.mean((fields.u_pred(params, t_bc, x_bc) - u_bc) ** 2)
        return L_r, L_ic, L_bc

    group_losses = jit(losses)

    @jit
    def step(params, opt_state, batch, w_r, w_ic, w_bc):
        def lossfn(p):
            L_r, L_ic, L_bc = losses(p, batch)
            return w_r * L_r + w_ic * L_ic + w_bc * L_bc

        g = grad(lossfn)(params)
        upd, opt_state = tx.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state

    sa = SelfAdaptive(fields, pde, N_R, N_IC, seed) if method == 'sa_pinn' else None
    tx_w = optax.adam(lr)

    @jit
    def step_sa(params, opt_state, w, w_state, batch):
        g_p, g_w = grad(sa.loss, argnums=(0, 1))(params, w, batch)
        upd, opt_state = tx.update(g_p, opt_state, params)
        params = optax.apply_updates(params, upd)
        upd_w, w_state = tx_w.update(jax.tree.map(lambda a: -a, g_w), w_state, w)
        return params, opt_state, optax.apply_updates(w, upd_w), w_state

    key = random.PRNGKey(seed)
    # resolve batch for the readout; never used for training, so the resolve is
    # out of sample for every method (SA-PINN trains on its own fixed set below)
    fixed_batch = pdes.sample_batch(random.PRNGKey(seed + 777), pde, coef, N_R, N_IC, N_BC)
    lr_rule = make_lr_anneal(fields) if method == 'lr_anneal' else None
    relo = ReLoBRaLo(seed=seed) if method == 'relobralo' else None
    if method == 'sa_pinn':
        # the reference keeps SA-PINN on one training set for the whole run
        sa_batch = pdes.sample_batch(random.PRNGKey(seed + 888), pde, coef, N_R, N_IC, N_BC)
        w = sa.w
        w_state = tx_w.init(w)
    lam_ic = lam_bc = 1.0
    if weights is not None:
        # an adaptive rule would overwrite these within a few iterations
        assert method == 'vanilla', f'constant weights need method=vanilla, got {method}'
        lam_ic, lam_bc = float(weights[0]), float(weights[1])
    peak_lam = max(lam_ic, lam_bc)

    for it in range(iters):
        if method == 'sa_pinn':
            params, opt_state, w, w_state = step_sa(params, opt_state, w, w_state, sa_batch)
            continue
        key, sub = random.split(key)
        batch = pdes.sample_batch(sub, pde, coef, N_R, N_IC, N_BC)
        w_r = 1.0
        if method == 'relobralo':
            lam = relo.update([float(a) for a in group_losses(params, batch)])
            w_r, lam_ic, lam_bc = float(lam[0]), float(lam[1]), float(lam[2])
        params, opt_state = step(params, opt_state, batch, w_r, lam_ic, lam_bc)
        if method == 'ntk_trace' and it % NTK_EVERY == 0:
            lam_ic, lam_bc = ntk_weights(fields, params, batch)
        elif method == 'lr_anneal' and it % ANNEAL_EVERY == 0:
            li, lb = lr_rule(params, batch, lam_ic, lam_bc)
            lam_ic = (1 - ANNEAL_BETA) * li + ANNEAL_BETA * lam_ic
            lam_bc = (1 - ANNEAL_BETA) * lb + ANNEAL_BETA * lam_bc
        peak_lam = max(peak_lam, lam_ic, lam_bc)

    # scoring fence: the exact solution enters here and only here
    TT, XX, u_ref = pdes.eval_grid(pde, coef, 100)
    t_flat = jnp.asarray(TT.reshape(-1)); x_flat = jnp.asarray(XX.reshape(-1))
    u_flat = onp.asarray(u_ref).reshape(-1)
    pre_L2 = _rel_l2(onp.asarray(fields.u_pred(params, t_flat, x_flat)), u_flat)

    new_params, diag = sgr.sgr_readout(fields, params, fixed_batch)
    post_L2 = _rel_l2(onp.asarray(fields.u_pred(new_params, t_flat, x_flat)), u_flat)
    if dump_field:
        onp.savez(dump_field, u_pred=onp.asarray(fields.u_pred(params, t_flat, x_flat)),
                  u_post=onp.asarray(fields.u_pred(new_params, t_flat, x_flat)),
                  u_exact=u_flat, TT=onp.asarray(TT), XX=onp.asarray(XX),
                  rel_l2=pre_L2, rel_l2_post=post_L2)

    # span floor: best rel-L2 any last layer reaches on the frozen backbone
    A_grid = onp.asarray(sgr.features(params, t_flat, x_flat), onp.float64)
    u64 = onp.asarray(u_flat, onp.float64)
    c, *_ = onp.linalg.lstsq(A_grid, u64, rcond=None)
    floor_test = _rel_l2(A_grid @ c, u64)

    result = {'pde': name, 'method': method, 'seed': int(seed), 'iters': int(iters),
              'coef': coef, 'pre_L2': pre_L2, 'post_L2': post_L2,
              'floor_test': floor_test, 'lam_hat': float(diag['lam_hat']),
              'valley_contrast': diag['valley_contrast'], 'leakage': diag['leakage'],
              'wall_time_s': time.time() - start_time,
              # the loss weights act on group means and the frozen system on raw
              # sums, so the two conventions differ by n_r / n_group
              'n_r': N_R, 'n_ic': N_IC, 'n_bc': N_BC}
    if method == 'sa_pinn':
        # per-point weights: no scalar lambda to record, so those keys stay out
        result['sa_ratio_final'] = sa.ratio(w)
        result['sa_weight_means'] = {'res': float(jnp.mean(w[0])),
                                     'ic': float(jnp.mean(w[1])) if sa.weighted_ic else None}
    else:
        result.update(peak_lambda=peak_lam, final_lam_ic=lam_ic, final_lam_bc=lam_bc)
    if method == 'relobralo':
        lam = relo.lambdas
        result['relo_weights'] = {'res': float(lam[0]), 'ic': float(lam[1]),
                                  'bc': float(lam[2])}
        # the other rules hold the residual weight at 1, so divide through to put
        # the recorded condition weights on the same footing
        result['final_lam_ic'] = float(lam[1]) / float(lam[0])
        result['final_lam_bc'] = float(lam[2]) / float(lam[0])
    if return_state:
        state = {'fields': fields, 'params': params, 'post_params': new_params,
                 'batch': fixed_batch, 'grid': (TT, XX, u_ref)}
        return result, state
    return result
