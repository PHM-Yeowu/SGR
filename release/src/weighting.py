"""Training-weight rules for the five baseline schemes, all solution-free.

These set the training weights in L = L_r + lambda_ic L_ic + lambda_bc L_bc, not
the readout weight of SGR, which runs post hoc on whichever baseline trained.
Each rule follows the authors' released code:

  vanilla   : constant (1, 1)                                (inline in train.py)
  ntk_trace : trace-balanced NTK weights                     (Wang et al. 2022)
  lr_anneal : gradient-statistics annealing                  (Wang et al. 2021)
  relobralo : softmax balancing of relative loss changes     (Bischof & Kraus 2021)
  sa_pinn   : per-point ascent weights                       (McClenny & Braga-Neto 2023)
"""
import jax.numpy as jnp
from jax import grad, jit, random, vmap
from jax.flatten_util import ravel_pytree
from jax.nn import softmax

N_SUB = 64      # points per group in the trace estimate


def trace_ntk(field_fn, params, t, x):
    """Tr(K) = sum_i ||d field(t_i, x_i) / d theta||^2 over every weight and bias.

    The summed diagonal of the empirical NTK block.  The reference builds the
    full Jacobian and takes tr(J J^T), which is the same number without the
    (n x p) array."""
    def sq(t_i, x_i):
        g = grad(field_fn)(params, t_i, x_i)
        flat, _ = ravel_pytree(g)
        return jnp.sum(flat ** 2)

    return jnp.sum(vmap(sq)(t, x))


def ntk_weights(fields, params, batch, trace=trace_ntk):
    """(lambda_ic, lambda_bc) = Tr_rr / Tr_g on a point subsample.

    The reference sets lambda_g = Tr_total / Tr_g on every term, the residual
    included; dividing through by lambda_r leaves the same relative weighting
    with the residual weight pinned at 1.  On the second-order-in-time benchmark
    Tr_ic folds in the u_t trace, which the reference keeps as a group of its
    own, so that every rule runs in the three-group split of the paper.  trace
    picks the kernel: the reference one by default, the last-layer one the
    theory is posed in for the analyses.
    """
    t_ic, x_ic, _, t_bc, x_bc, _, t_r, x_r = batch

    def sub(t, x):
        # evenly spread subsample: the condition groups are ordered edge
        # concatenations, so a prefix slice would cover one edge only
        idx = jnp.linspace(0, len(t) - 1, N_SUB).round().astype(int)
        return t[idx], x[idx]

    tr = lambda fn, t, x: float(trace(fn, params, t, x))

    Tr_rr = tr(fields.residual_net, *sub(t_r, x_r))
    Tr_ic = tr(fields.u_net, *sub(t_ic, x_ic))
    if fields.has_ut_ic:
        Tr_ic += tr(fields.u_t_net, *sub(t_ic, x_ic))
    Tr_bc = tr(fields.u_net, *sub(t_bc, x_bc))
    return Tr_rr / (Tr_ic + 1e-12), Tr_rr / (Tr_bc + 1e-12)


def make_lr_anneal(fields):
    """Learning-rate annealing (Wang et al. 2021, released code).

    lambda_hat_g = max|grad L_r| / (lambda_g mean|grad L_g|), the instantaneous
    ratio against the already-weighted group loss; the caller then applies the
    moving average lambda_g <- 0.1 lambda_hat_g + 0.9 lambda_g every 10
    iterations.
    """
    @jit
    def grad_stats(params, batch):
        t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_r, x_r = batch
        L_r = lambda p: jnp.mean(fields.r_pred(p, t_r, x_r) ** 2)

        def L_ic(p):
            L = jnp.mean((fields.u_pred(p, t_ic, x_ic) - u_ic) ** 2)
            return L + jnp.mean(fields.u_t_pred(p, t_ic, x_ic) ** 2) if fields.has_ut_ic else L
        L_bc = lambda p: jnp.mean((fields.u_pred(p, t_bc, x_bc) - u_bc) ** 2)
        g_r, g_ic, g_bc = grad(L_r)(params), grad(L_ic)(params), grad(L_bc)(params)
        # weight matrices only, biases dropped: max over all entries of all
        # layers, mean of the per-layer means
        mx = jnp.max(jnp.array([jnp.max(jnp.abs(W)) for W, _ in g_r]))
        m_ic = jnp.mean(jnp.array([jnp.mean(jnp.abs(W)) for W, _ in g_ic]))
        m_bc = jnp.mean(jnp.array([jnp.mean(jnp.abs(W)) for W, _ in g_bc]))
        return mx, m_ic, m_bc

    def rule(params, batch, lam_ic, lam_bc):
        mx, m_ic, m_bc = grad_stats(params, batch)
        return (float(mx) / (lam_ic * float(m_ic) + 1e-12),
                float(mx) / (lam_bc * float(m_bc) + 1e-12))

    return rule


class ReLoBRaLo:
    """Relative Loss Balancing with Random Lookback (Bischof & Kraus 2021).

    n loss terms with the residual included, lambda_hat = softmax(L_i / (l_i T +
    1e-12)) * n against the previous step's losses l_i, a lookback lambda0_hat
    against the losses saved after step 1, a Bernoulli(rho) switch between the
    two, and the released alpha warm-up (1 at step 0, then 0 at step 1, a hard
    reset to the instantaneous softmax, then alpha).  At the reference defaults
    alpha = 0.999, T = 1, rho = 1 the switch never fires, so the lookback term
    stays inactive; rho is the authors' knob.
    """

    def __init__(self, alpha=0.999, T=1.0, rho=1.0, n=3, seed=0):
        self.alpha_final, self.T, self.rho, self.n = alpha, T, rho, n
        self.key = random.PRNGKey(seed)
        self.lambdas = jnp.ones(n)
        self.l = jnp.ones(n)        # previous-step losses (reference init: 1)
        self.l0 = jnp.ones(n)       # lookback baseline (snapshot after step 1)
        self.step = 0

    def update(self, losses):
        L = jnp.asarray(losses)
        alpha = 1.0 if self.step == 0 else (0.0 if self.step == 1 else self.alpha_final)
        lam_hat = softmax(L / (self.l * self.T + 1e-12)) * self.n
        lam0_hat = softmax(L / (self.l0 * self.T + 1e-12)) * self.n
        self.key, subkey = random.split(self.key)
        rho_t = (random.uniform(subkey) < self.rho).astype(jnp.float32)
        self.lambdas = (rho_t * alpha * self.lambdas
                        + (1.0 - rho_t) * alpha * lam0_hat
                        + (1.0 - alpha) * lam_hat)
        self.l = L
        if self.step == 1:
            self.l0 = L
        self.step += 1
        return self.lambdas


class SelfAdaptive:
    """Self-adaptive PINN weights (McClenny & Braga-Neto 2023, released code).

    One weight per collocation point and per initial-condition point, multiplied
    inside the square and raised by gradient ascent on the same loss through
    their own optimizer.  The boundary group stays unweighted, and the steady
    Helmholtz benchmark, which has no genuine initial condition, carries weights
    on the residual only, as in the reference examples.
    """

    def __init__(self, fields, pde, n_r, n_ic, seed):
        self.fields = fields
        self.weighted_ic = fields.has_ut_ic or pde != 'helmholtz'
        k_r, k_ic = random.split(random.PRNGKey(seed + 555), 2)
        self.w = (random.uniform(k_r, (n_r,)),               # collocation ~ U(0,1)
                  100.0 * random.uniform(k_ic, (n_ic,)))     # IC ~ 100 U(0,1)

    def loss(self, params, w, batch):
        t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_r, x_r = batch
        w_r, w_i = w
        f = self.fields
        w_i = w_i if self.weighted_ic else 1.0      # steady benchmark: residual only
        L = jnp.mean((w_r * f.r_pred(params, t_r, x_r)) ** 2)
        L += jnp.mean((w_i * (f.u_pred(params, t_ic, x_ic) - u_ic)) ** 2)
        if f.has_ut_ic:
            L += jnp.mean((w_i * f.u_t_pred(params, t_ic, x_ic)) ** 2)
        return L + jnp.mean((f.u_pred(params, t_bc, x_bc) - u_bc) ** 2)

    def ratio(self, w):
        """Aggregate condition-to-residual weight, comparable to the scalar rules."""
        w_i = float(jnp.mean(w[1] ** 2)) if self.weighted_ic else 1.0
        return 0.5 * (w_i + 1.0) / (float(jnp.mean(w[0] ** 2)) + 1e-30)
