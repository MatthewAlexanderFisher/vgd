from __future__ import annotations
from typing import Optional, Tuple, Dict
import jax, jax.numpy as jnp
from jax import Array, lax

from vgd.util import _median_lengthscale_subset, _replace_lengthscale
from vgd.distribution import MixtureLike
from vgd.model import Model
from vgd.kernel import Kernel, KernelParams
from vgd.loss import Loss

def make_vgd_step(
    *,
    model: Model,                          # Posterior/Model wrapper (likelihood + prior), data already bound
    loss: Loss,                                # Loss: .grad(model, Q) -> (n,d)
    kernel: Kernel,                            # Kernel: __call__(X, Y, params) -> (K:(n,n), G:(n,n,d))
    base_kparams: KernelParams,                # has 'lengthscale'; used via _replace_lengthscale
    eps: float,                                # step size
    Q0: MixtureLike,                       # MixtureLike with normalised weights (particles,w)
    max_points: int = 256,                     # for median lengthscale
    update_every: int = 10,                    # how often to refresh lengthscale
    ema_alpha: float = 1.0,                    # EMA smoothing for lengthscale (1.0 = no smoothing)
    clamp: Tuple[float, float] = (0.5, 2.0),
    repulse_coef: float = 1.0,
    step_clip: Optional[float] = 0.1,
    use_median_lengthscale: bool = True,
    return_metrics: bool = False,             
):
    """
    Returns: step_fn that updates (particles, ell, t) -> (particles', ell', t+1),
    and emits a metrics dict per step when return_metrics=True.

    If return_metrics=True, per-step metrics contain:
      - 'loss'       : scalar (evaluated on current Q before the update)
      - 'drift_norm' : mean L2 norm of the update direction (phi)
      - 'particles'  : (n, d) particles after the update
    """
    # baseline lengthscale from initial particles
    if use_median_lengthscale:
        ell0 = _median_lengthscale_subset(Q0.particles, max_points)
        lo, hi = clamp[0] * ell0, clamp[1] * ell0
    else:
        ell0 = jnp.asarray(base_kparams.lengthscale)
        lo, hi = clamp[0] * ell0, clamp[1] * ell0

    # Close over fixed, normalised weights; avoid any per-step normalisation.
    w_fixed = Q0.w

    def _one_step(carry: Tuple[MixtureLike, Array, Array]):
        # mixture dist, lengthscale, time
        Q_cur, ell, t = carry

        # Update lengthscale (optional EMA + clamp)
        if use_median_lengthscale:
            def recompute(_):
                ell_hat = _median_lengthscale_subset(Q_cur.particles, max_points)
                ell_new = (1.0 - ema_alpha) * ell + ema_alpha * ell_hat
                return jnp.clip(ell_new, lo, hi)
            ell_next = lax.cond((t % update_every) == 0, recompute, lambda _: ell, operand=None)
        else:
            ell_next = ell

        # Kernel params for this step
        kparams = _replace_lengthscale(base_kparams, ell_next)

        # Wasserstein gradient and prior score
        wass_grad = loss.grad(model, Q_cur)               # (n,d)
        s_prior   = model.score_prior(Q_cur.particles)      # (n,d)

        # Kernel drift & repulsion (weighted by fixed w)
        K, G = kernel(Q_cur.particles, Q_cur.particles, kparams)  # (n,n), (n,n,d)
        drift   = K.T @ (w_fixed[:, None] * (wass_grad + s_prior))     # (n,d)
        repulse = repulse_coef * jnp.sum(w_fixed[:, None, None] * G, axis=0)  # (n,d)
        phi = drift + repulse   # update direction

        if step_clip is not None:
            nrm = jnp.linalg.norm(phi, axis=1, keepdims=True)
            phi = phi * jnp.minimum(1.0, step_clip / (nrm + 1e-12))

        Q_next = Q_cur.replace_particles(Q_cur.particles + eps * phi)

        if return_metrics:
            loss_val = loss.eval(model, Q_cur)  # evaluated BEFORE update (matches mfld behaviour)
            drift_norm = jnp.mean(jnp.linalg.norm(phi, axis=-1))
            metrics: Dict[str, Array] = {
                "loss": loss_val,
                "drift_norm": drift_norm,
                "particles": Q_next.particles,   # after update; scan stacks to (T, n, d)
            }
        else:
            metrics = {}

        return (Q_next, ell_next, t + 1), metrics

    @jax.jit
    def step(carry, _):
        return _one_step(carry)

    return step, ell0


def vgd(
    *,
    model: Model,
    loss: Loss,
    kernel: Kernel,
    kparams: KernelParams,
    eps: float,
    Q0: MixtureLike,                               # MixtureLike with normalized w
    steps: int = 1000,
    max_points: int = 256,
    update_every: int = 10,
    ema_alpha: float = 1.0,
    clamp: Tuple[float, float] = (0.5, 2.0),
    repulse_coef: float = 1.0,
    step_clip: Optional[float] = 0.1,
    use_median_lengthscale: bool = True,
    return_metrics: bool = False,                      
):
    """
    Run Variational Gradient Descent with kernelised transport.

    Returns:
      - particles_T : (n, d) final particles
      - history     : {} if return_metrics=False, else dict with:
            'loss'           : (steps,)
            'drift_norm'     : (steps,)
            'particles'      : (steps, n, d)        # after each step
            'particles_traj' : (steps+1, n, d)      # includes initial Q0.particles at index 0
    """
    step, ell0 = make_vgd_step(
        model=model, loss=loss, kernel=kernel, base_kparams=kparams, eps=eps,
        Q0=Q0, max_points=max_points, update_every=update_every, ema_alpha=ema_alpha,
        clamp=clamp, repulse_coef=repulse_coef, step_clip=step_clip,
        use_median_lengthscale=use_median_lengthscale,
        return_metrics=return_metrics,
    )

    carry0 = (Q0, ell0, jnp.array(0, dtype=jnp.int32))

    # Run scan; outputs will be stacked per key in the metrics dict
    (Q_final, _, _), history = lax.scan(step, carry0, xs=None, length=steps)

    if return_metrics:
        particles_traj = jnp.concatenate(
            [Q0.particles[jnp.newaxis, ...], history["particles"]],
            axis=0,
        )
        history = {**history, "particles_traj": particles_traj}
    else:
        history = {}

    return Q_final, history
