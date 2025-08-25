from __future__ import annotations
from typing import Optional, Tuple, Dict
import jax, jax.numpy as jnp
from jax import lax, Array

from vgd.util import _median_lengthscale_subset, _replace_lengthscale
from vgd.distribution import DiscreteMixture, Posterior
from vgd.kernel import Kernel, KernelParams
from vgd.loss import Loss


# ------------------------------------------------------

def make_vgd_weight_step(
    *,
    model: Posterior,
    loss: Loss,                                # provides wasserstein gradient .grad(model, Q) -> (n,d)
    kernel: Kernel,
    base_kparams: KernelParams,
    eps: float,
    Q0: DiscreteMixture,                       # initial particles & (normalised) weights
    # lengthscale control
    max_points: int = 256,
    update_every: int = 10,
    ema_alpha: float = 1.0,
    clamp: Tuple[float, float] = (0.5, 2.0),
    use_median_lengthscale: bool = True,
    # kernel gradient coefficient / step size clip
    repulse_coef: float = 1.0,
    step_clip: Optional[float] = 0.1,
    # weight update
    weight_ema: float = 0.1,                   # EMA amount β in [0,1]
    weight_update_every: int = 1,              # update weights every k steps
    # misc
    return_metrics: bool = False,
):
    """
    Interleaves a convex weight update (EMA) with a kernel VGD step.

    Carry: (particles, w, ell, t)
      - weights are *updated* each step via EMA (or per weight_update_every).
      - loss.grad is evaluated with the **current updated weights**.
    """

    # baseline lengthscale
    if use_median_lengthscale:
        ell0 = _median_lengthscale_subset(Q0.particles, max_points)
    else:
        ell0 = jnp.asarray(base_kparams.lengthscale)

    def _ema_weights(Q: DiscreteMixture, t: Array) -> Array:
        def do_update(_):
            r = loss.weights(model, Q, mean=True)  # (n,)
            w_new = (1.0 - weight_ema) * Q.w + weight_ema * r
            return w_new
        return lax.cond((t % weight_update_every) == 0,
                        do_update, lambda _: Q.w, operand=None)


    def _one_step(carry: Tuple[DiscreteMixture, Array, Array]):
        # Qmixture, lengthscale, time
        Q_cur, ell, t = carry

        # lengthscale refresh
        if use_median_lengthscale:
            def recompute(_):
                ell_hat = _median_lengthscale_subset(Q_cur.particles, max_points)
                ell_new = (1.0 - ema_alpha) * ell + ema_alpha * ell_hat
                lo, hi = clamp[0] * ell0, clamp[1] * ell0
                return jnp.clip(ell_new, lo, hi)
            ell_next = lax.cond((t % update_every) == 0,
                                recompute, lambda _: ell, operand=None)
        else:
            ell_next = ell

        # smooth convex weight update
        w_next = _ema_weights(Q_cur, t)           # (n,)

        # kernel params for this step
        kparams = _replace_lengthscale(base_kparams, ell_next)

        # build Q for loss.grad using UPDATED weights
        Q_int = DiscreteMixture(Q_cur.particles, w_next)

        # Wasserstein field + prior score
        wass_grad = loss.grad(model, Q_int)              # (n,d)
        s_prior   = model.score_prior(Q_int.particles)         # (n,d)

        # kernel drift & repulsion with current weights
        K, G = kernel(Q_int.particles, Q_int.particles, kparams)     # (n,n), (n,n,d)
        drift   = K.T @ (w_next[:, None] * (wass_grad + s_prior))           # (n,d)
        repulse = repulse_coef * jnp.sum(w_next[:, None, None] * G, axis=0) # (n,d)
        phi = drift + repulse

        if step_clip is not None:
            nrm = jnp.linalg.norm(phi, axis=1, keepdims=True)
            phi = phi * jnp.minimum(1.0, step_clip / (nrm + 1e-12))

        Q_next = Q_int.replace_particles(Q_int.particles + eps * phi)

        if return_metrics:
            # Evaluate loss with current (before move/weight updates) state
            loss_val   = loss.eval(model, Q_cur)
            drift_norm = jnp.mean(jnp.linalg.norm(phi, axis=-1))
            metrics: Dict[str, Array] = {
                "loss": loss_val,
                "drift_norm": drift_norm,
                "particles": Q_next.particles,   # after update; scan stacks to (T, n, d)
                "weights": Q_next.w,   # after update; scan stacks to (T, n)
            }
        else:
            metrics = {}

        return (Q_next, ell_next, t + 1), metrics

    @jax.jit
    def step(carry, _):
        return _one_step(carry)

    return step, ell0


def vgd_weight(
    *,
    model: Posterior,
    loss: Loss,
    kernel: Kernel,
    kparams: KernelParams,
    eps: float,
    Q0: DiscreteMixture,                               # (particles, normalised w)
    steps: int = 1000,
    max_points: int = 256,
    update_every: int = 10,
    ema_alpha: float = 1.0,
    clamp: Tuple[float, float] = (0.5, 2.0),
    repulse_coef: float = 1.0,
    step_clip: Optional[float] = 0.1,
    use_median_lengthscale: bool = True,
    # weight update knobs
    weight_ema: float = 0.1,
    weight_update_every: int = 1,
    # metrics
    return_metrics: bool = False,
):
    step, ell0 = make_vgd_weight_step(
        model=model, loss=loss, kernel=kernel, base_kparams=kparams, eps=eps,
        Q0=Q0, max_points=max_points, update_every=update_every, ema_alpha=ema_alpha,
        clamp=clamp, repulse_coef=repulse_coef, step_clip=step_clip,
        use_median_lengthscale=use_median_lengthscale,
        weight_ema=weight_ema, weight_update_every=weight_update_every,
        return_metrics=return_metrics,
    )

    carry0 = (Q0, ell0, jnp.array(0, dtype=jnp.int32))
    (Q_final, _, _), history = lax.scan(step, carry0, xs=None, length=steps)

    if return_metrics:
        # reformat particle trajectories
        particles_traj = jnp.concatenate(
            [Q0.particles[jnp.newaxis, ...], history["particles"]],
            axis=0,
        )

        # reformat weight trajectories
        weight_traj = jnp.concatenate(
            [Q0.w[jnp.newaxis, ...], history["weights"]],
            axis=0,
        )

        history = {**history, "particles_traj": particles_traj, "weights_traj": weight_traj}
    else:
        history = {}

    return Q_final, history
