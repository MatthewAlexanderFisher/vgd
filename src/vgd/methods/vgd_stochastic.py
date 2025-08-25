from __future__ import annotations
from typing import Optional, Tuple, Dict
import jax, jax.numpy as jnp
from jax import Array, lax, random
from torch import dtype

from vgd.util import _median_lengthscale_subset, _replace_lengthscale
from vgd.distribution import DiscreteMixture, Posterior
from vgd.kernel import Kernel, KernelParams
from vgd.loss import Loss


def make_vgd_random_step(
    *,
    model: Posterior,                            # Posterior/Model wrapper (likelihood + prior), data already bound
    loss: Loss,                                  # Loss: .grad(model, Q) -> (n,d)
    kernel: Kernel,                              # Kernel: __call__(X, Y, params) -> (K:(n,n), G:(n,n,d))
    base_kparams: KernelParams,                  # has 'lengthscale'; used via _replace_lengthscale
    eps: float,                                  # step size (dt)
    Q0: DiscreteMixture,                         # DiscreteMixture with normalised weights (particles, w)
    max_points: int = 256,                     # for median lengthscale
    update_every: int = 10,                    # how often to refresh lengthscale
    ema_alpha: float = 1.0,                    # EMA smoothing for lengthscale (1.0 = no smoothing)
    clamp: Tuple[float, float] = (0.5, 2.0),
    lambd: float = 0.0,
    step_clip: Optional[float] = 0.1,
    use_median_lengthscale: bool = True,
    noise_scale: float = 1.0,                  
    return_metrics: bool = False,              
):
    """
    Stochastic VGD step (overdamped):
        x <- x + eps * phi(x) + sqrt(2 * temperature * eps) * xi

    If return_metrics=True, emits per-step dict with:
      - 'loss'        : scalar (evaluated on current Q before update)
      - 'drift_norm'  : mean L2 norm of phi
      - 'particles'   : (n, d) particles AFTER the update
    """

    # baseline lengthscale from initial particles
    if use_median_lengthscale:
        ell0 = _median_lengthscale_subset(Q0.particles, max_points)
    else:
        ell0 = jnp.asarray(base_kparams.lengthscale)

    # Close over fixed, normalised weights; avoid any per-step normalisation.
    w_fixed = Q0.w

    dtype = Q0.particles.dtype
    eps_ = jnp.asarray(eps, dtype)

    def _one_step(carry: Tuple[DiscreteMixture, Array, Array, Array]):
        # mixture dist, lengthscale, time, key
        Q_cur, ell, t, key = carry
        dtype = Q_cur.particles.dtype

        # Update lengthscale (optional EMA + clamp)
        if use_median_lengthscale:
            def recompute(_):
                ell_hat = _median_lengthscale_subset(Q_cur.particles, max_points)
                ell_new = (1.0 - ema_alpha) * ell + ema_alpha * ell_hat
                lo, hi = clamp[0] * ell0, clamp[1] * ell0
                return jnp.clip(ell_new, lo, hi)
            ell_next = lax.cond((t % update_every) == 0, recompute, lambda _: ell, operand=None)
        else:
            ell_next = ell

        # Kernel params for this step
        kparams = _replace_lengthscale(base_kparams, ell_next)

        # Wasserstein gradient and prior score
        wass_grad = loss.grad(model, Q_cur)               # (n,d)
        s_prior   = model.score_prior(Q_cur.particles)      # (n,d)

        # Strang Split (1/2 Noise -> Kernel Grad -> 1/2 Noise)
        particles = Q_cur.particles
        key, key_noise1, key_noise2 = random.split(key, 3)
        sigma = jnp.sqrt(jnp.asarray((1 - lambd) * eps_, dtype)) * jnp.asarray(noise_scale, dtype)

        # Langevin update 1:
        noise1 = random.normal(key_noise1, particles.shape, dtype=dtype) * sigma
        particles_next = particles + eps_ * (1 - lambd) * s_prior + noise1

        # Kernel VGD update (fixed weights) 
        K, G = kernel(particles_next, particles_next, kparams)  # (n,n), (n,n,d)
        drift   = K.T @ (w_fixed[:, None] * (wass_grad + lambd * s_prior))     # (n,d)
        repulse = lambd * jnp.sum(w_fixed[:, None, None] * G, axis=0)  # (n,d)
        phi = drift + repulse   # deterministic transport direction

        if step_clip is not None: # perform step clipping
            nrm = jnp.linalg.norm(phi, axis=1, keepdims=True)
            phi = phi * jnp.minimum(1.0, step_clip / (nrm + 1e-12))

        # Perform update + Langevin update 2:
        noise2 = random.normal(key_noise2, particles.shape, dtype=dtype) * sigma
        particles_next = particles + eps_ * (phi + (1 - lambd) * s_prior) + noise2

        Q_next = Q_cur.replace_particles(particles_next)

        if return_metrics:
            loss_val = loss.eval(model, Q_cur)  # BEFORE update
            drift_norm = jnp.mean(jnp.linalg.norm(phi, axis=-1))
            metrics: Dict[str, Array] = {
                "loss": loss_val,
                "drift_norm": drift_norm,
                "particles": Q_next.particles,
            }
        else:
            metrics = {}

        return (Q_next, ell_next, t + 1, key), metrics

    @jax.jit
    def step(carry, _):
        return _one_step(carry)
    
    return step, ell0


def vgd_random(
    *,
    model: Posterior,
    loss: Loss,
    kernel: Kernel,
    kparams: KernelParams,
    eps: float,
    Q0: DiscreteMixture,                       # DiscreteMixture with normalized w
    key: Array,                                
    steps: int = 1000,
    max_points: int = 256,
    update_every: int = 10,
    ema_alpha: float = 1.0,
    clamp: Tuple[float, float] = (0.5, 2.0),
    lambd: float = 0.,
    step_clip: Optional[float] = 0.1,
    use_median_lengthscale: bool = True,
    noise_scale: float = 1.0,                  
    return_metrics: bool = False,              
):
    """
    Run Stochastic VGD (overdamped Langevinised VGD).

    Returns:
      - particles_T : (n, d) final particles
      - history     : {} if return_metrics=False, else dict with:
            'loss'           : (steps,)
            'drift_norm'     : (steps,)
            'particles'      : (steps, n, d)        # after each step
            'particles_traj' : (steps+1, n, d)      # includes initial Q0.particles at index 0
    """
    step, ell0 = make_vgd_random_step(
        model=model, loss=loss, kernel=kernel, base_kparams=kparams, eps=eps,
        Q0=Q0, max_points=max_points, update_every=update_every, ema_alpha=ema_alpha,
        clamp=clamp, lambd=lambd, step_clip=step_clip,
        use_median_lengthscale=use_median_lengthscale,
        noise_scale=noise_scale, return_metrics=return_metrics,
    )

    carry0 = (Q0, ell0, jnp.array(0, dtype=jnp.int32), key)

    # Run scan; outputs will be stacked per key in the metrics dict
    (particles_T, _, _, _), history = lax.scan(step, carry0, xs=None, length=steps)

    if return_metrics:
        particles_traj = jnp.concatenate(
            [Q0.particles[jnp.newaxis, ...], history["particles"]],
            axis=0,
        )
        history = {**history, "particles_traj": particles_traj}
    else:
        history = {}

    return particles_T, history


def make_vgd_naive_random_step(
    *,
    model: Posterior,                            # Posterior/Model wrapper (likelihood + prior), data already bound
    loss: Loss,                                  # Loss: .grad(model, Q) -> (n,d)
    kernel: Kernel,                              # Kernel: __call__(X, Y, params) -> (K:(n,n), G:(n,n,d))
    base_kparams: KernelParams,                  # has 'lengthscale'; used via _replace_lengthscale
    eps: float,                                  # step size (dt)
    Q0: DiscreteMixture,                         # DiscreteMixture with normalised weights (particles, w)
    max_points: int = 256,                       # for median lengthscale
    update_every: int = 10,                      # how often to refresh lengthscale
    ema_alpha: float = 1.0,                      # EMA smoothing for lengthscale (1.0 = no smoothing)
    clamp: Tuple[float, float] = (0.5, 2.0),
    repulse_coef: float = 1.0,
    step_clip: Optional[float] = 0.1,
    use_median_lengthscale: bool = True,
    temperature: float = 1.0,                  
    noise_scale: float = 1.0,                  
    return_metrics: bool = False,              
):
    """
    Stochastic VGD step (overdamped):
        x <- x + eps * phi(x) + sqrt(2 * temperature * eps) * xi

    If return_metrics=True, emits per-step dict with:
      - 'loss'        : scalar (evaluated on current Q before update)
      - 'drift_norm'  : mean L2 norm of phi
      - 'particles'   : (n, d) particles AFTER the update
    """

    # baseline lengthscale from initial particles
    if use_median_lengthscale:
        ell0 = _median_lengthscale_subset(Q0.particles, max_points)
    else:
        ell0 = jnp.asarray(base_kparams.lengthscale)

    # Close over fixed, normalised weights; avoid any per-step normalisation.
    w_fixed = Q0.w

    def _one_step(particles: Array, ell: Array, t: Array, key: Array):
        dtype = particles.dtype

        # Update lengthscale (optional EMA + clamp)
        if use_median_lengthscale:
            def recompute(_):
                ell_hat = _median_lengthscale_subset(particles, max_points)
                ell_new = (1.0 - ema_alpha) * ell + ema_alpha * ell_hat
                lo, hi = clamp[0] * ell0, clamp[1] * ell0
                return jnp.clip(ell_new, lo, hi)
            ell_next = lax.cond((t % update_every) == 0, recompute, lambda _: ell, operand=None)
        else:
            ell_next = ell

        # Kernel params for this step
        kparams = _replace_lengthscale(base_kparams, ell_next)

        # Q with current particles (weights fixed)
        Q = Q0.replace_particles(particles)

        # Wasserstein gradient and prior score
        wass_grad = loss.grad(model, Q)               # (n,d)
        s_prior   = model.score_prior(particles)      # (n,d)

        # Kernel drift & repulsion (weighted by fixed w)
        K, G = kernel(particles, particles, kparams)  # (n,n), (n,n,d)
        drift   = K.T @ (w_fixed[:, None] * (wass_grad + s_prior))     # (n,d)
        repulse = repulse_coef * jnp.sum(w_fixed[:, None, None] * G, axis=0)  # (n,d)
        phi = drift + repulse   # deterministic transport direction

        if step_clip is not None:
            nrm = jnp.linalg.norm(phi, axis=1, keepdims=True)
            phi = phi * jnp.minimum(1.0, step_clip / (nrm + 1e-12))

        # Langevin noise (dtype-stable)
        sigma = jnp.sqrt(jnp.asarray(2.0 * temperature * eps, dtype)) * jnp.asarray(noise_scale, dtype)
        key, key_noise = random.split(key)
        noise = random.normal(key_noise, particles.shape, dtype=dtype) * sigma

        particles_next = particles + jnp.asarray(eps, dtype) * phi + noise

        if return_metrics:
            loss_val = loss.eval(model, Q)  # BEFORE update
            drift_norm = jnp.mean(jnp.linalg.norm(phi, axis=-1))
            metrics: Dict[str, Array] = {
                "loss": loss_val,
                "drift_norm": drift_norm,
                "particles": particles_next,
            }
        else:
            metrics = {}

        return particles_next, ell_next, key, metrics

    @jax.jit
    def step(carry, _):
        particles, ell, t, key = carry
        particles_next, ell_next, key_next, metrics = _one_step(particles, ell, t, key)
        return (particles_next, ell_next, t + 1, key_next), metrics

    return step, ell0


def vgd_naive_random(
    *,
    model: Posterior,
    loss: Loss,
    kernel: Kernel,
    kparams: KernelParams,
    eps: float,
    Q0: DiscreteMixture,                       # DiscreteMixture with normalized w
    key: Array,                                
    steps: int = 1000,
    max_points: int = 256,
    update_every: int = 10,
    ema_alpha: float = 1.0,
    clamp: Tuple[float, float] = (0.5, 2.0),
    repulse_coef: float = 1.0,
    step_clip: Optional[float] = 0.1,
    use_median_lengthscale: bool = True,
    temperature: float = 1.0,                  
    noise_scale: float = 1.0,                  
    return_metrics: bool = False,              
):
    """
    Run Stochastic VGD (just adding random noise).

    Returns:
      - particles_T : (n, d) final particles
      - history     : {} if return_metrics=False, else dict with:
            'loss'           : (steps,)
            'drift_norm'     : (steps,)
            'particles'      : (steps, n, d)        # after each step
            'particles_traj' : (steps+1, n, d)      # includes initial Q0.particles at index 0
    """
    step, ell0 = make_vgd_naive_random_step(
        model=model, loss=loss, kernel=kernel, base_kparams=kparams, eps=eps,
        Q0=Q0, max_points=max_points, update_every=update_every, ema_alpha=ema_alpha,
        clamp=clamp, repulse_coef=repulse_coef, step_clip=step_clip,
        use_median_lengthscale=use_median_lengthscale,
        temperature=temperature, noise_scale=noise_scale,
        return_metrics=return_metrics,
    )

    carry0 = (Q0.particles, ell0, jnp.array(0, dtype=jnp.int32), key)

    # Run scan; outputs will be stacked per key in the metrics dict
    (particles_T, _, _, _), history = lax.scan(step, carry0, xs=None, length=steps)

    if return_metrics:
        particles_traj = jnp.concatenate(
            [Q0.particles[jnp.newaxis, ...], history["particles"]],
            axis=0,
        )
        history = {**history, "particles_traj": particles_traj}
    else:
        history = {}

    return particles_T, history
