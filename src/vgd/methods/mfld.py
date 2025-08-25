from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import random, Array, lax
from typing import Optional, Dict, Any, Callable, Tuple

from vgd.loss import Loss
from vgd.distribution import DiscreteMixture, Posterior

def make_mfld_step(
    *,
    model: Posterior,                      # Posterior/Model wrapper (likelihood + prior), data already bound
    loss: Loss,                            # Loss: .eval(model, Q)->scalar, .grad(model, Q)->(n,d) VGD/Wasserstein field
    lr: float = 1e-3,                      # step size (Euler–Maruyama)
    temperature: float = 1.0,
    noise_scale: float = 1.0,
    friction: float = 0.0,                 # momentum/underdamped if > 0
    return_metrics: bool = False,
) -> Callable[
    [Tuple[DiscreteMixture, Array, Array], Array],
    Tuple[Tuple[DiscreteMixture, Array, Array], Dict[str, Array]],
]:
    """
    One MFLD scan step.
      carry  = (Q, velocities) with velocities.shape == Q.particles.shape
      key    = PRNGKey for Gaussian noise
      output = ((Q_next, velocities_next), metrics_dict)  # metrics_dict is {} if return_metrics=False

    Overdamped (friction == 0):
        x <- x + lr * drift + sqrt(2 * temperature * lr) * ξ
    Underdamped (friction > 0):
        v <- (1 - friction) * v + lr * drift + sqrt(2 * temperature * lr) * ξ
        x <- x + v

    drift = loss.grad(model, Q) + model.score_prior(Q.particles)
    """

    def _one_step(carry: Tuple[DiscreteMixture, Array, Array]) -> Tuple[Tuple[DiscreteMixture, Array, Array], Dict[str, Array]]:
        # mixture dist, lengthscale, time, key
        Q_cur, velocities, key = carry

        particles = Q_cur.particles  # assume Q.w already normalised and fixed
        dtype = particles.dtype

        # Wasserstein data field + prior score
        field   = loss.grad(model, Q_cur)                 # (n,d)
        s_prior = model.score_prior(particles)        # (n,d)
        drift   = field + s_prior                     # (n,d)

        # Gaussian noise (dtype-stable)
        key, key_noise = random.split(key)

        sigma = jnp.sqrt(jnp.asarray(2.0 * temperature * lr, dtype)) * jnp.asarray(noise_scale, dtype)
        noise = random.normal(key_noise, particles.shape, dtype=dtype) * sigma

        if friction > 0.0:
            new_velocities = (jnp.asarray(1.0 - friction, dtype) * velocities
                              + jnp.asarray(lr, dtype) * drift + noise)
            new_particles  = particles + new_velocities
        else:
            new_velocities = velocities
            new_particles  = particles + jnp.asarray(lr, dtype) * drift + noise

        # replace particles only - keep weights unchanged without re-normalising
        Q_next = Q_cur.replace_particles(new_particles)

        if return_metrics:
            loss_val   = loss.eval(model, Q_cur)  # scalar (uses current Q)
            drift_norm = jnp.mean(
                jnp.linalg.norm(drift.reshape((drift.shape[0], -1)), axis=-1)
            )
            metrics: Dict[str, Array] = {
                "loss": loss_val,
                "drift_norm": drift_norm,
                # record per-step particles so scan stacks to (T, n, d)
                "particles": Q_next.particles,
            }
        else:
            metrics: Dict[str, Array] = {}  # stable empty PyTree for lax.scan

        return (Q_next, new_velocities, key), metrics

    @jax.jit
    def step(carry, _):
        return _one_step(carry)

    return step


def mfld(
    *,
    model: Posterior,
    loss: Loss,
    key: Array,
    Q0: DiscreteMixture,
    steps: int,
    lr: float = 1e-3,
    temperature: float = 1.0,
    noise_scale: float = 1.0,
    friction: float = 0.0,
    return_metrics: bool = False,
) -> Tuple[DiscreteMixture, Dict[str, Array]]:
    """
    Run Mean-Field Langevin Dynamics.

    Returns:
      - final DiscreteMixture
      - final velocities array (same shape as Q0.particles)
      - history dict stacked over time; {} if return_metrics=False

    If return_metrics=True, history contains:
      - 'loss'         : (steps,)
      - 'drift_norm'   : (steps,)
      - 'particles'    : (steps, n, d)          # particles *after* each step
      - 'particles_traj': (steps+1, n, d)       # includes initial Q0.particles at index 0
    """
    velocities0 = jnp.zeros_like(Q0.particles)

    step = make_mfld_step(
        model=model,
        loss=loss,
        lr=lr,
        temperature=temperature,
        noise_scale=noise_scale,
        friction=friction,
        return_metrics=return_metrics,
    )

    carry0 = (Q0, velocities0, key)
    (Q_final, v_final, _), history = lax.scan(step, carry0, xs=None, length=steps)

    if return_metrics:
        # Add full trajectory including the initial state
        particles_traj = jnp.concatenate(
            [Q0.particles[jnp.newaxis, ...], history["particles"]],
            axis=0,
        )
        history = {**history, "particles_traj": particles_traj}

    return Q_final, history
