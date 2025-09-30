from typing import Optional, Tuple, Dict
import jax
import jax.numpy as jnp
from jax import Array, lax
from functools import partial

from vgd.distribution import SignedMixtureLike
from vgd.loss import ISFLoss
from vgd.util import _median_lengthscale_subset, _replace_lengthscale


# Interpolated Stein Flow (ISF) stepper
# --------------------------------------
def make_isf_step(
    *,
    model,                                   # provides score_ref_plus/minus
    loss: ISFLoss,                          # provides grad_alpha, psi_theta
    kernel,                                  # __call__(X,Y,params)->(K,G)
    base_kparams,
    eps_theta: float,                        # step for locations
    eps_alpha: float,                        # step for weights
    Q0: SignedMixtureLike,                   # signed mixture (particles, alpha)
    lambda_skl: float = 1.0,                 # weight on SKL Stein field
    max_points: int = 256,
    update_every: int = 10,
    ema_alpha: float = 1.0,
    clamp: Tuple[float, float] = (0.5, 2.0),
    step_clip: Optional[float] = 0.1,
    use_median_lengthscale: bool = True,
    normalise_skl_by_sign_mass: bool = True, # "true expectations" vs unnormalised
    return_metrics: bool = False,
):
    # Convert to JAX arrays to ensure they're traced correctly
    eps_theta_ = jnp.asarray(eps_theta)
    eps_alpha_ = jnp.asarray(eps_alpha)
    lambda_skl_ = jnp.asarray(lambda_skl)
    ema_alpha_ = jnp.asarray(ema_alpha)
    
    # lengthscale bootstrap
    if use_median_lengthscale:
        ell0 = _median_lengthscale_subset(Q0.particles, max_points)
        lo = jnp.asarray(clamp[0]) * ell0
        hi = jnp.asarray(clamp[1]) * ell0
    else:
        ell0 = jnp.asarray(base_kparams.lengthscale)
        lo = jnp.asarray(clamp[0]) * ell0
        hi = jnp.asarray(clamp[1]) * ell0

    # Handle step_clip outside of JIT
    if step_clip is None:
        step_clip_val = jnp.inf
        use_clipping = False
    else:
        step_clip_val = jnp.asarray(step_clip)
        use_clipping = True

    def _stein_field_signed(Q: SignedMixtureLike, K: Array, G: Array) -> Array:
        """
        Build φ_SKL at all i, then gate by sign:
            φ_plus (for i with α_i>0) and φ_minus (for α_i<0).
        """
        X = Q.particles
        s_plus  = model.score_ref_plus(X)    # (n,d)
        s_minus = model.score_ref_minus(X)   # (n,d)

        w_plus, w_minus = Q.per_sign_weights(normalise=normalise_skl_by_sign_mass)

        # Drift terms: K^T @ (w * score)
        drift_plus  = K.T @ (w_plus[:,  None]  * s_plus)   # (n,d)
        drift_minus = K.T @ (w_minus[:, None] * s_minus)   # (n,d)

        # Repulsion terms: sum_j w_j * ∇_{Y_j} k(X_i, Y_j)
        rep_plus  = jnp.sum(w_plus[:,  None, None]  * G, axis=0)  # (n,d)
        rep_minus = jnp.sum(w_minus[:, None, None] * G, axis=0)   # (n,d)

        phi_plus  = drift_plus  + rep_plus
        phi_minus = drift_minus + rep_minus

        m_plus, m_minus = Q.sign_masks()
        phi = (m_plus[:, None]  * phi_plus) + (m_minus[:, None] * phi_minus)
        return phi  # (n,d)

    def _rkhs_field_cheap(Q: SignedMixtureLike, G: Array, psi_theta: Array) -> Array:
        """
        Implement (11):  \dot θ_i ≈ - Σ_ℓ α_ℓ ∇_θ k(θ_i, θ_ℓ) ψ(θ_ℓ)
        Using G with shape (n,n,d) carrying ∇_{Y} k(X_i, Y_j),
        the sum is   sum_j (alpha_j * psi_j) * G[j, i, :]  (sum over axis=0).
        """
        beta = Q.alpha * psi_theta               # (n,)
        loc = - jnp.sum(beta[:, None, None] * G, axis=0)  # (n,d)
        return loc

    def _one_step(carry):
        Q_cur, ell, t = carry

        # refresh lengthscale (optional)
        if use_median_lengthscale:
            def recompute(_):
                ell_hat = _median_lengthscale_subset(Q_cur.particles, max_points)
                ell_new = (1.0 - ema_alpha_) * ell + ema_alpha_ * ell_hat
                return jnp.clip(ell_new, lo, hi)
            
            def no_recompute(_):
                return ell
            
            ell_next = lax.cond(
                (t % update_every) == 0, 
                recompute, 
                no_recompute, 
                operand=None
            )
        else:
            ell_next = ell
        
        kparams = _replace_lengthscale(base_kparams, ell_next)

        # Kernel blocks
        K, G = kernel(Q_cur.particles, Q_cur.particles, kparams)  # (n,n), (n,n,d)

        # 1) RKHS cheap update (11)
        g_alpha   = loss.grad_alpha(model, Q_cur)    # (n,)
        psi_theta = loss.psi_theta(model, Q_cur)     # (n,)
        phi_rkhs  = _rkhs_field_cheap(Q_cur, G, psi_theta)  # (n,d)

        # 2) Signed SKL Stein field
        phi_skl = _stein_field_signed(Q_cur, K, G)   # (n,d)

        # Combine and step
        phi_total = phi_rkhs + lambda_skl_ * phi_skl  # directions add

        # Apply clipping if needed
        if use_clipping:
            def clip_fn(phi):
                nrm = jnp.linalg.norm(phi, axis=1, keepdims=True)
                return phi * jnp.minimum(1.0, step_clip_val / (nrm + 1e-12))
            
            def no_clip_fn(phi):
                return phi
            
            phi_total = lax.cond(
                jnp.isfinite(step_clip_val),
                clip_fn,
                no_clip_fn,
                phi_total
            )

        # Update particles
        particles_next = Q_cur.particles + eps_theta_ * phi_total
        Q_next = Q_cur.replace_particles(particles_next)

        # weight step (separate step size)
        alpha_next = Q_next.alpha - eps_alpha_ * g_alpha
        Q_next = Q_next.replace_alpha(alpha_next)

        # Always compute metrics (they'll be filtered later if not needed)
        metrics = {
            "loss": loss.eval(model, Q_cur),
            "rkhs_step_norm": jnp.mean(jnp.linalg.norm(phi_rkhs, axis=-1)),
            "stein_step_norm": jnp.mean(jnp.linalg.norm(phi_skl, axis=-1)),
            "particles": Q_next.particles,
            "alpha": Q_next.alpha,
        }

        return (Q_next, ell_next, t + 1), metrics

    # JIT compile the step function
    @jax.jit
    def step(carry, _):
        return _one_step(carry)

    return step, ell0


def isf(
    *,
    model,
    loss: ISFLoss,
    kernel,
    kparams,
    eps_theta: float,
    eps_alpha: float,
    Q0: SignedMixtureLike,
    steps: int = 1000,
    lambda_skl: float = 1.0,
    max_points: int = 256,
    update_every: int = 10,
    ema_alpha: float = 1.0,
    clamp: Tuple[float, float] = (0.5, 2.0),
    step_clip: Optional[float] = 0.1,
    use_median_lengthscale: bool = True,
    normalise_skl_by_sign_mass: bool = True,
    return_metrics: bool = False,
):
    step, ell0 = make_isf_step(
        model=model, 
        loss=loss, 
        kernel=kernel, 
        base_kparams=kparams,
        eps_theta=eps_theta, 
        eps_alpha=eps_alpha, 
        Q0=Q0,
        lambda_skl=lambda_skl, 
        max_points=max_points, 
        update_every=update_every,
        ema_alpha=ema_alpha, 
        clamp=clamp, 
        step_clip=step_clip,
        use_median_lengthscale=use_median_lengthscale,
        normalise_skl_by_sign_mass=normalise_skl_by_sign_mass,
        return_metrics=return_metrics,
    )

    # Initialize carry
    carry0 = (Q0, ell0, jnp.array(0, dtype=jnp.int32))
    
    # Run scan
    (Q_final, _, _), history = lax.scan(step, carry0, xs=None, length=steps)

    if return_metrics:
        # Properly concatenate trajectories
        particles_traj = jnp.concatenate(
            [Q0.particles[None, ...], history["particles"]], 
            axis=0
        )
        alpha_traj = jnp.concatenate(
            [Q0.alpha[None, ...], history["alpha"]], 
            axis=0
        )
        history = {
            **history, 
            "particles_traj": particles_traj, 
            "alpha_traj": alpha_traj
        }
    else:
        # Return empty dict if metrics not requested
        history = {}

    return Q_final, history