from __future__ import annotations
from typing import Optional, Protocol, Any, Callable, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from vgd.distribution import DiscreteMixture, MixtureLike, SignedMixtureLike
from vgd.model import Model, Posterior, Predictor
from vgd.util import logmeanexp_weighted


class Loss(Protocol):
    """
    Base class for loss functions.
    Subclasses should implement the methods for sampling, log probability, and gradients.
    """
    name: str = "base_loss"
    display_name: str = "base_loss"

    def eval(self, model: Model, Q: MixtureLike) -> Array: ...
    def grad(self, model: Model, Q: MixtureLike) -> Array: ...
    def weights(self, model: Model, Q: MixtureLike, mean: bool) -> Array: ...


class PredictionCentricLoss:
    name = "pc"; display_name = "prediction-centric"

    def __init__(self, alpha: float = 1.0):  
        # alpha is a tempering variable
        self.alpha = alpha

    def eval(self, model: Posterior, Q: MixtureLike) -> Array:
        logw = jnp.log(Q.w + 1e-300)
        LL = model.loglik_pointwise(Q.particles)                    # (n,N)
        # −∑_i log ∫ p(y_i|θ) dQ(θ) = −∑_i log ∑_j w_j e^{ℓ_{j,i}}
        val = - jnp.sum(logmeanexp_weighted(LL, logw, axis=0))
        return val

    def grad(self, model: Posterior, Q: MixtureLike) -> Array:
        r = self.weights(model, Q, mean=False)  # (n,N)

        S_i = model.score_like_pointwise(Q.particles)               # (n,N,d)
        return jnp.einsum('jI,jId->jd', r, S_i)             # (n,d)

    def weights(self, model: Posterior, Q: MixtureLike, mean: bool = True) -> Array:
        """r = average over data of per-datum responsibilities (PC):
        ρ_{j,i} ∝ w_j * exp(alpha * ℓ_{j,i});  r_j = mean_i ρ_{j,i}.
        """
        logliks  = model.loglik_pointwise(Q.particles)             # (n, N)
        a   = jnp.log(Q.w + 1e-300)[:, None] + self.alpha * logliks     # (n, N)
        rho = jax.nn.softmax(a, axis=0)                     # (n, N)

        if mean is True:
            r   = jnp.mean(rho, axis=1)                         # (n,)
        else:
            r   = rho                                         # (n, N)

        return jax.lax.stop_gradient(r)

class PredictionCentricLoss_n:
    name = "pc"; display_name = "prediction-centric"

    def __init__(self, alpha: float = 1.0):  
        # alpha is a tempering variable
        self.alpha = alpha

    def eval(self, model: Posterior, Q: MixtureLike) -> Array:
        logw = jnp.log(Q.w + 1e-300)
        LL = model.loglik_pointwise(Q.particles)                    # (n,N)
        # −∑_i log ∫ p(y_i|θ) dQ(θ) = −∑_i log ∑_j w_j e^{ℓ_{j,i}}
        val = - jnp.sum(logmeanexp_weighted(LL, logw, axis=0))
        return model.n() * val

    def grad(self, model: Posterior, Q: MixtureLike) -> Array:
        r = self.weights(model, Q, mean=False)  # (n,N)

        S_i = model.score_like_pointwise(Q.particles)               # (n,N,d)
        return model.n() * jnp.einsum('jI,jId->jd', r, S_i)             # (n,d)

    def weights(self, model: Posterior, Q: MixtureLike, mean: bool = True) -> Array:
        """r = average over data of per-datum responsibilities (PC):
        ρ_{j,i} ∝ w_j * exp(alpha * ℓ_{j,i});  r_j = mean_i ρ_{j,i}.
        """
        logliks  = model.loglik_pointwise(Q.particles)             # (n, N)
        a   = jnp.log(Q.w + 1e-300)[:, None] + self.alpha * logliks     # (n, N)
        rho = jax.nn.softmax(a, axis=0)                     # (n, N)

        if mean is True:
            r   = jnp.mean(rho, axis=1)                         # (n,)
        else:
            r   = rho                                         # (n, N)

        return jax.lax.stop_gradient(r)


class SequenceLoss:
    name = "seq"; display_name = "sequence"

    def __init__(self, alpha: float = 1.0):  # tempering for stability if needed
        self.alpha = alpha

    def eval(self, model, Q: MixtureLike) -> Array:
        logw = jnp.log(Q.w + 1e-300)                    # (n,)
        ell  = model.loglik_joint(Q.particles)          # (n,)
        # -log ∑_j w_j e^{ℓ_j}
        return -logsumexp(logw + ell)

    def grad(self, model, Q: MixtureLike) -> Array:
        r = self.weights(model, Q)  # (n,)
        S = model.score_like_joint(Q.particles)      # (n,d)
        return r[:, None] * S                           # (n,d)

    def weights(self, model: Posterior, Q: MixtureLike,  mean: bool=True) -> Array:
        """r_j ∝ w_j * exp(alpha * ℓ_j),  ℓ_j = sum_i log p(y_i|θ_j)."""
        loglik_joint = model.loglik_joint(Q.particles)                 # (n,)
        a   = jnp.log(Q.w + 1e-300) + self.alpha * loglik_joint             # (n,)
        r   = jax.nn.softmax(a, axis=0)                     # (n,)
        return jax.lax.stop_gradient(r)


@dataclass
class PosteriorKLLoss:
    name = "svgd"; display_name = "posterior-kl"

    def eval(self, model, Q: MixtureLike) -> Array:
        ell = model.loglik_joint(Q.particles)           # (n,)
        lp  = model.log_prior(Q.particles)              # (n,)
        return -jnp.sum(Q.w * (ell + lp))

    def grad(self, model, Q: MixtureLike) -> Array:
        # SVGD wasserstein gradient: ∇_θ log p(y|θ)
        return model.score_like_joint(Q.particles)      # (n,d)


# ---- loss and grads ----- 

def squared_loss_and_grad(y: Array, f: Array):
    r = f - y
    return 0.5 * r**2, r

def logistic_loss_and_grad(y: Array, f: Array):
    sig = jnp.clip(1 / (1 + jnp.exp(-f)), 1e-12, 1 - 1e-12)
    ell = -(y * jnp.log(sig) + (1 - y) * jnp.log(1 - sig))
    return ell, (sig - y)


# =============================================
# Deterministic risk loss (for regression/classification)
# =============================================


@dataclass
class DeterministicRiskLoss:
    """
    L(Q) = sum_i ell(y_i, f_Q(x_i))
    - model: any data-bound predictor with predict(.) and dfeatures_pointwise(.)
    - No KL/prior here (VGD handles prior via model.prior separately).
    """
    name: str = "dr"
    display_name: str = "deterministic-risk"

    # Provide a callable: loss_and_grad(y, f) -> (ell_i, dℓ/df_i), both (N,)
    loss_and_grad: Callable[[Array, Array], tuple] = squared_loss_and_grad  # default to regression

    def eval(self, model: Predictor, Q: MixtureLike) -> Array:
        y = jnp.asarray(model.data[model.y_key])   # (N,)
        f = model.predict(Q)                       # (N,)
        ell, _ = self.loss_and_grad(y, f)         # (N,)
        return jnp.sum(ell)                        # keep 'sum' to match your other losses

    def grad(self, model: Predictor, Q: MixtureLike) -> Array:
        """
        ∂L/∂θ_j = Σ_i (∂ℓ_i/∂f_i) * w_j * ∂φ_{j,i}/∂θ_j
        Returns (n, dθ) — particle gradients only (for SVGD update).
        """
        y = jnp.asarray(model.data[model.y_key])     # (N,)
        f = model.predict(Q)                         # (N,)
        _, dL_df = self.loss_and_grad(y, f)          # (N,)

        dphi = model.dfeatures_pointwise(Q)          # (n, N, dθ)
        w = Q.w[:, None, None]                       # (n, 1, 1)
        g = dL_df[None, :, None]                     # (1, N, 1)

        # sum over data -> (n, dθ)
        dtheta = jnp.sum(w * dphi * g, axis=1)
        return -dtheta

# Interpolated Stein Flow loss

class ISFLoss:
    # Must return (n,) each
    def grad_alpha(self, model, Q: SignedMixtureLike) -> Array: ...
    def psi_theta(self, model, Q: SignedMixtureLike) -> Array: ...
    # (Optional) scalar loss for logging:
    def eval(self, model, Q: SignedMixtureLike) -> Array: ...


@dataclass
class ISFDeterministicRiskLoss:
    """
    Interpolated Stein Flow loss for:
        L(Q) = sum_i ell(y_i, f_Q(x_i))
    Protocol:
        - grad_alpha(model, Q): (n,)
        - psi_theta(model, Q): (n,)   (functional derivative evaluated at θ_j)
        - eval(model, Q): scalar
    Assumes the model supplies:
        - model.data with keys (model.x_key, model.y_key)
        - model.predict(Q): (N,)
        - model.features_pointwise(Q): (n, N)  # φ_j(x_i)
    """
    name: str = "isf"
    display_name: str = "interpolated-stein-flow"
    loss_and_grad: Callable[[Array, Array], Tuple[Array, Array]] = squared_loss_and_grad

    # ----- utilities -----
    def _y_f(self, model, Q: SignedMixtureLike) -> Tuple[Array, Array]:
        y = jnp.asarray(model.data[model.y_key])   # (N,)
        f = model.predict(Q)                       # (N,)
        return y, f

    def _grad_signal(self, model, Q: SignedMixtureLike) -> Array:
        """g_i := ∂ℓ/∂f_i, shape (N,)"""
        y, f = self._y_f(model, Q)
        _, dL_df = self.loss_and_grad(y, f)
        return dL_df

    # ----- protocol methods -----
    def eval(self, model, Q: SignedMixtureLike) -> Array:
        y, f = self._y_f(model, Q)
        ell, _ = self.loss_and_grad(y, f)
        return jnp.sum(ell)

    def grad_alpha(self, model, Q: SignedMixtureLike) -> Array:
        """
        ∂L/∂α_j = Σ_i g_i φ_j(x_i)  ==> (n,)
        """
        g = self._grad_signal(model, Q)                 # (N,)
        Phi = model.features_pointwise(Q)               # (n, N)
        return Phi @ g                                  # (n,)

    def psi_theta(self, model, Q: SignedMixtureLike) -> Array:
        """
        ψ(θ_j) = Σ_i g_i φ_{θ_j}(x_i)  ==> identical to grad_alpha for linear-in-α models.
        """
        return self.grad_alpha(model, Q)

    # direct θ-gradient, for debugging vs. RKHS cheap update:
    def grad_theta_direct(self, model, Q: SignedMixtureLike) -> Array:
        """
        ∂L/∂θ_j = Σ_i g_i * α_j * ∂φ_{j,i}/∂θ_j  ==> (n, dθ)
        Not used by the ISF RKHS cheap update, but handy for checks.
        """
        g = self._grad_signal(model, Q)                 # (N,)
        dphi = model.dfeatures_pointwise(Q)             # (n, N, dθ)
        return jnp.sum((Q.alpha[:, None, None] * dphi) * g[None, :, None], axis=1)
