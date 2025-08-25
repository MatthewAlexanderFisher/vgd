from jax import grad
from functools import partial
from typing import Optional, Protocol

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from vgd.distribution import Posterior, DiscreteMixture
from vgd.util import logmeanexp_weighted


class Loss(Protocol):
    """
    Base class for loss functions.
    Subclasses should implement the methods for sampling, log probability, and gradients.
    """
    name: str = "base_loss"
    display_name: str = "base_loss"

    def eval(self, model: Posterior, Q: DiscreteMixture) -> Array: ...
    def grad(self, model: Posterior, Q: DiscreteMixture) -> Array: ...
    def weights(self, model: Posterior, Q: DiscreteMixture, mean: bool) -> Array: ...


class PredictionCentricLoss:
    name = "pc"; display_name = "prediction-centric"

    def __init__(self, alpha: float = 1.0):  
        # alpha is a tempering variable
        self.alpha = alpha

    def eval(self, model: Posterior, Q: DiscreteMixture) -> Array:
        logw = jnp.log(Q.w + 1e-300)
        LL = model.loglik_pointwise(Q.particles)                    # (n,N)
        # −∑_i log ∫ p(y_i|θ) dQ(θ) = −∑_i log ∑_j w_j e^{ℓ_{j,i}}
        val = - jnp.sum(logmeanexp_weighted(LL, logw, axis=0))
        return val

    def grad(self, model: Posterior, Q: DiscreteMixture) -> Array:
        r = self.weights(model, Q, mean=False)  # (n,N)

        S_i = model.score_like_pointwise(Q.particles)               # (n,N,d)
        return jnp.einsum('jI,jId->jd', r, S_i)             # (n,d)

    def weights(self, model: Posterior, Q: DiscreteMixture, mean: bool = True) -> Array:
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

    def eval(self, model, Q) -> Array:
        logw = jnp.log(Q.w + 1e-300)                    # (n,)
        ell  = model.loglik_joint(Q.particles)          # (n,)
        # -log ∑_j w_j e^{ℓ_j}
        return -logsumexp(logw + ell)

    def grad(self, model, Q) -> Array:
        r = self.weights(model, Q)  # (n,)
        S = model.score_like_joint(Q.particles)      # (n,d)
        return r[:, None] * S                           # (n,d)

    def weights(self, model: Posterior, Q: DiscreteMixture,  mean: bool=True) -> Array:
        """r_j ∝ w_j * exp(alpha * ℓ_j),  ℓ_j = sum_i log p(y_i|θ_j)."""
        loglik_joint = model.loglik_joint(Q.particles)                 # (n,)
        a   = jnp.log(Q.w + 1e-300) + self.alpha * loglik_joint             # (n,)
        r   = jax.nn.softmax(a, axis=0)                     # (n,)
        return jax.lax.stop_gradient(r)



class PosteriorKLLoss:
    name = "svgd"; display_name = "posterior-kl"

    def eval(self, model, Q) -> Array:
        ell = model.loglik_joint(Q.particles)           # (n,)
        lp  = model.log_prior(Q.particles)              # (n,)
        return -jnp.sum(Q.w * (ell + lp))

    def grad(self, model, Q) -> Array:
        # SVGD wasserstein gradient: ∇_θ log p(y|θ)
        return model.score_like_joint(Q.particles)      # (n,d)
