from typing import Optional, Protocol, Tuple, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


class Distribution(Protocol):
    def sample(self, key: Array, shape: Tuple[int, ...] = ()) -> Array: ...
    def log_prob(self, x: Array) -> Array: ...
    def grad_log_prob(self, x: Array) -> Array: ...   # shape == x


class Likelihood(Protocol):
    # joint over the full dataset in `data`
    def loglik_joint(self, theta: Array, data: Optional[Any] = None) -> Array: ...
    def score_like_joint(self, theta: Array, data: Optional[Any] = None) -> Array: ...
    # optional per-observation pieces (needed for prediction-centric losses)
    def loglik_pointwise(self, theta: Array, data: Optional[Any] = None) -> Array: ...
    def score_like_pointwise(self, theta: Array, data: Optional[Any] = None) -> Array: ...

@dataclass
class Posterior:
    like: Likelihood
    prior: Distribution
    data: Optional[Any] = None  # e.g. dict(x=..., y=..., sigma=...)

    # Prior pieces (just delegate to Distribution)
    def log_prior(self, theta: Array) -> Array:
        return self.prior.log_prob(theta)
    def score_prior(self, theta: Array) -> Array:
        return self.prior.grad_log_prob(theta)

    # Posterior pieces (for standard SVGD / monitoring)
    def log_posterior(self, theta: Array) -> Array:
        return self.like.loglik_joint(theta, self.data) + self.log_prior(theta)
    def score_posterior(self, theta: Array) -> Array:
        return self.like.score_like_joint(theta, self.data) + self.score_prior(theta)

    # Convenience passthroughs used by losses
    def loglik_joint(self, theta: Array) -> Array:
        return self.like.loglik_joint(theta, self.data)
    def score_like_joint(self, theta: Array) -> Array:
        return self.like.score_like_joint(theta, self.data)
    def loglik_pointwise(self, theta: Array) -> Array:
        return self.like.loglik_pointwise(theta, self.data)
    def score_like_pointwise(self, theta: Array) -> Array:
        return self.like.score_like_pointwise(theta, self.data)

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class DiscreteMixture:
    particles: Array          # (..., n,d) or (..., n)
    w: Array                  # (..., n)

    def __init__(self, particles: Array, w: Optional[Array] = None):
        particles = jnp.asarray(particles)
        if particles.ndim == 1:
            particles = particles[:, None]  # (n,1)

        n_particles = particles.shape[-2]       # n
        w_shape = particles.shape[:-1]          # (..., n)

        if w is None:
            w_arr = jnp.full(w_shape, 1.0 / n_particles)
        else:
            w = jnp.asarray(w)
            s = jnp.sum(w, axis=-1, keepdims=True)
            w_arr = w / (s + 1e-300)

        self.particles = particles
        self.w = w_arr

    def normalised(self) -> "DiscreteMixture":
        # idempotent: normalise along last axis of w
        s = jnp.sum(self.w, axis=-1, keepdims=True)
        return DiscreteMixture(self.particles, self.w / (s + 1e-300))

    # helper to update particles without touching / renormalising weights
    def replace_particles(self, new_particles: Array) -> "DiscreteMixture":
        obj = object.__new__(type(self))
        obj.particles = new_particles
        obj.w = self.w
        return obj

    # pytree plumbing 
    def tree_flatten(self):
        return (self.particles, self.w), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)  # bypass __init__
        obj.particles, obj.w = children
        return obj

    @classmethod
    def axes(cls, p_axis, w_axis):
        """Build an in_axes safely (ints as leaves; no __init__/__post_init__)."""
        obj = object.__new__(cls)
        obj.particles = p_axis
        obj.w = w_axis # can vmap e.g. f_batched = jax.vmap(f, in_axes=(DiscreteMixture.axes(0, 0),), out_axes=0)
        return obj
