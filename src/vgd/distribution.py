from __future__ import annotations
from typing import Optional, Protocol, Tuple, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


class Distribution(Protocol):
    def sample(self, key: Array, shape: Tuple[int, ...] = ()) -> Array: ...
    def log_prob(self, x: Array) -> Array: ...
    def grad_log_prob(self, x: Array) -> Array: ...   # shape == x

class MixtureLike(Protocol):
    """A lightweight 'distribution' of particles with weights."""
    particles: Array   # (n, dθ)
    w: Array           # (n,)

    def replace_particles(self, new_particles: Array) -> "MixtureLike":
        obj = object.__new__(type(self))
        obj.particles = new_particles
        obj.w = self.w
        return obj
    
    def replace_weights(self, new_w: Array) -> "MixtureLike":
        obj = object.__new__(type(self))
        obj.particles = self.particles
        obj.w = new_w
        return obj

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

    def replace_weights(self, new_w: Array) -> "MixtureLike":
        obj = object.__new__(type(self))
        obj.particles = self.particles
        obj.w = new_w
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

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class FourierParticles:
    particles: Array   # (n, d+1): [omega (d,), b (1,)]
    w: Array           # (n,)

    def __init__(self, particles: Array, w: Optional[Array] = None):
        particles = jnp.asarray(particles)
        if particles.ndim != 2:
            raise ValueError("particles must have shape (n, d+1)")
        n = particles.shape[0]
        if w is None:
            w_arr = jnp.full((n,), 1.0 / n)
        else:
            w = jnp.asarray(w)
            w_arr = w / (jnp.sum(w) + 1e-300)
        self.particles = particles
        self.w = w_arr

    def normalised(self) -> "FourierParticles":
        s = jnp.sum(self.w, keepdims=True)
        return FourierParticles(self.particles, self.w / (s + 1e-300))

    def replace_particles(self, new_particles: Array) -> "FourierParticles":
        obj = object.__new__(type(self))
        obj.particles = new_particles
        obj.w = self.w
        return obj

    def replace_weights(self, new_w: Array) -> "FourierParticles":
        obj = object.__new__(type(self))
        obj.particles = self.particles
        obj.w = new_w
        return obj

    # pytree plumbing
    def tree_flatten(self):
        return (self.particles, self.w), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.particles, obj.w = children
        return obj

    @classmethod
    def axes(cls, p_axis, w_axis):
        obj = object.__new__(cls)
        obj.particles = p_axis
        obj.w = w_axis
        return obj


@jax.tree_util.register_pytree_node_class
@dataclass
class SignedMixtureLike:
    particles: Array   # (n, dθ)
    alpha: Array       # (n,) signed weights

    # make sure fields are JAX arrays
    def __post_init__(self):
        self.particles = jnp.asarray(self.particles)
        self.alpha     = jnp.asarray(self.alpha)

    # immutable-style updaters
    def replace_particles(self, new_particles: Array) -> "SignedMixtureLike":
        return SignedMixtureLike(new_particles, self.alpha)

    def replace_alpha(self, new_alpha: Array) -> "SignedMixtureLike":
        return SignedMixtureLike(self.particles, new_alpha)

    # convenience
    def sign_masks(self):
        m_plus  = (self.alpha > 0.0)
        m_minus = (self.alpha < 0.0)
        return m_plus, m_minus

    def per_sign_weights(self, normalise: bool = True, eps: float = 1e-12):
        a = self.alpha
        a_plus  = jnp.where(a > 0,  a, 0.0)
        a_minus = jnp.where(a < 0, -a, 0.0)
        if normalise:
            m_plus  = jnp.sum(a_plus)  + eps
            m_minus = jnp.sum(a_minus) + eps
            w_plus  = a_plus  / m_plus
            w_minus = a_minus / m_minus
        else:
            w_plus, w_minus = a_plus, a_minus
        return w_plus, w_minus

    # ---- PyTree plumbing ----
    def tree_flatten(self):
        # children must be JAX types
        return (self.particles, self.alpha), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        particles, alpha = children
        return cls(particles=particles, alpha=alpha)

@jax.tree_util.register_pytree_node_class
@dataclass
class PAdicMixtureLike:
    Xi: Array   # (n,d) int64 in [0, M-1]
    phi: Array  # (n,)   real phases (unused when use_sin=True)
    alpha: Array  # (n,) or (2n,) if cos+sin
    def replace_particles(self, new_Xi):   return PAdicMixtureLike(new_Xi, self.phi, self.alpha)
    def replace_alpha(self, new_alpha):    return PAdicMixtureLike(self.Xi, self.phi, new_alpha)
    def replace_phi(self, new_phi):        return PAdicMixtureLike(self.Xi, new_phi, self.alpha)
    def sign_masks(self):
        a = self.alpha
        return (a > 0.0), (a < 0.0)
    def per_sign_weights(self, normalise: bool = True, eps: float = 1e-12):
        a = self.alpha
        a_plus  = jnp.where(a > 0,  a, 0.0)
        a_minus = jnp.where(a < 0, -a, 0.0)
        if normalise:
            m_plus  = jnp.sum(a_plus)  + eps
            m_minus = jnp.sum(a_minus) + eps
            w_plus  = a_plus  / m_plus
            w_minus = a_minus / m_minus
        else:
            w_plus, w_minus = a_plus, a_minus
        return w_plus, w_minus
    # pytree plumbing
    def tree_flatten(self): return (self.Xi, self.phi, self.alpha), None
    @classmethod
    def tree_unflatten(cls, aux, children):
        Xi, phi, alpha = children
        return cls(Xi=Xi, phi=phi, alpha=alpha)




# ============= Simple distributions

# Normal with diagonal covariance
@dataclass
class NormalDiag(Distribution):
    mean: Array | float = 0.0
    sd:   Array | float = 1.0

    def _as_arrays(self):
        m = jnp.asarray(self.mean)
        s = jnp.asarray(self.sd)
        return m, s

    def sample(self, key: Array, shape: Tuple[int, ...] = ()) -> Array:
        m, s = self._as_arrays()
        z = jax.random.normal(key, shape)
        return m + s * z

    def log_prob(self, x: Array) -> Array:
        x = jnp.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        m, s = self._as_arrays()
        diff = x - m
        lp_elem = -0.5 * (diff / s) ** 2 - jnp.log(s) - 0.5 * jnp.log(2.0 * jnp.pi)
        return jnp.sum(lp_elem, axis=-1)

    def grad_log_prob(self, x: Array) -> Array:
        x = jnp.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        m, s = self._as_arrays()
        return -(x - m) / (s ** 2)
