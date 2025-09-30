from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Any, Tuple, Callable

import jax
from jax import Array
import jax.numpy as jnp

from vgd.distribution import Distribution
from vgd.model import Likelihood, Model, Posterior
from vgd.loss import Loss

def pred_mean(theta: Array, x: Array) -> Array:
    """
    Predictive mean matrix with broadcasting:
      theta shape: (n,) or (n,d) with d=1
      x     shape: (N,)
    returns: (n, N)
    """
    theta = jnp.asarray(theta)
    if theta.ndim == 1:
        theta = theta[:, None]  # (n,1)
    x = jnp.asarray(x)[None, :]  # (1,N)
    return theta @ (x ** 2)      # (n,N)


#  Prior: diagonal Normal 
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

# Toy Likelihood using a given pred_mean_fn 
@dataclass
class ToyLikelihood(Likelihood):
    pred_mean_fn: Callable[[Array, Array], Array]  # (theta_batch, x_vec) -> (n, N)

    def _ensure_2d(self, theta: Array) -> Array:
        theta = jnp.asarray(theta)
        return theta[:, None] if theta.ndim == 1 else theta  # (n,d)

    def loglik_pointwise(self, theta: Array, data: Optional[Any] = None) -> Array:
        theta = self._ensure_2d(theta)              # (n,d)
        x = jnp.asarray(data["x"])                  # (N,)
        y = jnp.asarray(data["y"])                  # (N,)
        sigma = jnp.asarray(data["sigma"])          # scalar or (N,)
        mean = self.pred_mean_fn(theta, x)          # (n,N)
        resid = y[None, :] - mean                   # (n,N)
        # Up to constants in y, which cancel in responsibilities across particles
        return -0.5 * (resid / sigma) ** 2          # (n,N)

    def score_like_pointwise(self, theta: Array, data: Optional[Any] = None) -> Array:
        """
        Chain rule: for Gaussian with fixed σ,
          ∂/∂θ log p(y|θ) per obs = ((y - μ)/σ^2) * ∂μ/∂θ.
        We build ∂μ/∂θ via jacobian of pred_mean_fn w.r.t. θ.
        """
        theta = self._ensure_2d(theta)              # (n,d)
        x = jnp.asarray(data["x"])                  # (N,)
        y = jnp.asarray(data["y"])                  # (N,)
        sigma = jnp.asarray(data["sigma"])          # scalar or (N,)

        # μ(θ, x) for all particles
        mean = self.pred_mean_fn(theta, x)          # (n,N)
        err = y[None, :] - mean                     # (n,N)

        # Jacobian J(θ): (n, N, d) where J[i] = ∂μ(θ_i, x)/∂θ_i  (N,d)
        def mu_single(th: Array) -> Array:
            th = jnp.asarray(th)
            if th.ndim == 0:
                th = th[None]        # (1,)
            return self.pred_mean_fn(th[None, ...], x)[0]   # (N,)

        J = jax.vmap(jax.jacrev(mu_single))(theta)  # (n, N, d)

        factor = (err / (sigma ** 2))[:, :, None]   # (n,N,1)
        s_i = factor * J                             # (n,N,d)
        return s_i

    def loglik_joint(self, theta: Array, data: Optional[Any] = None) -> Array:
        return jnp.sum(self.loglik_pointwise(theta, data), axis=1)  # (n,)

    def score_like_joint(self, theta: Array, data: Optional[Any] = None) -> Array:
        s_i = self.score_like_pointwise(theta, data)                # (n,N,d)
        return jnp.sum(s_i, axis=1)                                 # (n,d)

# Helper to generate synthetic data using a pred_mean_fn
def sample_toy_dataset(
    key: Array,
    *,
    x: Array,                                  # (N,)
    prior: Distribution,
    pred_mean_fn: Callable[[Array, Array], Array] = pred_mean,
    true_theta: float | Array | None = None,   # scalar or (d,)
    obs_sd: float = 1.0,
    theta_sd: float = 0.0,                     # per-datum jitter
    per_datum_theta: bool = False,
):
    """
    y_i = μ(θ_i, x_i) + ε_i,  ε_i ~ N(0, obs_sd^2).
    If per_datum_theta=False, a single θ is used across all i (well-specified).
    If per_datum_theta=True, each datum gets its own θ_i (misspecified).
    """
    x = jnp.asarray(x)
    N = x.shape[0]
    key, k_theta, k_obs = jax.random.split(key, 3)

    # Draw θ or θ_i
    if true_theta is None:
        if per_datum_theta:
            # Let prior decide d via shape=(N,d?) — here we use (N,1) by default
            theta = prior.sample(k_theta, shape=(N, 1))  # (N,1) -> treat as (N,d)
        else:
            theta = prior.sample(k_theta, shape=(1, 1)).squeeze()  # scalar or (d,) if you prefer
    else:
        base = jnp.asarray(true_theta)
        if per_datum_theta:
            if base.ndim == 0:  # scalar param
                base = base[None]            # (1,)
            base_row = jnp.broadcast_to(base, (N,) + base.shape)  # (N,d)
            if theta_sd > 0:
                theta = base_row + theta_sd * jax.random.normal(k_theta, base_row.shape)
            else:
                theta = base_row
        else:
            theta = base.reshape(())  # scalar (or reshape to (d,) if vector param)

    # Mean: handle per-datum θ_i vs single θ
    if per_datum_theta:
        # μ_i = pred_mean_fn(θ_i, x_i)
        def mu_one(th_i: Array, x_i: Array) -> Array:
            th_i = jnp.atleast_1d(th_i)                 # (d,)
            mu_i = pred_mean_fn(th_i[None, ...], x_i[None, ...])  # (1,1)
            return mu_i[0, 0]
        mean = jax.vmap(mu_one)(theta, x)               # (N,)
    else:
        # θ shared: μ(θ, x) → (N,)
        th = jnp.atleast_1d(theta)                      # (d,)
        mean = pred_mean_fn(th[None, ...], x)[0]        # (N,)

    y = mean + obs_sd * jax.random.normal(k_obs, (N,))  # (N,)
    data = dict(x=x, y=y, sigma=jnp.asarray(obs_sd))
    return key, theta, data
