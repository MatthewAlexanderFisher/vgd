
from __future__ import annotations
from typing import Optional, Protocol, Tuple, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from vgd.distribution import Distribution, FourierParticles, MixtureLike, SignedMixtureLike, PAdicMixtureLike

class Model(Protocol):

    @property
    def prior(self) -> Distribution: ...

    def sample(self, key: Array, shape: Tuple[int, ...] = ()) -> Array: ...
    def log_prob(self, x: Array) -> Array: ...
    def grad_log_prob(self, x: Array) -> Array: ...   # shape == x

    # Prior pieces (just delegate to Distribution)
    def log_prior(self, theta: Array) -> Array:
        return self.prior.log_prob(theta)
    def score_prior(self, theta: Array) -> Array:
        return self.prior.grad_log_prob(theta)


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
    
    def n(self):
        return self.data["x"].shape[0] if self.data is not None else 0


class Predictor(Protocol):
    """Data-bound model. Provides features and predictions for its bound data."""
    prior: Distribution
    data: Any                  # e.g. dict(x=..., y=...)
    x_key: str
    y_key: str

    def set_data(self, data: Any) -> "Predictor": ...
    def n(self) -> int: ...

    # All use predictor.data internally (no 'data' arg needed externally)
    def features_pointwise(self, Q: MixtureLike) -> Array: ...
    def dfeatures_pointwise(self, Q: MixtureLike) -> Array: ...
    def predict(self, Q: MixtureLike) -> Array: ...
    def predict_on_x(self, Q: MixtureLike, x: Array) -> Array: ...

    def score_prior(self, theta: Array) -> Array:
        return self.prior.grad_log_prob(theta)

# --------------------------------
# The R^d "Fourier" predictor
# --------------------------------
@dataclass
class FourierPredictor:
    prior_plus:  "Distribution"   # grad_log_prob(theta) -> (n, d+1)
    prior_minus: "Distribution"
    data: Any
    x_key: str = "x"
    y_key: str = "y"

    def set_data(self, data: Any) -> "FourierPredictor":
        self.data = data; return self

    # reference scores for SKL Stein fields (SHOULD be (n, d+1))
    def score_ref_plus(self, theta: Array) -> Array:
        return self.prior_plus.grad_log_prob(theta)   # (n, d+1)

    def score_ref_minus(self, theta: Array) -> Array:
        return self.prior_minus.grad_log_prob(theta)  # (n, d+1)

    def n(self) -> int:
        return int(jnp.asarray(self.data[self.x_key]).shape[0])

    def _X(self) -> Array:
        return jnp.asarray(self.data[self.x_key])   # (N,d)

    def _proj(self, Q: SignedMixtureLike, X: Array) -> Array:
        omega = Q.particles[:, :-1]                 # (n,d)
        b     = Q.particles[:, -1]                  # (n,)
        return (X @ omega.T + b[None, :]).T         # (n,N)

    def features_pointwise(self, Q: SignedMixtureLike) -> Array:
        X = self._X()
        return jnp.cos(self._proj(Q, X))            # (n,N)

    def dfeatures_pointwise(self, Q: SignedMixtureLike) -> Array:
        X = self._X()
        proj = self._proj(Q, X)                     # (n,N)
        S = -jnp.sin(proj)                          # (n,N)
        dphi_domega = S[:, :, None] * X[None, :, :] # (n,N,d)
        dphi_db     = S[:, :, None]                 # (n,N,1)
        return jnp.concatenate([dphi_domega, dphi_db], axis=-1)  # (n,N,d+1)

    def predict(self, Q: SignedMixtureLike) -> Array:
        Phi = self.features_pointwise(Q)            # (n,N)
        return Phi.T @ Q.alpha                      # (N,)

    def predict_on_x(self, Q: SignedMixtureLike, X: Array) -> Array:
        """
        Predict on an explicit input array X (ignores self.data).

        Args:
            Q: SignedMixtureLike with fields `particles` (n, d+1) and `alpha` (n,)
            X: (N,d) array, or:
               - (d,) -> treated as a single d-dim sample
               - (N,) with d==1 -> treated as N scalar samples

        Returns:
            (N,) predictions f_Q(X)
        """
        X = jnp.asarray(X)
        d = int(Q.particles.shape[1] - 1)  # feature dimension

        if X.ndim == 1:
            if X.size == d:
                X = X[None, :]
            elif d == 1:
                X = X[:, None]
            else:
                raise ValueError(
                    f"Ambiguous 1D X of shape {X.shape} for d={d}. "
                    "Provide (N,d) or (d,) explicitly."
                )

        proj = (X @ Q.particles[:, :-1].T + Q.particles[:, -1][None, :])  # (N,n)
        feats = jnp.cos(proj)                                              # (N,n)
        return feats @ Q.alpha                                             # (N,)

# --------------------------------
# The p-adic "Fourier" predictor
# --------------------------------
@dataclass
class PAdicFourierPredictor:
    p: int
    m: int
    data: Any
    x_key: str = "x"
    y_key: str = "y"
    use_sin: bool = True  # <-- default True: cos & sin features

    def set_data(self, data: Any): self.data = data; return self
    def modulus(self) -> int: return int(self.p ** self.m)
    def n(self) -> int: return int(jnp.asarray(self._X()).shape[0])

    def _X(self) -> jnp.ndarray:
        X = jnp.asarray(self.data[self.x_key])
        if X.ndim == 1:
            X = X[:, None]
        return X.astype(jnp.int64)

    def _mod_inner(self, X: jnp.ndarray, Xi: jnp.ndarray) -> jnp.ndarray:
        M = self.modulus()
        prod = (X[:, None, :] * Xi[None, :, :]) % M   # (N,n,d)
        return jnp.sum(prod, axis=-1) % M             # (N,n)

    def _angle(self, Q: PAdicMixtureLike, X: jnp.ndarray) -> jnp.ndarray:
        M = self.modulus()
        T = self._mod_inner(X.astype(jnp.int64), Q.Xi.astype(jnp.int64))  # (N,n)
        phase = Q.phi  # no modulus for differentiation; cos is periodic
        return (2.0 * jnp.pi / M) * (T + phase[None, :])

    def features_pointwise(self, Q: PAdicMixtureLike) -> jnp.ndarray:
        X = self._X()
        ang = self._angle(Q, X)    # (N,n)
        C = jnp.cos(ang)           # (N,n)
        if not self.use_sin:
            return C.T             # (n,N)
        S = jnp.sin(ang)           # (N,n)
        return jnp.concatenate([C, S], axis=1).T  # (2n, N)

    def predict(self, Q: PAdicMixtureLike) -> jnp.ndarray:
        Phi = self.features_pointwise(Q)     # (n or 2n, N)
        return (Phi.T @ Q.alpha).reshape(-1)

    def predict_on_x(self, Q: PAdicMixtureLike, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.asarray(X)
        if X.ndim == 1: X = X[:, None]
        ang = self._angle(Q, X.astype(jnp.int64))  # (N,n)
        C = jnp.cos(ang)
        feats = C if not self.use_sin else jnp.concatenate([C, jnp.sin(ang)], axis=1)
        return (feats @ Q.alpha).reshape(-1)

    def sample_frequencies(self, n: int, d: int, key: Array,
                           min_scale: int = 0, max_scale: Optional[int] = None) -> jnp.ndarray:
        """(Kept for convenience; not used in the spectrum init path.)"""
        M = self.modulus()
        rmax = self.m - 1 if max_scale is None else int(max_scale)
        if not (0 <= min_scale <= rmax <= self.m - 1):
            raise ValueError("Require 0 <= min_scale <= rmax <= m-1.")
        k1, k2, k3 = jax.random.split(key, 3)
        r = jax.random.randint(k1, shape=(n, d), minval=min_scale, maxval=rmax + 1)
        base = jax.random.randint(k2, shape=(n, d), minval=0, maxval=self.p ** (self.m - r))
        Xi = (base * (self.p ** r)).astype(jnp.int64) % M
        all_zero = jnp.all(Xi == 0, axis=1, keepdims=True)
        Xi = jnp.where(all_zero, jax.random.randint(k3, (n, d), 0, M), Xi)
        return Xi

    def pick_freqs_via_sampled_dft(self, X_residues, y, M: int, k: int) -> jnp.ndarray:
        """
        Select top-k frequencies by | sum_i y_i exp(-2πi k X_i / M) |.
        Returns shape (k,1) int64 frequency indices in [0, M-1].
        """
        X = X_residues.reshape(-1).astype(jnp.int64)  # (N,)
        y = y.reshape(-1)                              # (N,)
        ks = jnp.arange(M, dtype=jnp.int64)           # (M,)
        angles = (2.0 * jnp.pi / M) * (X[:, None] * ks[None, :])  # (N,M)
        # exp(-i * angles) = cos - i sin
        E = jnp.cos(angles) - 1j * jnp.sin(angles)                 # (N,M), complex
        Ghat = jnp.sum(y[:, None] * E, axis=0)                     # (M,), complex
        mags = jnp.abs(Ghat)                                       # (M,)
        mags = mags.at[0].set(0.0)                                 
        topk = jnp.argsort(mags)[-k:]
        return topk.astype(jnp.int64)[:, None]  # (k,1)
