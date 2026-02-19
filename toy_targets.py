from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Any, Tuple, Callable

import jax
from jax import Array
import jax.numpy as jnp


def logpi_std_normal(x: Array) -> Array:
    """Standard Gaussian target in R^D (up to const)."""
    return -0.5 * jnp.sum(x**2, axis=-1)

def logpi_mixture(x: Array) -> Array:
    """Simple two-mode Gaussian mixture (unnormalised log)."""
    def logsumexp(a, axis=-1):
        m = jnp.max(a, axis=axis, keepdims=True)
        return (m + jnp.log(jnp.sum(jnp.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)
    comp1 = -0.5 * jnp.sum((x - 3.0)**2, axis=-1)
    comp2 = -0.5 * jnp.sum((x + 3.0)**2, axis=-1)
    # log(0.5 e^{comp1} + 0.5 e^{comp2})
    return logsumexp(jnp.stack([comp1, comp2], axis=-1) - jnp.log(2.0), axis=-1)
