import functools
import jax, jax.numpy as jnp
from jax import Array
from typing import  NamedTuple

class KernelParams(NamedTuple):
    lengthscale: Array
    amplitude:  Array
    gamma:      Array | None = None
    beta:       Array | None = None


class Kernel:
    name: str = "base_kernel"
    display_name: str = "base_kernel"

    @staticmethod
    def _phi(r: Array, params: KernelParams) -> Array:
        raise NotImplementedError

    @staticmethod
    def _psi(r: Array, params: KernelParams) -> Array:
        """Return φ′(r)/r (finite at r=0)."""
        raise NotImplementedError

    def _kernel_pair(self, x: Array, y: Array, params: KernelParams) -> Array:
        r = jnp.linalg.norm(x - y)
        return self._phi(r, params)

    def _grad_kernel_pair(self, x: Array, y: Array, params: KernelParams) -> Array:
        u = x - y
        r = jnp.linalg.norm(u)
        return self._psi(r, params) * u

    def __call__(self, X: Array, Y: Array, params: KernelParams) -> tuple[Array, Array]:
        k_pair = functools.partial(self._kernel_pair, params=params)
        g_pair = functools.partial(self._grad_kernel_pair, params=params)
        K = jax.vmap(lambda x: jax.vmap(lambda y: k_pair(x, y))(Y))(X)  # (n,m)
        G = jax.vmap(lambda x: jax.vmap(lambda y: g_pair(x, y))(Y))(X)  # (n,m,d)
        return K, G

class Matern52Kernel(Kernel):
    name: str = "Matern52"
    display_name: str = "Matérn 5/2"

    @staticmethod
    def _phi(r, params: KernelParams) -> Array:
        ell = params.lengthscale
        A   = params.amplitude
        c = jnp.sqrt(5.0) / ell
        return A * (1.0 + c * r + (c**2) * (r**2) / 3.0) * jnp.exp(-c * r)

    @staticmethod
    def _psi(r, params: KernelParams) -> Array:
        ell = params.lengthscale
        amp  = params.amplitude
        c = jnp.sqrt(5.0) / ell
        # ψ(r) = φ′(r)/r, finite at r=0
        return -amp * (c**2 / 3.0) * (1.0 + c * r) * jnp.exp(-c * r)


class Matern72Kernel(Kernel):
    name: str = "Matern72"
    display_name: str = "Matérn 7/2"

    @staticmethod
    def _phi(r, params: KernelParams) -> Array:

        ell = params.lengthscale
        amp   = params.amplitude

        c = jnp.sqrt(7.0) / ell  # c = √7 / ℓ
        return (
            amp
            * (1.0 + c * r + (c**2 / 3.0) * r**2 + (c**3 / 15.0) * r**3)
            * jnp.exp(-c * r)
        )

    @staticmethod
    def _psi(r, params: KernelParams):
        ell = params.lengthscale
        amp   = params.amplitude

        c = jnp.sqrt(7.0) / ell
        # ϕ′(r)/r = -σ² (c²/15) · (5 + 2 c r + (c r)²) · e^{-c r}
        return (
            -amp
            * (c**2 / 15.0)
            * (5.0 + 2.0 * c * r + (c * r) ** 2)
            * jnp.exp(-c * r)
        )

class RBFKernel(Kernel):
    name: str = "Gaussian"
    display_name: str = "Gaussian (RBF)"

    @staticmethod
    def _phi(r: Array, params: KernelParams) -> Array:
        ell = params.lengthscale
        A   = params.amplitude
        z = -0.5 * (r / ell) ** 2
        return A * jnp.exp(z)

    @staticmethod
    def _psi(r: Array, params: KernelParams) -> Array:
        ell = params.lengthscale
        A   = params.amplitude
        z = -0.5 * (r / ell) ** 2
        base = A * jnp.exp(z)
        return -base / ell**2


class IMQKernel(Kernel):
    name: str = "IMQ"
    display_name: str = "Inverse Multiquadric (IMQ)"

    @staticmethod
    def _phi(r: Array, params: KernelParams) -> Array:

        if params.gamma is None or params.beta is None:
            raise ValueError("IMQKernel needs non-None gamma and beta")

        ell = params.lengthscale
        amp   = params.amplitude
        g   = params.gamma
        b   = params.beta
        u = g**2 + (r / ell) ** 2
        return amp * u ** (-b)

    @staticmethod
    def _psi(r: Array, params: KernelParams) -> Array:

        if params.gamma is None or params.beta is None:
            raise ValueError("IMQKernel needs non-None gamma and beta")

        ell = params.lengthscale
        A   = params.amplitude
        g   = params.gamma
        b   = params.beta
        u = g**2 + (r / ell) ** 2
        # φ(r) = A u^{-b}; dφ/dr = A(-b) u^{-b-1} * (2r/ell^2)  ⇒  (φ′/r) = A(-b)*2/ell^2 * u^{-b-1}
        return -2.0 * b * A * (1.0 / ell**2) * u ** (-(b + 1.0))
