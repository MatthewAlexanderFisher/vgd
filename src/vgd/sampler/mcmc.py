from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad
from jax import Array


LogProbFn = Callable[[Array], Array]  # maps (batch,D) -> (batch,)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def mvn_iso_logpdf(x: Array, mean: Array, std: float) -> Array:
    """
    Isotropic Gaussian log-density for each row x ~ N(mean, std^2 I).
    Shapes: x, mean: (B,D). Returns (B,)
    """
    D = x.shape[-1]
    diff = (x - mean) / std
    return -0.5 * jnp.sum(diff**2, axis=-1) - D * jnp.log(std) - 0.5 * D * jnp.log(2.0 * jnp.pi)

def stable_logsigmoid(z: Array) -> Array:
    return -jax.nn.softplus(-z)

# ------------------------------------------------------------
# Acceptance rules
# ------------------------------------------------------------

def mh_accept_prob(logpi_x: Array, logpi_y: Array, logq_xy: Array, logq_yx: Array) -> Array:
    """
    Metropolis–Hastings acceptance: alpha = min(1, exp(Δ))
    where Δ = logπ(y)-logπ(x) + log q(y->x) - log q(x->y)
    """
    delta = (logpi_y - logpi_x) + (logq_yx - logq_xy)
    # log-accept = min(0, Δ), then exp
    return jnp.exp(jnp.minimum(0.0, -jax.nn.softplus(-delta) + delta))  # numerically stable = jnp.minimum(1, jnp.exp(delta))

def barker_accept_prob(logpi_x: Array, logpi_y: Array, logq_xy: Array, logq_yx: Array) -> Array:
    """
    Barker acceptance: alpha = 1 / (1 + exp(-Δ)) = sigmoid(Δ)
    """
    delta = (logpi_y - logpi_x) + (logq_yx - logq_xy)
    return jax.nn.sigmoid(delta)

# ------------------------------------------------------------
# Proposals
#   Each returns: (y, logq_xy, logq_yx)
#   Inputs:
#     key      : PRNGKey
#     x        : (B,D)
#     params   : dict of hyperparameters
#     logpi_fn : optional, only needed for Langevin drift via grad logπ
# ------------------------------------------------------------

def rwm_propose(key: Array,
                x: Array,
                std: float) -> Tuple[Array, Array, Array]:
    """
    Random-Walk Metropolis (isotropic Gaussian).
    y ~ N(x, std^2 I)
    """
    B, D = x.shape
    key_eps = key
    eps = random.normal(key_eps, shape=(B, D)) * std
    y = x + eps
    # symmetric, so logq(x->y) == logq(y->x)
    logq_xy = mvn_iso_logpdf(y, x, std)
    logq_yx = mvn_iso_logpdf(x, y, std)
    return y, logq_xy, logq_yx

def mala_propose(key: Array,
                 x: Array,
                 logpi_fn: LogProbFn,
                 step_size: float) -> Tuple[Array, Array, Array]:
    """
    Unadjusted Langevin / MALA proposal:
      y ~ N(x + (h/2) * ∇ log π(x), h I), with h = step_size^2
    """
    h = step_size**2
    score = grad(lambda z: logpi_fn(z).sum())(x)     # (B,D)
    mean_xy = x + 0.5 * h * score
    eps = random.normal(key, shape=x.shape) * step_size
    y = mean_xy + eps

    # reverse proposal densities need ∇ log π(y)
    score_y = grad(lambda z: logpi_fn(z).sum())(y)
    mean_yx = y + 0.5 * h * score_y

    logq_xy = mvn_iso_logpdf(y, mean_xy, step_size)
    logq_yx = mvn_iso_logpdf(x, mean_yx, step_size)
    return y, logq_xy, logq_yx


# ------------------------------------------------------------
# End-to-end step helpers
# ------------------------------------------------------------

def mh_step(key: Array,
            x: Array,
            logpi_fn: LogProbFn,
            propose_fn: Callable[..., Tuple[Array, Array, Array]],
            accept_fn: Callable[[Array, Array, Array, Array], Array],
            *prop_args, **prop_kwargs) -> Array:
    """
    One vectorised MH-like step using a given proposal and acceptance.
    """
    y, logq_xy, logq_yx = propose_fn(key, x, *prop_args, **prop_kwargs)
    logpi_x = logpi_fn(x)
    logpi_y = logpi_fn(y)
    alpha = accept_fn(logpi_x, logpi_y, logq_xy, logq_yx)  # (B,)
    u = random.uniform(key, shape=alpha.shape)
    accept = (u < alpha)
    x_next = jnp.where(accept[:, None], y, x)
    return x_next



def softplus(x: Array) -> Array:
    return jax.nn.softplus(x)

# ------------------------------------------------------------
# Algorithm 1 (1D Barker proposal) - vectorised
# Draw z ~ mu_sigma on R_+, choose direction with p = sigmoid(z * dlogpi(x)),
# return y = x + b * z
# ------------------------------------------------------------
def sample_mu_sigma_abs_normal(key: Array, shape, sigma: float) -> Array:
    """Default choice for μ_σ: folded Normal(|N(0,σ^2)|) on R_+."""
    return jnp.abs(random.normal(key, shape=shape)) * sigma

def barker_alg1_1d(
    key: Array,
    x: Array,               # shape (B,)  -- 1D points
    dlogpi_x: Array,        # shape (B,)  -- derivative ∂ log π(x)
    sigma: float,
    mu_sampler: Callable[[Array, Tuple[int, ...], float], Array] = sample_mu_sigma_abs_normal,
) -> Array:
    """
    Vectorised Algorithm 1 on R:
      1) z ~ μ_σ (on R_+)
      2) p = sigmoid(z * dlogpi(x))
      3) b in {+1,-1} with P(+1)=p
      4) y = x + b * z
    """
    B = x.shape[0]
    k_z, k_u = random.split(key)
    z = mu_sampler(k_z, (B,), sigma)             # (B,)
    p = jax.nn.sigmoid(z * dlogpi_x)             # (B,)
    u = random.uniform(k_u, shape=(B,))
    b = jnp.where(u < p, 1.0, -1.0)              # (B,)
    y = x + b * z
    return y

# ------------------------------------------------------------
# Algorithm 2 (Rd Barker proposal) - coordinate-wise independent Alg 1
# ------------------------------------------------------------
def barker_alg2_propose(
    key: Array,
    x: Array,               # (B,D)
    logpi_fn: LogProbFn,    # (B,D)->(B,)
    sigma: float,
    mu_sampler_1d: Callable[[Array, Tuple[int, ...], float], Array] = sample_mu_sigma_abs_normal,
) -> Array:
    """
    Implements Algorithm 2 proposal on R^D:
    For each coordinate i, apply Algorithm 1 with ∂_i log π(x).
    Returns proposed y (B,D).
    """
    B, D = x.shape
    # ∇ log π(x) (B,D)
    score_x = grad(lambda Z: logpi_fn(Z).sum())(x)
    # Make per-coordinate keys
    keys = random.split(key, D)
    # vmap Alg1 across coordinates; each coordinate uses its ∂_i log π
    def one_coord(k_i, x_i, grad_i):
        # shapes (B,)
        return barker_alg1_1d(k_i, x_i, grad_i, sigma, mu_sampler_1d)

    # Move last axis to first to vmap over coords, then transpose back
    x_T = jnp.swapaxes(x, 0, 1)          # (D,B)
    g_T = jnp.swapaxes(score_x, 0, 1)    # (D,B)
    y_T = jax.vmap(one_coord)(keys, x_T, g_T)  # (D,B)
    y = jnp.swapaxes(y_T, 0, 1)          # (B,D)
    return y

# ------------------------------------------------------------
# Barker acceptance α_B(x,y) (their Eq. 18) - vectorised
# α_B = min(1, π(y)/π(x) * Π_i [(1+e^{(x_i-y_i)∂_i logπ(x)})/(1+e^{(y_i-x_i)∂_i logπ(y)})])
# We compute in log-space stably using softplus.
# ------------------------------------------------------------
def barker_alpha_alg2(
    x: Array, y: Array, logpi_fn: LogProbFn
) -> Array:
    """
    Compute Barker acceptance (Eq. 18) in a numerically stable log-domain.
    Returns alpha in (0,1], shape (B,).
    """
    logpi_x = logpi_fn(x)                          # (B,)
    logpi_y = logpi_fn(y)                          # (B,)
    score_x = grad(lambda Z: logpi_fn(Z).sum())(x) # (B,D)
    score_y = grad(lambda Z: logpi_fn(Z).sum())(y) # (B,D)

    diff_xy = x - y                                 # (B,D)
    term_num  = softplus(diff_xy * score_x).sum(axis=-1)   # Σ_i log(1+exp((x_i - y_i)*∂_i logπ(x)))
    term_denom= softplus((-diff_xy) * score_y).sum(axis=-1)# Σ_i log(1+exp((y_i - x_i)*∂_i logπ(y)))

    log_ratio = (logpi_y - logpi_x) + (term_num - term_denom)   # (B,)
    # alpha = min(1, exp(log_ratio))
    # Compute stably: if log_ratio >= 0 => alpha=1; else exp(log_ratio)
    alpha = jnp.where(log_ratio >= 0.0, 1.0, jnp.exp(log_ratio))
    return alpha

# ------------------------------------------------------------
# One Barker step (Algorithm 2): propose via Alg 2, accept with α_B
# ------------------------------------------------------------
def barker_step_alg2(
    key: Array,
    x: Array,               # (B,D)
    logpi_fn: LogProbFn,
    sigma: float,
    mu_sampler_1d: Callable[[Array, Tuple[int, ...], float], Array] = sample_mu_sigma_abs_normal,
) -> Array:
    """
    One transition of Algorithm 2:
      y ~ Barker proposal (coordinate-wise Alg 1),
      accept with α_B(x,y).
    """
    k_prop, k_u = random.split(key)
    y = barker_alg2_propose(k_prop, x, logpi_fn, sigma, mu_sampler_1d)
    alpha = barker_alpha_alg2(x, y, logpi_fn)          # (B,)
    u = random.uniform(k_u, shape=alpha.shape)
    accept = (u < alpha)
    x_next = jnp.where(accept[:, None], y, x)
    return x_next
