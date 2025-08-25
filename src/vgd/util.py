import jax, jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from typing import Optional, Any
from jax.scipy.special import logsumexp

def _subset_indices(n: int, max_points: int) -> Array:
    """
    Deterministic, evenly spaced indices in [0, n-1], length s = min(n, max_points).
    n must be a Python int (use theta.shape[0] inside jit), max_points is a Python int.
    """
    s = max(1, min(n, max_points))
    # evenly spaced positions, inclusive of 0 and n-1
    idx = jnp.floor(jnp.linspace(0, n - 1, s)).astype(jnp.int32)
    return idx  # (s,)

def _upper_tri_median(d2_sub: Array) -> Array:
    """
    Median over i<j of a (s,s) squared-distance matrix, without boolean indexing.
    Returns scalar jnp.array.
    """
    s = d2_sub.shape[0]
    # mask upper triangle by setting others to +inf
    I = jnp.arange(s)[:, None]
    J = jnp.arange(s)[None, :]
    v = jnp.where(I < J, d2_sub, jnp.inf).ravel()  # (s*s,)
    m = s * (s - 1) // 2
    k = m // 2
    part_k  = jnp.partition(v, k)
    if (m % 2) == 1:
        med = part_k[k]
    else:
        part_km1 = jnp.partition(v, k - 1)
        med = 0.5 * (part_k[k] + part_km1[k - 1])
    return med

def _median_lengthscale_subset(theta: Array, max_points: int) -> Array:
    """
    Approx median heuristic on a static-size subset:
      ell = sqrt( median_offdiag(||θ_i-θ_j||^2) / (2 log(s+1)) )
    """
    n = theta.shape[0]                     # static int inside jit
    idx = _subset_indices(n, max_points)   # (s,)
    sub = theta[idx]                       # (s,d)
    x2  = jnp.sum(sub * sub, axis=1, keepdims=True)
    d2  = x2 + x2.T - 2.0 * (sub @ sub.T)  # (s,s)
    med_d2 = _upper_tri_median(d2)
    s = sub.shape[0]                       # static int
    h2 = jnp.maximum(med_d2 / (2.0 * jnp.log(s + 1.0)), 1e-12)
    return jnp.sqrt(h2)

def _replace_lengthscale(kparams, ell: Array):
    """Replace 'lengthscale' in a NamedTuple/dataclass params object."""
    if hasattr(kparams, "_replace"):
        return kparams._replace(lengthscale=ell)
    d = dict(kparams.__dict__)
    d["lengthscale"] = ell
    return type(kparams)(**d)

def tempered_ratio_from_loglik(
    loglik: Array,
    *,
    alpha: float = 1.0,        # tempering exponent; try 1/N
    stopgrad: bool = True,     # treat r as fixed weights
    r_min: float | None = None,
    r_max: float | None = None
) -> Array:
    """
    r_i = p_i^alpha / mean_j p_j^alpha, computed stably from log-likelihoods.
    Satisfies sum_i r_i = n for any alpha.
    """
    loglik = jnp.asarray(loglik)
    n = loglik.shape[0]
    a = alpha * loglik
    denom = logsumexp(a) - jnp.log(n)      # log(mean_j p_j^alpha)
    r = jnp.exp(a - denom)                 # sums to n

    if r_min is not None or r_max is not None:
        if r_min is None: r_min = -jnp.inf
        if r_max is None: r_max =  jnp.inf
        r = jnp.clip(r, r_min, r_max)
        # (Optionally renormalise back to sum n)
        r = r * (n / jnp.sum(r))

    if stopgrad:
        r = jax.lax.stop_gradient(r)
    return r

def _stable_ratio_from_loglik(loglik: Array, alpha: float = 0.001) -> Array:
    """Tempered version of r_j = p_j^alpha / mean_k p_k^alpha (stable)."""
    return tempered_ratio_from_loglik(loglik, alpha=alpha, stopgrad=True)

def logmeanexp_weighted(logv: Array, logw: Array, axis: int) -> Array:
    """
    Compute log ∑_j w_j exp(logv) along `axis`, with `logw` of shape (size along axis,).
    """
    # reshape logw to broadcast along `axis`
    shp = [1] * logv.ndim
    shp[axis] = logw.shape[0]
    logw_b = logw.reshape(shp)
    return logsumexp(logv + logw_b, axis=axis)

def seq_weights(loglik_total: Array, logw: Array, alpha: float = 1.0) -> Array:
    # r_j ∝ w_j * exp(α ℓ_j);  Σ r_j = 1
    a = logw + alpha * loglik_total
    r = jnp.exp(a - logsumexp(a))
    return jax.lax.stop_gradient(r)


# Plotting code

def plot_1d_output(Q, history, posterior, algo_name="VGD", figsize=(18,4), pred_mean_alpha=None):
    particles = jnp.array(Q.particles).reshape(-1)  # flatten to (n,)
    trajectory = history["particles_traj"]

    tmin, tmax = np.percentile(particles, [0.5, 99.5])
    pad = 0.25 * (tmax - tmin + 1e-8)
    grid = np.linspace(tmin - pad, tmax + pad, 600)

    # unnormalised posterior
    lp = posterior.log_posterior(grid)  
    lp = np.asarray(lp)
    logZ = lp.max()
    post_unnorm = np.exp(lp - logZ)   # rescaled density curve

    fig, axs = plt.subplots(1, 5, figsize=figsize)

    # VGD particle trajectories
    axs[0].set_title(algo_name + " particle trajectories")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("$\\theta$")

    if trajectory.shape[1] <= 50:  # only plot if not too many particles
        axs[0].plot(trajectory[:, :, 0], alpha=0.4, linewidth=0.2)
        axs[0].set_title(algo_name + " Particle Trajectories")
    else:
        mean_traj = jnp.reshape(trajectory.mean(axis=1), (-1, ))
        std_traj = jnp.reshape(trajectory.std(axis=1), (-1, ))

        for i in range(10):
            axs[0].fill_between(np.arange(mean_traj.shape[0]), mean_traj - (i + 1) / 4 * std_traj, mean_traj + (i + 1) / 4 * std_traj, alpha=0.05, color="C0")

    # KDE of VGD Particles
    particles = jnp.array(particles).reshape(-1)
    sns.kdeplot(x=particles, fill=True, bw_adjust=0.5, label=algo_name + " KDE", ax=axs[1])
    axs[1].set_title(algo_name + " KDE of Particles")
    axs[1].legend()

    # Posterior curve
    axs[2].plot(grid, post_unnorm, label="Posterior")
    axs[2].set_title("Posterior Density")
    axs[2].set_xlabel("$\\theta$")
    axs[2].legend()

    # Loss
    axs[3].plot(history["loss"], label="Loss")
    axs[3].set_title("Loss Curve")
    axs[3].set_xlabel("Iteration")
    axs[3].set_ylabel("Loss")
    axs[3].legend()

    # Predictive Means

    x, y = posterior.data["x"], posterior.data["y"]
    x_min, x_max = x.min(), x.max()
    grid = jnp.linspace(x_min, x_max, 100)
    predictive_means = posterior.like.pred_mean_fn(particles, grid)  # (n_particles, len(grid))

    if pred_mean_alpha is None:
        axs[4].plot(grid, predictive_means.T, color='red', alpha=0.4)
    else:
        for i, (mean_i, w_i) in enumerate(zip(predictive_means, Q.w)):
            axs[4].plot(grid, mean_i, color='red', alpha=float(w_i) * 2)

    axs[4].scatter(x, y, s=5, alpha=.3, label='Sampled Data', zorder=5)
    axs[4].set_title("Predictive Means")
    axs[4].set_xlabel("$x$")
    axs[4].set_ylabel("$y$")

    plt.tight_layout()

    return fig, axs
