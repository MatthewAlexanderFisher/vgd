from __future__ import annotations

import jax, jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
from typing import Optional, Any
from jax.scipy.special import logsumexp
import math
from collections import defaultdict, Counter
import numpy as np


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

def plot_1d_output(Q, history, posterior, algo_name="VGD", figsize=(18,4), use_weights=False):
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
    if use_weights is False:
        sns.kdeplot(x=particles, fill=True, bw_adjust=0.5, label=algo_name + " KDE", ax=axs[1])
    else:
        sns.kdeplot(x=particles, fill=True, bw_adjust=0.5, label=algo_name + " KDE", ax=axs[1], weights=Q.w)
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

    if use_weights is False:
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


def plot_1d_predictive_output(Q, history, model, algo_name="VGD", figsize=(18,4), use_weights=False):
    particles = jnp.array(Q.particles).reshape(-1)  # flatten to (n,)
    trajectory = history["particles_traj"]

    tmin, tmax = np.percentile(particles, [0.5, 99.5])
    pad = 0.25 * (tmax - tmin + 1e-8)
    grid = np.linspace(tmin - pad, tmax + pad, 600)

    fig, axs = plt.subplots(1, 4, figsize=figsize)

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
    if use_weights is False:
        sns.kdeplot(x=particles, fill=True, bw_adjust=0.5, label=algo_name + " KDE", ax=axs[1])
    else:
        sns.kdeplot(x=particles, fill=True, bw_adjust=0.5, label=algo_name + " KDE", ax=axs[1], weights=Q.w)
    axs[1].set_title(algo_name + " KDE of Particles")
    axs[1].legend()

    # Loss
    axs[2].plot(history["loss"], label="Loss")
    axs[2].set_title("Loss Curve")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Loss")
    axs[2].legend()

    # Predictive Means

    x, y = model.data["x"], model.data["y"]
    x_min, x_max = x.min(), x.max()
    grid = jnp.linspace(x_min, x_max, 100)
    predictive_means = model.predict_on_x(Q, grid)  # (n_particles, len(grid))

    if use_weights is False:
        axs[3].plot(grid, predictive_means.T, color='red', alpha=0.4)
    else:
        for i, (mean_i, w_i) in enumerate(zip(predictive_means, Q.w)):
            axs[3].plot(grid, mean_i, color='red', alpha=float(w_i) * 2)

    axs[3].scatter(x, y, s=5, alpha=.3, label='Sampled Data', zorder=5)
    axs[3].set_title("Predictive Means")
    axs[3].set_xlabel("$x$")
    axs[3].set_ylabel("$y$")

    plt.tight_layout()

    return fig, axs


def plot_isf_results(
    predictor,
    data,
    Q0,
    Qhat,
    metrics=None,
    grid_lim=2.5,
    grid_n=200,
    title_prefix="ISF"
):
    def _to_np(x):
        return np.asarray(x)

    def _grid_2d(lim=2.5, n=200):
        xs = np.linspace(-lim, lim, n)
        ys = np.linspace(-lim, lim, n)
        XX, YY = np.meshgrid(xs, ys)
        XY = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (n*n, 2)
        return XX, YY, XY

    def _predict_probs(predictor, Q, XY):
        logits = _to_np(predictor.predict_on_x(Q, XY))   # (M,)
        return 1.0 / (1.0 + np.exp(-logits))

    X = _to_np(data["x"])
    y = _to_np(data["y"]).astype(int)

    XX, YY, XY = _grid_2d(lim=grid_lim, n=grid_n)

    P0 = _predict_probs(predictor, Q0, XY).reshape(XX.shape)
    Ph = _predict_probs(predictor, Qhat, XY).reshape(XX.shape)
    dP = Ph - P0


    # One figure, 2x3 grid
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])

    # --- Axes ---
    ax_init  = fig.add_subplot(gs[0, 0])
    ax_final = fig.add_subplot(gs[0, 1])
    ax_diff  = fig.add_subplot(gs[0, 2])
    ax_loss  = fig.add_subplot(gs[1, 0])
    ax_norms = fig.add_subplot(gs[1, 1])
    ax_hist  = fig.add_subplot(gs[1, 2])

    # Common plotting helper for surfaces
    def _plot_surface(ax, Z, title, draw_points=True, levels=25, vmin=0.0, vmax=1.0, show_cb=True):
        im = ax.contourf(XX, YY, Z, levels=levels, vmin=vmin, vmax=vmax, alpha=0.85)
        cs = ax.contour(XX, YY, Z, levels=[0.5], colors="k", linewidths=1.2)
        ax.clabel(cs, inline=True, fontsize=8, fmt="0.5")
        if draw_points:
            ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, c="C0", label="class 0", alpha=0.8, edgecolors="none")
            ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, c="C1", label="class 1", alpha=0.8, edgecolors="none")
            ax.legend(loc="upper right", frameon=True, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.set_xlim(-grid_lim, grid_lim); ax.set_ylim(-grid_lim, grid_lim)
        if show_cb:
            fig.colorbar(im, ax=ax, shrink=0.8)

    # Top row: initial, final, delta
    _plot_surface(ax_init,  P0, f"{title_prefix}: initial $p(y=1)$")
    _plot_surface(ax_final, Ph, f"{title_prefix}: final $p(y=1)$")

    # For difference, symmetric color range around 0
    vmax = np.max(np.abs(dP))
    imd = ax_diff.contourf(XX, YY, dP, levels=25, vmin=-vmax, vmax=vmax, alpha=0.9, cmap="coolwarm")
    csd = ax_diff.contour(XX, YY, Ph, levels=[0.5], colors="k", linewidths=1.0)
    ax_diff.clabel(csd, inline=True, fontsize=8, fmt="final 0.5")
    ax_diff.set_title(f"{title_prefix}: $\Delta p$ = final - initial")
    ax_diff.set_xlabel("x1"); ax_diff.set_ylabel("x2")
    ax_diff.set_xlim(-grid_lim, grid_lim); ax_diff.set_ylim(-grid_lim, grid_lim)
    fig.colorbar(imd, ax=ax_diff, shrink=0.8)

    # Bottom-left: loss curve (if available)
    if metrics is not None and "loss" in metrics:
        loss = _to_np(metrics["loss"])
        ax_loss.plot(loss, lw=1.5)
        ax_loss.set_title("training loss")
        ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss")
    else:
        ax_loss.axis("off")
        ax_loss.set_title("training loss (unavailable)")

    # Bottom-middle: step norms (if available)
    has_r = metrics is not None and "rkhs_step_norm" in metrics
    has_s = metrics is not None and "stein_step_norm" in metrics
    if has_r or has_s:
        if has_r: ax_norms.plot(_to_np(metrics["rkhs_step_norm"]), label="rkhs_step_norm", lw=1.5)
        if has_s: ax_norms.plot(_to_np(metrics["stein_step_norm"]), label="stein_step_norm", lw=1.5)
        ax_norms.set_title("step norms")
        ax_norms.set_xlabel("step"); ax_norms.set_ylabel("norm")
        ax_norms.legend()
    else:
        ax_norms.axis("off")
        ax_norms.set_title("step norms (unavailable)")

    # Bottom-right: alpha histogram (if available)
    if hasattr(Qhat, "alpha"):
        al = _to_np(Qhat.alpha)
        ax_hist.hist(al, bins=20, edgecolor="white")
        ax_hist.set_title("final weights (alpha)")
        ax_hist.set_xlabel("alpha"); ax_hist.set_ylabel("count")
    else:
        ax_hist.axis("off")
        ax_hist.set_title("final weights (unavailable)")

    fig.suptitle(title_prefix, y=0.99, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()



def base_p_digits_least_first(n: int, p: int, m: int):
    """Return the least-significant-first base-p digits of n modulo p^m, length m."""
    digits = []
    for _ in range(m):
        digits.append(n % p)
        n //= p
    return digits  # [d0 (units), d1, ..., d{m-1}]

def build_tree_positions(p: int, m: int):
    """
    Build deterministic (x, y) positions for each prefix node at each depth k=0..m.
    Returns:
      pos: dict mapping prefix tuple (d0,...,d_{k-1}) to (x, y)
      parent: dict mapping child_prefix -> parent_prefix (for drawing edges)
    """
    pos = {}
    parent = {}
    pos[tuple()] = (0.5, 1.0)  # root at top center
    for k in range(1, m + 1):
        num_nodes = p ** k
        xs = (np.arange(num_nodes) + 0.5) / num_nodes
        y = 1.0 - k / (m + 0.3)
        for idx in range(num_nodes):
            # decode idx to base-p vector of length k, least-first
            digits = []
            t = idx
            for _ in range(k):
                digits.append(t % p)
                t //= p
            prefix = tuple(digits)
            pos[prefix] = (float(xs[idx]), float(y))
            parent[prefix] = tuple(digits[:-1])
    return pos, parent

def aggregate_leaf_stats(X: np.ndarray, vals: np.ndarray, p: int, m: int):
    """
    Aggregate 'vals' for each residue class (leaf) seen in X.
    Returns dict leaf_prefix -> dict(mean=..., count=...).
    """
    assert X.ndim == 1 and vals.ndim == 1 and X.shape[0] == vals.shape[0]
    M = p ** m
    X_mod = np.asarray(X, dtype=np.int64) % M
    vals = np.asarray(vals, dtype=float)

    by_leaf = defaultdict(list)
    for xi, vi in zip(X_mod, vals):
        digits = base_p_digits_least_first(int(xi), p, m)
        leaf = tuple(digits)
        by_leaf[leaf].append(vi)

    stats = {}
    for leaf, vs in by_leaf.items():
        arr = np.asarray(vs, float)
        stats[leaf] = {"mean": float(arr.mean()), "count": int(arr.size)}
    return stats

def plot_p_adic_tree_with_predictor(X: np.ndarray, y: np.ndarray, Q, predictor,
                                    p: int, m: int, title=None,
                                    cmap="viridis", diff_cmap="coolwarm",
                                    size_base=18.0, size_scale=70.0):
    """
    Draw three panels on one figure:
      (A) leaf color = mean(y),   size = count(X==leaf)
      (B) leaf color = mean(yhat), size = same counts as (A)
      (C) leaf color = mean(yhat) - mean(y), size = counts
    Only leaves that appear in X are emphasized (others are drawn tiny/transparent).

    Works for 1D residues (Z / p^m Z). For multi-D inputs, first encode indices.
    """
    # Positions and topology
    pos, parent = build_tree_positions(p, m)
    M = p ** m

    # Predict on the same sample support (so counts match)
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(predictor.predict_on_x(Q, X)).reshape(-1)

    stats_y    = aggregate_leaf_stats(X, y,    p, m)
    stats_yhat = aggregate_leaf_stats(X, yhat, p, m)

    # Build per-leaf arrays (in fixed order across panels)
    leaves = [prefix for prefix in pos.keys() if len(prefix) == m]
    leaf_x = np.array([pos[L][0] for L in leaves])
    leaf_y = np.array([pos[L][1] for L in leaves])

    means_y = np.full(len(leaves), np.nan)
    means_h = np.full(len(leaves), np.nan)
    counts  = np.zeros(len(leaves), dtype=int)

    for i, L in enumerate(leaves):
        if L in stats_y:
            means_y[i] = stats_y[L]["mean"]
            counts[i]  = stats_y[L]["count"]
        if L in stats_yhat:
            means_h[i] = stats_yhat[L]["mean"]

    # Shared color scale for data & prediction
    vmin = np.nanmin(np.concatenate([means_y[~np.isnan(means_y)],
                                     means_h[~np.isnan(means_h)]])) if np.any(~np.isnan(means_y)) or np.any(~np.isnan(means_h)) else 0.0
    vmax = np.nanmax(np.concatenate([means_y[~np.isnan(means_y)],
                                     means_h[~np.isnan(means_h)]])) if np.any(~np.isnan(means_y)) or np.any(~np.isnan(means_h)) else 1.0
    # Diff panel symmetric around 0
    diffs = means_h - means_y
    dmax = np.nanmax(np.abs(diffs[~np.isnan(diffs)])) if np.any(~np.isnan(diffs)) else 1.0

    # Sizing: emphasize observed leaves
    sizes = size_base + size_scale * np.sqrt(np.maximum(counts, 0))
    sizes[np.where(counts == 0)] = 2.0  # tiny for unseen

    # --- plotting helpers ---
    def draw_edges(ax):
        for child, par in parent.items():
            x0, y0 = pos[par]
            x1, y1 = pos[child]
            ax.plot([x0, x1], [y0, y1], linewidth=0.6, alpha=0.35, color="gray")

    def draw_internals(ax):
        for prefix, (xx, yy) in pos.items():
            if len(prefix) < m:
                ax.scatter([xx], [yy], s=8, alpha=0.4, color="gray")

    def panel(ax, values, cbar_label, cm, vmin_, vmax_, title_):
        draw_edges(ax)
        draw_internals(ax)
        sc = ax.scatter(leaf_x, leaf_y, s=sizes, c=values, cmap=cm,
                        vmin=vmin_, vmax=vmax_, edgecolors="none")
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_title(title_)

    # --- figure ---
    if title is None:
        title = f"p-ary tree view (p={p}, m={m}) - data vs predictor and error"
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    fig.suptitle(title, y=1.03, fontsize=12)

    panel(axs[0], means_y, "mean $y$", cmap, vmin, vmax, "Data (mean at leaf)")
    panel(axs[1], means_h, "mean $\hat{y}$", cmap, vmin, vmax, "Prediction (mean at leaf)")
    panel(axs[2], diffs,   "$\hat{y}$ - y", diff_cmap, -dmax, dmax, "Error (prediction - data)")

    plt.show()


def plot_1d_fit(X, y, Q, predictor, topk_amplitudes=None):
    """
    One figure with:
      (1) noisy y and prediction vs x (sorted),
      (2) residuals vs x,
      (3) per-frequency amplitudes (if use_sin=True) or raw alphas (if cos-only).

    Args:
      topk_amplitudes: if not None and use_sin=True, show only the top-K amplitudes for readability.
    """
    # ----- Sort x for clean curves -----
    x = np.asarray(X).reshape(-1)
    y_np = np.asarray(y).reshape(-1)
    order_x = np.argsort(x)
    x_sorted = x[order_x]
    y_sorted = y_np[order_x]

    # Prediction on sorted x
    yhat_sorted = np.asarray(predictor.predict_on_x(Q, x_sorted))

    # Residuals
    resid = y_sorted - yhat_sorted

    # ----- Coefficients / amplitudes -----
    Xi = np.asarray(Q.Xi).reshape(-1)         # (n,)
    alpha = np.asarray(Q.alpha).reshape(-1)   # (n,) or (2n,)
    order_k = np.argsort(Xi)
    Xi_sorted = Xi[order_k]

    if getattr(predictor, "use_sin", False):
        n = Xi.shape[0]
        alpha_cos_sorted = alpha[:n][order_k]
        alpha_sin_sorted = alpha[n:][order_k]
        # Amplitude per frequency: a cos + b sin = R cos(· - φ), R = sqrt(a^2+b^2)
        amps_sorted = np.sqrt(alpha_cos_sorted**2 + alpha_sin_sorted**2)
        bar_vals = amps_sorted
        bar_label = "amplitude per frequency"
    else:
        # Cos-only: just show |alpha|
        bar_vals = np.abs(alpha[order_k])
        bar_label = "|alpha| per frequency"

    # Optionally keep only top-K amplitudes for readability
    if topk_amplitudes is not None and getattr(predictor, "use_sin", False):
        k = int(topk_amplitudes)
        idx_top = np.argsort(bar_vals)[-k:]
        Xi_sorted = Xi_sorted[idx_top]
        bar_vals = bar_vals[idx_top]
        # sort by frequency for nicer x-axis
        srt = np.argsort(Xi_sorted)
        Xi_sorted = Xi_sorted[srt]
        bar_vals = bar_vals[srt]

    # ----- Plot: single figure with 3 subplots -----
    M = predictor.modulus()
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    # (1) y and yhat vs x
    axs[0].scatter(x_sorted, y_sorted, s=10, alpha=0.6, label="y (noisy)")
    axs[0].plot(x_sorted, yhat_sorted, lw=2, label="prediction")
    axs[0].set_xlabel(r"$x \;(\mathrm{mod}\; p^m)$")
    axs[0].set_ylabel("signal")
    axs[0].set_title("Noisy target and fitted prediction")
    axs[0].legend()

    # (2) residuals vs x
    axs[1].scatter(x_sorted, resid, s=10, alpha=0.7)
    axs[1].axhline(0.0, color="k", ls="--", lw=1)
    axs[1].set_xlabel(r"$x \;(\mathrm{mod}\; p^m)$")
    axs[1].set_ylabel("residual (y - yhat)")
    axs[1].set_title("Residuals")

    # (3) amplitudes / |alpha| vs frequency index (sorted by Xi)
    # Use frequency labels sparingly to avoid clutter
    xs = np.arange(len(Xi_sorted))
    axs[2].bar(xs, bar_vals, width=0.9)
    axs[2].set_xlabel("frequencies (sorted $k$)")
    axs[2].set_ylabel(bar_label)
    axs[2].set_title(f"Coefficients over frequencies (M={M})")

    # Show a few tick labels with actual k's to orient the axis
    if len(Xi_sorted) > 0:
        tick_step = max(1, len(Xi_sorted)//10)
        sel = np.arange(0, len(Xi_sorted), tick_step)
        axs[2].set_xticks(sel)
        axs[2].set_xticklabels([str(int(k)) for k in Xi_sorted[sel]], rotation=45, ha="right")

    plt.show()
