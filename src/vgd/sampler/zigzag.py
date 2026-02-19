from __future__ import annotations
import jax.numpy as jnp
import jax
from jax import lax, vmap, Array
from functools import partial
import numpy as np

# Zig-Zag process with Poisson thinning
# assumes target distribution with potential U(x)
# and grad U(x) = \nabla U(x) available and bounded by a linear function (Lipschitz)
# i.e., for each coordinate i, |dU/dx_i| <= a_i + b_i |x_i|
# where a_i, b_i >= 0 are constants provided

def _solve_linear_intensity_time(key, a, b):
    """
    Draw T for inhomogeneous Poisson with lambda(t) = a + b t, a>=0, b>=0
    using inverse CDF of integrated hazard H(t) = a t + 0.5 b t^2.
    If a=b=0, return +inf (no events under the bound).
    """
    u = jax.random.uniform(key, minval=0.0, maxval=1.0)
    s = -jnp.log(u)  # Exp(1)
    # Cases:
    # b > 0 : 0.5 b t^2 + a t - s = 0  -> t = (-a + sqrt(a^2 + 2 b s)) / b
    # b = 0, a > 0 : a t = s -> t = s / a
    # a = b = 0 : inf
    sqrt_term = jnp.sqrt(jnp.maximum(a*a + 2.0*b*s, 0.0))
    t_quad = (sqrt_term - a) / jnp.maximum(b, 1e-32)
    t_lin  = s / jnp.maximum(a, 1e-32)
    t = jnp.where(b > 0.0, t_quad, jnp.where(a > 0.0, t_lin, jnp.inf))
    return t

# Vectorise over the batch dimension (keys)
_solve_linear_intensity_time_vmap = vmap(_solve_linear_intensity_time, in_axes=(0, 0, 0))

def _draw_candidate_times(key, v, g0, L, refresh_rate) -> tuple[Array | tuple[Array, ...], Array | tuple[Array, ...]]:
    """
    Draw candidate event times for each coordinate under linear upper bounds
    and an optional refresh event.
    """
    a = jnp.maximum(0.0, v * g0)       # intercepts for bounds
    b = L                              # slopes for bounds
    k_dim = a.shape[0]
    key_i, key_ref = jax.random.split(key, 2)

    # Per-coordinate candidate times
    keys_dim = jax.random.split(key_i, k_dim)
    t_dim = _solve_linear_intensity_time_vmap(keys_dim, a, b)

    # Optional refresh event (constant rate gamma)
    # If gamma=0, set t_ref=+inf
    u_ref = jax.random.uniform(key_ref, ())
    t_ref = jnp.where(refresh_rate > 0.0, -jnp.log(u_ref) / refresh_rate, jnp.inf)
    return t_dim, t_ref

def _advance(x, v, tau):
    return x + tau * v

def _flip(v, i):
    return v.at[i].set(-v[i])

def _accept_flip(key, v_i, g_i_new, a_i, b_i, tau):
    lam_true = jnp.maximum(0.0, v_i * g_i_new)
    lam_bar  = a_i + b_i * tau
    acc = jnp.where(lam_bar > 0.0, lam_true / lam_bar, 0.0)
    u = jax.random.uniform(key, ())
    return u < jnp.clip(acc, 0.0, 1.0), acc

@partial(jax.jit, static_argnames=("grad_potential","num_events"))
def run_zigzag(
    key,
    grad_potential,
    x0: Array,
    v0: Array,
    L: Array,
    num_events: int,
    refresh_rate: float = 0.0,
):
    """
    Simulate `num_events` Zig-Zag events.
    """
    d = x0.shape[0]

    def body(carry, k):
        key, x, v, t = carry
        key0, key1, key2, key3 = jax.random.split(key, 4)

        g0 = grad_potential(x)

        # Candidate times for each coordinate and a potential refresh
        t_dim, t_ref = _draw_candidate_times(key0, v, g0, L, refresh_rate)

        # Find min time and argmin (over d + optional 1 refresh)
        t_all = jnp.concatenate([t_dim, jnp.array([t_ref])], axis=0)
        i_min = jnp.argmin(t_all)
        tau = t_all[i_min]

        # Advance to candidate time
        x_new = _advance(x, v, tau)
        t_new = t + tau

        # Event type: refresh if i_min == d, else coordinate i_min
        is_refresh = jnp.equal(i_min, d)

        def coord_event(args):
            key1, key2, x_new, v, g0, tau, i_min = args
            g_new = grad_potential(x_new)
            a_i = jnp.maximum(0.0, v[i_min] * g0[i_min])
            b_i = L[i_min]
            accept, acc = _accept_flip(key1, v[i_min], g_new[i_min], a_i, b_i, tau)
            v_new = jnp.where(accept, _flip(v, i_min), v)
            return v_new, accept, i_min, acc

        def refresh_event(args):
            key1, key2, x_new, v, g0, tau, i_min = args
            u = jax.random.bernoulli(key2, 0.5, shape=v.shape)
            v_new = jnp.where(u, 1.0, -1.0)
            return v_new, True, jnp.int32(-1), jnp.float32(0.0)

        v_new, accepted, which_evt, acc_prob = lax.cond(
            is_refresh,
            refresh_event,
            coord_event,
            (key1, key2, x_new, v, g0, tau, i_min)
        )

        out_x = x_new
        out_v = v_new
        out_t = t_new

        return (key3, out_x, out_v, out_t), (out_x, out_v, out_t, accepted, which_evt, acc_prob)

    init_carry = (key, x0, v0, jnp.array(0.0))
    (key_f, x_f, v_f, t_f), (xs1, vs1, ts1, acc1, which1, accp1) = lax.scan(body, init_carry, jnp.arange(num_events))

    # Prepend initial state
    xs = jnp.vstack([x0[None, :], xs1])
    vs = jnp.vstack([v0[None, :], vs1])
    ts = jnp.concatenate([jnp.array([0.0]), ts1])
    return xs, vs, ts, acc1, which1, accp1


def flip_coord(v, i):
    return v.at[i].set(-v[i])

def lambda_vec(x, v, grad_potential, speed_fn, grad_speed_fn, gamma: float = 0.1, e_fn=None):
    """
    λ_i(x,v) = ( v_i * ( s(x)*∂_i U(x) - ∂_i s(x) ) )_+  +  s(x)*γ*e_i(x)
    If refresh is off, set gamma=0. 
    """
    s = speed_fn(x)                      # scalar
    gU = grad_potential(x)               # (d,)
    gs = grad_speed_fn(x)                # ∇s(x) ∈ R^d
    core = v * (s * gU - gs)             # elementwise
    lam = jnp.clip(core, a_min=0.0)      # (_)+
    
    e = e_fn(x) if e_fn is not None else jnp.ones_like(v)
    refresh_term = s * gamma * e
    lam = jnp.where(gamma != 0.0, lam + refresh_term, lam)
    
    return lam



@partial(jax.jit, static_argnames=("grad_potential","potential","speed_fn","grad_speed_fn"))
def mazz_step(
    key,
    x, v,
    delta: float,
    grad_potential,
    potential,
    speed_fn,          # s(x): R^d -> R+
    grad_speed_fn,     # ∇s(x): R^d -> R^d   (use grad_speed_fn = jax.grad(speed_fn))
    gamma: float = 0.0,
    e_fn = None,       # optional e(x): R^d -> R^d for refresh weights
):
    """
    Algorithm 3 (Strang split, non-reversible MH) with Prop. 3.4 time-change.
    """
    d = x.shape[0]
    k1, k2 = jax.random.split(key, 2)

    # Half flow: X_half = X + (δ/2) * s(X) * V
    s0 = speed_fn(x)
    x_half = x + 0.5 * delta * s0 * v

    # Coordinate sweep flips at X_half with current Ṽ
    def flip_body(carry, i):
        key_i, v_tilde = carry
        key_i, k = jax.random.split(key_i)
        lam_i = lambda_vec(x_half, v_tilde, grad_potential, speed_fn, grad_speed_fn, gamma, e_fn)[i]
        p = 1.0 - jnp.exp(-delta * lam_i)               # 1 - exp(-δ λ_i)
        do_flip = jax.random.bernoulli(k, jnp.clip(p, 0.0, 1.0))
        v_next = jnp.where(do_flip, flip_coord(v_tilde, i), v_tilde)
        return (key_i, v_next), None

    (_, v_tilde), _ = lax.scan(flip_body, (k1, v), jnp.arange(d))

    # Second half flow: X~ = X_half + (δ/2) * s(X_half) * Ṽ
    s1 = speed_fn(x_half)
    x_tilde = x_half + 0.5 * delta * s1 * v_tilde

    # Acceptance:
    # log α = -[U(X~)-U(X)] + δ * sum_j [ λ_j(X_half, V) - λ_j(X_half, -Ṽ) ]
    dU = potential(x_tilde) - potential(x)
    lam_fwd = lambda_vec(x_half,  v,        grad_potential, speed_fn, grad_speed_fn, gamma, e_fn)
    lam_rev = lambda_vec(x_half, -v_tilde,  grad_potential, speed_fn, grad_speed_fn, gamma, e_fn)
    correction = delta * jnp.sum(lam_fwd - lam_rev)

    log_alpha = jnp.clip(-dU + correction, -80.0, 80.0)
    alpha = jnp.minimum(1.0, jnp.exp(log_alpha))
    u = jax.random.uniform(k2)
    accept = u < alpha

    x_new = jnp.where(accept, x_tilde, x)
    v_new = jnp.where(accept, v_tilde, -v)              # non-reversible reject move

    stats = {
        "accepted": accept,
        "acc_prob": alpha,
        "dU": dU,
        "corr": correction,
        "s0": s0,
        "s1": s1,
    }
    return x_new, v_new, stats

@partial(jax.jit, static_argnames=("grad_potential","potential","speed_fn","grad_speed_fn","num_steps"))
def run_mazz(
    key, x0, v0, delta: float, num_steps: int,
    grad_potential, potential,
    speed_fn, grad_speed_fn,
    gamma: float = 0.0, e_fn = None,
):
    def body(carry, _):
        key, x, v = carry
        key, k = jax.random.split(key)
        x, v, stats = mazz_step(
            k, x, v, delta, grad_potential, potential,
            speed_fn, grad_speed_fn, gamma, e_fn
        )
        return (key, x, v), (x, v, stats)

    (_, xf, vf), (xs, vs, stats) = lax.scan(body, (key, x0, v0), None, length=num_steps)
    xs = jnp.vstack([x0[None,:], xs])
    vs = jnp.vstack([v0[None,:], vs])
    return xs, vs, stats
