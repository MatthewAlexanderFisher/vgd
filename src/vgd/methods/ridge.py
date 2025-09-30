import jax.numpy as jnp

def solve_alpha_ridge(Phi_n_by_N, y, lam=1e-3):
    """
    Solve: min_alpha ||Phi^T alpha - y||^2 + lam ||alpha||^2
    where Phi is (n, N), y is (N,), returning alpha (n,).
    """
    A = Phi_n_by_N @ Phi_n_by_N.T + lam * jnp.eye(Phi_n_by_N.shape[0])
    b = Phi_n_by_N @ y
    return jnp.linalg.solve(A, b)