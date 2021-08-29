""" Grid for BIE

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix


def grid(N, mesh_grid=False):
    """
    θ, Δθ, S, T = grid(N, mesh_grid=False)

    Return the N equi-space point of the grid [-π/2, 3π/2).

    Parameters
    ----------
    N : int
        number of grid points
    mesh_grid : bool (default False)
        if you want the result of meshgrid(θ, θ)

    Returns
    -------
    θ : array
        N point grid of [-π/2, 3π/2)
    Δθ : float
        grid size
    S : array (if mesh_grid=True)
        second output of meshgrid(θ, θ)
    T : array (if mesh_grid=True)
        first output of meshgrid(θ, θ)
    """

    θ, Δθ = np.linspace(-np.pi / 2, 3 * np.pi / 2, num=N, endpoint=False, retstep=True)

    if mesh_grid:
        T, S = np.meshgrid(θ, θ)
        return θ, Δθ, S, T

    return θ, Δθ


def half_grid(N, mesh_grid=False):
    """
    θ, Δθ, S, T = half_grid(N, mesh_grid=False)

    Return the N equi-space point of the grid [-π/2, π/2] which return the same N÷2+1
    points than grid(N).

    Parameters
    ----------
    N : int
        number of grid points
    mesh_grid : bool (default False)
        if you want the result of meshgrid(θ, θ)

    Returns
    -------
    θ : array
        N point grid of [-π/2, 3π/2)
    Δθ : float
        grid size
    S : array (if mesh_grid=True)
        second output of meshgrid(θ, θ)
    T : array (if mesh_grid=True)
        first output of meshgrid(θ, θ)
    """

    θ, Δθ = np.linspace(-np.pi / 2, np.pi / 2, num=N // 2 + 1, retstep=True)

    if mesh_grid:
        T, S = np.meshgrid(θ, θ)
        return θ, Δθ, S, T

    return θ, Δθ


def parity_base(N):
    """
    P, Q = parity_base(N)

    Return the matrix corresponding to the parity even/odd with respect of the major-
    axis of the ellipse from the grid function.

    Parameters
    ----------
    N : int
        size of the matrix

    Returns
    -------
    P : array
        projection on the even/odd sub-space
    Q : array
        Q = P⁻¹
    """

    n = N // 2

    pv = [
        *np.ones(n + 1, dtype=int),
        *np.full(n - 1, -1, dtype=int),
        *np.ones(2 * (n - 1), dtype=int),
    ]
    pi = [*np.arange(N), *np.arange(1, n), *np.arange(N - 1, N - n, -1)]
    pj = [
        *np.arange(n + 1),
        *np.arange(N - 1, N - n, -1),
        *np.arange(n + 1, N),
        *np.arange(1, n),
    ]

    qv = [1, *np.full(n - 1, 0.5), 1, *np.full(n - 1, -0.5), *np.full(2 * (n - 1), 0.5)]
    qi = [
        *np.arange(n + 1),
        *np.arange(N - 1, N - n, -1),
        *np.arange(1, n),
        *np.arange(n + 1, N),
    ]
    qj = [*np.arange(N), *np.arange(N - 1, N - n, -1), *np.arange(1, n)]

    return (
        coo_matrix((pv, (pi, pj)), shape=(N, N)),
        coo_matrix((qv, (qi, qj)), shape=(N, N)),
    )
