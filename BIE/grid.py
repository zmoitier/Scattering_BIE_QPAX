""" Grid for BIE

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany

    Last modified: 06/04/2021
"""
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.special import ellipe, ellipeinc


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


def grid_arc_length(ε, N, mesh_grid=False):
    """
    s, Δs, S, T = grid_arc_length(ε, N, mesh_grid=False)

    Return the N equi-space point in arc length of the grid [0, L) where L is the
    length of the ellipse of semi-major-axis 1 and semi-minor-axis ε.

    Parameters
    ----------
    N : int
        number of grid points
    mesh_grid : bool (default False)
        if you want the result of meshgrid(θ, θ)

    Returns
    -------
    s : array
        N point grid of [0, L)
    Δs : float
        grid size
    S : array (if mesh_grid=True)
        second output of meshgrid(s, s)
    T : array (if mesh_grid=True)
        first output of meshgrid(s, s)
    """

    m = 1 - ε * ε
    n = (N - 4) // 4

    πo2 = np.pi / 2
    Lo4 = ellipe(m)

    vL, Δs = np.linspace(0, Lo4, num=n + 2, retstep=True)

    vx = np.array([brentq(lambda s: ellipeinc(s, m) - l, 0, πo2) for l in vL[1:-1]])
    vy = np.array([*(-vx[::-1]), 0, *vx])
    vs = np.array([-πo2, *vy, πo2, *(np.pi - vy[::-1])])

    if mesh_grid:
        T, S = np.meshgrid(vs, vs)
        return vs, Δs, S, T

    return vs, Δs


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
