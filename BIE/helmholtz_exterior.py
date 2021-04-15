""" BIE for Helmholtz exterior

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany

    Last modified: 15/04/2021
"""
from math import atan

import numpy as np
from numpy.linalg import cond
from scipy.linalg import solve
from scipy.sparse import identity
from scipy.special import hankel1, jv

from .grid import grid, parity_base
from .matrix_LH1 import matrix_H1_ev, matrix_H1_od
from .quadrature import kress_weight


def helmholtz_pqr(boundary, k, trace, nb, matrix=False):
    """
    μ, c, θ = helmholtz_pqr(boundary, k, trace, nb, matrix=False)

    The Product Quadrature Rule (PQR) method, by Kress, for the Helmholtz problem.

    Parameters
    ----------
    boundary : Boundary
        Boundary object
    k : float
        wavenumber
    trace : function
        the function θ ↦ f(θ) the source term of the BIE
    nb : int
        number of grid points
    matrix : bool (default False)
        c is the condition number if matrix=False and the matrix if matrix=True

    Returns
    -------
    μ : vector
        the solution
    c : float or matrix
        the condition number if matrix=False and the matrix if matrix=True
    θ : vector
        θ-grid of the boundary
    """

    θ, Δθ, S, T = grid(nb, mesh_grid=True)

    x_s, y_s = boundary.gamma(S)
    x_t, y_t = boundary.gamma(T)
    jac = boundary.jacobian(T)
    ν_x, ν_y = boundary.normal_ext(T)

    xdiff, ydiff = x_s - x_t, y_s - y_t
    dist = np.hypot(xdiff, ydiff)

    not_zero = np.where(np.greater(np.abs(S - T), Δθ / 2))

    cos_term = np.empty((nb, nb))
    cos_term[not_zero] = (
        ν_x[not_zero] * xdiff[not_zero] + ν_y[not_zero] * ydiff[not_zero]
    ) / dist[not_zero]

    L1 = np.empty((nb, nb))
    L1[not_zero] = (
        (-k / (4 * np.pi))
        * cos_term[not_zero]
        * jv(1, k * dist[not_zero])
        * jac[not_zero]
    )
    L1[(range(nb), range(nb))] = 0

    L2 = np.empty((nb, nb), dtype=complex)
    L2[not_zero] = (
        (0.25 * 1j * k)
        * hankel1(1, k * dist[not_zero])
        * cos_term[not_zero]
        * jac[not_zero]
    )
    L2[not_zero] -= L1[not_zero] * np.log(
        4 * np.sin(0.5 * (S[not_zero] - T[not_zero])) ** 2
    )
    L2[(range(nb), range(nb))] = (
        (-1 / (4 * np.pi)) * jac[(range(nb), range(nb))] * boundary.curvature(θ)
    )
    L2 *= Δθ

    L = kress_weight(nb) * L1 + L2
    A = 0.5 * identity(nb) - L

    μ = solve(A, trace(*boundary.gamma(θ)))

    if matrix:
        return μ, A, θ

    return μ, cond(A), θ


def helmholtz_mpqr(boundary, k, trace, nb, ε, matrix=False):
    """
    μ, c, θ = helmholtz_mpqr(boundary, k, trace, nb, ε, matrix=False)

    The Modified Product Quadrature Rule (MPQR) method for the Helmholtz problem.

    Parameters
    ----------
    boundary : Boundary
        Boundary object
    k : float
        wavenumber
    trace : function
        the function θ ↦ f(θ) the source term of the BIE
    nb : int
        number of grid points
    ε : float
        semi-minor axis of the ellipse
    matrix : bool (default False)
        c is the condition number if matrix=False and the matrix if matrix=True

    Returns
    -------
    μ : vector
        the solution
    c : float or matrix
        the condition number if matrix=False and the matrix if matrix=True
    θ : vector
        θ-grid of the boundary
    """

    θ, Δθ, S, T = grid(nb, mesh_grid=True)

    x_s, y_s = boundary.gamma(S)
    x_t, y_t = boundary.gamma(T)
    jac = boundary.jacobian(T)
    ν_x, ν_y = boundary.normal_ext(T)

    xdiff, ydiff = x_s - x_t, y_s - y_t
    dist = np.hypot(xdiff, ydiff)

    not_zero = np.where(np.abs(S - T) > Δθ / 2)

    cos_term = np.empty((nb, nb))
    cos_term[not_zero] = (
        ν_x[not_zero] * xdiff[not_zero] + ν_y[not_zero] * ydiff[not_zero]
    ) / dist[not_zero]

    L1 = np.empty((nb, nb))
    L1[not_zero] = (
        (-k / (4 * np.pi))
        * cos_term[not_zero]
        * jv(1, k * dist[not_zero])
        * jac[not_zero]
    )
    L1[(range(nb), range(nb))] = 0

    L2 = np.empty((nb, nb), dtype=complex)
    L2[not_zero] = (
        (0.25 * 1j * k)
        * hankel1(1, k * dist[not_zero])
        * cos_term[not_zero]
        * jac[not_zero]
    )
    L2[not_zero] -= L1[not_zero] * np.log(
        4 * np.sin(0.5 * (S[not_zero] - T[not_zero])) ** 2
    )
    L2[(range(nb), range(nb))] = (
        (-1 / (4 * np.pi)) * jac[(range(nb), range(nb))] * boundary.curvature(θ)
    )
    L2 *= Δθ

    quasi_sing = np.where(
        np.less(np.abs(np.remainder(S + T, 2 * np.pi) - np.pi), Δθ / 2)
    )
    L2[quasi_sing] = -atan(Δθ / (4 * ε)) / np.pi

    L = kress_weight(nb) * L1 + L2
    A = 0.5 * identity(nb) - L

    μ = solve(A, trace(*boundary.gamma(θ)))

    if matrix:
        return μ, A, θ

    return μ, cond(A), θ


def helmholtz_qpax(ε, k, expan_trace, N, matrix=False):
    """
    μ, c, θ = helmholtz_qpax(boundary, k, trace, nb, ε, matrix=False)

    The Quadrature by Parity Asymptotic eXpansions (QPAX) method for the Helmholtz
    problem.

    Parameters
    ----------
    boundary : Boundary
        Boundary object
    k : float
        wavenumber
    expan_trace : tuple{function, function}
        the tuple of function (θ ↦ u₀(θ), θ ↦ u₁(θ)) the source term of the BIE
    nb : int
        number of grid points
    ε : float
        semi-minor axis of the ellipse
    matrix : bool (default False)
        c is the condition number if matrix=False and the matrix if matrix=True

    Returns
    -------
    μ : vector
        the solution
    c : float or matrix
        the condition number if matrix=False and the matrix if matrix=True
    θ : vector
        θ-grid of the boundary
    """

    n = N // 2
    θ, _ = grid(N)

    H1_ev = matrix_H1_ev(k, N)
    H1_od = matrix_H1_od(k, N)

    μ = np.zeros(N, dtype=complex)
    if expan_trace[0] is not None:
        f_ev = expan_trace[0](θ[: n + 1])

        μ_ev = np.zeros(N, dtype=f_ev.dtype)
        μ_ev[: n + 1] += f_ev - ε * (H1_ev @ f_ev)
        μ_ev[n + 1 :] += μ_ev[n - 1 : 0 : -1]

        μ += μ_ev

    if expan_trace[1] is not None:
        f_od = expan_trace[1](θ[1:n])

        μ_od = np.zeros(N, dtype=f_od.dtype)
        μ_od[1:n] += solve(H1_od, f_od)
        μ_od[n + 1 :] -= μ_od[n - 1 : 0 : -1]

        μ += μ_od

    if matrix:
        P, Q = parity_base(N)
        L = np.block(
            [[H1_ev, np.zeros((n + 1, n - 1))], [np.zeros((n - 1, n + 1)), H1_od]]
        )
        return μ, P @ (L @ Q), θ

    return μ, cond(H1_od), θ


def far_field(N, ε, k, u):
    θ, Δθ, S, T = grid(N, mesh_grid=True)

    pc = np.cos(T) * np.cos(S)
    ps = np.sin(T) * np.sin(S)
    A = (Δθ * 0.25 * k) * (pc + ε * ps) * np.exp(-1j * k * (ps + ε * pc))

    return A @ u
