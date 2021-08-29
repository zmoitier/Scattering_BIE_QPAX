""" BIE for Laplace interior

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, 
"""
from math import atan

import numpy as np
from numpy.linalg import cond
from scipy.linalg import solve
from scipy.sparse import identity

from .grid import grid, parity_base
from .matrix_LH1 import matrix_L1_ev, matrix_L1_od


def laplace_ptr(boundary, trace, nb, matrix=False):
    """
    μ, c, θ = laplace_ptr(boundary, trace, nb, matrix=False)

    The PTR method for the Laplace problem.

    Parameters
    ----------
    boundary : Boundary
        Boundary object
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
    dist2 = xdiff ** 2 + ydiff ** 2

    not_zero = np.where(np.greater(np.abs(S - T), Δθ / 2))

    kernel = np.empty((nb, nb))
    kernel[not_zero] = (
        (ν_x[not_zero] * xdiff[not_zero] + ν_y[not_zero] * ydiff[not_zero])
        * jac[not_zero]
        / (dist2[not_zero] * nb)
    )
    kernel[(range(nb), range(nb))] = (
        (-1 / (2 * nb)) * jac[(range(nb), range(nb))] * boundary.curvature(θ)
    )

    A = -0.5 * identity(nb) + kernel
    μ = solve(A, trace(θ))

    if matrix:
        return μ, A, θ

    return μ, cond(A), θ


def laplace_mtr(boundary, trace, nb, matrix=False):
    """
    μ, c, θ = laplace_mtr(boundary, trace, nb, matrix=False)

    The MTR method for the Laplace problem.

    Parameters
    ----------
    boundary : Boundary
        Boundary object
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

    ε = boundary.ε
    θ, Δθ, S, T = grid(nb, mesh_grid=True)

    x_s, y_s = boundary.gamma(S)
    x_t, y_t = boundary.gamma(T)
    jac = boundary.jacobian(T)
    ν_x, ν_y = boundary.normal_ext(T)

    xdiff, ydiff = x_s - x_t, y_s - y_t
    dist2 = xdiff ** 2 + ydiff ** 2

    not_zero = np.where(np.greater(np.abs(S - T), Δθ / 2))

    kernel = np.empty((nb, nb))
    kernel[not_zero] = (
        (ν_x[not_zero] * xdiff[not_zero] + ν_y[not_zero] * ydiff[not_zero])
        * jac[not_zero]
        / (dist2[not_zero] * nb)
    )
    kernel[(range(nb), range(nb))] = (
        (-1 / (2 * nb)) * jac[(range(nb), range(nb))] * boundary.curvature(θ)
    )

    quasi_sing = np.where(
        np.less(np.abs(np.remainder(S + T, 2 * np.pi) - np.pi), Δθ / 2)
    )
    kernel[quasi_sing] = -atan(Δθ / (4 * ε)) / np.pi

    A = -0.5 * identity(nb) + kernel
    μ = solve(A, trace(θ))

    if matrix:
        return μ, A, θ

    return μ, cond(A), θ


def laplace_qpax(boundary, trace_ev_od, N, matrix=False):
    """
    μ, c, θ = laplace_asy(boundary, trace_ev_od, N, matrix=False)

    The QPAX method for the Laplace problem.

    Parameters
    ----------
    boundary : Boundary
        Boundary object
    trace_ev_od : Tuple{Function, Function}
        trace_ev_od[0] contain None or the function θ ↦ f(θ) the even source term
        trace_ev_od[1] contain None or the function θ ↦ f(θ) the odd source term
    N : int
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

    ε = boundary.ε
    n = N // 2
    θ, _ = grid(N)

    L1_ev = matrix_L1_ev(n + 1)
    L1_od = matrix_L1_od(n - 1)

    μ = np.zeros(N)
    if trace_ev_od[0] is not None:
        f_ev = trace_ev_od[0](θ[: n + 1])

        μ_ev = np.zeros(N, dtype=f_ev.dtype)
        μ_ev[: n + 1] += -f_ev + ε * (L1_ev @ f_ev)
        μ_ev[n + 1 :] += μ_ev[n - 1 : 0 : -1]

        μ += μ_ev

    if trace_ev_od[1] is not None:
        f_od = trace_ev_od[1](θ[1:n])

        μ_od = np.zeros(N, dtype=f_od.dtype)
        μ_od[1:n] += (-1 / ε) * solve(L1_od, f_od)
        μ_od[n + 1 :] -= μ_od[n - 1 : 0 : -1]

        μ += μ_od

    if matrix:
        P, Q = parity_base(N)
        L = np.block(
            [[L1_ev, np.zeros((n + 1, n - 1))], [np.zeros((n - 1, n + 1)), L1_od]]
        )
        return μ, P @ (L @ Q), θ

    return μ, cond(L1_od), θ