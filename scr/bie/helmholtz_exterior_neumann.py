""" BIE for Helmholtz exterior Neumann

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
import numpy as np
from numpy.linalg import cond
from scipy.linalg import solve
from scipy.sparse import identity

from .double_layer import double_layer_mpqr, double_layer_pqr
from .grid import grid, parity_base
from .matrix_LH1 import matrix_H1_ev, matrix_H1_od


def helmholtz_neumann_pqr(boundary, k, trace, nb, matrix=False):
    """
    μ, c, θ = helmholtz_pqr(boundary, k, trace, nb, matrix=False)

    The PQR (Kress) method for the Helmholtz problem.

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

    L, θ = double_layer_pqr(boundary, k, nb)
    A = 0.5 * identity(nb) - L

    μ = solve(A, trace(θ))

    if matrix:
        return μ, A, θ

    return μ, cond(A), θ


def helmholtz_neumann_mpqr(boundary, k, trace, nb, matrix=False):
    """
    μ, c, θ = helmholtz_mpqr(boundary, k, trace, nb, ε, matrix=False)

    The MPQR method for the Helmholtz problem.

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

    L, θ = double_layer_mpqr(boundary, k, nb)
    A = 0.5 * identity(nb) - L

    μ = solve(A, trace(θ))

    if matrix:
        return μ, A, θ

    return μ, cond(A), θ


def helmholtz_neumann_qpax(boundary, k, expan_trace, N, matrix=False):
    """
    μ, c, θ = helmholtz_qpax(boundary, k, trace, nb, ε, matrix=False)

    The method based on the asymptotic expansion of the operator for the Helmholtz
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
    ε = boundary.ε
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
