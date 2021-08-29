""" Matrix of the double layer potential

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from math import atan

import numpy as np
from scipy.special import hankel1, jv

from .grid import grid
from .quadrature import kress_weight


def double_layer_pqr(boundary, k, nb):
    """
    D, θ = double_layer_pqr(boundary, k, nb)

    Return the double layer matrix.

    Parameters
    ----------
    boundary : Boundary
        Boundary object
    k : float
        wavenumber
    nb : int
        number of grid points

    Returns
    -------
    D : matrix
        matrix of the double layer potential
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

    return (kress_weight(nb) * L1 + L2, θ)


def double_layer_mpqr(boundary, k, nb):
    """
    D, θ = double_layer_mpqr(boundary, k, trace, nb, ε, matrix=False)

    The MPQR method for the Helmholtz problem.

    Parameters
    ----------
    boundary : Boundary
        Boundary object
    k : float
        wavenumber
    nb : int
        number of grid points

    Returns
    -------
    D : matrix
        matrix of the double layer potential
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

    return (kress_weight(nb) * L1 + L2, θ)
