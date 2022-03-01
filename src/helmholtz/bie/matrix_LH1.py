""" Utils for BIE

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from numpy import (
    absolute,
    arange,
    cos,
    eye,
    full,
    log,
    logical_not,
    meshgrid,
    ones,
    pi,
    sin,
    sqrt,
    where,
)
from numpy.typing import NDArray
from scipy.sparse import diags
from scipy.special import hankel1, jv

from .grid import _mesh_grid, half_grid
from .quadrature import kress_weight_ev


def matrix_L1_ev(N: int) -> NDArray:
    """
    L = matrix_L1_ev(N)

    Return the even part of the matrix 𝕃₁.

    Parameters
    ----------
    N : Int
        number of grid point

    Returns
    -------
    L : Matrix
    """

    m = arange(N)
    M, K = meshgrid(m, m)

    A = cos((pi / (N - 1)) * M * K)
    A[:, 0] *= 0.5
    A[:, -1] *= 0.5
    A *= sqrt(2 / (N - 1))

    return A @ (diags(-m) @ A)


def matrix_L1_od(N):
    """
    L = matrix_L1_od(N)

    Return the odd part of the matrix 𝕃₁.

    Parameters
    ----------
    N : Int
        number of grid point

    Returns
    -------
    L : Matrix
    """

    m = arange(1, N + 1)
    M, K = meshgrid(m, m)

    A = sqrt(2 / (N + 1)) * sin((pi / (N + 1)) * M * K)

    return A @ (diags(m) @ A)


def matrix_H1_ev(k, N):
    """
    H = matrix_H1_ev(N)

    Return the even part of the matrix ℍ₁.

    Parameters
    ----------
    N : Int
        number of grid point

    Returns
    -------
    H : Matrix
    """

    n = N // 2 + 1
    S, T = _mesh_grid(half_grid(N)[0])

    Z = k * absolute(sin(S) - sin(T))
    Z_not_0 = where(logical_not(eye(n, dtype=bool)))

    Φ = full((n, n), 0.5)
    Φ[Z_not_0] = jv(1, Z[Z_not_0])

    Ψ = ones((n, n), dtype=complex)
    Ψ[Z_not_0] = (
        0.5
        * Z[Z_not_0]
        * (
            (1j * pi) * hankel1(1, Z[Z_not_0])
            + Φ[Z_not_0] * log(((2 / k) * Z[Z_not_0]) ** 2)
        )
    )

    Φ[Z_not_0] /= Z[Z_not_0]
    Φ *= (0.5 * k * k / pi) * (1 - sin(S) * sin(T))

    return matrix_L1_ev(n) * Ψ - kress_weight_ev(n) * Φ


def matrix_H1_od(k, N):
    """
    H = matrix_H1_od(N)

    Return the odd part of the matrix ℍ₁.

    Parameters
    ----------
    N : Int
        number of grid point

    Returns
    -------
    H : Matrix
    """

    n = N // 2 - 1
    S, T = _mesh_grid(half_grid(N)[0])

    Z = k * absolute(sin(S[1:-1, 1:-1]) - sin(T[1:-1, 1:-1]))
    Z_not_0 = where(logical_not(eye(n, dtype=bool)))

    Φ = full((n, n), 0.5)
    Φ[Z_not_0] = jv(1, Z[Z_not_0])

    Ψ = ones((n, n), dtype=complex)
    Ψ[Z_not_0] = (
        0.5
        * Z[Z_not_0]
        * (
            (1j * pi) * hankel1(1, Z[Z_not_0])
            + Φ[Z_not_0] * log(((2 / k) * Z[Z_not_0]) ** 2)
        )
    )

    Φ[Z_not_0] /= Z[Z_not_0]
    Φ *= (-0.5 * k * k / pi) * cos(S[1:-1, 1:-1]) * cos(T[1:-1, 1:-1])

    return matrix_L1_od(n) * Ψ - kress_weight_ev(n + 2)[1:-1, 1:-1] * Φ
