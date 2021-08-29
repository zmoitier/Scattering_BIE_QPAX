""" Utils for BIE

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
import numpy as np
from scipy.sparse import diags
from scipy.special import hankel1, jv

from .grid import half_grid
from .quadrature import kress_weight_ev


def matrix_L1_ev(N):
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

    m = np.arange(N)
    M, K = np.meshgrid(m, m)

    A = np.cos((np.pi / (N - 1)) * M * K)
    A[:, 0] *= 0.5
    A[:, -1] *= 0.5
    A *= np.sqrt(2 / (N - 1))

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

    m = np.arange(1, N + 1)
    M, K = np.meshgrid(m, m)

    A = np.sqrt(2 / (N + 1)) * np.sin((np.pi / (N + 1)) * M * K)

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
    _, _, S, T = half_grid(N, mesh_grid=True)

    Z = k * np.abs(np.sin(S) - np.sin(T))
    Z_not_0 = np.where(np.logical_not(np.eye(n, dtype=bool)))

    Φ = np.full((n, n), 0.5)
    Φ[Z_not_0] = jv(1, Z[Z_not_0])

    Ψ = np.ones((n, n), dtype=complex)
    Ψ[Z_not_0] = (
        0.5
        * Z[Z_not_0]
        * (
            (1j * np.pi) * hankel1(1, Z[Z_not_0])
            + Φ[Z_not_0] * np.log(((2 / k) * Z[Z_not_0]) ** 2)
        )
    )

    Φ[Z_not_0] /= Z[Z_not_0]
    Φ *= (0.5 * k * k / np.pi) * (1 - np.sin(S) * np.sin(T))

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
    _, _, S, T = half_grid(N, mesh_grid=True)

    Z = k * np.abs(np.sin(S[1:-1, 1:-1]) - np.sin(T[1:-1, 1:-1]))
    Z_not_0 = np.where(np.logical_not(np.eye(n, dtype=bool)))

    Φ = np.full((n, n), 0.5)
    Φ[Z_not_0] = jv(1, Z[Z_not_0])

    Ψ = np.ones((n, n), dtype=complex)
    Ψ[Z_not_0] = (
        0.5
        * Z[Z_not_0]
        * (
            (1j * np.pi) * hankel1(1, Z[Z_not_0])
            + Φ[Z_not_0] * np.log(((2 / k) * Z[Z_not_0]) ** 2)
        )
    )

    Φ[Z_not_0] /= Z[Z_not_0]
    Φ *= (-0.5 * k * k / np.pi) * np.cos(S[1:-1, 1:-1]) * np.cos(T[1:-1, 1:-1])

    return matrix_L1_od(n) * Ψ - kress_weight_ev(n + 2)[1:-1, 1:-1] * Φ
