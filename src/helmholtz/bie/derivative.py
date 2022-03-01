""" Spectrally accurate second derivative

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from numpy import arange, cos, meshgrid, pi, sin, sqrt
from numpy.typing import NDArray
from scipy.fft import irfft
from scipy.linalg import toeplitz
from scipy.sparse import diags


def mat_d2(N: int) -> NDArray:
    """
    W = mat_der(N)

    For a vector v = (f(θᵢ)) where f is a 2π-periodic function and θ an equi-space grid
    of ℝ / 2πℤ, the matrix vector product W @ v return a spectrally accurate
    approximation of the second derivative of f.

    Parameters
    ----------
    N : int
        number of grid points

    Returns
    -------
    W : array
        weight
    """

    coeffs = -arange(N // 2 + 1) ** 2
    R = irfft(coeffs, n=N)
    return toeplitz(R)


def mat_d2_ev(N: int) -> NDArray:
    """
    W = mat_d2_ev(N)

    For a vector v = (f(θᵢ)) where f is an even 2π-periodic function and θ an equi-space
    grid of [-π/2, π/2], the matrix vector product W @ v return a spectrally accurate
    approximation of the second derivative of f.

    Parameters
    ----------
    N : int
        number of grid points

    Returns
    -------
    W : array
        weight
    """

    m = arange(N)
    M, K = meshgrid(m, m)

    A = cos((pi / (N - 1)) * M * K)
    A[:, 0] *= 0.5
    A[:, -1] *= 0.5
    A *= sqrt(2 / (N - 1))

    return A @ (diags(-(m**2)) @ A)


def mat_d2_od(N: int) -> NDArray:
    """
    W = mat_d2_od(N)

    For a vector v = (f(θᵢ)) where f is an odd 2π-periodic function and θ an equi-space
    grid of (-π/2, π/2), the matrix vector product W @ v return a spectrally accurate
    approximation of the second derivative of f.

    Parameters
    ----------
    N : int
        number of grid points

    Returns
    -------
    W : array
        weight
    """

    m = arange(1, N + 1)
    M, K = meshgrid(m, m)

    A = sqrt(2 / (N + 1)) * sin((pi / (N + 1)) * M * K)

    return A @ (diags(-(m**2)) @ A)
