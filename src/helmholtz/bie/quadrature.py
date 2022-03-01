""" Quadrature for BIE

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from numpy import arange, cos, meshgrid, pi, sin, sqrt, zeros
from numpy.typing import NDArray
from scipy.fft import irfft
from scipy.linalg import toeplitz
from scipy.sparse import diags


def kress_weight(N: int) -> NDArray:
    """
    W = kress_weight(N)

    Return the weights for the Kress quadrature for the logarithmic part.

    Parameters
    ----------
    N : int
        size of the matrix

    Returns
    -------
    W : matrix
        the Kress weights
    """

    coeffs = zeros(N // 2 + 1)
    coeffs[1:] = -2 * pi / arange(1, N // 2 + 1)
    R = irfft(coeffs, n=N)
    return toeplitz(R)


def kress_weight_ev(N: int) -> NDArray:
    """
    W = kress_weight_ev(N)

    Return the even weights for the Kress quadrature for the logarithmic part.

    Parameters
    ----------
    N : int
        size of the matrix

    Returns
    -------
    W : matrix
        the Kress weights
    """

    m = arange(N)
    M, K = meshgrid(m, m)

    A = cos((pi / (N - 1)) * M * K)
    A[:, 0] *= 0.5
    A[:, -1] *= 0.5
    A *= sqrt(2 / (N - 1))

    coeffs = zeros(N)
    coeffs[1:] = -2 * pi / m[1:]
    return A @ (diags(coeffs) @ A)


def kress_weight_od(N: int) -> NDArray:
    """
    W = kress_weight_od(N)

    Return the odd weights for the Kress quadrature for the logarithmic part.

    Parameters
    ----------
    N : int
        size of the matrix

    Returns
    -------
    W : matrix
        the Kress weights
    """

    m = arange(1, N + 1)
    M, K = meshgrid(m, m)

    A = sqrt(2 / (N + 1)) * sin((pi / (N + 1)) * M * K)

    return A @ (diags(-2 * pi / m) @ A)
