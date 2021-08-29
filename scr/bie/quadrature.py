""" Quadrature for BIE

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
import numpy as np
from scipy.fft import irfft
from scipy.linalg import toeplitz
from scipy.sparse import diags


def kress_weight(N):
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

    coeffs = np.zeros(N // 2 + 1)
    coeffs[1:] = -2 * np.pi / np.arange(1, N // 2 + 1)
    R = irfft(coeffs, n=N)
    return toeplitz(R)


def kress_weight_ev(N):
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

    m = np.arange(N)
    M, K = np.meshgrid(m, m)

    A = np.cos((np.pi / (N - 1)) * M * K)
    A[:, 0] *= 0.5
    A[:, -1] *= 0.5
    A *= np.sqrt(2 / (N - 1))

    coeffs = np.zeros(N)
    coeffs[1:] = -2 * np.pi / m[1:]
    return A @ (diags(coeffs) @ A)


def kress_weight_od(N):
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

    m = np.arange(1, N + 1)
    M, K = np.meshgrid(m, m)

    A = np.sqrt(2 / (N + 1)) * np.sin((np.pi / (N + 1)) * M * K)

    return A @ (diags(-2 * np.pi / m) @ A)
