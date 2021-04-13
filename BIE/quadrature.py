""" Quadrature for BIE

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany

    Last modified: 05/04/2021
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


def pqr_eps_weight(ε, nb, S, T):
    """
    W = eps_pqr(ε, nb, S, T)

    Return the weight of the εPQR method.

    Parameters
    ----------
    ε : float
        semi-minor axis of the ellipse
    nb : int
        nb // 2 is the troncation of the Fourier interpolant
    S : array_like
        s variable
    T : array_like
        t variable

    Returns
    -------
    W : array_like
    """

    N = nb // 2
    SpT = S + T
    coeff = (-0.25 / N) * ((ε - 1) / (ε + 1)) ** np.abs(np.arange(N + 1))

    W = coeff[0] * np.ones_like(SpT)
    W += sum([cm * np.cos(m * SpT) for m, cm in enumerate(2 * coeff[1:], start=1)])

    return W
