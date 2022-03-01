""" Plane wave

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from typing import Callable

from numpy import zeros
from numpy.typing import NDArray
from scipy.special import mathieu_even_coef, mathieu_odd_coef


def _get_mat_AB(q: float, N: int, coef_fct: Callable, start: int) -> NDArray:
    """Get the matrix A or B."""

    mat = zeros((N, N))

    for i, m in enumerate(range(start, 2 * N + start, 2)):
        coef = coef_fct(m, q)
        n = coef.size
        if n < N:
            mat[i, :n] = coef
        else:
            mat[i, :] = coef[:N]

    return mat


def _get_mat_A_even(q: float, N: int) -> NDArray:
    """Get the matrix A_even."""
    return _get_mat_AB(q, N, mathieu_even_coef, 0)


def _get_mat_A_odd(q: float, N: int) -> NDArray:
    """Get the matrix A_odd."""
    return _get_mat_AB(q, N, mathieu_even_coef, 1)


def _get_mat_B_even(q: float, N: int) -> NDArray:
    """Get the matrix B_even."""
    return _get_mat_AB(q, N, mathieu_odd_coef, 2)


def _get_mat_B_odd(q: float, N: int) -> NDArray:
    """Get the matrix B_odd."""
    return _get_mat_AB(q, N, mathieu_odd_coef, 1)
