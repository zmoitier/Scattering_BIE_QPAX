""" Grid for BIE

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from typing import Tuple

from numpy import arange, full, linspace, meshgrid, ones, pi
from numpy.typing import NDArray
from scipy.sparse import coo_matrix


def grid(N: int) -> Tuple[NDArray, float]:
    """
    Return the N equi-space point of the grid [-π/2, 3π/2).
    """

    θ, Δθ = linspace(-pi / 2, 3 * pi / 2, num=N, endpoint=False, retstep=True)
    return θ, Δθ


def half_grid(N: int) -> Tuple[NDArray, float]:
    """
    Return the N equi-space point of the grid [-π/2, π/2] which return the same N÷2+1
    points than grid(N).
    """

    θ, Δθ = linspace(-pi / 2, pi / 2, num=N // 2 + 1, retstep=True)
    return θ, Δθ


def _mesh_grid(θ: NDArray) -> Tuple[NDArray, NDArray]:
    """Mesh grid"""

    T, S = meshgrid(θ, θ)
    return S, T


def parity_base(N: int):
    """
    P, Q = parity_base(N)

    Return the matrix corresponding to the parity even/odd with respect of the major-
    axis of the ellipse from the grid function.

    Parameters
    ----------
    N : int
        size of the matrix

    Returns
    -------
    P : array
        projection on the even/odd sub-space
    Q : array
        Q = P⁻¹
    """

    n = N // 2

    p_v = [
        *ones(n + 1, dtype=int),
        *full(n - 1, -1, dtype=int),
        *ones(2 * (n - 1), dtype=int),
    ]
    p_i = [*arange(N), *arange(1, n), *arange(N - 1, N - n, -1)]
    p_j = [
        *arange(n + 1),
        *arange(N - 1, N - n, -1),
        *arange(n + 1, N),
        *arange(1, n),
    ]

    q_v = [1, *full(n - 1, 0.5), 1, *full(n - 1, -0.5), *full(2 * (n - 1), 0.5)]
    q_i = [
        *arange(n + 1),
        *arange(N - 1, N - n, -1),
        *arange(1, n),
        *arange(n + 1, N),
    ]
    q_j = [*arange(N), *arange(N - 1, N - n, -1), *arange(1, n)]

    return (
        coo_matrix((p_v, (p_i, p_j)), shape=(N, N)),
        coo_matrix((q_v, (q_i, q_j)), shape=(N, N)),
    )
