""" Matrix of the double layer potential

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from math import atan
from typing import Callable, Tuple

from numpy import (
    absolute,
    diag_indices,
    empty,
    eye,
    hypot,
    less,
    log,
    logical_not,
    pi,
    remainder,
    sin,
    where,
)
from numpy.typing import NDArray
from scipy.special import hankel1, jv

from ..obstacle import EllipseParametrization
from .grid import _mesh_grid, grid
from .quadrature import kress_weight


def _distance(
    S: NDArray, T: NDArray, gamma: Callable
) -> Tuple[NDArray, NDArray, NDArray]:
    """Return distance and difference."""

    x_s, y_s = gamma(S)
    x_t, y_t = gamma(T)

    xdiff, ydiff = x_s - x_t, y_s - y_t
    dist = hypot(xdiff, ydiff)

    return dist, xdiff, ydiff


def double_layer_mpqr(
    ellipse: EllipseParametrization, k: float, N: int
) -> Tuple[NDArray, NDArray]:
    """Double layer"""
    # pylint: disable=too-many-locals

    θ, Δθ = grid(N)
    S, T = _mesh_grid(θ)

    dist, xdiff, ydiff = _distance(S, T, ellipse.gamma)

    jac = ellipse.jacobian(T)
    ν_x, ν_y = ellipse.normal_ext(T)

    dist_eq_0 = diag_indices(N)
    dist_not_0 = where(logical_not(eye(N, dtype=bool)))

    cos_term = empty((N, N))
    cos_term[dist_not_0] = (
        ν_x[dist_not_0] * xdiff[dist_not_0] + ν_y[dist_not_0] * ydiff[dist_not_0]
    ) / dist[dist_not_0]

    L1 = empty((N, N))
    L1[dist_not_0] = (
        (-k / (4 * pi))
        * cos_term[dist_not_0]
        * jv(1, k * dist[dist_not_0])
        * jac[dist_not_0]
    )
    L1[dist_eq_0] = 0

    L2 = empty((N, N), dtype=complex)
    L2[dist_not_0] = (
        (0.25 * 1j * k)
        * hankel1(1, k * dist[dist_not_0])
        * cos_term[dist_not_0]
        * jac[dist_not_0]
    )
    L2[dist_not_0] -= L1[dist_not_0] * log(
        4 * sin(0.5 * (S[dist_not_0] - T[dist_not_0])) ** 2
    )
    L2[dist_eq_0] = (-1 / (4 * pi)) * jac[dist_eq_0] * ellipse.curvature(θ)
    L2 *= Δθ

    quasi_sing = where(less(absolute(remainder(S + T, 2 * pi) - pi), Δθ / 2))
    L2[quasi_sing] = -atan(Δθ / (4 * ellipse.ɛ)) / pi

    return (kress_weight(N) * L1 + L2, θ)
