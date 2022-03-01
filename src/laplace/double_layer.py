""" Double layer Laplace

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology,
"""

from math import atan
from typing import Callable, Tuple

from numpy import (
    absolute,
    diag_indices,
    empty,
    eye,
    less,
    logical_not,
    pi,
    power,
    remainder,
    where,
)
from numpy.typing import NDArray

from ..helmholtz.bie.grid import _mesh_grid, grid
from ..helmholtz.ellipse_parametrization import EllipseParametrization


def _distance2(
    S: NDArray, T: NDArray, gamma: Callable
) -> Tuple[NDArray, NDArray, NDArray]:
    """Return distance and difference."""

    x_s, y_s = gamma(S)
    x_t, y_t = gamma(T)

    xdiff, ydiff = x_s - x_t, y_s - y_t
    dist2 = power(xdiff, 2) + power(ydiff, 2)

    return dist2, xdiff, ydiff


def double_layer_pqr(
    ellipse: EllipseParametrization, N: int
) -> Tuple[NDArray, NDArray]:
    """Double layer"""
    # pylint: disable=too-many-locals

    θ, _ = grid(N)
    S, T = _mesh_grid(θ)

    dist2, xdiff, ydiff = _distance2(S, T, ellipse.gamma)
    jac = ellipse.jacobian(T)
    ν_x, ν_y = ellipse.normal_ext(T)

    dist_eq_0 = diag_indices(N)
    dist_not_0 = where(logical_not(eye(N, dtype=bool)))

    kernel = empty((N, N))
    kernel[dist_not_0] = (
        (ν_x[dist_not_0] * xdiff[dist_not_0] + ν_y[dist_not_0] * ydiff[dist_not_0])
        * jac[dist_not_0]
        / (dist2[dist_not_0] * N)
    )
    kernel[dist_eq_0] = (-1 / (2 * N)) * jac[dist_eq_0] * ellipse.curvature(θ)

    return kernel, θ


def double_layer_mpqr(
    ellipse: EllipseParametrization, N: int
) -> Tuple[NDArray, NDArray]:
    """Double layer"""
    # pylint: disable=too-many-locals

    θ, Δθ = grid(N)
    S, T = _mesh_grid(θ)

    dist2, xdiff, ydiff = _distance2(S, T, ellipse.gamma)
    jac = ellipse.jacobian(T)
    ν_x, ν_y = ellipse.normal_ext(T)

    dist_eq_0 = diag_indices(N)
    dist_not_0 = where(logical_not(eye(N, dtype=bool)))

    kernel = empty((N, N))
    kernel[dist_not_0] = (
        (ν_x[dist_not_0] * xdiff[dist_not_0] + ν_y[dist_not_0] * ydiff[dist_not_0])
        * jac[dist_not_0]
        / (dist2[dist_not_0] * N)
    )
    kernel[dist_eq_0] = (-1 / (2 * N)) * jac[dist_eq_0] * ellipse.curvature(θ)

    quasi_sing = where(less(absolute(remainder(S + T, 2 * pi) - pi), Δθ / 2))
    kernel[quasi_sing] = -atan(Δθ / (4 * ellipse.ɛ)) / pi

    return kernel, θ
