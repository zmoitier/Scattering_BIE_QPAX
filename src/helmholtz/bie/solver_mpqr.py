"""Solver for MPQR

    Author: Zoïs Moitier
        Karlsruhe Institute of Technology, Germany
"""

from numpy import block, sqrt, zeros
from scipy.linalg import solve
from scipy.sparse import identity

from ..ellipse_parametrization import EllipseParametrization
from ..obstacle import BoundaryType, Obstacle
from ..trace import Trace
from .layer_mpqr import double_layer_mpqr
from .layer_pqr import single_layer_pqr
from .solution import BieSolution


def _solver_mpqr_neu(
    eparam: EllipseParametrization, k: float, trace: Trace, N: int
) -> BieSolution:
    """The PQR (Kress) method for the Neumann Helmholtz problem."""

    L, θ = double_layer_mpqr(eparam, k, N)
    A = 0.5 * identity(N) - L

    U = solve(A, trace[0](θ))

    return BieSolution(trace=U, scaled_normal_trace=zeros(N), grid=θ, N=N)


def _solver_mpqr_dir(
    eparam: EllipseParametrization, k: float, trace: Trace, N: int
) -> BieSolution:
    """The PQR (Kress) method for the Dirichlet Helmholtz problem."""

    L, θ = double_layer_mpqr(eparam, k, N)
    A = 0.5 * identity(N) + L.transpose()

    U = solve(A, trace[1](θ))

    return BieSolution(trace=zeros(N), scaled_normal_trace=U, grid=θ, N=N)


def _solver_mpqr_pen(obs: Obstacle, k_out: float, trace: Trace, N: int) -> BieSolution:
    """The PQR (Kress) method for the Penetrable Helmholtz problem."""

    k_inn = k_out * sqrt(obs.ρ / obs.σ)

    D_inn, θ = double_layer_mpqr(obs.param, k_inn, N)
    D_out, _ = double_layer_mpqr(obs.param, k_out, N)

    S_inn, _ = single_layer_pqr(obs.param, k_inn, N)
    S_out, _ = single_layer_pqr(obs.param, k_out, N)

    A = block(
        [
            [0.5 * identity(N) - D_out, S_out],
            [0.5 * identity(N) + D_inn, (-1 / obs.σ) * S_inn],
        ]
    )
    F = block([trace[0](θ), zeros(N)])
    U = solve(A, F)

    return BieSolution(trace=U[:N], scaled_normal_trace=U[N:], grid=θ, N=N)


def get_total_field_mpqr(obs: Obstacle, k: float, trace: Trace, N: int) -> BieSolution:
    """The PQR (Kress) method for the Helmholtz problem."""

    if obs.boundary_type == BoundaryType.NEUMANN:
        return _solver_mpqr_neu(obs.param, k, trace, N)

    if obs.boundary_type == BoundaryType.DIRICHLET:
        return _solver_mpqr_dir(obs.param, k, trace, N)

    if obs.boundary_type == BoundaryType.PENETRABLE:
        return _solver_mpqr_pen(obs, k, trace, N)

    raise ValueError(f"{obs.boundary_type} must be of type BoundaryType.")
