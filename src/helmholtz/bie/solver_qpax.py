"""Solver for QPAX

    Author: Zoïs Moitier
        Karlsruhe Institute of Technology, Germany
"""

from numpy import zeros
from scipy.linalg import solve

from ..obstacle import BoundaryType, Obstacle
from ..trace import TraceExpansion
from .grid import grid
from .matrix_LH1 import matrix_H1_ev, matrix_H1_od
from .solution import BieSolution


def _solver_qpax_neu(
    ɛ: float, k: float, trace_expan: TraceExpansion, N: int
) -> BieSolution:
    """The QPAX method for the Neumann Helmholtz problem."""

    H1_ev = matrix_H1_ev(k, N)
    H1_od = matrix_H1_od(k, N)

    n = N // 2
    θ, _ = grid(N)

    μ = zeros(N, dtype=complex)
    if trace_expan[0] is not None:
        f_ev = trace_expan[0](θ[: n + 1])

        μ_ev = zeros(N, dtype=complex)
        μ_ev[: n + 1] += f_ev - ɛ * (H1_ev @ f_ev)
        μ_ev[n + 1 :] += μ_ev[n - 1 : 0 : -1]

        μ += μ_ev

    if trace_expan[1] is not None:
        f_od = trace_expan[1](θ[1:n])

        μ_od = zeros(N, dtype=complex)
        μ_od[1:n] += solve(H1_od, f_od)
        μ_od[n + 1 :] -= μ_od[n - 1 : 0 : -1]

        μ += μ_od

    return BieSolution(trace=μ, scaled_normal_trace=zeros(N), grid=θ, N=N)


def _solver_qpax_dir(
    ɛ: float, k: float, trace_expan: TraceExpansion, N: int
) -> BieSolution:
    """The QPAX method for the Dirichlet Helmholtz problem."""

    H1_ev = matrix_H1_ev(k, N)
    H1_od = matrix_H1_od(k, N)

    n = N // 2
    θ, _ = grid(N)

    μ = zeros(N, dtype=complex)
    if trace_expan[1] is not None:
        f_od = trace_expan[1](θ[1:n])

        μ_od = zeros(N, dtype=complex)
        μ_od[1:n] += f_od + ɛ * (H1_od @ f_od)
        μ_od[n + 1 :] -= μ_od[n - 1 : 0 : -1]

        μ += μ_od

    if trace_expan[2] is not None:
        f_ev = trace_expan[2](θ[: n + 1])

        μ_ev = zeros(N, dtype=complex)
        μ_ev[: n + 1] -= solve(H1_ev, f_ev)
        μ_ev[n + 1 :] += μ_ev[n - 1 : 0 : -1]

        μ += μ_ev

    return BieSolution(trace=zeros(N), scaled_normal_trace=μ, grid=θ, N=N)


def _solver_qpax_pen(trace_expan: TraceExpansion, N: int) -> BieSolution:
    """The QPAX method for the Dirichlet Helmholtz problem."""

    θ, _ = grid(N)
    μ = trace_expan[0](θ)
    ν = trace_expan[1](θ)

    return BieSolution(trace=μ, scaled_normal_trace=ν, grid=θ, N=N)


def get_total_field_qpax(
    obs: Obstacle, k: float, trace_expan: TraceExpansion, N: int
) -> BieSolution:
    """The QPAX method for the Helmholtz problem."""

    if obs.boundary_type == BoundaryType.NEUMANN:
        return _solver_qpax_neu(obs.param.ɛ, k, trace_expan, N)

    if obs.boundary_type == BoundaryType.DIRICHLET:
        return _solver_qpax_dir(obs.param.ɛ, k, trace_expan, N)

    if obs.boundary_type == BoundaryType.PENETRABLE:
        return _solver_qpax_pen(trace_expan, N)

    raise ValueError(f"{obs.boundary_type} must be of type BoundaryType.")
