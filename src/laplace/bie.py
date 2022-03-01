""" BIE for Laplace interior

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology,
"""

from typing import Callable, Tuple

from numpy import zeros
from scipy.linalg import solve
from scipy.sparse import identity

from ..helmholtz.bie.grid import grid
from ..helmholtz.bie.matrix_LH1 import matrix_L1_ev, matrix_L1_od
from ..helmholtz.ellipse_parametrization import EllipseParametrization
from .double_layer import double_layer_mpqr, double_layer_pqr


def laplace_ptr(ellipse: EllipseParametrization, trace: Callable, N: int):
    """BIE Laplace PTR"""

    L, θ = double_layer_pqr(ellipse, N)
    A = -0.5 * identity(N) + L

    U = solve(A, trace(θ))

    return U, θ, N


def laplace_mtr(ellipse: EllipseParametrization, trace: Callable, N: int):
    """BIE Laplace MPTR"""

    L, θ = double_layer_mpqr(ellipse, N)
    A = -0.5 * identity(N) + L

    μ = solve(A, trace(θ))

    return μ, θ, N


def laplace_qpax(
    ellipse: EllipseParametrization, trace_ev_od: Tuple[Callable, Callable], N: int
):
    """BIE Laplace QPAX"""

    ε = ellipse.ɛ
    n = N // 2
    θ, _ = grid(N)

    L1_ev = matrix_L1_ev(n + 1)
    L1_od = matrix_L1_od(n - 1)

    μ = zeros(N)
    if trace_ev_od[0] is not None:
        f_ev = trace_ev_od[0](θ[: n + 1])

        μ_ev = zeros(N, dtype=f_ev.dtype)
        μ_ev[: n + 1] += -f_ev + ε * (L1_ev @ f_ev)
        μ_ev[n + 1 :] += μ_ev[n - 1 : 0 : -1]

        μ += μ_ev

    if trace_ev_od[1] is not None:
        f_od = trace_ev_od[1](θ[1:n])

        μ_od = zeros(N, dtype=f_od.dtype)
        μ_od[1:n] += (-1 / ε) * solve(L1_od, f_od)
        μ_od[n + 1 :] -= μ_od[n - 1 : 0 : -1]

        μ += μ_od

    return μ, θ, N
