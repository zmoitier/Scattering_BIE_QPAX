""" Plane wave

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from typing import Callable, Tuple

from numpy import arange, array, block, zeros, zeros_like
from numpy.typing import NDArray
from scipy.linalg import solve
from scipy.sparse import diags

from ..obstacle import BoundaryType, Obstacle
from .field import IncidentField, MonoField, ParityField, ScatterField
from .mathieu_api import Mce1, Mce3, Mse1, Mse3, RadialMathieu
from .solver_utils import (
    _get_mat_A_even,
    _get_mat_A_odd,
    _get_mat_B_even,
    _get_mat_B_odd,
)


def _solve_obstacle_mono(
    ξ0: float, p: int, M_e3: RadialMathieu, mfield: MonoField
) -> MonoField:
    """Solve obstacle for Monofield."""

    q = mfield.q
    rf = mfield.radfct

    index = mfield.index
    coeff = array(
        [
            -c * rf(m, q, ξ0, p=p) / M_e3(m, q, ξ0, p=p)
            for m, c in zip(index, mfield.coeff)
        ]
    )

    return MonoField(q=q, angfct=mfield.angfct, radfct=M_e3, index=index, coeff=coeff)


def _solve_obstacle(obs: Obstacle, in_field: IncidentField) -> ScatterField:
    """Solve obstacle."""

    if obs.boundary_type == BoundaryType.DIRICHLET:
        p = 0
    elif obs.boundary_type == BoundaryType.NEUMANN:
        p = 1
    else:
        raise ValueError(f"{obs.boundary_type} must be of type BoundaryType.")

    out_field = ParityField(
        even=_solve_obstacle_mono(obs.boundary_ξ, p, Mce3, in_field.even),
        odd=_solve_obstacle_mono(obs.boundary_ξ, p, Mse3, in_field.odd),
    )

    return ScatterField(obs.boundary_ξ, inn_field=None, out_field=out_field)


def _split_coeff(
    index: NDArray, coeff: NDArray, even: bool
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split the coeffients."""

    if index.size == 0:
        return array([]), array([]), array([]), array([])

    N = index.max() + 1
    if even:
        m_tt = arange(0, N)
        idx_ev = arange(0, N, 2)
        idx_od = arange(1, N, 2)
        shift = 0
    else:
        m_tt = arange(1, N)
        idx_ev = arange(1, N - 1, 2)
        idx_od = arange(0, N - 1, 2)
        shift = 1

    coef_pad = zeros(N, dtype=coeff.dtype)
    for m, c in zip(index, coeff):
        coef_pad[m - shift] = c

    return m_tt, coef_pad, idx_ev, idx_od


def _solve_penetrable_mono_part(
    σ: float,
    q_inn: float,
    q_out: float,
    ξ: float,
    vm: NDArray,
    c_in: NDArray,
    radfct: RadialMathieu,
    get_mat: Callable,
    M_e1: RadialMathieu,
    M_e3: RadialMathieu,
) -> Tuple[NDArray, NDArray]:
    """Solve part."""
    # pylint: disable=too-many-locals

    N = vm.size

    A_inn = get_mat(q_inn, N).T
    A_out = get_mat(q_out, N).T

    A = block(
        [
            [
                A_inn @ diags(M_e1(vm, q_inn, ξ, p=0)),
                -A_out @ diags(M_e3(vm, q_out, ξ, p=0)),
            ],
            [
                σ * A_inn @ diags(M_e1(vm, q_inn, ξ, p=1)),
                -A_out @ diags(M_e3(vm, q_out, ξ, p=1)),
            ],
        ]
    )

    C_in = zeros(2 * N, dtype=complex)
    C_in[:N] = A_out @ (radfct(vm, q_out, ξ, p=0) * c_in)
    C_in[N:] = A_out @ (radfct(vm, q_out, ξ, p=1) * c_in)

    S = solve(A, C_in)

    return S[:N], S[N:]


def _solve_penetrable_mono(
    ξ: float, τ: float, σ: float, mfield: MonoField, even: bool
) -> Tuple[MonoField, MonoField]:
    """Solve for a MonoField."""
    # pylint: disable=too-many-locals

    q_out = mfield.q
    q_inn = q_out * τ

    m_tt, c_in, idx_ev, idx_od = _split_coeff(mfield.index, mfield.coeff, even)

    if even:
        get_mat_ev = _get_mat_A_even
        get_mat_od = _get_mat_A_odd
        M_e1 = Mce1
        M_e3 = Mce3
    else:
        get_mat_ev = _get_mat_B_even
        get_mat_od = _get_mat_B_odd
        M_e1 = Mse1
        M_e3 = Mse3

    c_inn = zeros_like(c_in, dtype=complex)
    c_out = zeros_like(c_in, dtype=complex)

    if idx_ev.size != 0:
        c_inn[idx_ev], c_out[idx_ev] = _solve_penetrable_mono_part(
            σ,
            q_inn,
            q_out,
            ξ,
            m_tt[idx_ev],
            c_in[idx_ev],
            mfield.radfct,
            get_mat_ev,
            M_e1,
            M_e3,
        )
    if idx_od.size != 0:
        c_inn[idx_od], c_out[idx_od] = _solve_penetrable_mono_part(
            σ,
            q_inn,
            q_out,
            ξ,
            m_tt[idx_od],
            c_in[idx_od],
            mfield.radfct,
            get_mat_od,
            M_e1,
            M_e3,
        )

    return (
        MonoField(q=q_inn, angfct=mfield.angfct, radfct=M_e1, index=m_tt, coeff=c_inn),
        MonoField(q=q_out, angfct=mfield.angfct, radfct=M_e3, index=m_tt, coeff=c_out),
    )


def _solve_penetrable(obs: Obstacle, in_field: IncidentField) -> ScatterField:
    """Solve penetrable"""

    ξ = obs.boundary_ξ
    τ = obs.ρ / obs.σ

    f_even_inn, f_even_out = _solve_penetrable_mono(ξ, τ, obs.σ, in_field.even, True)
    f_odd_inn, f_odd_out = _solve_penetrable_mono(ξ, τ, obs.σ, in_field.odd, False)

    return ScatterField(
        obs.boundary_ξ,
        inn_field=ParityField(even=f_even_inn, odd=f_odd_inn),
        out_field=ParityField(even=f_even_out, odd=f_odd_out),
    )


def get_scatter_field(obs: Obstacle, in_field: IncidentField) -> ScatterField:
    """Return the scattering field object."""

    if obs.boundary_type == BoundaryType.PENETRABLE:
        return _solve_penetrable(obs, in_field)

    return _solve_obstacle(obs, in_field)
