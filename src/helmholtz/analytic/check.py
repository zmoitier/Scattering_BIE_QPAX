"""Check for analytic solution

    Author: Zoïs Moitier
        Karlsruhe Institute of Technology, Germany
"""

from typing import Tuple

from numpy import absolute, array
from numpy.typing import NDArray

from ..obstacle import BoundaryType, Obstacle
from .eval_field import _eval_parityfield
from .field import IncidentField, ScatterField


def check_neumann(
    obs: Obstacle, in_field: IncidentField, sc_field: ScatterField, η: NDArray
) -> NDArray:
    """
    Check analytical solution for Neumann boundary condition by returning the normal
    derivative of the total field ∂ₙu.
    """

    if obs.boundary_type != BoundaryType.NEUMANN:
        raise ValueError("BoundaryType must be NEUMANN")

    ξ = array(obs.boundary_ξ)
    return absolute(
        _eval_parityfield(sc_field.out_field, ξ, η, 1, 0)
        + _eval_parityfield(in_field, ξ, η, 1, 0)
    )


def check_dirichlet(
    obs: Obstacle, in_field: IncidentField, sc_field: ScatterField, η: NDArray
) -> NDArray:
    """
    Check analytical solution for Dirichlet boundary condition by returning the trace
    of the total field u.
    """

    if obs.boundary_type != BoundaryType.DIRICHLET:
        raise ValueError("BoundaryType must be DIRICHLET")

    ξ = array(obs.boundary_ξ)
    return absolute(
        _eval_parityfield(sc_field.out_field, ξ, η, 0, 0)
        + _eval_parityfield(in_field, ξ, η, 0, 0)
    )


def check_penetrable(
    obs: Obstacle, in_field: IncidentField, sc_field: ScatterField, η: NDArray
) -> Tuple[NDArray, NDArray]:
    """
    Check analytical solution for scattering by a penetrable ellipse by returning the
    trace u and the normal derivative ∂ₙu of the total field.
    """

    ξ = array(obs.boundary_ξ)
    if sc_field.inn_field is not None:
        u0_inn = _eval_parityfield(sc_field.inn_field, ξ, η, 0, 0)
        u0_out = _eval_parityfield(sc_field.out_field, ξ, η, 0, 0) + _eval_parityfield(
            in_field, ξ, η, 0, 0
        )
        rel_err_0 = 2 * absolute((u0_inn - u0_out) / (u0_inn + u0_out))

        u1_inn = obs.σ * _eval_parityfield(sc_field.inn_field, ξ, η, 1, 0)
        u1_out = _eval_parityfield(sc_field.out_field, ξ, η, 1, 0) + _eval_parityfield(
            in_field, ξ, η, 1, 0
        )
        rel_err_1 = 2 * absolute((u1_inn - u1_out) / (u1_inn + u1_out))

        return (rel_err_0, rel_err_1)

    raise ValueError("must have inner field.")
