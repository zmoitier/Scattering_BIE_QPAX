""" Plane wave

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from typing import Union

from numpy import broadcast_shapes, greater_equal, less, where, zeros
from numpy.typing import NDArray

from .field import IncidentField, MonoField, ParityField, ScatterField


def _eval_monofield(
    mfield: MonoField, ξ: NDArray, η: NDArray, pξ: int, pη: int
) -> Union[int, NDArray]:
    """Eval the Monofield for the elliptical coordinates."""

    q = mfield.q
    return sum(
        (
            c * mfield.angfct(m, q, η, p=pη) * mfield.radfct(m, q, ξ, p=pξ)
            for m, c in zip(mfield.index, mfield.coeff)
        )
    )


def _eval_parityfield(
    pfield: ParityField, ξ: NDArray, η: NDArray, pξ: int, pη: int
) -> Union[int, NDArray]:
    """Eval the ParityField for the elliptical coordinates."""

    return _eval_monofield(pfield.even, ξ, η, pξ, pη) + _eval_monofield(
        pfield.odd, ξ, η, pξ, pη
    )


def eval_incident_field(
    in_field: IncidentField, ξ: NDArray, η: NDArray, pξ: int = 0, pη: int = 0
) -> Union[int, NDArray]:
    """Return the incident field for the elliptical coordinates."""

    return _eval_parityfield(in_field, ξ, η, pξ, pη)


def eval_scatter_field(
    in_field: IncidentField,
    sc_field: ScatterField,
    ξ: NDArray,
    η: NDArray,
    pξ: int = 0,
    pη: int = 0,
) -> NDArray:
    """Return the scatter field for the elliptical coordinates."""

    u = zeros(broadcast_shapes(ξ.shape, η.shape), dtype=complex)

    if sc_field.inn_field is not None:
        inner = where(less(ξ, sc_field.boundary_ξ))
        u[inner] = _eval_parityfield(
            sc_field.inn_field, ξ[inner], η[inner], pξ, pη
        ) - _eval_parityfield(in_field, ξ[inner], η[inner], pξ, pη)

    outer = where(greater_equal(ξ, sc_field.boundary_ξ))
    u[outer] = _eval_parityfield(sc_field.out_field, ξ[outer], η[outer], pξ, pη)

    return u


def eval_total_field(
    in_field: IncidentField,
    sc_field: ScatterField,
    ξ: NDArray,
    η: NDArray,
    pξ: int = 0,
    pη: int = 0,
) -> NDArray:
    """Return the total field for the elliptical coordinates."""

    u = zeros(broadcast_shapes(ξ.shape, η.shape), dtype=complex)

    if sc_field.inn_field is not None:
        inner = where(less(ξ, sc_field.boundary_ξ))
        u[inner] = _eval_parityfield(sc_field.inn_field, ξ[inner], η[inner], pξ, pη)

    outer = where(greater_equal(ξ, sc_field.boundary_ξ))
    u[outer] = _eval_parityfield(
        sc_field.out_field, ξ[outer], η[outer], pξ, pη
    ) + _eval_parityfield(in_field, ξ[outer], η[outer], pξ, pη)

    return u
