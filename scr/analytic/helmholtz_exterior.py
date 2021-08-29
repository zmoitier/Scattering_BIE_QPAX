""" Dataclass and functions for Helmholtz exterior

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
from dataclasses import dataclass

from numpy import broadcast_shapes, shape, zeros

from .mathieu import Mce1, Mce2, Mce3, Mce4, Mse1, Mse2, Mse3, Mse4, ce, se


@dataclass(frozen=True)
class Field:
    """
    Dataclass that describe the coefficients of field in term of Modified Mathieu
    functions.

    Attributes
    ----------
    ε : float
        parameter for elliptical coordinates
    c : float
        (0, ±c) focal point of the Elliptic coordinates
    k : float
        Wavenumber
    q : float
        The q parameter in Mathieu ODEs, q = c² * k² / 4
    coef_ce : list{list{tuple{int, number}}}
        Coefficients in front of  Mce⁽ⁱ⁾ₘ(ξ, q) * ceₘ(η, q)
    coef_se : list{list{tuple{int, number}}}
        Coefficients in front of Mse⁽ⁱ⁾ₘ(ξ, q) * seₘ(η, q)
    """

    ε: float
    c: float
    k: float
    q: float
    coef_ce: list
    coef_se: list


def create_field(ε, c, k, coef_ce, coef_se):
    """
    field = create_field(c, k, coef_ce, coef_se)

    Create Field dataclass form a wavenumber and list of coefficients.

    Parameters
    ----------
    ε : float
        parameter for elliptical coordinates
    c : float
        (0, ±c) focal point of the Elliptic coordinates
    k : float
        Wavenumber.
    coef_ce : list{list{tuple{int, number}}}
        Coefficients in front of ceₘ(q, η) * Mce⁽ⁱ⁾ₘ(q, ξ)
    coef_se : list{list{tuple{int, number}}}
        Coefficients in front of seₘ(q, η) * Mse⁽ⁱ⁾ₘ(q, ξ)

    Returns
    -------
    field : Field
    """

    return Field(ε, c, k, c * c * k * k / 4, coef_ce, coef_se)


def solve_field(incident_field, ξ0, p):
    """
    field = solve_scattering(incident_field, ξ0, p)

    Create Field dataclass for the solution of scattering Helmholtz with
    homogeneous Dirichlet or Neuman boundary condition.

    Parameters
    ----------
    incident_field : Field
        Incident field
    ξ0 : float
        radial Elliptic coordinate of the boundary
    p : 0 or 1
        0 for Dirichlet condition and 1 for Neumann conditions

    Returns
    -------
    field : Field
    """

    ε = incident_field.ε
    c = incident_field.c
    k = incident_field.k
    q = incident_field.q

    coef_c3 = []
    coef_s3 = []

    for coef_c_, Mce_ in zip(incident_field.coef_ce, (Mce1, Mce2, Mce3, Mce4)):
        for m, d in coef_c_:
            coef_c3.append((m, -d * Mce_(m, q, ξ0, p=p) / Mce3(m, q, ξ0, p=p)))

    for coef_s_, Mse_ in zip(incident_field.coef_se, (Mse1, Mse2, Mse3, Mse4)):
        for m, d in coef_s_:
            coef_s3.append((m, -d * Mse_(m, q, ξ0, p=p) / Mse3(m, q, ξ0, p=p)))

    return Field(ε, c, k, q, [[], [], coef_c3, []], [[], [], coef_s3, []])


def eval_field(field, ξ, η, *, d_ξ=0, d_η=0):
    """
    u = eval_field(field, ξ, η, *, d_ξ=0, d_η=0)

    Compute the field in the Elliptic coordinates.

    Parameters
    ----------
    field : Field
        field to be evaluated
    ξ : array_like
        radial Elliptic coordinate
    η : array_like
        angular Elliptic coordinate
    d_ξ : 0 or 1 or 2 (default 0)
        0 for the value of the field
        1 for the ∂_ξ derivative of the field
        2 for the ∂_ξξ derivative of the field
    d_η : 0 or 1 or 2 (default 0)
        0 for the value of the field
        1 for the ∂_η derivative of the field
        2 for the ∂_ηη derivative of the field

    Returns
    -------
    u : array_like
        values of the field
    """

    q = field.q

    u = zeros(broadcast_shapes(shape(ξ), shape(η)), dtype=complex)

    for coef_c_, Mce_ in zip(field.coef_ce, (Mce1, Mce2, Mce3, Mce4)):
        for m, c in coef_c_:
            u += c * ce(m, q, η, p=d_η) * Mce_(m, q, ξ, p=d_ξ)

    for coef_s_, Mse_ in zip(field.coef_se, (Mse1, Mse2, Mse3, Mse4)):
        for m, c in coef_s_:
            u += c * se(m, q, η, p=d_η) * Mse_(m, q, ξ, p=d_ξ)

    return u


def field_even_part(field):
    """
    field_ev = field_even_part(field)

    Create the even part of the field object.

    Parameters
    ----------
    field : Field
        Field object from the dataclass

    Returns
    -------
    field : Field
    """

    if not (
        field.coef_ce[0] or field.coef_ce[1] or field.coef_ce[2] or field.coef_ce[3]
    ):
        raise Warning("The even part of the field is zero.")

    return Field(field.ε, field.c, field.k, field.q, field.coef_ce, [[], [], [], []])


def field_odd_part(field):
    """
    field_od = field_odd_part(field)

    Create the odd part of the field object.

    Parameters
    ----------
    field : Field
        Field object from the dataclass

    Returns
    -------
    field : Field
    """

    if not (
        field.coef_se[0] or field.coef_se[1] or field.coef_se[2] or field.coef_se[3]
    ):
        raise Warning("The odd part of the field is zero.")

    return Field(field.ε, field.c, field.k, field.q, [[], [], [], []], field.coef_se)
