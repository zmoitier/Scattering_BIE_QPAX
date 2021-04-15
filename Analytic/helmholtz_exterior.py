""" Dataclass and functions for Helmholtz exterior

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany

    Last modified: 15/04/2021
"""
from dataclasses import dataclass

import numpy as np

from .mathieu import Mce1, Mce2, Mce3, Mce4, Mse1, Mse2, Mse3, Mse4, ce, se


@dataclass(frozen=True)
class Field:
    """
    Dataclass that describe the coefficients of field in term of Modified Mathieu
    functions.

    Attributes
    ----------
    k : float
        Wavenumber.
    q : float
        The q parameter in Mathieu ODEs, q = c² * k² / 4.
    coef_ce : list{list{tuple{int, number}}}
        Coefficients in front of  Mce⁽ⁱ⁾ₘ(ξ, q) * ceₘ(η, q)
    coef_se : list{list{tuple{int, number}}}
        Coefficients in front of Mse⁽ⁱ⁾ₘ(ξ, q) * seₘ(η, q)
    """

    k: float
    q: float
    coef_ce: list
    coef_se: list


def create_field(c, k, coef_ce, coef_se):
    """
    field = create_field(c, k, coef_ce, coef_se)

    Create Field dataclass form a wavenumber and list of coefficients.

    Parameters
    ----------
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

    return Field(k, c * c * k * k / 4, coef_ce, coef_se)


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

    q = incident_field.q

    coef_c3 = []
    coef_s3 = []

    for coef_c_, Mce_ in zip(incident_field.coef_ce, (Mce1, Mce2, Mce3, Mce4)):
        for m, c in coef_c_:
            coef_c3.append((m, -c * Mce_(m, q, ξ0, p) / Mce3(m, q, ξ0, p)))

    for coef_s_, Mse_ in zip(incident_field.coef_se, (Mse1, Mse2, Mse3, Mse4)):
        for m, c in coef_s_:
            coef_s3.append((m, -c * Mse_(m, q, ξ0, p) / Mse3(m, q, ξ0, p)))

    return Field(incident_field.k, q, [[], [], coef_c3, []], [[], [], coef_s3, []])


def eval_field(field, ξ, η, p=0):
    """
    u = eval_field(field, ξ, η)

    Compute the field in the Elliptic coordinates.

    Parameters
    ----------
    field : Field
        field to be evaluated
    ξ : array_like
        radial Elliptic coordinate
    η : array_like
        angular Elliptic coordinate

    Returns
    -------
    u : array_like
        values of the field
    """

    q = field.q

    u = np.zeros(np.broadcast_shapes(np.shape(ξ), np.shape(η)), dtype=complex)

    for coef_c_, Mce_ in zip(field.coef_ce, (Mce1, Mce2, Mce3, Mce4)):
        for m, c in coef_c_:
            u += c * ce(m, q, η) * Mce_(m, q, ξ, p=p)

    for coef_s_, Mse_ in zip(field.coef_se, (Mse1, Mse2, Mse3, Mse4)):
        for m, c in coef_s_:
            u += c * se(m, q, η) * Mse_(m, q, ξ, p=p)

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

    return Field(field.k, field.q, field.coef_ce, [[], [], [], []])


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

    return Field(field.k, field.q, [[], [], [], []], field.coef_se)


def expansion_trace(field):
    """
    u0, u1 = expansion_trace(field)

    Return the fist two terms of the asymptotic expansion of the trace along the
    ellipse.

    Parameters
    ----------
    field : Field
        Field object from the dataclass

    Returns
    -------
    u0 : Function
        θ ↦ uⁱⁿᶜ(0, sin(θ))
    u1 : Function
        θ ↦ cos(θ) ∂ₓ uⁱⁿᶜ(0, sin(θ))
    """

    return (
        lambda θ: eval_field(field, 0, np.pi / 2 - θ),
        lambda θ: eval_field(field, 0, np.pi / 2 - θ, p=1),
    )


def eval_far_field(field, θ):
    """
    u = eval_far_field(field, θ)

    Return the far field pattern from a field.

    Parameters
    ----------
    field : Field
        Field object from the dataclass
    θ : array_like
        Field object from the dataclass

    Returns
    -------
    u : array_like
        θ ↦ far field pattern
    """

    q = field.q
    η = np.pi / 2 - θ

    u = np.zeros_like(η, dtype=complex)

    for m, c in field.coef_ce[2]:
        u += (c * (-1j) ** m) * ce(m, q, η)

    for m, c in field.coef_se[2]:
        u += (c * (-1j) ** m) * se(m, q, η)

    return u
