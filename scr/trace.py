""" Trace for ellipses

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
from numpy import cos, cosh, pi, sqrt

from .analytic import ellipse_param, eval_field


def trace(field):
    """
    u0 = trace_dir(field, ξ0)

    Return the Dirichlet trace of the field u along the ellipse.

    Parameters
    ----------
    field : Field
        Field class

    Returns
    -------
    u0 : Function
        t ↦ u(ε cos(t), sin(t))
    """
    _, ξ0 = ellipse_param(1, field.ε)
    return lambda t: eval_field(field, ξ0, pi / 2 - t)


def normal_trace(field):
    """
    u1 = normal_trace(field, ellipse)

    Return the normal trace of the field u along the ellipse.

    Parameters
    ----------
    field : Field
        Field class

    Returns
    -------
    u_dir : Function
        t ↦ ∂ₙu(ε cos(t), sin(t))
    """
    c, ξ0 = ellipse_param(1, field.ε)
    fact = sqrt(2) / c
    return lambda t: (
        fact
        * eval_field(field, ξ0, pi / 2 - t, d_ξ=1)
        / sqrt(cosh(2 * ξ0) - cos(pi - 2 * t))
    )


def scaled_normal_trace(field):
    """
    u1 = scaled_normal_trace(field, ellipse)

    Return the scaled normal trace of the field u along the ellipse.

    Parameters
    ----------
    field : Field
        Field class

    Returns
    -------
    u_dir : Function
        t ↦ |γ´(t)| ∂ₙu(ε cos(t), sin(t))
    """
    _, ξ0 = ellipse_param(1, field.ε)
    return lambda t: eval_field(field, ξ0, pi / 2 - t, d_ξ=1)
