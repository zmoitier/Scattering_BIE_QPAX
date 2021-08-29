""" Trace expansion for ellipses

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
from numpy import pi

from .analytic import eval_field


def trace_neumann_expansion(field):
    """
    u0, u1 = trace_neu_asy(field)

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
        lambda θ: eval_field(field, 0, pi / 2 - θ),
        lambda θ: eval_field(field, 0, pi / 2 - θ, d_ξ=1),
    )


def trace_dirichlet_expansion(field):
    """
    u0, u1 = trace_dir_asy(field)

    Return the fist two terms of the asymptotic expansion of the trace along the
    ellipse.

    Parameters
    ----------
    field : Field
        Field object from the dataclass

    Returns
    -------
    u0 : Function
        θ ↦ cos(θ) ∂ₓ uⁱⁿᶜ(0, sin(θ))
    u1 : Function
        θ ↦ cos(θ)² ∂ₓₓ uⁱⁿᶜ(0, sin(θ)) + sin(θ) ∂_y uⁱⁿᶜ(0, sin(θ))
    """

    return (
        lambda θ: eval_field(field, 0, pi / 2 - θ, d_ξ=1),
        lambda θ: eval_field(field, 0, pi / 2 - θ, d_ξ=2),
    )
