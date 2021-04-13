""" Compute analytic solution for laplace

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany

    Last modified: 16/02/2021
"""
from numpy import cos, sin
from sympy import lambdify, symbols


def laplace_sol_eve(m, ε):
    """
    scr, sol = laplace_sol_eve(m, ε)

    Return the even source term and solution to the Laplace problem.

    Parameters
    ----------
    m : int
        number of oscillations
    ε : float
        semi-minor axis of the ellipse

    Returns
    -------
    scr : Function
        cos(m θ) or sin(m θ)
    sol : Function
        a_ε cos(m θ) or b_ε sin(m θ)
    """

    if (not float(m).is_integer()) or (m < 0):
        raise ValueError("m should be a non-negative interger")

    e = symbols("e", real=True, positive=True)
    ρ = (e - 1) / (e + 1)

    if m % 2:
        c = (-2 / (1 - ρ ** m)).factor()
        cossin = sin
    else:
        c = (-2 / (1 + ρ ** m)).factor()
        cossin = cos

    fct = lambdify(e, c)
    return (lambda θ: cossin(m * θ), lambda θ: fct(ε) * cossin(m * θ))


def laplace_sol_odd(m, ε):
    """
    scr, sol = laplace_sol_odd(m, ε)

    Return the odd source term and solution to the Laplace problem.

    Parameters
    ----------
    m : int
        number of oscillations
    ε : float
        semi-minor axis of the ellipse

    Returns
    -------
    scr : Function
        cos(m θ) or sin(m θ)
    sol : Function
        a_ε cos(m θ) or b_ε sin(m θ)
    """

    if (not float(m).is_integer()) or (m < 1):
        raise ValueError("m should be a positive interger")

    e = symbols("e", real=True, positive=True)
    ρ = (e - 1) / (e + 1)

    if m % 2:
        c = (-2 / (1 + ρ ** m)).factor()
        cossin = cos
    else:
        c = (-2 / (1 - ρ ** m)).factor()
        cossin = sin

    fct = lambdify(e, c)
    return (lambda θ: cossin(m * θ), lambda θ: fct(ε) * cossin(m * θ))
