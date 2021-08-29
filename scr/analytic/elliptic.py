""" Utilities for elliptical coordinates

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
from math import atanh, sqrt

from numpy import arccosh, imag, real


def to_elliptic(c, x, y):
    """
    ξ, η = to_elliptic(c, x, y)

    Transform Cartesian coordinates (x, y) into Elliptic coordinates (ξ, η).
    The Elliptic coordinates depand of a positive real number c and, for (x, y) ∈ ℝ²,
    they are define via the relation
        x = c sh(ξ) sin(η)
        y = c ch(ξ) cos(η)
    where (ξ, η) ∈ ℝ₊ × ℝ/2πℤ.

    Parameters
    ----------
    c : float
        (0, ±c) are the focal points of the ellipses and hyperbolas
    x : array_like
        first Cartesian coordinate
    y : array_like
        second Cartesian coordinate

    Returns
    -------
    ξ : array_like
        ``radial'' coordinate
    η : array_like
        ``angular'' coordinate
    """

    w = arccosh((y + 1j * x) / abs(c))
    return (real(w), imag(w))


def ellipse_param(a, b):
    """
    c, ξ0 = ellipse_param(a, b)

    Compute the focal points and the ``radial'' cordinate in Elliptic coordinate of the
    ellipse define by the semi-major axis a and semi-minor axis b and center at (0, 0).

    Parameters
    ----------
    a : float
        semi-major axis
    b : float
        semi-minor axis

    Returns
    -------
    c : float
        (0, ±c) are the focal points of the ellipse
    ξ0 : float
        ``radial'' coordinate of the ellipse in Elliptic coordinate
    """

    if (a < 0) or (b < 0):
        raise ValueError(
            f"The semi-major axis value {a} and the semi-minor axis value {b} should "
            "be positives."
        )

    if a <= b:
        raise ValueError(
            f"The semi-major axis value {a} should be bigger than the semi-minor axis "
            f"value {b}."
        )

    return (sqrt(a * a - b * b), atanh(b / a))
