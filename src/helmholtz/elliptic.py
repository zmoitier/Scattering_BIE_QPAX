""" Utilities for elliptical coordinates

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from typing import Tuple

from numpy import arccosh, imag, real
from numpy.typing import NDArray


def to_elliptic(c: float, x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
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
