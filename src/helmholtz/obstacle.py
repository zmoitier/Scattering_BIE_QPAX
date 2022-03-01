"""Class for problem

    Author: Zoïs Moitier
        Karlsruhe Institute of Technology, Germany
"""

from dataclasses import dataclass
from enum import Enum, auto
from math import atanh, sqrt
from typing import Tuple

from .ellipse_parametrization import EllipseParametrization


class BoundaryType(Enum):
    """Enum for boundary type."""

    DIRICHLET = auto()
    NEUMANN = auto()
    PENETRABLE = auto()


@dataclass(frozen=True)
class Obstacle:
    """Plobelm class"""

    c: float
    boundary_ξ: float
    param: EllipseParametrization
    boundary_type: BoundaryType
    σ: float
    ρ: float


def _get_cξ(a: float, b: float) -> Tuple[float, float]:
    """
    Compute the focal points and the ``radial'' cordinate in Elliptic coordinate of the
    ellipse define by the semi-major axis a and semi-minor axis b and center at (0, 0).
    """

    return (sqrt(a * a - b * b), atanh(b / a))


def create_obstacle(
    *, ɛ: float, bdy_type: BoundaryType, σ: float = 0, ρ: float = 0
) -> Obstacle:
    """
    obs = create_obstacle(ε, bdy_type, σ, ρ)

    Create the obstacle object.

    Parameters
    ----------
    ε : float
        semi-minor axis of the ellipse boundary
    bdy_type: BoundaryType
        Boundary type
    σ: float ≠ 0
        value σ
    ρ: float ≠ 0
        value ρ

    Returns
    -------
    obs : Obstacle
        Obstacle object
    """

    eparam = EllipseParametrization(ɛ)
    c, ξ0 = _get_cξ(1, ɛ)

    if bdy_type == BoundaryType.PENETRABLE:
        if σ == 0:
            raise ValueError(f"The value σ = {σ} must be non zero.")

        if ρ == 0:
            raise ValueError(f"The value ρ = {ρ} must be non zero.")

    return Obstacle(c=c, boundary_ξ=ξ0, param=eparam, boundary_type=bdy_type, σ=σ, ρ=ρ)
