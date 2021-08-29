""" Create boundary for ellipse

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
from dataclasses import dataclass
from typing import Tuple

from numpy import cos, hypot, ndarray, sin

from .boundary import Boundary


@dataclass(frozen=True)
class Ellipse(Boundary):
    """
    Class for an ellipse of semi-major axis 1 and semi-minor axis ε.

    Attributes
    ----------
    ε : float
        semi-minor axis of the ellipse
    """

    ε: float

    def gamma(self, t: ndarray) -> Tuple[ndarray, ndarray]:
        """
        gamma(t) → (γ₁(t), γ₂(t))

        Return the coordinates (γ₁(t), γ₂(t)) ∈ ℝ² of the curve γ.
        """
        return (self.ε * cos(t), sin(t))

    def gamma_prime(self, t: ndarray) -> Tuple[ndarray, ndarray]:
        """
        gamma_prime(t) → (γ₁´(t), γ₂´(t))

        Return the coordinates (γ₁´(t), γ₂´(t)) ∈ ℝ² of the curve γ.
        """
        return (-self.ε * sin(t), cos(t))

    def jacobian(self, t: ndarray) -> ndarray:
        """
        jacobian(t) → |γ′(t)|

        Return the jacobian |γ′(t)| of the curve γ.
        """
        return hypot(cos(t), self.ε * sin(t))

    def normal_ext(self, t: ndarray) -> Tuple[ndarray, ndarray]:
        """
        normal_ext(t) → νₑₓₜ(t)

        Return the unitary exterior normal νₑₓₜ(t) ∈ 𝕊¹ of the curve γ.
        """
        jac = hypot(cos(t), self.ε * sin(t))
        return (cos(t) / jac, self.ε * sin(t) / jac)

    def curvature(self, t: ndarray) -> ndarray:
        """
        curvature(t) → κ(t)

        Return the curvature κ(t) of the curve γ.
        """
        return self.ε / hypot(cos(t), self.ε * sin(t)) ** 3
