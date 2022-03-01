"""Geometry class

    Author: Zoïs Moitier
        Karlsruhe Institute of Technology, Germany
"""

from dataclasses import dataclass
from typing import Tuple

from numpy import cos, hypot, sin
from numpy.typing import NDArray


@dataclass(frozen=True)
class EllipseParametrization:
    """
    Class for an ellipse of semi-major axis 1 and semi-minor axis ɛ.

    Attributes
    ----------
    ɛ : float
        semi-minor axis of the ellipse
    """

    ɛ: float

    def gamma(self, t: NDArray) -> Tuple[NDArray, NDArray]:
        """
        gamma(t) → (γ₁(t), γ₂(t))

        Return the coordinates (γ₁(t), γ₂(t)) ∈ ℝ² of the curve γ.
        """
        return (self.ɛ * cos(t), sin(t))

    def gamma_prime(self, t: NDArray) -> Tuple[NDArray, NDArray]:
        """
        gamma_prime(t) → (γ₁´(t), γ₂´(t))

        Return the coordinates (γ₁´(t), γ₂´(t)) ∈ ℝ² of the curve γ.
        """
        return (-self.ɛ * sin(t), cos(t))

    def jacobian(self, t: NDArray) -> NDArray:
        """
        jacobian(t) → |γ′(t)|

        Return the jacobian |γ′(t)| of the curve γ.
        """
        return hypot(cos(t), self.ɛ * sin(t))

    def normal_ext(self, t: NDArray) -> Tuple[NDArray, NDArray]:
        """
        normal_ext(t) → νₑₓₜ(t)

        Return the unitary exterior normal νₑₓₜ(t) ∈ 𝕊¹ of the curve γ.
        """
        ct, ɛst = cos(t), self.ɛ * sin(t)
        jac = hypot(ct, ɛst)
        return (ct / jac, ɛst / jac)

    def curvature(self, t: NDArray) -> NDArray:
        """
        curvature(t) → κ(t)

        Return the curvature κ(t) of the curve γ.
        """
        return self.ɛ / hypot(cos(t), self.ɛ * sin(t)) ** 3

    def __post_init__(self):
        if self.ɛ < 0:
            raise ValueError(f"The semi-minor axis value {self.ɛ} should be positives.")

        if self.ɛ >= 1:
            raise ValueError(
                f"The semi-minor axis {self.ɛ} must be smaller or equal 1."
            )
