""" Boundary class

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
from abc import ABC, abstractmethod
from typing import Tuple

from numpy import ndarray


class Boundary(ABC):
    """Dataclass that describe quantities of a simple close curve."""

    @abstractmethod
    def gamma(self, t: ndarray) -> Tuple[ndarray, ndarray]:
        """
        gamma(t) → (γ₁(t), γ₂(t))

        Return the coordinates (γ₁(t), γ₂(t)) ∈ ℝ² of the curve γ.
        """

    @abstractmethod
    def gamma_prime(self, t: ndarray) -> Tuple[ndarray, ndarray]:
        """
        gamma_prime(t) → (γ₁´(t), γ₂´(t))

        Return the coordinates (γ₁´(t), γ₂´(t)) ∈ ℝ² of the curve γ.
        """

    @abstractmethod
    def jacobian(self, t: ndarray) -> ndarray:
        """
        jacobian(t) → |γ′(t)|

        Return the jacobian |γ′(t)| of the curve γ.
        """

    @abstractmethod
    def normal_ext(self, t: ndarray) -> Tuple[ndarray, ndarray]:
        """
        normal_ext(t) → νₑₓₜ(t)

        Return the unitary exterior normal νₑₓₜ(t) ∈ 𝕊¹ of the curve γ.
        """

    @abstractmethod
    def curvature(self, t: ndarray) -> ndarray:
        """
        curvature(t) → κ(t)

        Return the curvature κ(t) of the curve γ.
        """
