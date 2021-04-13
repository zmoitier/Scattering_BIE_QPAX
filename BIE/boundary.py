""" Boundary class

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany

    Last modified: 17/02/2021
"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Boundary:
    """
    Dataclass that describe quantities of a simple close curve.

    Attributes
    ----------
    gamma : Callable
        the function θ ↦ γ(θ) ∈ ℝ² the coordinates of the curve γ
    jacobian : Callable
        the function θ ↦ |γ′(θ)| the jacobian of the curve γ
    normal_ext : Callable
        the function θ ↦ n(θ) ∈ 𝕊¹ the unitary exterior normal of the curve γ
    curvature : Callable
        the function θ ↦ κ(θ) the curvature of the curve γ
    """

    gamma: Callable
    jacobian: Callable
    normal_ext: Callable
    curvature: Callable
