""" Create boundary for ellipse

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany

    Last modified: 02/04/2021
"""
import numpy as np

from .boundary import Boundary


def ellipse(ε):
    """
    boundary = ellipse(ε)

    Return the Boundary object corresponding to the ellipse with semi-major
    axis 1 and semi-minor axis ε.

    Parameters
    ----------
    ε : float
        semi-minor axis of the ellipse

    Returns
    -------
    boundary : Boundary
    """

    def gamma(t):
        return (ε * np.cos(t), np.sin(t))

    def jacobian(t):
        return np.hypot(np.cos(t), ε * np.sin(t))

    def normal_ext(t):
        jac = np.hypot(np.cos(t), ε * np.sin(t))
        return (np.cos(t) / jac, ε * np.sin(t) / jac)

    def curvature(t):
        return ε / np.hypot(np.cos(t), ε * np.sin(t)) ** 3

    return Boundary(gamma, jacobian, normal_ext, curvature)
