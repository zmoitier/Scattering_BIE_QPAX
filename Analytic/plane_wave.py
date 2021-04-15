""" Plane wave

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany

    Last modified: 15/04/2021
"""

from math import cos, sin

import numpy as np

from .helmholtz_exterior import create_field
from .mathieu import ce, se


def plane_wave(α, k, x, y):
    return np.exp(1j * k * (cos(α) * x + sin(α) * y))


def plane_wave_trace(α, k, a, b, η, p):
    """
    u = plane_wave_trace(α, k, a, b, η, p)

    Compute the Dirichlet or Neumann trace of a plane wave with direction
    (cos(α), sin(α)) and wavenumber k on the ellipse of semi-major axis a and
    semi-minor axis b.

    Parameters
    ----------
    α : float
        (cos(α), sin(α)) direction of the plane wave
    k : float > 0
        wavenumber of the plane wave
    a : float > 0
        semi-major axis of the ellipse
    b : float > 0
        semi-minor axis of the ellipse
    η : array_like
        angular Elliptic coordinate of the ellipse
    p : 0 or 1
        0 for the Dirichlet trace and 1 for the Neumann trace

    Returns
    -------
    u : array_like
        trace
    """

    if p == 0:
        return np.exp(1j * k * (cos(α) * a * np.cos(η) + sin(α) * b * np.sin(η)))

    cα, sα = cos(α), sin(α)
    cη, sη = np.cos(η), np.sin(η)
    return (
        1j
        * k
        * (b * cη * cα + a * sη * sα)
        * np.exp(1j * k * (cα * a * cη + sα * b * sη))
        / np.hypot(a * sη, b * cη)
    )


def field_plane_wave(α, k, c, M):
    """
    field = field_plane_wave(α, k, c, M)

    Compute the Field corresponding to a plane wave with direction (cos(α), sin(α)) and
    wavenumber k on the ellipse of focal points (0, ±c).

    Parameters
    ----------
    α : float
        (cos(α), sin(α)) direction of the plane wave
    k : float > 0
        wavenumber of the plane wave
    c : float > 0
        (0, ±c) focal ponits of the ellipse
    M : int
        number of modes in the expansion

    Returns
    -------
    field : Field
        Field object corresponding to the approximation of the plane wave
    """

    q = c * c * k * k / 4
    απ2 = np.pi / 2 - α

    coef_c1 = []
    for m in range(M + 1):
        c_ce = 2 * 1j ** m * ce(m, q, απ2)
        if abs(c_ce) > 1e-8:
            coef_c1.append((m, c_ce))

    coef_s1 = []
    for m in range(1, M + 1):
        c_se = 2 * 1j ** m * se(m, q, απ2)
        if abs(c_se) > 1e-8:
            coef_s1.append((m, c_se))

    return create_field(c, k, [coef_c1, [], [], []], [coef_s1, [], [], []])
