""" Plane wave

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from math import cos, sin

from numpy import cos as np_cos
from numpy import exp, hypot, pi
from numpy import sin as np_sin

from .helmholtz_exterior import create_field
from .mathieu import ce, se


def plane_wave(α, k, x, y):
    """
    u = plane_wave(α, k, x, y)

    Compute the  plane wave with direction (cos(α), sin(α)) and wavenumber k.

    Parameters
    ----------
    α : float
        (cos(α), sin(α)) direction of the plane wave
    k : float > 0
        wavenumber of the plane wave
    x : array_like
        x coordinate
    y : array_like
        y coordinate

    Returns
    -------
    u : array_like
        values of the plane wave
    """
    return exp(1j * k * (cos(α) * x + sin(α) * y))


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
        return exp(1j * k * (cos(α) * a * np_cos(η) + sin(α) * b * np_sin(η)))

    cα, sα = cos(α), sin(α)
    cη, sη = np_cos(η), np_sin(η)
    return (
        1j
        * k
        * (b * cη * cα + a * sη * sα)
        * exp(1j * k * (cα * a * cη + sα * b * sη))
        / hypot(a * sη, b * cη)
    )


def field_plane_wave(ε, c, α, k, M):
    """
    field = field_plane_wave(ε, c, α, k, M)

    Compute the Field corresponding to a plane wave with direction (cos(α), sin(α)) and
    wavenumber k on the ellipse of focal points (0, ±c).

    Parameters
    ----------
    ε : float > 0
        parameter for elliptical coordinates
    c : float > 0
        (0, ±c) focal ponits of the ellipse
    α : float
        (cos(α), sin(α)) direction of the plane wave
    k : float > 0
        wavenumber of the plane wave
    M : int
        number of modes in the expansion

    Returns
    -------
    field : Field
        Field object corresponding to the approximation of the plane wave
    """

    q = c * c * k * k / 4
    απ2 = pi / 2 - α

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

    return create_field(ε, c, k, [coef_c1, [], [], []], [coef_s1, [], [], []])
