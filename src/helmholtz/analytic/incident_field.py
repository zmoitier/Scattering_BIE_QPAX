""" Plane wave

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from numpy import arange, array, pi
from numpy.typing import NDArray

from ..obstacle import Obstacle
from .field import IncidentField, MonoField
from .mathieu_api import (
    AngularMathieu,
    Mce1,
    Mce2,
    Mce3,
    Mce4,
    Mse1,
    Mse2,
    Mse3,
    Mse4,
    RadialMathieu,
    ce,
    se,
)


def comp_monofield(
    q: float, α: float, am: AngularMathieu, rm: RadialMathieu, m_vec: NDArray
) -> MonoField:
    """Compute a monofield."""
    c_vec = 2 * 1j**m_vec * am(m_vec, q, α)

    index = []
    coeff = []
    for m, c in zip(m_vec, c_vec):
        if abs(c) > 1e-8:
            index.append(m)
            coeff.append(c)

    return MonoField(q=q, angfct=am, radfct=rm, index=array(index), coeff=array(coeff))


def plane_wave_field(obs: Obstacle, α: float, k: float, M: int) -> IncidentField:
    """
    Compute the Field corresponding to a plane wave with direction (cos(α), sin(α)) and
    wavenumber k on the ellipse of focal points (0, ±c).
    """

    q = obs.c * obs.c * k * k / 4
    απ2 = pi / 2 - α

    f_even = comp_monofield(q, απ2, ce, Mce1, arange(M + 1))
    f_odd = comp_monofield(q, απ2, se, Mse1, arange(1, M + 1))

    return IncidentField(even=f_even, odd=f_odd)


def _choose_Mc(i: int):
    """Choose Mce⁽ⁱ⁾."""

    if i == 1:
        return Mce1
    if i == 2:
        return Mce2
    if i == 3:
        return Mce3
    if i == 4:
        return Mce4

    raise ValueError(f"i = {i} must be 1, 2, 3, or 4.")


def _choose_Ms(i: int):
    """Choose Mse⁽ⁱ⁾."""

    if i == 1:
        return Mse1
    if i == 2:
        return Mse2
    if i == 3:
        return Mse3
    if i == 4:
        return Mse4

    raise ValueError(f"i = {i} must be 1, 2, 3, or 4.")


def create_one_field(
    obs: Obstacle, k: float, parity: str, i: int, m: int
) -> IncidentField:
    """Create a field with one basis function."""

    q = obs.c * obs.c * k * k / 4
    if parity.startswith("e"):
        angfct = ce
        radfct = _choose_Mc(i)

        even = MonoField(
            q, angfct=angfct, radfct=radfct, index=array([m]), coeff=array([1])
        )
        odd = MonoField(
            q, angfct=angfct, radfct=radfct, index=array([]), coeff=array([])
        )

    elif parity.startswith("o"):
        angfct = se
        radfct = _choose_Ms(i)

        even = MonoField(
            q, angfct=angfct, radfct=radfct, index=array([]), coeff=array([])
        )
        odd = MonoField(
            q, angfct=angfct, radfct=radfct, index=array([m]), coeff=array([1])
        )

    else:
        raise ValueError(f"parity = '{parity}' must be 'even' or 'odd'")

    return IncidentField(even=even, odd=odd)
