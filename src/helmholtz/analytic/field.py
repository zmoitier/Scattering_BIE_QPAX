"""Field

    Author: Zoïs Moitier
        Karlsruhe Institute of Technology, Germany
"""

from dataclasses import dataclass
from typing import Optional

from numpy.typing import NDArray

from .mathieu_api import AngularMathieu, RadialMathieu


@dataclass(frozen=True)
class MonoField:
    """MonoField"""

    q: float

    angfct: AngularMathieu
    radfct: RadialMathieu

    index: NDArray
    coeff: NDArray


@dataclass(frozen=True)
class ParityField:
    """ParityField"""

    even: MonoField
    odd: MonoField


@dataclass(frozen=True)
class Field:
    """Field"""

    boundary_ξ: float
    inn_field: Optional[ParityField]
    out_field: ParityField


IncidentField = ParityField
ScatterField = Field
