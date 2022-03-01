"""Source code for scattering by ellipse

    Author: Zoïs Moitier
        Karlsruhe Institute of Technology, Germany
"""

from . import analytic, bie
from .ellipse_parametrization import EllipseParametrization
from .elliptic import to_elliptic
from .obstacle import BoundaryType, Obstacle, create_obstacle
from .trace import get_incident_trace, get_incident_trace_expansion, get_total_trace
