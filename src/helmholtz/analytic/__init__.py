"""Analytically compute scattering field by elliptic obstacle

    Author: Zoïs Moitier
        Karlsruhe Institute of Technology, Germany
"""

# from .check import check_dirichlet, check_neumann, check_penetrable
from .eval_field import eval_incident_field, eval_scatter_field, eval_total_field
from .field import IncidentField, ScatterField
from .incident_field import create_one_field, plane_wave_field
from .solver import get_scatter_field
