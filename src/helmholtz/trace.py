""" Get traces

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from typing import Callable, Tuple

from numpy import array, pi
from numpy.typing import NDArray

from .analytic.eval_field import _eval_parityfield
from .analytic.field import IncidentField, ScatterField
from .obstacle import Obstacle

Trace = Tuple[Callable, Callable]
TraceExpansion = Tuple[Callable, Callable, Callable]


def get_incident_trace(obs: Obstacle, in_field: IncidentField) -> Trace:
    """Incident field trace."""

    ξ = array(obs.boundary_ξ)

    def trace0(θ: NDArray):
        """Trace."""
        return _eval_parityfield(in_field, ξ, pi / 2 - θ, 0, 0)

    def trace1(θ: NDArray):
        """Scaled normal trace."""
        return _eval_parityfield(in_field, ξ, pi / 2 - θ, 1, 0)

    return (trace0, trace1)


def get_total_trace(
    obs: Obstacle, in_field: IncidentField, sc_field: ScatterField
) -> Trace:
    """Total field trace."""

    ξ = array(obs.boundary_ξ)

    def trace0(θ: NDArray):
        """Trace."""
        η = pi / 2 - θ
        return _eval_parityfield(in_field, ξ, η, 0, 0) + _eval_parityfield(
            sc_field.out_field, ξ, η, 0, 0
        )

    def trace1(θ: NDArray):
        """Scaled normal trace."""
        η = pi / 2 - θ
        return _eval_parityfield(in_field, ξ, η, 1, 0) + _eval_parityfield(
            sc_field.out_field, ξ, η, 1, 0
        )

    return (trace0, trace1)


def get_incident_trace_expansion(in_field: IncidentField) -> TraceExpansion:
    """Incident field trace."""

    def trace0(θ: NDArray):
        """Trace."""
        return _eval_parityfield(in_field, array([0]), pi / 2 - θ, 0, 0)

    def trace1(θ: NDArray):
        """Scaled normal trace."""
        return _eval_parityfield(in_field, array([0]), pi / 2 - θ, 1, 0)

    def trace2(θ: NDArray):
        """Scaled normal trace."""
        return _eval_parityfield(in_field, array([0]), pi / 2 - θ, 2, 0)

    return (trace0, trace1, trace2)
