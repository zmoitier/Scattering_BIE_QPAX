"""Solution class

    Author: Zoïs Moitier
        Karlsruhe Institute of Technology, Germany
"""

from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass(frozen=True)
class BieSolution:
    """Bie solution class"""

    trace: NDArray
    scaled_normal_trace: NDArray
    grid: NDArray
    N: int
