""" Utils for plots

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""

from numpy import array, geomspace, rint
from numpy.typing import NDArray


def logspace_quadrature(start: float, stop: float, num: int) -> NDArray:
    """
    vec = logspace_quadrature(start, stop, num)

    Return the `num' equi-space interger in log scale between `start' and `stop' with
    the constraint that they be divisible by 4.

    Parameters
    ----------
    start : int
        starting integer
    stop : int
        stoping integer
    num : int
        number of intergers

    Returns
    -------
    vec : vector
        equi-space interger in log scale
    """

    return 4 * rint(geomspace(start / 4, stop / 4, num=num)).astype(int)


def logspace_epsilon(start: float, stop: float, num: int) -> NDArray:
    """
    vec = logspace_epsilon(start, stop, num)

    Return the `num' equi-space float in log scale between `start' and `stop'.

    Parameters
    ----------
    start : int
        starting float
    stop : int
        stoping float
    num : int
        number of float

    Returns
    -------
    vec : vector
        equi-space interger in log scale
    """

    return array([float(f"{x:.1e}") for x in geomspace(start, stop, num=num)])
