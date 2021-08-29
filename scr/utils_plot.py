""" Utils for plots

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
import numpy as np


def logspace_quadrature(start, stop, num):
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

    return 4 * np.rint(np.geomspace(start / 4, stop / 4, num=num)).astype(int)


def logspace_epsilon(start, stop, num):
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

    return np.array([float(f"{x:.1e}") for x in np.geomspace(start, stop, num=num)])


def add_elem(v):
    """
    vec = add_elem(v)

    Create a new vector with the size `len(v)+1' periodically.

    Parameters
    ----------
    v : vector

    Returns
    -------
    vec : vector
        [*v, v[-1]]
    """

    u = np.empty(np.size(v) + 1, dtype=v.dtype)
    u[:-1] = v
    u[-1] = v[0]
    return u
