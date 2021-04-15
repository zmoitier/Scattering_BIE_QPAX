""" Utils for plots

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany

    Last modified: 15/04/2021
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
    ε_vec, ε_str = logspace_epsilon(start, stop, num)

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
    ε_vec : vector
        equi-space float in log scale
    ε_str : vector
        string representation of ε_vec
    """

    ε_str = [f"{x:.1e}" for x in np.geomspace(start, stop, num=num)[::-1]]
    ε_vec = [float(s) for s in ε_str]
    return (ε_vec, ε_str)


def linspace_alpha(start, stop, num):
    """
    α_vec, α_str = linspace_alpha(start, stop, num)

    Return the `num' equi-space float between `start'*π and `stop'*π.

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
    α_vec : vector
        equi-space float
    α_str : vector
        string representation of α_vec
    """

    α_ = [f"{x:.2f}" for x in np.linspace(start, stop, num=num)]
    α_vec = [float(s) * np.pi for s in α_]
    α_str = [fr"${s}\pi$" for s in α_]
    return (α_vec, α_str)


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
