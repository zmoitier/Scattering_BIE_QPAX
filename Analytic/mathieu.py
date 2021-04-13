""" Mathieu functions

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany

    Last modified: 16/02/2021
"""
from numpy import degrees  # (╬ಠ益ಠ)
from scipy.special import (
    mathieu_cem,
    mathieu_modcem1,
    mathieu_modcem2,
    mathieu_modsem1,
    mathieu_modsem2,
    mathieu_sem,
)


def ce(m, q, η):
    """
    v = ce(m, q, η)

    Compute the value of the even angular Mathieu function ceₘ(q, η).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    η : array like
        angular coordinate in the Elliptic coordinates

    Returns
    -------
    v : array like
        value of ceₘ(q, η)
    """

    return mathieu_cem(m, q, degrees(η))[0]


def se(m, q, η):
    """
    v = se(m, q, η)

    Compute the value of the odd angular Mathieu function seₘ(q, η).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    η : array like
        ``angular'' coordinate in the Elliptic coordinates

    Returns
    -------
    v : array like
        value of seₘ(q, η)
    """

    return mathieu_sem(m, q, degrees(η))[0]


def Mce1(m, q, ξ, p=0):
    """
    v = Mce1(m, q, ξ, p=0)

    Compute the value of the even Radial Mathieu function of the first kind
    Mce⁽¹⁾ₘ(q, ξ).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    ξ : array like
        ``radial'' coordinate in the Elliptic coordinates
    p : 0 or 1 (default 0)
        0 for the value of the function, 1 for its derivative

    Returns
    -------
    v : array like
        value of Mce⁽¹⁾ₘ(q, ξ) or Mce⁽¹⁾ₘ′(q, ξ)
    """

    return mathieu_modcem1(m, q, ξ)[p]


def Mse1(m, q, ξ, p=0):
    """
    v = Mse1(m, q, ξ, p=0)

    Compute the value of the odd Radial Mathieu function of the first kind
    Mse⁽¹⁾ₘ(q, ξ).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    ξ : array like
        radial coordinate in the Elliptic coordinates
    p : 0 or 1 (default 0)
        0 for the value of the function, 1 for its derivative

    Returns
    -------
    v : array like
        value of Mse⁽¹⁾ₘ(q, ξ) or Mse⁽¹⁾ₘ′(q, ξ)
    """

    return mathieu_modsem1(m, q, ξ)[p]


def Mce2(m, q, ξ, p=0):
    """
    v = Mce2(m, q, ξ, p=0)

    Compute the value of the even Radial Mathieu function of the second kind
    Mce⁽²⁾ₘ(q, ξ).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    ξ : array like
        ``radial'' coordinate in the Elliptic coordinates
    p : 0 or 1 (default 0)
        0 for the value of the function, 1 for its derivative

    Returns
    -------
    v : array like
        value of Mce⁽²⁾ₘ(q, ξ) or Mce⁽²⁾ₘ′(q, ξ)
    """

    return mathieu_modcem2(m, q, ξ)[p]


def Mse2(m, q, ξ, p=0):
    """
    v = Mse2(m, q, ξ, p=0)

    Compute the value of the even Radial Mathieu function of the second kind
    Mse⁽²⁾ₘ(q, ξ).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    ξ : array like
        ``radial'' coordinate in the Elliptic coordinates
    p : 0 or 1 (default 0)
        0 for the value of the function, 1 for its derivative

    Returns
    -------
    v : array like
        value of Mse⁽²⁾ₘ(q, ξ) or Mse⁽²⁾ₘ′(q, ξ)
    """

    return mathieu_modsem2(m, q, ξ)[p]


def Mce3(m, q, ξ, p=0):
    """
    v = Mce3(m, q, ξ, p=0)

    Compute the value of the even Radial Mathieu function of the third kind
    Mce⁽³⁾ₘ(q, ξ).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    ξ : array like
        ``radial'' coordinate in the Elliptic coordinates
    p : 0 or 1 (default 0)
        0 for the value of the function, 1 for its derivative

    Returns
    -------
    v : array like
        value of Mce⁽³⁾ₘ(q, ξ) or Mce⁽³⁾ₘ′(q, ξ)
    """

    return mathieu_modcem1(m, q, ξ)[p] + 1j * mathieu_modcem2(m, q, ξ)[p]


def Mse3(m, q, ξ, p=0):
    """
    v = Mse3(m, q, ξ, p=0)

    Compute the value of the even Radial Mathieu function of the third kind
    Mse⁽³⁾ₘ(q, ξ).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    ξ : array like
        ``radial'' coordinate in the Elliptic coordinates
    p : 0 or 1 (default 0)
        0 for the value of the function, 1 for its derivative

    Returns
    -------
    v : array like
        value of Mse⁽³⁾ₘ(q, ξ) or Mse⁽³⁾ₘ′(q, ξ)
    """

    return mathieu_modsem1(m, q, ξ)[p] + 1j * mathieu_modsem2(m, q, ξ)[p]


def Mce4(m, q, ξ, p=0):
    """
    v = Mce4(m, q, ξ, p=0)

    Compute the value of the even Radial Mathieu function of the fourth kind
    Mce⁽⁴⁾ₘ(q, ξ).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    ξ : array like
        ``radial'' coordinate in the Elliptic coordinates
    p : 0 or 1 (default 0)
        0 for the value of the function, 1 for its derivative

    Returns
    -------
    v : array like
        value of Mce⁽⁴⁾ₘ(q, ξ) or Mce⁽⁴⁾ₘ′(q, ξ)
    """

    return mathieu_modcem1(m, q, ξ)[p] - 1j * mathieu_modcem2(m, q, ξ)[p]


def Mse4(m, q, ξ, p=0):
    """
    v = Mse4(m, q, ξ, p=0)

    Compute the value of the even Radial Mathieu function of the fourth kind
    Mse⁽⁴⁾ₘ(q, ξ).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    ξ : array like
        ``radial'' coordinate in the Elliptic coordinates
    p : 0 or 1 (default 0)
        0 for the value of the function, 1 for its derivative

    Returns
    -------
    v : array like
        value of Mse⁽⁴⁾ₘ(q, ξ) or Mse⁽⁴⁾ₘ′(q, ξ)
    """

    return mathieu_modsem1(m, q, ξ)[p] - 1j * mathieu_modsem2(m, q, ξ)[p]
