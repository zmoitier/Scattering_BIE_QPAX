""" Mathieu functions

    Author: Zoïs Moitier
            Karlsruhe Institute of Technology, Germany
"""
from numpy import degrees  # (╬ಠ益ಠ)
from numpy import cos, cosh
from scipy.special import (
    mathieu_a,
    mathieu_b,
    mathieu_cem,
    mathieu_modcem1,
    mathieu_modcem2,
    mathieu_modsem1,
    mathieu_modsem2,
    mathieu_sem,
)


def ce(m, q, η, *, p=0):
    """
    v = ce(m, q, η, *, p=0)

    Compute the value of the even angular Mathieu function ceₘ(q, η).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    η : array like
        angular coordinate in the Elliptic coordinates
    p : 0 or 1 or 2 (default 0)
        0 for the function,
        1 for the first derivative
        2 for the second derivative

    Returns
    -------
    v : array like
        value of ceₘ(q, η) or ceₘ´(q, η) or ceₘ´´(q, η)
    """

    if p == 0:
        return mathieu_cem(m, q, degrees(η))[0]

    if p == 1:
        return mathieu_cem(m, q, degrees(η))[1]

    if p == 2:
        return ((2 * q) * cos(2 * η) - mathieu_a(m, q)) * mathieu_cem(m, q, degrees(η))[
            0
        ]

    raise ValueError("The value p must be 0, 1, or 2.")


def se(m, q, η, *, p=0):
    """
    v = se(m, q, η, *, p=0)

    Compute the value of the odd angular Mathieu function seₘ(q, η).

    Parameters
    ----------
    m : array_like
        interger order of the Mathieu function
    q : array_like
        positive parameter in the Mathieu differential equation
    η : array like
        ``angular'' coordinate in the Elliptic coordinates
    p : 0 or 1 or 2 (default 0)
        0 for the function,
        1 for the first derivative
        2 for the second derivative

    Returns
    -------
    v : array like
        value of seₘ(q, η) or seₘ´(q, η) or seₘ´´(q, η)
    """

    if p == 0:
        return mathieu_sem(m, q, degrees(η))[0]

    if p == 1:
        return mathieu_sem(m, q, degrees(η))[1]

    if p == 2:
        return ((2 * q) * cos(2 * η) - mathieu_b(m, q)) * mathieu_sem(m, q, degrees(η))[
            0
        ]

    raise ValueError("The value p must be 0, 1, or 2.")


def Mce1(m, q, ξ, *, p=0):
    """
    v = Mce1(m, q, ξ, *, p=0)

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
    p : 0 or 1 or 2 (default 0)
        0 for the function,
        1 for the first derivative
        2 for the second derivative

    Returns
    -------
    v : array like
        value of Mce⁽¹⁾ₘ(q, ξ) or Mce⁽¹⁾ₘ′(q, ξ) or Mce⁽¹⁾ₘ′´(q, ξ)
    """

    if p == 0:
        return mathieu_modcem1(m, q, ξ)[0]

    if p == 1:
        return mathieu_modcem1(m, q, ξ)[1]

    if p == 2:
        return (mathieu_a(m, q) - (2 * q) * cosh(2 * ξ)) * mathieu_modcem1(m, q, ξ)[0]

    raise ValueError("The value p must be 0, 1, or 2.")


def Mse1(m, q, ξ, *, p=0):
    """
    v = Mse1(m, q, ξ, *, p=0)

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
    p : 0 or 1 or 2 (default 0)
        0 for the function,
        1 for the first derivative
        2 for the second derivative

    Returns
    -------
    v : array like
        value of Mse⁽¹⁾ₘ(q, ξ) or Mse⁽¹⁾ₘ′(q, ξ) or Mse⁽¹⁾ₘ′´(q, ξ)
    """

    if p == 0:
        return mathieu_modsem1(m, q, ξ)[0]

    if p == 1:
        return mathieu_modsem1(m, q, ξ)[1]

    if p == 2:
        return (mathieu_b(m, q) - (2 * q) * cosh(2 * ξ)) * mathieu_modsem1(m, q, ξ)[0]

    raise ValueError("The value p must be 0, 1, or 2.")


def Mce2(m, q, ξ, *, p=0):
    """
    v = Mce2(m, q, ξ, *, p=0)

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
    p : 0 or 1 or 2 (default 0)
        0 for the function,
        1 for the first derivative
        2 for the second derivative

    Returns
    -------
    v : array like
        value of Mce⁽²⁾ₘ(q, ξ) or Mce⁽²⁾ₘ′(q, ξ) or Mce⁽²⁾ₘ′´(q, ξ)
    """

    if p == 0:
        return mathieu_modcem2(m, q, ξ)[0]

    if p == 1:
        return mathieu_modcem2(m, q, ξ)[1]

    if p == 2:
        return (mathieu_a(m, q) - (2 * q) * cosh(2 * ξ)) * mathieu_modcem2(m, q, ξ)[0]

    raise ValueError("The value p must be 0, 1, or 2.")


def Mse2(m, q, ξ, *, p=0):
    """
    v = Mse2(m, q, ξ, *, p=0)

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
    p : 0 or 1 or 2 (default 0)
        0 for the function,
        1 for the first derivative
        2 for the second derivative

    Returns
    -------
    v : array like
        value of Mse⁽²⁾ₘ(q, ξ) or Mse⁽²⁾ₘ′(q, ξ) or Mse⁽²⁾ₘ′´(q, ξ)
    """

    if p == 0:
        return mathieu_modsem2(m, q, ξ)[0]

    if p == 1:
        return mathieu_modsem2(m, q, ξ)[1]

    if p == 2:
        return (mathieu_b(m, q) - (2 * q) * cosh(2 * ξ)) * mathieu_modsem2(m, q, ξ)[0]

    raise ValueError("The value p must be 0, 1, or 2.")


def Mce3(m, q, ξ, *, p=0):
    """
    v = Mce3(m, q, ξ, *, p=0)

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
    p : 0 or 1 or 2 (default 0)
        0 for the function,
        1 for the first derivative
        2 for the second derivative

    Returns
    -------
    v : array like
        value of Mce⁽³⁾ₘ(q, ξ) or Mce⁽³⁾ₘ′(q, ξ) or Mce⁽³⁾ₘ′´(q, ξ)
    """

    if p == 0:
        return mathieu_modcem1(m, q, ξ)[0] + 1j * mathieu_modcem2(m, q, ξ)[0]

    if p == 1:
        return mathieu_modcem1(m, q, ξ)[1] + 1j * mathieu_modcem2(m, q, ξ)[1]

    if p == 2:
        return (mathieu_a(m, q) - (2 * q) * cosh(2 * ξ)) * (
            mathieu_modcem1(m, q, ξ)[0] + 1j * mathieu_modcem2(m, q, ξ)[0]
        )

    raise ValueError("The value p must be 0, 1, or 2.")


def Mse3(m, q, ξ, *, p=0):
    """
    v = Mse3(m, q, ξ, *, p=0)

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
    p : 0 or 1 or 2 (default 0)
        0 for the function,
        1 for the first derivative
        2 for the second derivative

    Returns
    -------
    v : array like
        value of Mse⁽³⁾ₘ(q, ξ) or Mse⁽³⁾ₘ′(q, ξ) or Mse⁽³⁾ₘ′´(q, ξ)
    """

    if p == 0:
        return mathieu_modsem1(m, q, ξ)[0] + 1j * mathieu_modsem2(m, q, ξ)[0]

    if p == 1:
        return mathieu_modsem1(m, q, ξ)[1] + 1j * mathieu_modsem2(m, q, ξ)[1]

    if p == 2:
        return (mathieu_b(m, q) - (2 * q) * cosh(2 * ξ)) * (
            mathieu_modsem1(m, q, ξ)[0] + 1j * mathieu_modsem2(m, q, ξ)[0]
        )

    raise ValueError("The value p must be 0, 1, or 2.")


def Mce4(m, q, ξ, *, p=0):
    """
    v = Mce4(m, q, ξ, *, p=0)

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
    p : 0 or 1 or 2 (default 0)
        0 for the function,
        1 for the first derivative
        2 for the second derivative

    Returns
    -------
    v : array like
        value of Mce⁽⁴⁾ₘ(q, ξ) or Mce⁽⁴⁾ₘ′(q, ξ) or Mce⁽⁴⁾ₘ′´(q, ξ)
    """

    if p == 0:
        return mathieu_modcem1(m, q, ξ)[0] - 1j * mathieu_modcem2(m, q, ξ)[0]

    if p == 1:
        return mathieu_modcem1(m, q, ξ)[1] - 1j * mathieu_modcem2(m, q, ξ)[1]

    if p == 2:
        return (mathieu_a(m, q) - (2 * q) * cosh(2 * ξ)) * (
            mathieu_modcem1(m, q, ξ)[0] - 1j * mathieu_modcem2(m, q, ξ)[0]
        )

    raise ValueError("The value p must be 0, 1, or 2.")


def Mse4(m, q, ξ, *, p=0):
    """
    v = Mse4(m, q, ξ, *, p=0)

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
    p : 0 or 1 or 2 (default 0)
        0 for the function,
        1 for the first derivative
        2 for the second derivative

    Returns
    -------
    v : array like
        value of Mse⁽⁴⁾ₘ(q, ξ) or Mse⁽⁴⁾ₘ′(q, ξ) or Mse⁽⁴⁾ₘ′´(q, ξ)
    """

    if p == 0:
        return mathieu_modsem1(m, q, ξ)[0] - 1j * mathieu_modsem2(m, q, ξ)[0]

    if p == 1:
        return mathieu_modsem1(m, q, ξ)[1] - 1j * mathieu_modsem2(m, q, ξ)[1]

    if p == 2:
        return (mathieu_b(m, q) - (2 * q) * cosh(2 * ξ)) * (
            mathieu_modsem1(m, q, ξ)[0] - 1j * mathieu_modsem2(m, q, ξ)[0]
        )

    raise ValueError("The value p must be 0, 1, or 2.")
