"""
This is a gamma-law equation of state: p = rho e (gamma - 1), where
gamma is the constant ratio of specific heats.
"""
from compressible.eos.eos_module import eos, eos_init
from compressible.eos.eos_type_module import eos_t, eos_input_re, eos_input_pe
from compressible.eos.network import network_init
import numpy as np

# a wrapper for the eos module

eos_init()
network_init()


def pres(gamma, dens, eint):
    """
    Given the density and the specific internal energy, return the
    pressure

    Parameters
    ----------
    gamma : float
        The ratio of specific heats
    dens : float
        The density
    eint : float
        The specific internal energy

    Returns
    -------
    out : float
       The pressure

    """
    s = eos_t()
    s.gam1 = gamma
    s.xn = [1]
    p = np.zeros_like(dens)

    nx, ny = np.shape(dens)
    for i in range(nx):
        for j in range(ny):
            s.rho = dens[i,j]
            s.e = eint[i,j]
            s.gam1 = gamma

            eos(eos_input_re, s)

            p[i,j] = s.p

    return p


def dens(gamma, pres, eint):
    """
    Given the pressure and the specific internal energy, return
    the density

    Parameters
    ----------
    gamma : float
        The ratio of specific heats
    pres : float
        The pressure
    eint : float
        The specific internal energy

    Returns
    -------
    out : float
       The density

    """

    s = eos_t()
    s.gam1 = gamma
    dens = np.zeros_like(pres)

    nx, ny = np.shape(dens)
    for i in range(nx):
        for j in range(ny):
            s.p = pres[i,j]
            s.e = eint[i,j]

            eos(eos_input_pe, s)

            dens[i,j] = s.rho

    return dens


def rhoe(gamma, pres):
    """
    Given the pressure, return (rho * e)

    Parameters
    ----------
    gamma : float
        The ratio of specific heats
    pres : float
        The pressure

    Returns
    -------
    out : float
       The internal energy density, rho e

    """
    rhoe = pres/(gamma - 1.0)
    return rhoe
