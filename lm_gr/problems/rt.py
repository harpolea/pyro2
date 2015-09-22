from __future__ import print_function

import math
import numpy as np

import sys
import mesh.patch as patch
from util import msg

def init_data(my_data, base, rp, metric):
    """ initialize the Rayleigh-Taylor problem """

    msg.bold("initializing the Rayleigh-Taylor problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in rt.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    enth = my_data.get_var("enthalpy")
    xvel = my_data.get_var("x-velocity")
    yvel = my_data.get_var("y-velocity")
    eint = my_data.get_var("eint")

    gamma = rp.get_param("eos.gamma")

    g = rp.get_param("lm-gr.grav")
    c = rp.get_param("lm-gr.c")
    R = rp.get_param("lm-gr.radius")

    dens1 = rp.get_param("rt.dens1")
    dens2 = rp.get_param("rt.dens2")
    p0 = rp.get_param("rt.p0")
    amp = rp.get_param("rt.amp")
    sigma = rp.get_param("rt.sigma")


    # initialize the components, remember, that ener here is
    # rho*eint + 0.5*rho*v**2, where eint is the specific
    # internal energy (erg/g)
    xvel.d[:,:] = 0.0
    yvel.d[:,:] = 0.0
    enth.d[:,:] = 0.
    dens.d[:,:] = 0.0

    # set the density to be stratified in the y-direction
    myg = my_data.grid

    ycentre = 0.5*(myg.ymin + myg.ymax)

    p = myg.scratch_array()

    dens.d[:,:] = dens2
    p.d[:,:] = p0 - dens1 * g * ycentre - \
               dens2 * g * (myg.y[np.newaxis,:] - ycentre)

    idx = (myg.y[:] < ycentre)
    idx *= (myg.y[:] >= myg.y[myg.jlo])

    dens.d[:, idx] = dens1
    p.d[:, idx] = p0 - dens1 * g * myg.y[idx]

    """j = myg.jlo
    while j <= myg.jhi:
        if (myg.y[j] < ycentre):
            dens.d[:,j] = dens1
            p.d[:,j] = p0 + dens1*grav*myg.y[j]

        else:
            dens.d[:,j] = dens2
            p.d[:,j] = p0 + dens1*grav*ycentre + dens2*grav*(myg.y[j] - ycentre)


        j += 1
    """

    yvel.d[:,:] = amp * np.cos(2.0 * math.pi * myg.x2d /
        (myg.xmax - myg.xmin)) * np.exp(-(myg.y2d - ycentre)**2/sigma**2)

    # set the energy (P = cs2*dens)
    eint.d[:,:] = p.d[:,:]/(gamma - 1.0)/dens.d[:,:]
    enth.d[:, :] = 1. + eint.d + p.d / dens.d

    my_data.fill_BC_all()

    # do the base state
    p0 = base["p0"]
    old_p0 = base["old_p0"]
    D0 = base["D0"]
    Dh0 = base["Dh0"]
    D0.d[:] = np.mean(dens.d, axis=0)
    Dh0.d[:] = np.mean(enth.d, axis=0)
    p0.d[:] = np.mean(p.d, axis=0)

    u0 = metric.calcu0()
    p0.d[:] = (D0.d / u0.d1d())**gamma

    for i in range(myg.jlo, myg.jhi+1):
        p0.d[i] = p0.d[i-1] - \
                  myg.dy * Dh0.d[i] * g / (R * c**2 * metric.alpha.d[i] **2 * u0.d1d()[i])

    # multiply by correct u0s
    dens.d[:, :] *= u0.d  # rho * u0
    enth.d[:, :] *= dens.d  # rho * h * u0
    D0.d[:] *= u0.d1d()
    Dh0.d[:] *= D0.d
    old_p0 = p0.copy()

    my_data.fill_BC_all()


def finalize():
    """ print out any information to the user at the end of the run """
    pass
