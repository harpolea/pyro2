from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg


def init_data(my_data, base, rp):
    """ initialize the Gresho vortex problem """

    msg.bold("initializing the Gresho vortex problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in bubble.py")
        print(my_data.__class__)
        sys.exit()

    # get the density and velocities
    dens = my_data.get_var("density")
    xvel = my_data.get_var("x-velocity")
    yvel = my_data.get_var("y-velocity")
    eint = my_data.get_var("eint")

    grav = rp.get_param("lm-atmosphere.grav")

    gamma = rp.get_param("eos.gamma")

    scale_height = rp.get_param("gresho.scale_height")
    dens_base = rp.get_param("gresho.dens_base")
    dens_cutoff = rp.get_param("gresho.dens_cutoff")

    R = rp.get_param("gresho.r")
    u0 = rp.get_param("gresho.u0")
    # p0 = rp.get_param("gresho.p0")

    # initialize the components -- we'll get a pressure too
    # but that is used only to initialize the base state
    xvel[:, :] = 0.0
    yvel[:, :] = 0.0
    dens[:, :] = dens_cutoff

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    pres = myg.scratch_array()

    for j in range(myg.jlo, myg.jhi+1):
        dens[:, j] = max(dens_base*np.exp(-myg.y[j]/scale_height),
                         dens_cutoff)

    cs2 = scale_height*abs(grav)

    # set the pressure (P = cs2*dens)
    pres = cs2*dens
    eint[:, :] = pres/(gamma - 1.0)/dens

    x_centre = 0.5 * (myg.x[0] + myg.x[-1])
    y_centre = 0.5 * (myg.y[0] + myg.y[-1])

    r = np.sqrt((myg.x2d - x_centre)**2 + (myg.y2d - y_centre)**2)

    pres[r <= R] += 0.5 * (u0 * r[r <= R]/R)**2
    pres[(r > R) & (r <= 2*R)] += u0**2 * (0.5 *
        (r[(r > R) & (r <= 2*R)]/R)**2 +
        4 * (1 - r[(r > R) & (r <= 2*R)]/R +
        np.log(r[(r > R) & (r <= 2*R)]/R)))
    pres[r > 2*R] += u0**2 * (4 * np.log(2) - 2)
    #
    uphi = np.zeros_like(pres)
    uphi[r <= R] = u0 * r[r <= R]/R
    uphi[(r > R) & (r <= 2*R)] = u0 * (2 - r[(r > R) & (r <= 2*R)]/R)

    xvel[:, :] = -uphi[:, :] * (myg.y2d - y_centre) / r[:, :]
    yvel[:, :] = uphi[:, :] * (myg.x2d - x_centre) / r[:, :]

    dens[:, :] = pres[:, :]/(eint[:, :]*(gamma - 1.0))

    # make relativistic
    # U2 = xvel**2 + yvel**2
    # idx = (U2 < 1.e-15)
    # W = np.ones_like(xvel)
    # W[~idx] = np.sqrt(0.5/U2[~idx] + np.sqrt(0.25/U2[~idx]**2 + 1.))
    W = 1.0 / np.sqrt(1.0 - xvel**2 - yvel**2)

    dens[:, :] *= W
    xvel[:, :] /= W
    yvel[:, :] /= W

    # do the base state
    base["rho0"].d[:] = np.mean(dens, axis=0)
    base["p0"].d[:] = np.mean(pres, axis=0)

    # redo the pressure via HSE
    for j in range(myg.jlo+1, myg.jhi):
        base["p0"].d[j] = base["p0"].d[j-1] + \
            0.5*myg.dy*(base["rho0"].d[j]/np.mean(W[:, j]) +
            base["rho0"].d[j-1]/np.mean(W[:, j-1]))*grav


def finalize():
    """ print out any information to the user at the end of the run """
    pass
