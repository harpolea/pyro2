from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg

def init_data(my_data, base, rp):
    """ initialize the bubble problem """

    msg.bold("initializing the bubble problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in bubble.py")
        print(my_data.__class__)
        sys.exit()

    # get the density and velocities
    dens = my_data.get_var("density")
    enth = my_data.get_var("enthalpy")
    xvel = my_data.get_var("x-velocity")
    yvel = my_data.get_var("y-velocity")
    eint = my_data.get_var("eint")

    g = rp.get_param("lm-gr.grav")
    c = rp.get_param("lm-gr.c")
    R = rp.get_param("lm-gr.radius")

    gamma = rp.get_param("eos.gamma")

    scale_height = rp.get_param("bubble.scale_height")
    dens_base = rp.get_param("bubble.dens_base")
    dens_cutoff = rp.get_param("bubble.dens_cutoff")

    x_pert = rp.get_param("bubble.x_pert")
    y_pert = rp.get_param("bubble.y_pert")
    r_pert = rp.get_param("bubble.r_pert")
    pert_amplitude_factor = rp.get_param("bubble.pert_amplitude_factor")

    # initialize the components -- we'll get a pressure too
    # but that is used only to initialize the base state
    xvel.d[:,:] = 0.0
    yvel.d[:,:] = 0.0
    dens.d[:,:] = dens_cutoff

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    pres = myg.scratch_array()

    # FIXME: do this properly for gr case, add alpha back in
    j = myg.jlo
    #for j in range(myg.jlo, myg.jhi+1):
    #    dens.d[:,j] = max(dens_base*np.exp(-myg.y[j]/scale_height),
    #                      dens_cutoff)
    dens.v()[:, :] = dens_base * \
        np.exp(-g * myg.y[np.newaxis, myg.jlo:myg.jhi+1] /
                (gamma * c**2 * R))
               #(gamma * c**2 * R * metric.alpha.v2d()**2))

    #cs2 = scale_height * abs(g)

    # set the pressure (P = cs2*dens)
    #pres = cs2 * dens
    pres.d[:,:] = dens.d**gamma
    eint.d[:,:] = pres.d/(gamma - 1.0)/dens.d
    enth.d[:, :] = eint.d + pres.d / dens.d # Newtonian

    # boost the specific internal energy, keeping the pressure
    # constant, by dropping the density
    r = np.sqrt((myg.x2d - x_pert)**2  + (myg.y2d - y_pert)**2)

    idx = r <= r_pert
    eint.d[idx] = eint.d[idx]*pert_amplitude_factor
    dens.d[idx] = pres.d[idx]/(eint.d[idx]*(gamma - 1.0))
    enth.d[idx] = eint.d[idx] + pres.d[idx]/dens.d[idx]

    # do the base state
    p0 = base["p0"]
    D0 = base["D0"]
    Dh0 = base["Dh0"]
    D0.d[:] = np.mean(dens.d, axis=0)
    Dh0.d[:] = np.mean(enth.d, axis=0)
    p0.d[:] = np.mean(pres.d, axis=0)

    # redo the pressure via TOV
    #for j in range(myg.jlo+1, myg.jhi):
    #    base["p0"].d[j] = base["p0"].d[j-1] + 0.5*myg.dy*(base["D0"].d[j] + base["D0"].d[j-1]) * grav
    # FIXME: do this properly
    u01d = p0.copy()
    u01d.d[:] = 1.
    alpha = u01d.copy()
    drp0 = u01d.copy()

    drp0.d[:] = g * (Dh0.d / u01d.d + 2. * p0.d * (1. - alpha.d**4)) / \
          (R * c**2 * alpha.d**2)

    p0.jp(1, buf=1)[:] = p0.v(buf=1) + 0.5 * myg.dy * \
                             (drp0.jp(1, buf=1) + drp0.v(buf=1))

    # FIXME: uncomment
    # multiply by correct u0s
    #dens.d[:, :] *= u0.d  # rho * u0
    #enth.d[:, :] *= u0.d * dens.d  # rho * h * u0

    my_data.fill_BC_all()


def finalize():
    """ print out any information to the user at the end of the run """
    pass
