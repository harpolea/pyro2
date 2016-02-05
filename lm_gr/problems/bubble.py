from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg

def init_data(my_data, aux_data, base, rp, metric):
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
    eint = aux_data.get_var("eint")
    scalar = my_data.get_var("scalar")
    T = my_data.get_var("temperature")
    DX = my_data.get_var("mass-frac")

    g = rp.get_param("lm-gr.grav")
    c = rp.get_param("lm-gr.c")
    R = rp.get_param("lm-gr.radius")

    gamma = rp.get_param("eos.gamma")
    K = rp.get_param("eos.k_poly")

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
    print('Resolution: ', myg.nx, ' x ', myg.ny)
    pres = myg.scratch_array()
    scalar.d[:,:] = 1.
    DX.d[:,:] = 0.

    # FIXME: do this properly for gr case, add alpha back in
    #j = myg.jlo
    #for j in range(myg.jlo, myg.jhi+1):
    #    dens.d[:,j] = max(dens_base*np.exp(-myg.y[j]/scale_height),
    #                      dens_cutoff)
    dens.v()[:, :] = dens_base * \
        np.exp(-g * myg.y[np.newaxis, myg.jlo:myg.jhi+1] /
                (gamma * c**2 * R * metric.alpha(myg).v2d()**2))

    #cs2 = scale_height * abs(g)

    # set the pressure (P = K dens^gamma)
    pres.d[:,:] = K * dens.d**gamma
    eint.d[:,:] = pres.d / (gamma - 1.0) / dens.d
    enth.d[:, :] = 1. + eint.d + pres.d / dens.d

    my_data.fill_BC_all()

    # do the base state
    p0 = base["p0"]
    old_p0 = base["old_p0"]
    D0 = base["D0"]
    Dh0 = base["Dh0"]
    D0.d[:] = np.mean(dens.d, axis=0)
    Dh0.d[:] = np.mean(enth.d, axis=0)
    p0.d[:] = np.mean(pres.d, axis=0)

    # boost the specific internal energy, keeping the pressure
    # constant, by dropping the density
    r = np.sqrt((myg.x2d - x_pert)**2  + (myg.y2d - y_pert)**2)

    idx = r <= r_pert
    eint.d[idx] += eint.d[idx] * (pert_amplitude_factor -  1.) * \
        0.5 * (1. + np.tanh((2. - r[idx]/r_pert)/0.9))# (2.*r_pert)))
    dens.d[idx] = pres.d[idx] / (eint.d[idx] * (gamma - 1.0))
    enth.d[idx] = 1. + eint.d[idx] + pres.d[idx] / dens.d[idx]
    scalar.d[idx] = 0.
    DX.d[idx] = 1.

    # redo the pressure via TOV
    u0 = metric.calcu0()
    p0.d[:] = K * D0.d**gamma

    for i in range(myg.jlo, myg.jhi+1):
        p0.d[i] = p0.d[i-1] - \
                  myg.dy * Dh0.d[i] * g / (R * c**2 * metric.alpha(myg).d[i] **2 * u0.d1d()[i])
                  #myg.dy * g * (2. * p0.d[i-1] * (1. + metric.alpha.d[i]**4) -
                  #Dh0.d[i] / u0.d1d()[i]) / (c**2 * metric.alpha.d[i]**2 * R)
    mu = 1./(2. * (1 - DX.d) + 4. * DX.d)
    # FIXME: hack to drive reactions
    mp_kB = 1.21147#e-8

    T.d[:,:] = p0.d2d() * mu * mp_kB / dens.d

    # multiply by correct u0s
    dens.d[:, :] *= u0.d  # rho * u0
    enth.d[:, :] *= dens.d  # rho * h * u0
    D0.d[:] *= u0.d1d()
    Dh0.d[:] *= D0.d
    old_p0 = p0.copy()
    scalar.d[:,:] *= dens.d
    DX.d[:,:] *= dens.d

    my_data.fill_BC_all()


def finalize():
    """ print out any information to the user at the end of the run """
    pass
