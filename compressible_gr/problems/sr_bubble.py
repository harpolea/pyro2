from __future__ import print_function

import sys
import mesh.patch as patch
from compressible_gr.unsplitFluxes import prim_to_cons
import numpy as np
from util import msg

def init_data(my_data, rp):
    """ initialize the SR bubble problem """

    msg.bold("initializing the bubble problem...")

    if rp.get_param("io.do_io"):
        print("Outputting to {}".format(rp.get_param("io.basename")))

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in bubble.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    D = my_data.get_var("D")
    Sx = my_data.get_var("Sx")
    Sy = my_data.get_var("Sy")
    tau = my_data.get_var("tau")
    DX = my_data.get_var("DX")
    rho = np.zeros_like(D.d)
    u = np.zeros_like(D.d)
    v = np.zeros_like(D.d)
    h = np.zeros_like(D.d)
    p = np.zeros_like(D.d)
    X = np.zeros_like(D.d)

    gamma = rp.get_param("eos.gamma")
    c = rp.get_param("eos.c")
    K = rp.get_param("eos.k_poly")

    dens_base = rp.get_param("sr-bubble.dens_base")
    dens_left = rp.get_param("sr-bubble.dens_left")
    eint_left = rp.get_param("sr-bubble.eint_left")

    x_pert = rp.get_param("sr-bubble.x_pert")
    y_pert = rp.get_param("sr-bubble.y_pert")
    r_pert = rp.get_param("sr-bubble.r_pert")
    pert_amplitude_factor = rp.get_param("sr-bubble.pert_amplitude_factor")
    u_vel = rp.get_param("sr-bubble.u_vel")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    # set the density to be stratified in the y-direction
    myg = my_data.grid

    rho[:,:] = dens_base
    u[:,:] = 0.0
    v[:,:] = 0.0
    p[:,:] = K * rho ** gamma
    h[:,:] = 1. + p * gamma / (rho * (gamma - 1.))
    X[:,:] = 1.0

    # set velocity on left side to non-zero
    idxl = myg.x2d <= 0.25*(xmin + xmax)
    u[idxl] = u_vel
    rho[idxl] = dens_left
    eint = eint_left
    p[idxl] = eint * (gamma - 1.) * rho[idxl]
    h[idxl] = 1. + p[idxl] * gamma / (rho[idxl] * (gamma - 1.))

    i = myg.ilo
    while i <= myg.ihi:

        j = myg.jlo
        while j <= myg.jhi:

            r = np.sqrt((myg.x[i] - x_pert)**2  + (myg.y[j] - y_pert)**2)

            if (r <= r_pert):
                # boost the specific internal energy, keeping the pressure
                # constant, by dropping the density
                eint = h[i,j] - 1. - p[i,j] / rho[i,j]

                eint = eint * pert_amplitude_factor
                rho[i,j] = p[i,j] / (eint * (gamma - 1.0))

                h[i,j] = 1. + eint + p[i,j] / rho[i,j]
                X[i,j] = 0.0

            j += 1
        i += 1

    (D.d[:,:], Sx.d[:,:], Sy.d[:,:], tau.d[:,:], DX.d[:,:]) = prim_to_cons((rho, u, v, h, p, X), c, gamma)


def finalize():
    """ print out any information to the user at the end of the run """
    pass
