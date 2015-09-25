from __future__ import print_function

import math

import mesh.patch as patch
import numpy as np
from util import msg

def init_data(my_data, base, rp, metric):
    """ initialize the Kelvin-Helmholtz problem """

    msg.bold("initializing the Kelvin-Helmholtz problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in kh.py")


    # get the density and velocities
    dens = my_data.get_var("density")
    enth = my_data.get_var("enthalpy")
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")
    eint = my_data.get_var("eint")
    scalar = my_data.get_var("scalar")

    g = rp.get_param("lm-gr.grav")
    c = rp.get_param("lm-gr.c")
    R = rp.get_param("lm-gr.radius")

    gamma = rp.get_param("eos.gamma")

    rho_1 = rp.get_param("kh.rho_1")
    u_1   = rp.get_param("kh.v_1")
    rho_2 = rp.get_param("kh.rho_2")
    u_2   = rp.get_param("kh.v_2")

    gamma = rp.get_param("eos.gamma")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    yctr = 0.5*(ymin + ymax)
    L_x = 0.025*(xmax - xmin)
    L = xmax - xmin
    rho_m = 0.5 * (rho_1 - rho_2)
    u_m = 0.5 * (u_1 - u_2)

    myg = my_data.grid
    print('Resolution: ', myg.nx, ' x ', myg.ny)

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    u.d[:,:] = u_1 - u_m * np.exp((myg.y[np.newaxis,:] - 0.5)/L_x)
    v.d[:,:] = 5.e-1 * u_1 * np.sin(4. * math.pi * (myg.x[:, np.newaxis]+0.5*L)/L)
    dens.d[:,:] = rho_1 - rho_m * np.exp((myg.y[np.newaxis,:] - 0.5)/L_x)
    scalar.d[:,:] = 0.

    idx = (myg.y2d[:,:] > yctr)
    dens.d[idx] = rho_2 + rho_m * np.exp((-myg.y2d[idx] + 0.5)/L_x)
    u.d[idx] = u_2 + u_m * np.exp((-myg.y2d[idx] + 0.5)/L_x)
    scalar.d[idx] = 1. #+ 0.5 * np.exp((-myg.y2d[idx] + 0.5)/L_x)

    #dens.v()[:, :] *= \
    #    np.exp(-g * myg.y[np.newaxis, myg.jlo:myg.jhi+1] /
    #            (gamma * c**2 * R * metric.alpha.v2d()**2))

    pres = myg.scratch_array()

    pres.d[:,:] = dens.d**gamma
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
    u.d[:,:] /= u0.d
    v.d[:,:] /= u0.d
    scalar.d[:,:] *= dens.d

    my_data.fill_BC_all()


def finalize():
    """ print out any information to the user at the end of the run """
    pass
