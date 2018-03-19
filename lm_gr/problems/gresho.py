from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg

def init_data(my_data, aux_data, base, rp, metric):
    """ initialize the gresho vortex problem """

    msg.bold("initializing the gresho vortex problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in gresho.py")
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

    dens_base = rp.get_param("gresho.dens_base")

    # initialize the components
    xvel.d[:,:] = 0.0
    yvel.d[:,:] = 0.0
    dens.d[:,:] = dens_base

    myg = my_data.grid
    print('Resolution: ', myg.nx, ' x ', myg.ny)
    pres = myg.scratch_array()
    scalar.d[:,:] = 1.
    DX.d[:,:] = 0.

    dens.v()[:, :] = dens_base

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
    r = np.sqrt((myg.x2d - 0.5*myg.xmax)**2  + (myg.y2d - 0.5*myg.ymax)**2)

    idx = r < 0.2
    xvel.d[idx] = -5. * (myg.y2d[idx] - 0.5*myg.ymax)
    yvel.d[idx] = 5. * (myg.x2d[idx]- 0.5*myg.xmax)
    pres.d[idx] += 12.5 * r[idx]**2

    idx = (r >= 0.2) * (r < 0.4)
    xvel.d[idx] = - (2 - 5. * r[idx]) * (myg.y2d[idx] - 0.5*myg.ymax) / r[idx]
    yvel.d[idx] = (2 - 5. * r[idx]) * (myg.x2d[idx] - 0.5*myg.xmax)/ r[idx]
    pres.d[idx] += 12.5 * r[idx]**2 + 4. * (1. - 5. * r[idx] - np.log(0.2) + np.log(r[idx]))

    idx = (r >= 0.4)
    xvel.d[idx] = 0.
    yvel.d[idx] = 0.
    pres.d[idx] += -2. + 4. * np.log(2.)

    eint.d[:,:] = pres.d / (gamma - 1.0) / dens.d
    enth.d[:, :] = 1. + eint.d + pres.d / dens.d
    scalar.d[:,:] = 0.

    # going to set up a blob so we can watch it move around as otherwise this is boring
    DX.d[:,:] = 1.
    r = np.sqrt((myg.x2d - 0.7*myg.xmax)**2  + (myg.y2d - 0.7*myg.ymax)**2)
    idx = r < 0.05
    DX.d[idx] = 0.0

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
