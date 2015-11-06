from __future__ import print_function

import math
import numpy as np

import sys
import mesh.patch as patch
from util import msg

def init_data(my_data, aux_data, base, rp, metric):
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
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")
    eint = aux_data.get_var("eint")
    scalar = my_data.get_var("scalar")
    T = my_data.get_var("temperature")
    DX = my_data.get_var("mass-frac")

    gamma = rp.get_param("eos.gamma")
    K = rp.get_param("eos.k_poly")

    g = rp.get_param("lm-gr.grav")
    c = rp.get_param("lm-gr.c")
    R = rp.get_param("lm-gr.radius")

    dens1 = rp.get_param("rt.dens1")
    dens2 = rp.get_param("rt.dens2")
    amp = rp.get_param("rt.amp")
    sigma = rp.get_param("rt.sigma")

    # initialize the components, remember eint is the specific
    # internal energy (erg/g)
    u.d[:,:] = 0.0
    v.d[:,:] = 0.0
    enth.d[:,:] = 0.

    myg = my_data.grid
    print('Resolution: ', myg.nx, ' x ', myg.ny)

    ycentre = 0.5 * (myg.ymin + myg.ymax)

    y_smooth = 0.04 * (myg.ymax - myg.ymin)

    p = myg.scratch_array()

    #dens.d[:,:] = dens2
    #dens.d[:, :int(np.round(0.5 * (myg.jlo+myg.jhi)))] = dens1

    # Some smoothing across boundary
    dens.d[:,:] = dens1 + (dens2 - dens1) * 0.5 * (1. + np.tanh(((myg.y2d - ycentre)/y_smooth)/0.9))
    y_smooth *= 1.e-5
    scalar.d[:,:] = 1. * 0.5 * (1. + np.tanh(((myg.y2d - ycentre)/(y_smooth))/0.9))
    DX.d[:,:] = 1. * 0.5 * (1. + np.tanh((-(myg.y2d - ycentre)/(y_smooth))/0.9))

    dens.v()[:, :] *= \
        np.exp(-g * myg.y[np.newaxis, myg.jlo:myg.jhi+1] /
                (gamma * c**2 * R * metric.alpha.v2d()**2))


    p.d[:,:] = K * dens.d**gamma

    v.d[:,:] = amp * np.cos(2.0 * math.pi * myg.x2d /
        (myg.xmax - myg.xmin)) * \
        np.exp(-(myg.y2d - ycentre)**2/sigma**2)
    # set the energy (P = cs2*dens)
    eint.d[:,:] = p.d[:,:]/(gamma - 1.0)/dens.d[:,:]
    enth.d[:, :] = 1. + eint.d + p.d / dens.d

    my_data.fill_BC_all()

    u0 = metric.calcu0(u=u, v=v)

    # do the base state
    p0 = base["p0"]
    old_p0 = base["old_p0"]
    D0 = base["D0"]
    Dh0 = base["Dh0"]
    D0.d[:] = np.mean(dens.d, axis=0)
    Dh0.d[:] = np.mean(enth.d, axis=0)
    p0.d[:] = np.mean(p.d, axis=0)

    p0.d[:] = K * (D0.d / u0.d1d())**gamma

    for i in range(myg.jlo, myg.jhi+1):
        p0.d[i] = p0.d[i-1] - \
                  myg.dy * Dh0.d[i] * g / (R * c**2 * metric.alpha.d[i] **2 * u0.d1d()[i])

    mu = 1./(2. + 4. * DX.d)
    mp_kB = 1.21147e-8

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
