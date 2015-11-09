from __future__ import print_function

import math

import mesh.patch as patch
import numpy as np
from util import msg

def init_data(my_data, aux_data, base, rp, metric):
    """ initialize the flame problem """

    msg.bold("initializing the flame problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in flame.py")

    # get the density and velocities
    dens = my_data.get_var("density")
    enth = my_data.get_var("enthalpy")
    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")
    eint = aux_data.get_var("eint")
    scalar = my_data.get_var("scalar")
    T = my_data.get_var("temperature")
    DX = my_data.get_var("mass-frac")

    g = rp.get_param("lm-gr.grav")
    c = rp.get_param("lm-gr.c")
    R = rp.get_param("lm-gr.radius")

    gamma = rp.get_param("eos.gamma")
    K = rp.get_param("eos.k_poly")

    dens_ratio = rp.get_param("flame.dens_ratio")
    print('dens ratio: ', dens_ratio)
    dens_base = rp.get_param("flame.dens_base")
    dens_cutoff = rp.get_param("flame.dens_cutoff")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5 * (xmin + xmax)
    L_x = 0.1 * (xmax - xmin)
    L = xmax - xmin

    myg = my_data.grid
    print('Resolution: ', myg.nx, ' x ', myg.ny)
    pres = myg.scratch_array()

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    u.d[:,:] = 0.
    v.d[:,:] = 0.
    dens.d[:,:] = dens_cutoff
    dens.v()[:,:] = dens_base * \
        np.exp(-g * myg.y[np.newaxis, myg.jlo:myg.jhi+1] /
                (gamma * c**2 * R * metric.alpha.v2d()**2))

    pres.d[:,:] = K * dens.d**gamma
    eint.d[:,:] = pres.d / (gamma - 1.0) / dens.d
    enth.d[:, :] = 1. + eint.d + pres.d / dens.d

    scalar.d[:,:] = 0.
    DX.d[:,:] = 0.

    # do burnt stuff - shall start flame on left, 10% of the way across the domain
    idx = (myg.x <= 0.2 * xctr)
    scalar.d[idx] = 1.
    DX.d[idx] = 1.
    # need to increase/decrease other quantities here as well to get the discontinuity - shall use Rankine-Hugoniot stuff
    p2 = myg.scratch_array()
    p2.d[:,:] = pres.d
    d2 = myg.scratch_array()
    d2.d[:,:] = dens.d
    dens.d[idx] *= dens_ratio

    pres.d[idx] = K * dens.d[idx]**gamma
    p1 = myg.scratch_array()
    p1.d[:,:] = pres.d
    d1 = myg.scratch_array()
    d1.d[:,:] = dens.d
    h2 = myg.scratch_array()
    h2.d[:,:] = enth.d

    # I think the weak deflagration must take the negative root
    enth.d[idx] = 0.5 * (p2.d[idx] - p1.d[idx]) / d1.d[idx] + np.sqrt(h2.d[idx]**2 + h2.d[idx] * (p1.d[idx] - p2.d[idx]) / d2.d[idx] + (0.5 * (p1.d[idx] - p2.d[idx]) / d1.d[idx])**2 )
    h1 = myg.scratch_array()
    h1.d[:,:] = enth.d

    # print(enth.d[5:15,5:15])

    # smooth
    #pres.d[:,:] += (pres.d-p2) * 0.5 * \
    #    (1. + np.tanh(((myg.x2d-0.2*xctr)/L_x)/0.9))
    #dens.d[:,:] += (dens.d-d2) * 0.5 * \
    #    (1. + np.tanh((0.1 * (myg.x2d-0.2*xctr)/L_x)/0.9))
    #enth.d[:,:] += (enth.d-h2) * 0.5 * \
    #    (1. + np.tanh(((myg.x2d-0.2*xctr)/L_x)/0.9))
    #DX.d[:,:] += 0.5 * (1. + np.tanh((-(myg.x2d-0.2*xctr)/L_x)/0.9))

    Jsq = myg.scratch_array()
    Jsq.d[idx] = - (p2.d[idx] - p1.d[idx])/ (h2.d[idx]/d2.d[idx] - h1.d[idx]/d1.d[idx])
    u.d[idx] = np.sqrt(Jsq.d[idx]) / d1.d[idx]

    # extend Jsq to outside the flame.
    Jsq = np.mean(Jsq.d[idx], axis=0, dtype=np.float64)
    u.d[~idx] = Jsq[np.newaxis, :]/dens.d[~idx]

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
    p0.d[:] = K * (D0.d / u0.d1d())**gamma

    for i in range(myg.jlo, myg.jhi+1):
        p0.d[i] = p0.d[i-1] - \
                  myg.dy * Dh0.d[i] * g / (R * c**2 * metric.alpha.d[i] **2 * u0.d1d()[i])

    mu = 1./(2. + 4. * DX.d)
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
