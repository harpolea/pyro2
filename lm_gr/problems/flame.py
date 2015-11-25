from __future__ import print_function

import math

import mesh.patch as patch
import numpy as np
from util import msg
import lm_gr.metric as metric
import scipy.optimize

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
    u.d[:,:] = 1.e-3
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

    # FIXME: make this into a scalar problem as it's not enjoying this

    d1_u = dens_base
    p1_u = K * d1_u**gamma
    e1_u = p1_u / (gamma - 1.0) / d1_u
    h1_u = 1. + e1_u + p1_u / d1_u
    # calculate flame speed here
    u1_u = 1.e2
    W1 = 1. / np.sqrt(1. - u1_u**2/c**2)
    J1 = d1_u * W1 * u1_u

    u1_b = u1_u*1.05
    v1 = 0.

    u1_b = scipy.optimize.fsolve(get_u1_b, u1_b, args=(h1_u*W1, d1_u*W1*u1_u, u1_u, h1_u, p1_u, v1, gamma, c))

    # hW, rhoWv, v_u, h_u, p_u, v, gamma, metric
    W1_b = 1. / np.sqrt(1. - u1_b**2/c**2)
    d1_b = J1 / (W1_b * u1_b)
    h1_b = h1_u * W1 / W1_b
    p1_b = p1_u - J1*2 * (h1_b/d1_b - h1_u/d1_u)

    # put variables back in
    idx = (myg.x <= 0.2 * xctr)
    pres.d[idx] = p1_b
    pres.d[~idx] = p1_u
    dens.d[idx] = d1_b
    dens.d[~idx] = d1_u
    enth.d[idx] = h1_b
    enth.d[~idx] = h1_u
    u.d[idx] = u1_b - u1_u
    u.d[~idx] = 0.

    dens.v()[:,:] *= np.exp(-g * myg.y[np.newaxis, myg.jlo:myg.jhi+1] /
                (gamma * c**2 * R * metric.alpha.v2d()**2))

    pres.d[:,:] = K * dens.d**gamma
    eint.d[:,:] = pres.d / (gamma - 1.0) / dens.d
    enth.d[:, :] = 1. + eint.d + pres.d / dens.d

    # do burnt stuff - shall start flame on left, 10% of the way across the domain
    scalar.d[idx] = 1.
    DX.d[idx] = 1.
    # need to increase/decrease other quantities here as well to get the discontinuity - shall use Rankine-Hugoniot stuff
    # u subscript = unburnt, b subscript = burnt
    """p_u = myg.scratch_array()
    p_b = myg.scratch_array()
    p_u.d[:,:] = pres.d
    d_u = myg.scratch_array()
    d_b = myg.scratch_array()
    d_u.d[:,:] = dens.d
    # need to calculate u_u = flame speed
    u_u = myg.scratch_array()
    u_u.d[:,:] = calc_flame_speed(myg).d
    # u_b = u - flame speed
    u_b = myg.scratch_array()
    h_u = myg.scratch_array()
    h_u.d[:,:] = enth.d
    h_b = myg.scratch_array()

    J = myg.scratch_array()
    W_u = metric.calcW(u=u_u, v=v)
    J.d[:,:] = dens.d * W_u.d * u_u.d

    # first estimate
    u_b.d[:,:] = u_u.d

    v_b = scipy.optimize.fsolve(get_v_b, u_b.d,  args=(h_u*W_u, d_u*W_u*u_u, u_u, h_u, p_u, v, gamma, metric, myg), maxfev=10)
    u_b.d[:,:] = v_b.reshape((myg.qx, myg.qy))

    # hW, rhoWv, v_u, h_u, p_u, v, gamma, metric
    W_b = metric.calcW(u=u_b, v=v)
    d_b.d[:,:] = J.d / (W_b.d * u_b.d)
    h_b.d[:,:] = h_u.d * W_u.d / W_b.d
    p_b.d[:,:] = p_u.d - J.d**2 * (h_b.d/d_b.d - h_u.d/d_u.d)

    # put variables back in
    pres.d[idx] = p_b.d[idx]
    pres.d[~idx] = p_u.d[~idx]
    dens.d[idx] = d_b.d[idx]
    dens.d[~idx] = d_u.d[~idx]
    enth.d[idx] = h_b.d[idx]
    enth.d[~idx] = h_u.d[~idx]
    u.d[idx] = u_b.d[idx] - u_u.d[idx]
    u.d[~idx] = 0.
    """
    my_data.fill_BC_all()

    # do the base state
    p0 = base["p0"]
    old_p0 = base["old_p0"]
    D0 = base["D0"]
    Dh0 = base["Dh0"]
    # take means of unburnt
    D0.d[:] = np.mean(dens.d[~idx], axis=0)
    Dh0.d[:] = np.mean(enth.d[~idx], axis=0)
    p0.d[:] = np.mean(pres.d[~idx], axis=0)

    u0 = metric.calcu0()
    p0.d[:] = K * (D0.d / u0.d1d())**gamma

    for i in range(myg.jlo, myg.jhi+1):
        p0.d[i] = p0.d[i-1] - \
                  myg.dy * Dh0.d[i] * g / (R * c**2 * metric.alpha.d[i] **2 * u0.d1d()[i])

    mu = 1./(2. * (1 - DX.d) + 4. * DX.d)
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

def get_v_b(v_b, hW, rhoWv, v_u, h_u, p_u, v, gamma, metric, myg):

    # method flattens array, so unflatten it and make into an
    # ArrayIndexer object so it's compatible with methods
    v_b_AI = myg.scratch_array()
    v_b_AI.d[:,:] = v_b.reshape(myg.qx, myg.qy)

    W_b = metric.calcW(v_b_AI, v)

    h_b = hW.d / W_b.d
    p_b = ((h_b - 1.) * rhoWv.d * (gamma - 1.)) / (gamma * W_b.d * v_b_AI.d)
    lhs = h_b**2 - h_u.d**2
    rhs = (hW.d/rhoWv.d) * (v_b_AI.d + v_u.d) * (p_b - p_u.d)

    residual = lhs - rhs

    return residual.flatten()

def get_u1_b(v_b, hW, rhoWv, v_u, h_u, p_u, v, gamma, c):

    W_b = 1. / np.sqrt(1. - v_b**2/c**2)

    h_b = hW / W_b
    p_b = ((h_b - 1.) * rhoWv * (gamma - 1.)) / (gamma * W_b * v_b)
    lhs = h_b**2 - h_u**2
    rhs = (hW/rhoWv) * (v_b + v_u) * (p_b - p_u)

    residual = lhs - rhs

    return residual

def calc_flame_speed(myg):
    s = myg.scratch_array()

    # FIXME: how do I do this???
    s.d[:,:] = 1.e-3

    return s

def finalize():
    """ print out any information to the user at the end of the run """
    pass
