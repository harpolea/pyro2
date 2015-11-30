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
    u_adv = rp.get_param("flame.u_adv")

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

    d1_u = dens_base
    p1_u = K * d1_u**gamma
    e1_u = p1_u / (gamma - 1.0) / d1_u
    Q_u = 0.
    h1_u = enthalpy(d1_u, p1_u, Q_u, gamma)
    # calculate flame speed here
    s = calc_flame_speed(myg)
    u1_u = u_adv - s
    W1 = W(u1_u, c)
    J1 = d1_u * W1 * u1_u

    d1_b_lims = np.array([0.01, 1.]) * d1_u

    print('lower lim: ', get_d_b(d1_b_lims[0], d1_u, u1_u, p1_u, gamma, c), '    upper lim: ', get_d_b(d1_b_lims[1], d1_u, u1_u, p1_u, gamma, c))

    d1_b = scipy.optimize.brentq(get_d_b, d1_b_lims[0], d1_b_lims[1], args=(d1_u, u1_u, p1_u, gamma, c))
    print('d1_b ', d1_b)
    Wv_b = J1 / d1_b
    W1_b = np.sqrt(1. + (Wv_b/c)**2)
    u1_b = Wv_b / W1_b
    Q_b = calcQ(d1_b, 1.)
    p1_b = pb_from_Rayleigh(d1_u, p1_u, d1_b, J1, Q_b, gamma)
    h1_b = enthalpy(d1_b, p1_b, Q_b, gamma)

    #u1_b = scipy.optimize.fsolve(get_u1_b, u1_b, args=(h1_u*W1, d1_u*W1*u1_u, u1_u, h1_u, p1_u, v1, gamma, c))

    # hW, rhoWv, v_u, h_u, p_u, v, gamma, metric
    #W1_b = 1. / np.sqrt(1. - u1_b**2/c**2)
    #d1_b = J1 / (W1_b * u1_b)
    #h1_b = h1_u * W1 / W1_b
    #p1_b = p1_u - J1*2 * (h1_b/d1_b - h1_u/d1_u)

    # put variables back in
    idx = (myg.x <= 0.2 * xctr)
    pres.d[idx] = p1_b
    pres.d[~idx] = p1_u
    dens.d[idx] = d1_b
    dens.d[~idx] = d1_u
    ul = u1_b + s
    ur = u1_u + s
    u.d[idx] = ul
    u.d[~idx] = ur
    hl = h1_b + 0.5 * d1_b * s**2
    hr = h1_u + 0.5 * d1_u * s**2
    enth.d[idx] = hl
    enth.d[~idx] = hr

    # do burnt stuff - shall start flame on left, 10% of the way across the domain
    scalar.d[idx] = 1.
    DX.d[idx] = 1.

    # ADD SMOOTHING
    # we're going to smooth between 0.05 and 0.15.
    smoo = (myg.x2d >= 0.1 * xctr) * (myg.x2d <= 0.3 * xctr)
    #smoo = (myg.x < -1000.)
    deltx = 0.1
    pres.d[smoo] = p1_b + (myg.x2d[smoo] - 0.1 * xctr) * (p1_u-p1_b) / deltx
    dens.d[smoo] = d1_b + (myg.x2d[smoo] - 0.1 * xctr) * (d1_u-d1_b) / deltx
    enth.d[smoo] = hl + (myg.x2d[smoo] - 0.1 * xctr) * (hr-hl) / deltx
    u.d[smoo] = ul + (myg.x2d[smoo] - 0.1 * xctr) * (ur-ul) / deltx
    scalar.d[smoo] = 1. + (myg.x2d[smoo] - 0.1 * xctr) * (0.-1.) / deltx
    DX.d[smoo] = 1. + (myg.x2d[smoo] - 0.1 * xctr) * (0.-1.) / deltx

    dens.v()[:,:] *= np.exp(-g * myg.y[np.newaxis, myg.jlo:myg.jhi+1] /
                (gamma * c**2 * R * metric.alpha.v2d()**2))

    pres.d[:,:] = K * dens.d**gamma
    eint.d[:,:] = pres.d / (gamma - 1.0) / dens.d
    enth.d[:, :] = 1. + eint.d + pres.d / dens.d

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

    u0 = metric.calcu0(u=myg.scratch_array())
    p0.d[:] = K * (D0.d / u0.d1d())**gamma

    # gravity = 0
    #for i in range(myg.jlo, myg.jhi+1):
    #    p0.d[i] = p0.d[i-1] - \
    #              myg.dy * Dh0.d[i] * g / (R * c**2 * metric.alpha.d[i] **2 * u0.d1d()[i])

    mu = 4./(8. * (1. - DX.d) + 3. * DX.d)
    mp_kB = 1.21147#e5#e-8

    T.d[:,:] = p0.d2d() * mu * mp_kB / dens.d

    # multiply by correct u0s
    dens.d[:, :] *= u0.d  # rho * u0
    enth.d[:, :] *= dens.d  # rho * h * u0
    D0.d[:] *= u0.d1d()
    Dh0.d[:] *= D0.d
    old_p0 = p0.copy()
    scalar.d[:,:] *= dens.d
    DX.d[:,:] *= dens.d
    v.d[:,:] = 0.

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

    W_b = W(v_b, c)

    h_b = hW / W_b
    p_b = ((h_b - 1.) * rhoWv * (gamma - 1.)) / (gamma * W_b * v_b)
    lhs = h_b**2 - h_u**2
    rhs = (hW/rhoWv) * (v_b + v_u) * (p_b - p_u)

    residual = lhs - rhs

    return residual

def calc_flame_speed(myg):
    #s = myg.scratch_array()

    # FIXME: how do I do this???
    #s.d[:,:] = 1.e-3
    s = -0.5

    return s

def calcQ(rho, X):
    #mu = 1./(2. * (1 - X) + 4. * X)
    # FIXME: hack to drive reactions
    #mp_kB = 1.21147#e-8
    #T = p * mu * mp_kB / rho

    # FIXME: hack to drive reactions
    #T9 = T * 1.e-9
    #r5 = rho * 1.e-5

    #Q = 5.3e18 * r5**2 * (X / T9)**3 * np.exp(-4.4 / T9)
    # FIXME: hackkkkk
    #Q *= 1.e12 # for bubble: 1.e9, else 1.e12
    Q = 5.3 * X * rho**2

    return Q

def get_d_b(d_b, d_u, u_u, p_u, gamma, c):
    W_u = W(u_u, c)
    J = d_u * W_u * u_u
    X_b = 1.
    X_u = 0.
    Q_b = calcQ(d_b, X_b)
    Q_u = calcQ(d_u, X_u)

    # Rayleigh line gives p_b
    p_b = pb_from_Rayleigh(d_u, p_u, d_b, J, Q_b, gamma)

    # EOS gives h
    h_b = enthalpy(d_b, p_b, Q_b, gamma)
    h_u = enthalpy(d_u, p_u, Q_u, gamma)

    # Hugoniot line gives the root-find relation
    residual = (h_u / d_u + h_b / d_b) * (p_b - p_u) - (h_u**2 - h_b**2)

    return residual


def pb_from_Rayleigh(d_u, p_u, d_b, J, Q_b, gamma):
    return d_b * \
           (J**2 * (d_b * (gamma * p_u + d_u * (gamma-1.)) -\
            d_u**2 * (gamma-1.) - d_u**2 * Q_b) + d_b * p_u *
            d_u**2 * (gamma-1.)) / \
            (d_u**2 * (J**2 * gamma + d_b**2 * (gamma-1.)))

def enthalpy(rho, p, Q, gamma):
    return 1.0 + gamma/(gamma-1.) * p / rho + Q/(gamma-1.)


def W(v, c):
    return 1. / np.sqrt(1 - v**2/c**2)


def finalize():
    """ print out any information to the user at the end of the run """
    pass
