from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg

def init_data(my_data, aux_data, base, rp, metric):
    """
    initialize the hydrostatic equilibrium test problem

    This is a static system with gravity and an atmosphere in hydrostatic
    equilibrium.
    """

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in test.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    rho = my_data.get_var("density")
    h = my_data.get_var("enthalpy")
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

    dens_base = rp.get_param("test.dens_base")
    myg = my_data.grid

    rho.d[:,:] = dens_base
    u.d[:,:] = 0.0
    v.d[:,:] = 0.0
    p = myg.scratch_array()

    rho.v()[:, :] = dens_base * \
        np.exp(-g * myg.y[np.newaxis, myg.jlo:myg.jhi+1] /
                (gamma * c**2 * R * metric.alpha(myg).v2d()**2))

    p.d[:,:] = K * rho.d ** gamma
    eint.d[:,:] = p.d / (gamma - 1.0) / rho.d
    h.d[:,:] = 1. + eint.d + p.d / rho.d
    DX.d[:,:] = 1.0
    scalar.d[:,:] = 1.0

    my_data.fill_BC_all()

    # do the base state
    p0 = base["p0"]
    old_p0 = base["old_p0"]
    D0 = base["D0"]
    Dh0 = base["Dh0"]
    D0.d[:] = np.mean(rho.d, axis=0)
    Dh0.d[:] = np.mean(h.d, axis=0)
    p0.d[:] = np.mean(p.d, axis=0)

    u0 = metric.calcu0()
    p0.d[:] = K * (D0.d / u0.d1d())**gamma

    for i in range(myg.jlo, myg.jhi+1):
        p0.d[i] = p0.d[i-1] - \
                  myg.dy * Dh0.d[i] * g / (R * c**2 * metric.alpha(myg).d[i] **2 * u0.d1d()[i])

    mu = 1./(2. * (1 - DX.d) + 4. * DX.d)
    mp_kB = 1.21147#e-8

    T.d[:,:] = p0.d2d() * mu * mp_kB / rho.d

    # multiply by correct u0s
    rho.d[:, :] *= u0.d  # rho * u0
    h.d[:, :] *= rho.d  # rho * h * u0
    D0.d[:] *= u0.d1d()
    Dh0.d[:] *= D0.d
    old_p0 = p0.copy()
    scalar.d[:,:] *= v.d
    DX.d[:,:] *= rho.d

    my_data.fill_BC_all()


def finalize():
    """ print out any information to the user at the end of the run """
    pass
