from __future__ import print_function

import math
import mesh.patch as patch
from compressible_gr.unsplitFluxes import prim_to_cons
import numpy as np
from util import msg

def init_data(my_data, rp):
    """ initialize the Kelvin-Helmholtz problem """

    msg.bold("initializing the Kelvin-Helmholtz problem...")

    if rp.get_param("io.do_io"):
        print("Outputting to {}".format(rp.get_param("io.basename")))

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in kh.py")
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

    rho_1 = rp.get_param("kh.rho_1")
    v_1   = rp.get_param("kh.v_1")
    rho_2 = rp.get_param("kh.rho_2")
    v_2   = rp.get_param("kh.v_2")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    yctr = 0.5 * (ymin + ymax)

    L_x = xmax - xmin

    myg = my_data.grid

    # upper half. v = 0, so don't need to worry about that.
    rho[:,:] = rho_2
    u[:,:] = v_2
    X[:,:] = 0.0 # unburnt above

    # lower half
    lower = myg.y2d < yctr #+ 0.01 * np.sin(10.0 * math.pi * myg.x2d / L_x)

    rho[lower] = rho_1
    u[lower] = v_1
    X[lower] = 1.0 # burnt below

    p[:,:] = K * rho ** gamma
    h[:,:] = 1. + p * gamma / (rho * (gamma - 1.))
    v[:,:] = 5.e-1 * v_1 * np.sin(4. * math.pi * (myg.x[:, np.newaxis] + 0.5 * L_x) / L_x)
    (D.d[:,:], Sx.d[:,:], Sy.d[:,:], tau.d[:,:], DX.d[:,:]) = prim_to_cons((rho, u, v, h, p, X), c, gamma)


def finalize():
    """ print out any information to the user at the end of the run """
    pass
