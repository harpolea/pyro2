from __future__ import print_function

import sys
import mesh.patch as patch
from compressible_gr.unsplitFluxes import prim_to_cons
import numpy as np
from util import msg

def init_data(my_data, rp):
    """ initialize the test problem """

    msg.bold("initializing the test problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in test.py")
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

    dens_base = rp.get_param("test.dens_base")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    myg = my_data.grid

    rho[:,:] = dens_base
    u[:,:] = 0.0
    v[:,:] = 0.0
    p[:,:] = K * rho ** gamma
    h[:,:] = 1. + p * gamma / (rho * (gamma - 1.))
    X[:,:] = 1.0

    (D.d[:,:], Sx.d[:,:], Sy.d[:,:], tau.d[:,:], DX.d[:,:]) = prim_to_cons((rho, u, v, h, p, X), c, gamma)



def finalize():
    """ print out any information to the user at the end of the run """
    pass
