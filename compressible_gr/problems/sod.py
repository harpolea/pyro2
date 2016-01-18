from __future__ import print_function

import sys

import mesh.patch as patch
from compressible_gr.unsplitFluxes import *
from util import msg
import numpy as np

def init_data(my_data, rp):
    """ initialize the sod problem """

    msg.bold("initializing the sod problem...")

    if rp.get_param("io.do_io"):
        print("Outputting to {}".format(rp.get_param("io.basename")))

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in sod.py")
        print(my_data.__class__)
        sys.exit()

    gamma = rp.get_param("eos.gamma")
    c = rp.get_param("eos.c")
    #K = rp.get_param("eos.k_poly")

    # get the sod parameters
    dens_left = rp.get_param("sod.dens_left")
    dens_right = rp.get_param("sod.dens_right")

    u_left = rp.get_param("sod.u_left")
    u_right = rp.get_param("sod.u_right")

    p_left = rp.get_param("sod.p_left")
    p_right = rp.get_param("sod.p_right")

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

    myg = my_data.grid

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    direction = rp.get_param("sod.direction")

    xctr = 0.5*(xmin + xmax)
    yctr = 0.5*(ymin + ymax)

    if direction == "x":

        # left
        idxl = myg.x2d <= xctr

        rho[idxl] = dens_left
        u[idxl] = u_left
        v[idxl] = 0.0
        p[idxl] = p_left
        X[idxl] = 1.0

        # right
        idxr = myg.x2d > xctr

        rho[idxr] = dens_right
        u[idxr] = u_right
        v[idxr] = 0.0
        p[idxr] = p_right

    else:

        # bottom
        idxb = myg.y2d <= yctr

        rho[idxb] = dens_left
        u[idxb] = 0.0
        v[idxb] = u_left
        p[idxb] = p_left
        X[idxb] = 1.0

        # top
        idxt = myg.y2d > yctr

        rho[idxt] = dens_right
        u[idxt] = 0.0
        v[idxt] = u_right
        p[idxt] = p

    h[:,:] = 1. + p * gamma / (rho * (gamma - 1.))

    (D.d[:,:], Sx.d[:,:], Sy.d[:,:], tau.d[:,:], DX.d[:,:]) = prim_to_cons((rho, u, v, h, p, X), c, gamma)


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The script analysis/sod_compare.py can be used to compare
          this output to the exact solution.  Some sample exact solution
          data is present as analysis/sod-exact.out
          """

    print(msg)
