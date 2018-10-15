from __future__ import print_function

import sys

import mesh.patch as patch
from util import msg


def init_data(swe_data, comp_data, aux_data, rp):
    """ initialize the dam problem """

    msg.bold("initializing the dam problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(swe_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in dam.py")
        print(swe_data.__class__)
        sys.exit()

    # get the dam parameters
    h_left = rp.get_param("dam.h_left")
    h_right = rp.get_param("dam.h_right")

    u_left = rp.get_param("dam.u_left")
    u_right = rp.get_param("dam.u_right")

    # get the height, momenta, and energy as separate variables
    h = swe_data.get_var("height")
    xmom = swe_data.get_var("x-momentum")
    ymom = swe_data.get_var("y-momentum")
    X = swe_data.get_var("fuel")

    rhobar = rp.get_param("swe.rhobar")
    z = rp.get_param("compressible.z")
    g = rp.get_param("multiscale.grav")

    # initialize the components
    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    direction = rp.get_param("dam.direction")

    xctr = 0.35 * (xmin + xmax)
    yctr = 0.35 * (ymin + ymax)

    myg = swe_data.grid

    if direction == "x":

        # left
        idxl = myg.x2d <= xctr

        h[idxl] = h_left
        xmom[idxl] = h_left * u_left
        ymom[idxl] = 0.0
        X[idxl] = 1.0

        # right
        idxr = myg.x2d > xctr

        h[idxr] = h_right
        xmom[idxr] = h_right * u_right
        ymom[idxr] = 0.0
        X[idxr] = 0.0

    else:

        # bottom
        idxb = myg.y2d <= yctr

        h[idxb] = h_left
        xmom[idxb] = 0.0
        ymom[idxb] = h_left * u_left
        X[idxb] = 1.0

        # top
        idxt = myg.y2d > yctr

        h[idxt] = h_right
        xmom[idxt] = 0.0
        ymom[idxt] = h_right * u_right
        X[idxt] = 0.0

    X[:, :] *= h

    # compressible initial data

    # get the density, momenta, and energy as separate variables
    dens = comp_data.get_var("density")
    xmom = comp_data.get_var("x-momentum")
    ymom = comp_data.get_var("y-momentum")
    ener = comp_data.get_var("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    gamma = rp.get_param("eos.gamma")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5 * (xmin + xmax)
    yctr = 0.5 * (ymin + ymax)

    dens[:, :] = 1
    xmom[:, :] = u_right
    ymom[:, :] = 0.0
    p = rhobar * g * (h_right - z)
    ener[:, :] = p / (gamma - 1.0) + 0.5 * (xmom[:, :]
                                            ** 2 + ymom[:, :]**2) / dens[:, :]



def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The script analysis/dam_compare.py can be used to compare
          this output to the exact solution.
          """

    print(msg)
