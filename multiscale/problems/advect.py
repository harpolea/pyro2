from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg


def init_data(swe_data, comp_data, aux_data, rp):
    """ initialize a smooth advection problem for testing convergence """

    msg.bold("initializing the advect problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(swe_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in advect.py")
        print(swe_data.__class__)
        sys.exit()

    # get the hity, momenta, and energy as separate variables
    h = swe_data.get_var("height")
    xmom = swe_data.get_var("x-momentum")
    ymom = swe_data.get_var("y-momentum")
    X = swe_data.get_var("fuel")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    h[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5 * (xmin + xmax)
    yctr = 0.5 * (ymin + ymax)

    # this is identical to the advection/smooth problem
    h[:, :] = 1.0 + np.exp(-60.0 * ((swe_data.grid.x2d - xctr)**2 +
                                    (swe_data.grid.y2d - yctr)**2))

    # velocity is diagonal
    u = 1.0
    v = 1.0
    xmom[:, :] = h[:, :] * u
    ymom[:, :] = h[:, :] * v

    X[:, :] = h**2 / np.max(h)

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

    # this is identical to the advection/smooth problem
    dens[:, :] = 1.0 + np.exp(-60.0 * ((comp_data.grid.x2d - xctr)**2 +
                                       (comp_data.grid.y2d - yctr)**2))

    # velocity is diagonal
    u = 1.0
    v = 1.0
    xmom[:, :] = dens[:, :] * u
    ymom[:, :] = dens[:, :] * v

    # pressure is constant
    p = 1.0
    ener[:, :] = p / (gamma - 1.0) + 0.5 * (xmom[:, :]
                                            ** 2 + ymom[:, :]**2) / dens[:, :]

    # mask
    mask = aux_data.get_var("multiscale_mask")
    mask[:, :] = True
    mask[swe_data.grid.x2d < xctr] = False


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          """

    print(msg)
