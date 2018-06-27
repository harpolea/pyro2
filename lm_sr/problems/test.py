from __future__ import print_function

import sys
import numpy as np
import mesh.patch as patch


def init_data(my_data, base, rp):
    """ an init routine for unit testing """

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in test.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xvel = my_data.get_var("x-velocity")
    yvel = my_data.get_var("y-velocity")
    eint = my_data.get_var("eint")

    grav = rp.get_param("lm-atmosphere.grav")
    scale_height = 1.0

    # initialize the components
    dens[:, :] = 1.0
    xvel[:, :] = 0.0
    yvel[:, :] = 0.0
    eint[:, :] = 2.5

    cs2 = scale_height*abs(grav)

    # set the pressure (P = cs2*dens)
    pres = cs2*dens

    # do the base state
    base["rho0"].d[:] = np.mean(dens, axis=0)
    base["p0"].d[:] = np.mean(pres, axis=0)


def finalize():
    """ print out any information to the user at the end of the run """
    pass
