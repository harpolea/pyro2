from __future__ import print_function

import sys
import mesh.patch as patch
import numpy
from util import msg


def init_data(my_data, rp):
    """ initialize the smooth advection problem """

    msg.bold("initializing the smooth advection problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in smooth.py")
        print(my_data.__class__)
        sys.exit()

    dens = my_data.get_var("density")

    xmin = my_data.grid.hxmin
    xmax = my_data.grid.hxmax

    ymin = my_data.grid.hymin
    ymax = my_data.grid.hymax

    xctr = 0.5 * (xmin + xmax)
    yctr = 0.5 * (ymin + ymax)

    dens[:, :] = 1.0 + numpy.exp(-60.0 * ((my_data.grid.hx2d - xctr)**2 +
                                          (my_data.grid.hy2d - yctr)**2))


def finalize():
    """ print out any information to the user at the end of the run """
    pass
