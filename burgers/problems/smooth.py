from __future__ import print_function

import sys
import mesh.patch as patch
# import numpy
from util import msg


def init_data(my_data, rp):
    """ initialize the smooth burgers problem """

    msg.bold("initializing the smooth burgers problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in smooth.py")
        print(my_data.__class__)
        sys.exit()

    u = my_data.get_var("xvel")
    v = my_data.get_var("yvel")

    xmin = my_data.grid.xmin
    xmax = my_data.grid.xmax

    # ymin = my_data.grid.ymin
    # ymax = my_data.grid.ymax

    xctr = 0.5 * (xmin + xmax)
    # yctr = 0.5 * (ymin + ymax)

    u[my_data.grid.x2d < xctr] = 1
    u[my_data.grid.x2d >= xctr] = 2
    # -0.5 + numpy.exp(-60.0*((my_data.grid.x2d-xctr)**2))# +
    # (my_data.grid.y2d-yctr)**2))

    v[:, :] = 0  # -0.5 + numpy.exp(-60.0*((my_data.grid.x2d-xctr)**2 +
    #                        (my_data.grid.y2d-yctr)**2))


def finalize():
    """ print out any information to the user at the end of the run """
    pass
