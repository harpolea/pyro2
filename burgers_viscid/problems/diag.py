from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg


def init_data(my_data, rp):
    """ initialize the sinusoidal burgers problem """

    msg.bold("initializing the sinusoidal burgers problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in sine.py")
        print(my_data.__class__)
        sys.exit()

    u = my_data.get_var("xvel")
    v = my_data.get_var("yvel")

    myg = my_data.grid

    xmin = myg.xmin
    xmax = myg.xmax
    ymin = myg.ymin
    ymax = myg.ymax


    mask = (myg.y2d > -myg.x2d + (xmax-xmin)/3) & (myg.y2d < -myg.x2d + (xmax-xmin)*2/3)

    u[:,:] = np.sqrt(2)
    v[:,:] = np.sqrt(2)

    r = np.sqrt(myg.x2d**2 + myg.y2d**2)

    u[mask] = np.sqrt(2) * (1 + 0.5 * np.sin(6*np.pi * (myg.y2d[mask] + myg.x2d[mask] - 1/3)))
    v[mask] = np.sqrt(2) * (1 + 0.5 * np.sin(6*np.pi * (myg.y2d[mask] + myg.x2d[mask] - 1/3)))



    # if direction == "x":
    #
    #     xmin = my_data.grid.xmin
    #     xmax = my_data.grid.xmax
    #
    #     x_third = (xmin + xmax) / 3
    #
    #     u[:, :] = 1
    #
    #     mask = (my_data.grid.x2d > x_third) & (my_data.grid.x2d < 2 * x_third)
    #     u[mask] = 1 + \
    #         0.5 * np.sin(6 * np.pi * (my_data.grid.x2d[mask] - 1 / 3))
    #
    #     v[:, :] = 0
    #
    # else:
    #
    #     ymin = my_data.grid.ymin
    #     ymax = my_data.grid.ymax
    #
    #     y_third = (ymin + ymax) / 3
    #
    #     v[:, :] = 1
    #
    #     mask = (my_data.grid.y2d > y_third) & (my_data.grid.y2d < 2 * y_third)
    #     v[mask] = 1 + \
    #         0.5 * np.sin(6 * np.pi * (my_data.grid.y2d[mask] - 1 / 3))
    #
    #     u[:, :] = 0


def finalize():
    """ print out any information to the user at the end of the run """
    pass
