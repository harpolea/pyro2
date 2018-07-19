from __future__ import print_function

import sys
import mesh.patch as patch
from util import msg
import numpy as np


def init_data(my_data, rp):
    """
    initialize the burgers problem for use with the script burgers_compare.

    This uses the initial data u_0 = x, which has the analytic
    solution at time t u(t, x) = x / (1 + t)
    """

    msg.bold("initializing the comparison burgers problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in compare.py")
        print(my_data.__class__)
        sys.exit()

    u = my_data.get_var("xvel")
    v = my_data.get_var("yvel")

    myg = my_data.grid

    a = 10
    nu = 0.01

    u[:, :] = 2 * nu * np.pi * np.sin(np.pi * myg.x2d) / (a + np.cos(np.pi * myg.x2d))

    v[:, :] = 0


def finalize():
    """ print out any information to the user at the end of the run """
    pass
