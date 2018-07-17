from __future__ import print_function

import sys
import mesh.patch as patch


def init_data(my_data, rp):
    """ an init routine for unit testing """

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in sedov.py")
        print(my_data.__class__)
        sys.exit()

    # get the separate variables
    u = my_data.get_var("xvel")
    v = my_data.get_var("yvel")

    # initialize the components
    u[:, :] = 1.0
    v[:, :] = 0.5


def finalize():
    """ print out any information to the user at the end of the run """
    pass
