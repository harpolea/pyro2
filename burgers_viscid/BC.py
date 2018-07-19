"""

"""

from util import msg


def user(bc_name, bc_edge, variable, ccdata):
    """
    A constant boundary.

    Upon exit, the ghost cells for the input variable will be set

    Parameters
    ----------
    bc_name : {'hse'}
        The descriptive name for the boundary condition -- this allows
        for pyro to have multiple types of user-supplied boundary
        conditions.  For this module, it needs to be 'hse'.
    bc_edge : {'ylb', 'yrb'}
        The boundary to update: ylb = lower y boundary; yrb = upper y
        boundary.
    variable : {'density', 'x-momentum', 'y-momentum', 'energy'}
        The variable whose ghost cells we are filling
    ccdata : CellCenterData2d object
        The data object

    """

    myg = ccdata.grid

    if bc_name == "constant":

        if bc_edge == "xlb":

            # lower x boundary

            if variable in ["xvel", 'u']:
                u = ccdata.get_var(variable)
                u[:myg.ilo, :] = 0

            elif variable in ["yvel", 'v']:
                v = ccdata.get_var(variable)
                i = myg.ilo - 1
                while i >= 0:
                    v[i, :] = v[myg.ilo, :]
                    i -= 1

            else:
                print(f'variable = {variable}')
                raise NotImplementedError("variable not defined")

        elif bc_edge == "xrb":

            # upper x boundary

            if variable in ["xvel", 'u']:
                u = ccdata.get_var(variable)
                u[myg.ihi + 1:, :] = 0

            elif variable in ["yvel", 'v']:
                v = ccdata.get_var(variable)

                for i in range(myg.ihi + 1, myg.ihi + myg.ng + 1):
                    v[i, :] = v[myg.ihi, :]

            else:
                raise NotImplementedError("variable not defined")

        elif bc_edge == "ylb":

            # lower y boundary

            if variable in ["xvel", 'u']:
                u = ccdata.get_var(variable)
                j = myg.jlo - 1
                while j >= 0:
                    u[:, j] = u[:, myg.jlo]
                    j -= 1

            elif variable in ["yvel", 'v']:
                v = ccdata.get_var(variable)

                v[:, :myg.jlo] = 0

            else:
                raise NotImplementedError("variable not defined")

        elif bc_edge == "yrb":

            # upper y boundary

            if variable in ["xvel", 'u']:
                u = ccdata.get_var(variable)
                for j in range(myg.jhi + 1, myg.jhi + myg.ng + 1):
                    u[:, j] = u[:, myg.jhi]

            elif variable in ["yvel", 'v']:
                v = ccdata.get_var(variable)
                v[:, myg.jhi + 1:] = 0

            else:
                raise NotImplementedError("variable not defined")

        else:
            msg.fail("error: hse BC not supported for xlb or xrb")

    else:
        msg.fail("error: bc type %s not supported" % (bc_name))
