import numpy as np
import multiscale.interface_swe_f as ifc
import mesh.array_indexer as ai
from util import msg


class Variables(object):
    """
    a container class for easy access to the different swe
    variables by an integer key
    """

    def __init__(self, myd):
        self.nvar = len(myd.names)

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.ih = myd.names.index("height")
        self.ixmom = myd.names.index("x-momentum")
        self.iymom = myd.names.index("y-momentum")

        # if there are any additional variables, we treat them as
        # passively advected scalars
        self.naux = self.nvar - 3
        if self.naux > 0:
            self.ihx = 3
        else:
            self.ihx = -1

        # primitive variables
        self.nq = 3 + self.naux

        self.ih = 0
        self.iu = 1
        self.iv = 2

        if self.naux > 0:
            self.ix = 3   # advected scalar
        else:
            self.ix = -1


def cons_to_prim(U, rp, ivars, myg):
    """
    Convert an input vector of conserved variables
    :math:`U = (h, hu, hv, {hX})`
    to primitive variables :math:`q = (h, u, v, {X})`.
    """

    q = myg.scratch_array(nvar=ivars.nq)

    q[:, :, ivars.ih] = U[:, :, ivars.ih]
    q[:, :, ivars.iu] = U[:, :, ivars.ixmom] / U[:, :, ivars.ih]
    q[:, :, ivars.iv] = U[:, :, ivars.iymom] / U[:, :, ivars.ih]

    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix + ivars.naux),
                          range(ivars.ihx, ivars.ihx + ivars.naux)):
            q[:, :, nq] = U[:, :, nu] / q[:, :, ivars.ih]

    return q


def prim_to_cons(q, rp, ivars, myg):
    """
    Convert an input vector of primitive variables :math:`q = (h, u, v, {X})`
    to conserved variables :math:`U = (h, hu, hv, {hX})`
    """

    U = myg.scratch_array(nvar=ivars.nvar)

    U[:, :, ivars.ih] = q[:, :, ivars.ih]
    U[:, :, ivars.ixmom] = q[:, :, ivars.iu] * U[:, :, ivars.ih]
    U[:, :, ivars.iymom] = q[:, :, ivars.iv] * U[:, :, ivars.ih]

    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix + ivars.naux),
                          range(ivars.ihx, ivars.ihx + ivars.naux)):
            U[:, :, nu] = q[:, :, nq] * q[:, :, ivars.ih]

    return U


def derive_primitives(myd, varnames):
    """
    derive desired primitive variables from conserved state
    """

    # get the variables we need
    h = myd.get_var("height")
    xmom = myd.get_var("x-momentum")
    ymom = myd.get_var("y-momentum")

    derived_vars = []

    u = xmom / h
    v = ymom / h

    g = myd.get_aux("g")

    if isinstance(varnames, str):
        wanted = [varnames]
    else:
        wanted = list(varnames)

    for var in wanted:

        if var == "velocity":
            derived_vars.append(u)
            derived_vars.append(v)

        elif var == "primitive":
            derived_vars.append(h)
            derived_vars.append(u)
            derived_vars.append(v)

        elif var == "soundspeed":
            derived_vars.append(np.sqrt(g * h))

    if len(derived_vars) > 1:
        return derived_vars
    else:
        return derived_vars[0]


def states(idir, data, ivars, dt, rp, q, ldx):
    """ wrapper for ifc states """

    myg = data.grid

    _V_l, _V_r = ifc.states(idir, myg.qx, myg.qy, myg.ng, myg.dx, dt,
                            ivars.ih, ivars.iu, ivars.iv, ivars.ix,
                            ivars.nvar, ivars.naux,
                            rp.get_param("swe.grav"),
                            q, ldx)

    V_l = ai.ArrayIndexer(d=_V_l, grid=myg)
    V_r = ai.ArrayIndexer(d=_V_r, grid=myg)

    return V_l, V_r


def riemannFunc(idir, data, ivars, solid, rp, U_l, U_r):
    """ Solve the Riemann problem """

    riemann = rp.get_param("swe.riemann")

    if riemann == "HLLC":
        func = ifc.riemann_hllc
    elif riemann == "Roe":
        func = ifc.riemann_roe
    else:
        msg.fail("ERROR: Riemann solver undefined")

    myg = data.grid

    _f = func(idir, myg.qx, myg.qy, myg.ng,
              ivars.nvar, ivars.ih, ivars.ixmom, ivars.iymom, ivars.ihx, ivars.naux,
              solid.xl, solid.xr,
              rp.get_param("swe.grav"), U_l, U_r)

    return ai.ArrayIndexer(d=_f, grid=myg)


def apply_sources(rp, my_data, my_aux, dt, ivars, U_xl, U_xr, U_yl, U_yr):
    """ Do nothing """
    return


def artificial_viscosity(my_data, myg, rp, ivars, q, F_x, F_y):
    """ apply artificial viscosity """
    return
