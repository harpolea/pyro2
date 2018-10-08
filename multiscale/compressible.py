import numpy as np

import compressible.eos as eos
import multiscale.interface_comp_f as ifc
import mesh.array_indexer as ai
from util import msg


class Variables(object):
    """
    a container class for easy access to the different compressible
    variable by an integer key
    """

    def __init__(self, myd):
        self.nvar = len(myd.names)

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.idens = myd.names.index("density")
        self.ixmom = myd.names.index("x-momentum")
        self.iymom = myd.names.index("y-momentum")
        self.iener = myd.names.index("energy")

        # if there are any additional variable, we treat them as
        # passively advected scalars
        self.naux = self.nvar - 4
        if self.naux > 0:
            self.irhox = 4
        else:
            self.irhox = -1

        # primitive variables
        self.nq = 4 + self.naux

        self.irho = 0
        self.iu = 1
        self.iv = 2
        self.ip = 3

        if self.naux > 0:
            self.ix = 4   # advected scalar
        else:
            self.ix = -1


def cons_to_prim(U, rp, ivars, myg):
    """ convert an input vector of conserved variables to primitive variables """

    gamma = rp.get_param("eos.gamma")

    q = myg.scratch_array(nvar=ivars.nq)

    q[:, :, ivars.irho] = U[:, :, ivars.idens]
    q[:, :, ivars.iu] = U[:, :, ivars.ixmom] / U[:, :, ivars.idens]
    q[:, :, ivars.iv] = U[:, :, ivars.iymom] / U[:, :, ivars.idens]

    e = (U[:, :, ivars.iener] -
         0.5 * q[:, :, ivars.irho] * (q[:, :, ivars.iu]**2 +
                                      q[:, :, ivars.iv]**2)) / q[:, :, ivars.irho]

    q[:, :, ivars.ip] = eos.pres(gamma, q[:, :, ivars.irho], e)

    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix + ivars.naux),
                          range(ivars.irhox, ivars.irhox + ivars.naux)):
            q[:, :, nq] = U[:, :, nu] / q[:, :, ivars.irho]

    return q


def prim_to_cons(q, rp, ivars, myg):
    """ convert an input vector of primitive variables to conserved variables """

    gamma = rp.get_param("eos.gamma")

    U = myg.scratch_array(nvar=ivars.nvar)

    U[:, :, ivars.idens] = q[:, :, ivars.irho]
    U[:, :, ivars.ixmom] = q[:, :, ivars.iu] * U[:, :, ivars.idens]
    U[:, :, ivars.iymom] = q[:, :, ivars.iv] * U[:, :, ivars.idens]

    rhoe = eos.rhoe(gamma, q[:, :, ivars.ip])

    U[:, :, ivars.iener] = rhoe + 0.5 * q[:, :, ivars.irho] * (q[:, :, ivars.iu]**2 +
                                                               q[:, :, ivars.iv]**2)

    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix + ivars.naux),
                          range(ivars.irhox, ivars.irhox + ivars.naux)):
            U[:, :, nu] = q[:, :, nq] * q[:, :, ivars.irho]

    return U


def derive_primitives(myd, varnames):
    """
    derive desired primitive variables from conserved state
    """

    # get the variables we need
    dens = myd.get_var("density")
    xmom = myd.get_var("x-momentum")
    ymom = myd.get_var("y-momentum")
    ener = myd.get_var("energy")

    derived_vars = []

    u = xmom / dens
    v = ymom / dens

    e = (ener - 0.5 * dens * (u * u + v * v)) / dens

    gamma = myd.get_aux("gamma")
    p = eos.pres(gamma, dens, e)

    if isinstance(varnames, str):
        wanted = [varnames]
    else:
        wanted = list(varnames)

    for var in wanted:

        if var == "velocity":
            derived_vars.append(u)
            derived_vars.append(v)

        elif var in ["e", "eint"]:
            derived_vars.append(e)

        elif var in ["p", "pressure"]:
            derived_vars.append(p)

        elif var == "primitive":
            derived_vars.append(dens)
            derived_vars.append(u)
            derived_vars.append(v)
            derived_vars.append(p)

        elif var == "soundspeed":
            derived_vars.append(np.sqrt(gamma * p / dens))

    if len(derived_vars) > 1:
        return derived_vars
    else:
        return derived_vars[0]


def states(idir, data, ivars, dt, rp, q, ldx):
    """ wrapper for ifc states """

    myg = data.grid

    _V_l, _V_r = ifc.states(idir, myg.qx, myg.qy, myg.ng, myg.dx, dt,
                            ivars.irho, ivars.iu, ivars.iv, ivars.ip, ivars.ix,
                            ivars.nvar, ivars.naux,
                            rp.get_param("eos.gamma"),
                            q, ldx)

    V_l = ai.ArrayIndexer(d=_V_l, grid=myg)
    V_r = ai.ArrayIndexer(d=_V_r, grid=myg)

    return V_l, V_r


def riemannFunc(idir, data, ivars, solid, rp, U_l, U_r):
    """ Solve Riemann problem """

    riemann = rp.get_param("compressible.riemann")

    if riemann == "HLLC":
        func = ifc.riemann_hllc
    elif riemann == "CGF":
        func = ifc.riemann_cgf
    else:
        msg.fail("ERROR: Riemann solver undefined")

    myg = data.grid

    _f = func(idir, myg.qx, myg.qy, myg.ng,
              ivars.nvar, ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener, ivars.irhox, ivars.naux,
              solid.xl, solid.xr,
              rp.get_param("eos.gamma"), U_l, U_r)

    return ai.ArrayIndexer(d=_f, grid=myg)


def apply_sources(rp, my_data, my_aux, dt, ivars, U_xl, U_xr, U_yl, U_yr):
    """ apply source terms """

    dens = my_data.get_var("density")
    ymom = my_data.get_var("y-momentum")
    grav = rp.get_param("compressible.grav")

    ymom_src = my_aux.get_var("ymom_src")
    ymom_src.v()[:, :] = dens.v() * grav
    my_aux.fill_BC("ymom_src")

    E_src = my_aux.get_var("E_src")
    E_src.v()[:, :] = ymom.v() * grav
    my_aux.fill_BC("E_src")

    # ymom_xl[i,j] += 0.5*dt*dens[i-1,j]*grav
    U_xl.v(buf=1, n=ivars.iymom)[:, :] += 0.5 * dt * ymom_src.ip(-1, buf=1)
    U_xl.v(buf=1, n=ivars.iener)[:, :] += 0.5 * dt * E_src.ip(-1, buf=1)

    # ymom_xr[i,j] += 0.5*dt*dens[i,j]*grav
    U_xr.v(buf=1, n=ivars.iymom)[:, :] += 0.5 * dt * ymom_src.v(buf=1)
    U_xr.v(buf=1, n=ivars.iener)[:, :] += 0.5 * dt * E_src.v(buf=1)

    # ymom_yl[i,j] += 0.5*dt*dens[i,j-1]*grav
    U_yl.v(buf=1, n=ivars.iymom)[:, :] += 0.5 * dt * ymom_src.jp(-1, buf=1)
    U_yl.v(buf=1, n=ivars.iener)[:, :] += 0.5 * dt * E_src.jp(-1, buf=1)

    # ymom_yr[i,j] += 0.5*dt*dens[i,j]*grav
    U_yr.v(buf=1, n=ivars.iymom)[:, :] += 0.5 * dt * ymom_src.v(buf=1)
    U_yr.v(buf=1, n=ivars.iener)[:, :] += 0.5 * dt * E_src.v(buf=1)


def artificial_viscosity(my_data, myg, rp, ivars, q, F_x, F_y):
    """ apply artificial viscosity """

    cvisc = rp.get_param("compressible.cvisc")

    _ax, _ay = ifc.artificial_viscosity(
        myg.qx, myg.qy, myg.ng, myg.dx, myg.dy,
        cvisc, q.v(n=ivars.iu, buf=myg.ng), q.v(n=ivars.iv, buf=myg.ng))

    avisco_x = ai.ArrayIndexer(d=_ax, grid=myg)
    avisco_y = ai.ArrayIndexer(d=_ay, grid=myg)

    b = (2, 1)

    for n in range(ivars.nvar):
        # F_x = F_x + avisco_x * (U(i-1,j) - U(i,j))
        var = my_data.get_var_by_index(n)

        F_x.v(buf=b, n=n)[:, :] += \
            avisco_x.v(buf=b) * (var.ip(-1, buf=b) - var.v(buf=b))

        # F_y = F_y + avisco_y * (U(i,j-1) - U(i,j))
        F_y.v(buf=b, n=n)[:, :] += \
            avisco_y.v(buf=b) * (var.jp(-1, buf=b) - var.v(buf=b))
