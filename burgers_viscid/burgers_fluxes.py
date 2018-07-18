import mesh.reconstruction as reconstruction
import numpy as np


def unsplit_fluxes(my_data, rp, ivars, dt):
    """
    Construct the fluxes through the interfaces for the linear burgers
    equations:

    .. math::

       u_t  + u u_x  + v u_y  = 0

       v_t  + u v_x  + v v_y  = 0

    We use a second-order (piecewise linear) unsplit Godunov method
    (following Colella 1990).

    In the pure burgers case, there is no Riemann problem we need to
    solve -- we just simply do upwinding.  So there is only one 'state'
    at each interface, and the zone the information comes from depends
    on the sign of the velocity.

    Our convention is that the fluxes are going to be defined on the
    left edge of the computational zones::

        |             |             |             |
        |             |             |             |
       -+------+------+------+------+------+------+--
        |     i-1     |      i      |     i+1     |

                 u_l,i  u_r,i   u_l,i+1


    u_r,i and u_l,i+1 are computed using the information in
    zone i,j.

    Parameters
    ----------
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    dt : float
        The timestep we are advancing through.
    scalar_name : str
        The name of the variable contained in my_data that we are
        advecting

    Returns
    -------
    out : ndarray, ndarray
        The fluxes on the x- and y-interfaces

    """

    myg = my_data.grid

    # get the burgers velocities
    u = my_data.get_var("xvel")
    v = my_data.get_var("yvel")

    q = my_data.data

    cx = u.v(buf=1) * dt / myg.dx
    cy = v.v(buf=1) * dt / myg.dy

    # --------------------------------------------------------------------------
    # monotonized central differences
    # --------------------------------------------------------------------------

    limiter = rp.get_param("burgers.limiter")
    nu = rp.get_param("burgers.visc")

    ldx = myg.scratch_array(nvar=ivars.nvar)
    ldy = myg.scratch_array(nvar=ivars.nvar)

    for n in range(ivars.nvar):
        ldx[:, :, n] = reconstruction.limit(q[:, :, n], myg, 1, limiter)
        ldy[:, :, n] = reconstruction.limit(q[:, :, n], myg, 2, limiter)

    visc_flx = viscous_flux(u, v, myg, nu, ivars)

    # upwind
    # x-direction
    q_xl = myg.scratch_array(nvar=ivars.nvar)
    q_xr = myg.scratch_array(nvar=ivars.nvar)
    q_x = myg.scratch_array(nvar=ivars.nvar)

    for n in range(ivars.nvar):
        q_xl.v(buf=1, n=n)[:, :] = q.ip(-1, buf=1, n=n) + \
            0.5 * (1.0 - cx) * ldx.ip(-1, buf=1, n=n) + \
            0.5 * dt * visc_flx.ip(-1, buf=1, n=n)
        q_xr.v(buf=1, n=n)[:, :] = q.v(buf=1, n=n) - \
            0.5 * (1.0 + cx) * ldx.v(buf=1, n=n) + \
            0.5 * dt * visc_flx.v(buf=1, n=n)

    # y-direction
    q_yl = myg.scratch_array(nvar=ivars.nvar)
    q_yr = myg.scratch_array(nvar=ivars.nvar)
    q_y = myg.scratch_array(nvar=ivars.nvar)

    for n in range(ivars.nvar):
        q_yl.v(buf=1, n=n)[:, :] = q.jp(-1, buf=1, n=n) + \
            0.5 * (1.0 - cy) * ldy.jp(-1, buf=1, n=n) + \
            0.5 * dt * visc_flx.jp(-1, buf=1, n=n)
        q_yr.v(buf=1, n=n)[:, :] = q.v(buf=1, n=n) - \
            0.5 * (1.0 + cy) * ldy.v(buf=1, n=n) + \
            0.5 * dt * visc_flx.v(buf=1, n=n)

    # compute the transverse flux differences. The flux is just (u q)
    for j in range(myg.qy):
        for i in range(myg.qx):
            # x-dir
            ul = q_xl[i, j, ivars.iu]
            ur = q_xr[i, j, ivars.iu]

            if ul > ur:
                S = 0.5 * (ul + ur)
                if S > 0:
                    q_x[i, j, :] = q_xl[i, j, :]
                elif S < 0:
                    q_x[i, j, :] = q_xr[i, j, :]
                else:
                    q_x[i, j, ivars.iv] = v[i, j]
            else:
                if ul > 0:
                    q_x[i, j, :] = q_xl[i, j, :]
                elif ur < 0:
                    q_x[i, j, :] = q_xr[i, j, :]
                else:
                    q_x[i, j, ivars.iv] = v[i, j]

            # y-dir
            ul = q_yl[i, j, ivars.iv]
            ur = q_yr[i, j, ivars.iv]

            if ul > ur:
                S = 0.5 * (ul + ur)
                if S > 0:
                    q_y[i, j, :] = q_yl[i, j, :]
                elif S < 0:
                    q_y[i, j, :] = q_yr[i, j, :]
                else:
                    q_y[i, j, ivars.iu] = u[i, j]
            else:
                if ul > 0:
                    q_y[i, j, :] = q_yl[i, j, :]
                elif ur < 0:
                    q_y[i, j, :] = q_yr[i, j, :]
                else:
                    q_y[i, j, ivars.iu] = u[i, j]

    F_xt = u[:, :, np.newaxis] * q_x
    F_yt = v[:, :, np.newaxis] * q_y

    F_x = myg.scratch_array(nvar=ivars.nvar)
    F_y = myg.scratch_array(nvar=ivars.nvar)

    dtdx2 = 0.5 * dt / myg.dx
    dtdy2 = 0.5 * dt / myg.dy

    for n in range(ivars.nvar):
        q_x.v(buf=1, n=n)[:, :] -= 0.5 * dtdy2 * (
            F_yt.jp(1, buf=1, n=n) - F_yt.jp(-1, buf=1, n=n))
        q_y.v(buf=1, n=n)[:, :] -= 0.5 * dtdx2 * (
            F_xt.ip(1, buf=1, n=n) - F_xt.ip(-1, buf=1, n=n))

    for n in range(ivars.nvar):
        F_x.v(buf=1, n=n)[:, :] = 0.5 * q_x.v(buf=1, n=n)**2

        F_y.v(buf=1, n=n)[:, :] = 0.5 * q_y.v(buf=1, n=n)**2

    return F_x, F_y


def viscous_flux(u, v, myg, nu, ivars):

    # nu = rp.get_param("burgers.visc")
    # myg = my_data.grid

    flux = myg.scratch_array(nvar=ivars.nvar)

    # u = my_data.get_var("xvel")
    # v = my_data.get_var("yvel")

    # x-dir
    flux.v(n=ivars.iu)[:, :] = nu * (
        u.ip(1) - 2 * u.v() + u.ip(-1)) / myg.dx**2

    # y-dir
    flux.v(n=ivars.iv)[:, :] = nu * (
        v.jp(1) - 2 * v.v() + v.jp(-1)) / myg.dy**2

    return flux
