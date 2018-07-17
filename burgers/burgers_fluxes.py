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

    Our convection is that the fluxes are going to be defined on the
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

    ldx = myg.scratch_array(nvar=ivars.nvar)
    ldy = myg.scratch_array(nvar=ivars.nvar)

    for n in range(ivars.nvar):
        ldx[:, :, n] = reconstruction.limit(q[:, :, n], myg, 1, limiter)
        ldy[:, :, n] = reconstruction.limit(q[:, :, n], myg, 2, limiter)

    q_xl = myg.scratch_array(nvar=ivars.nvar)
    q_xr = myg.scratch_array(nvar=ivars.nvar)
    q_x = myg.scratch_array(nvar=ivars.nvar)

    # upwind
    # idx = u.v(buf=1) < 0

    for n in range(ivars.nvar):
        q_xl.v(buf=1, n=n)[:, :] = q.v(buf=1, n=n) + \
            0.5 * (1.0 - cx) * ldx.v(buf=1, n=n)
        q_xr.v(buf=1, n=n)[:, :] = q.v(buf=1, n=n) - \
            0.5 * (1.0 + cx) * ldx.v(buf=1, n=n)
        # q_x.v(buf=1, n=n)[idx] = q.v(buf=1, n=n)[idx] - 0.5*(1.0 + cx[idx])*ldx.v(buf=1, n=n)[idx]
        # q_x.v(buf=1, n=n)[~idx] = q.ip(-1, buf=1, n=n)[~idx] + 0.5*(1.0 - cx[~idx])*ldx.ip(-1, buf=1, n=n)[~idx]

    # y-direction
    q_yl = myg.scratch_array(nvar=ivars.nvar)
    q_yr = myg.scratch_array(nvar=ivars.nvar)
    q_y = myg.scratch_array(nvar=ivars.nvar)

    # idx = v.v(buf=1) < 0

    # upwind
    for n in range(ivars.nvar):
        q_yl.v(buf=1, n=n)[:, :] = q.v(buf=1, n=n) + \
            0.5 * (1.0 - cy) * ldy.v(buf=1, n=n)
        q_yr.v(buf=1, n=n)[:, :] = q.v(buf=1, n=n) - \
            0.5 * (1.0 + cy) * ldy.v(buf=1, n=n)
        # q_y.v(buf=1, n=n)[idx] = q.v(buf=1, n=n)[idx] - 0.5*(1.0 + cy[idx])*ldy.v(buf=1, n=n)[idx]
        # q_y.v(buf=1, n=n)[~idx] = q.jp(-1, buf=1, n=n)[~idx] + 0.5*(1.0 - cy[~idx])*ldy.jp(-1, buf=1, n=n)[~idx]

    # compute the transverse flux differences.  The flux is just (u q)
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
                    q_x[i, j, ivars.iv] = q[i, j, ivars.iv]
            else:
                if ul > 0:
                    q_x[i, j, :] = q_xl[i, j, :]
                elif ur < 0:
                    q_x[i, j, :] = q_xr[i, j, :]
                else:
                    q_x[i, j, ivars.iv] = q[i, j, ivars.iv]

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
                    q_y[i, j, ivars.iu] = q[i, j, ivars.iu]
            else:
                if ul > 0:
                    q_y[i, j, :] = q_yl[i, j, :]
                elif ur < 0:
                    q_y[i, j, :] = q_yr[i, j, :]
                else:
                    q_y[i, j, ivars.iu] = q[i, j, ivars.iu]

    # S = myg.scratch_array(nvar=ivars.nvar)
    # # x-dir
    # S[:, :, :] = 0.5 * (q_xl[:,:,ivars.iu, np.newaxis] + q_xr[:,:,ivars.iu, np.newaxis])
    #
    # q_x[S > 0] = q_xl[S > 0]
    # q_x[S < 0] = q_xr[S < 0]
    #
    # # y-dir
    # S[:, :, :] = 0.5 * (q_yl[:,:,ivars.iv, np.newaxis] + q_yr[:,:,ivars.iv, np.newaxis])
    #
    # q_y[S > 0] = q_yl[S > 0]
    # q_y[S < 0] = q_yr[S < 0]

    F_xt = u[:, :, np.newaxis] * q_x
    F_yt = v[:, :, np.newaxis] * q_y

    F_x = myg.scratch_array(nvar=ivars.nvar)
    F_y = myg.scratch_array(nvar=ivars.nvar)

    dtdx2 = 0.5 * dt / myg.dx
    dtdy2 = 0.5 * dt / myg.dy

    for n in range(ivars.nvar):
        q_x.v(buf=1, n=n)[:, :] -= 0.5 * dtdy2 * (F_yt.jp(1, buf=1, n=n) -
                                                  F_yt.jp(-1, buf=1, n=n))
        q_y.v(buf=1, n=n)[:, :] -= 0.5 * dtdx2 * (F_xt.ip(1, buf=1, n=n) -
                                                  F_xt.ip(-1, buf=1, n=n))

    # the zone where we grab the transverse flux derivative from
    # depends on the sign of the advective velocity

    # idx = u.v(buf=1) <= 0

    for n in range(ivars.nvar):
        F_x.v(buf=1, n=n)[:, :] = 0.5 * q_x.v(buf=1, n=n)**2
        # F_x.v(buf=1, n=n)[idx] = u.v(buf=1, n=n)[idx]*(q_x.v(buf=1, n=n)[idx] -
        #                        dtdy2*(F_yt.ip_jp(0, 1, buf=1, n=n)[idx] -
        #                               F_yt.ip(0, buf=1, n=n)[idx]))
        # F_x.v(buf=1, n=n)[~idx] = u.v(buf=1, n=n)[~idx]*(q_x.v(buf=1, n=n)[~idx] -
        #                        dtdy2*(F_yt.ip_jp(-1, 1, buf=1, n=n)[~idx] -
        #                               F_yt.ip(-1, buf=1, n=n)[~idx]))

    # idx = v.v(buf=1) <= 0

    for n in range(ivars.nvar):
        F_y.v(buf=1, n=n)[:, :] = 0.5 * q_y.v(buf=1, n=n)**2
        # F_y.v(buf=1, n=n)[idx] = v.v(buf=1, n=n)[idx]*(q_y.v(buf=1, n=n)[idx] -
        #                        dtdx2*(F_xt.ip_jp(1, 0, buf=1, n=n)[idx] -
        #                               F_xt.jp(0, buf=1, n=n)[idx]))
        # F_y.v(buf=1, n=n)[~idx] = v.v(buf=1, n=n)[~idx]*(q_y.v(buf=1, n=n)[~idx] -
        #                        dtdx2*(F_xt.ip_jp(1, -1, buf=1, n=n)[~idx] -
        #                               F_xt.jp(-1, buf=1, n=n)[~idx]))

    return F_x, F_y
