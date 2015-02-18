from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg

def mac_vels(myg, dt, u, v, ldelta_ux, ldelta_vx, ldelta_uy,
    ldelta_vy, gradp_x, gradp_y, source):
    """
    Calculates the MAC velocities
    """

    # get the full u and v left and right states (including transverse terms) on
    # both the x- and y-interfaces

    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = get_interface_states(myg,
        dt, u, v, ldelta_ux, ldelta_vx, ldelta_uy, ldelta_vy, gradp_x, gradp_y,
                            source)

    u_MAC = riemann_and_upwind(myg, u_xl, u_xr)
    v_MAC = riemann_and_upwind(myg, v_xl, v_xr)

    return u_MAC, v_MAC






def states(myg, dt, u, v, ldelta_ux, ldelta_vx, ldelta_uy, ldelta_vy,
                  gradp_x, gradp_y, source, u_MAC, v_MAC):
    """
      ! this is similar to mac_vels, but it predicts the interface states
      ! of both u and v on both interfaces, using the MAC velocities to
      ! do the upwinding.
    """

    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = get_interface_states(myg,
            dt, u, v, ldelta_ux, ldelta_vx,ldelta_uy, ldelta_vy,gradp_x,
            gradp_y,source)

    u_xint = upwind(myg, u_xl, u_xr, u_MAC)
    v_xint = upwind(myg, v_xl, v_xr, u_MAC)
    u_yint = upwind(myg, u_yl, u_yr, v_MAC)
    v_yint = upwind(myg, v_yl, v_yr, v_MAC)

    return u_xint, v_xint, u_yint, v_yint





def get_interface_states(myg, dt, u, v, ldelta_ux, ldelta_vx, ldelta_uy,
                        ldelta_vy, gradp_x, gradp_y, source):

    """
      ! Compute the unsplit predictions of u and v on both the x- and
      ! y-interfaces.  This includes the transverse terms.

      ! note that the gradp_x, gradp_y should have any coefficients
      ! already included (e.g. beta_0/D)
    """

    #intialise some stuff
    u_xl = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    u_xr = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    u_yl = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    u_yr = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    v_xl = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    v_xr = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    v_yl = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    v_yr = np.zeros((myg.qx,myg.qy), dtype=np.float64)


    # first predict u and v to both interfaces, considering only the normal
    # part of the predictor.  These are the 'hat' states.


    dtdx = dt/myg.dx
    dtdy = dt/myg.dy
    """
    for j in range(myg.jlo-2, myg.jhi+2):
        for i in range(myg.ilo-2, myg.ihi+2):

            # u on x-edges
            u_xl[i+1,j] = u[i,j] + 0.5*(1. - dtdx*u[i,j])*ldelta_ux[i,j]
            u_xr[i  ,j] = u[i,j] - 0.5*(1. + dtdx*u[i,j])*ldelta_ux[i,j]

            # v on x-edges
            v_xl[i+1,j] = v[i,j] + 0.5*(1. - dtdx*u[i,j])*ldelta_vx[i,j]
            v_xr[i  ,j] = v[i,j] - 0.5*(1. + dtdx*u[i,j])*ldelta_vx[i,j]

            # u on y-edges
            u_yl[i,j+1] = u[i,j] + 0.5*(1. - dtdy*v[i,j])*ldelta_uy[i,j]
            u_yr[i,j] = u[i,j] - 0.5*(1. + dtdy*v[i,j])*ldelta_uy[i,j]

            # v on y-edges
            v_yl[i,j+1] = v[i,j] + 0.5*(1. - dtdy*v[i,j])*ldelta_vy[i,j]
            v_yr[i,j] = v[i,j] - 0.5*(1. + dtdy*v[i,j])*ldelta_vy[i,j]
    """
    # u on x-edges
    u_xl[myg.ilo-1:myg.ihi+4, myg.jlo-2:myg.jhi+3] = \
        u[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] + \
        0.5*(1. - dtdx * \
        u[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) * \
        ldelta_ux[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    u_xr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
        u[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] - \
        0.5*(1. + dtdx * \
        u[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) * \
        ldelta_ux[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    # v on x-edges
    v_xl[myg.ilo-1:myg.ihi+4, myg.jlo-2:myg.jhi+3] = \
        v[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] + \
        0.5*(1. - dtdx * \
        u[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) * \
        ldelta_vx[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    v_xr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
        v[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] - \
        0.5*(1. + dtdx * \
        u[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) * \
        ldelta_vx[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    # u on y-edges
    u_yl[myg.ilo-1:myg.ihi+4, myg.jlo-2:myg.jhi+3] = \
        u[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] + \
        0.5*(1. - dtdy * \
        v[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) * \
        ldelta_uy[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    u_yr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
        u[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] - \
        0.5*(1. + dtdy * \
        v[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) * \
        ldelta_uy[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    # v on y-edges
    v_yl[myg.ilo-1:myg.ihi+4, myg.jlo-2:myg.jhi+3] = \
        v[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] + \
        0.5*(1. - dtdy * \
        v[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) * \
        ldelta_vy[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    v_yr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
        v[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] - \
        0.5*(1. + dtdy * \
        v[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) * \
        ldelta_vy[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]



    # now get the normal advective velocities on the interfaces by solving
    # the Riemann problem.
    uhat_adv = riemann(myg, u_xl, u_xr)
    vhat_adv = riemann(myg, v_yl, v_yr)


    # now that we have the advective velocities, upwind the left and right
    # states using the appropriate advective velocity.

    # on the x-interfaces, we upwind based on uhat_adv
    u_xint = upwind(myg, u_xl, u_xr, uhat_adv)
    v_xint = upwind(myg, v_xl, v_xr, uhat_adv)

    # on the y-interfaces, we upwind based on vhat_adv
    u_yint = upwind(myg, u_yl, u_yr, vhat_adv)
    v_yint = upwind(myg, v_yl, v_yr, vhat_adv)

    # at this point, these states are the `hat' states -- they only
    # considered the normal to the interface portion of the predictor.

    # add the transverse flux differences to the preliminary interface states
    ubar = 0.5*(uhat_adv[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] + \
            uhat_adv[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3])
    vbar = 0.5*(vhat_adv[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] + \
            vhat_adv[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4])

    # v du/dy is the transverse term for the u states on x-interfaces
    vu_y = vbar[:,:]*(u_yint[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4] - \
            u_yint[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3])

    u_xl[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3] -= 0.5 * dtdy * \
        vu_y[:,:] - 0.5 * dt * gradp_x[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]
    u_xr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] -= 0.5 * dtdy * \
        vu_y[:,:] - 0.5 * dt * gradp_x[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    # v dv/dy is the transverse term for the v states on x-interfaces
    vv_y = vbar[:,:]*(v_yint[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4] - \
            v_yint[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3])

    v_xl[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3] -= 0.5 * dtdy * \
        vv_y[:,:] - 0.5 * dt * gradp_y[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]
    v_xr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] -= 0.5 * dtdy * \
        vv_y[:,:] - 0.5 * dt * gradp_y[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]
    # u du/dx is the transverse term for the v states on y-interfaces
    uv_x = ubar[:,:]*(v_xint[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4] - \
            v_xint[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3])

    v_yl[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3] -= 0.5 * dtdx * \
        uv_x[:,:] - 0.5 * dt * gradp_y[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]
    v_yr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] -= 0.5 * dtdx * \
        uv_x[:,:] - 0.5 * dt * gradp_y[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    # u du/dx is the transverse term for the u states on x-interfaces
    uu_x = ubar[:,:]*(u_xint[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4] - \
            u_xint[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3])

    u_yl[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3] -= 0.5 * dtdx * \
        uu_x[:,:] - 0.5 * dt * gradp_x[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]
    u_yr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] -= 0.5 * dtdx * \
        uu_x[:,:] - 0.5 * dt * gradp_x[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    """
    # add the transverse flux differences to the preliminary interface states
    for j in range(myg.jlo-2, myg.jhi+2):
        for i in range(myg.ilo-2, myg.ihi+2):

            ubar = 0.5*(uhat_adv[i,j] + uhat_adv(i+1,j))
            vbar = 0.5*(vhat_adv[i,j] + vhat_adv(i,j+1))

            # v du/dy is the transverse term for the u states on x-interfaces
            vu_y = vbar*(u_yint[i,j+1] - u_yint[i,j])

            u_xl[i+1,j] = u_xl[i+1,j] - 0.5*dtdy*vu_y - 0.5*dt*gradp_x[i,j]
            u_xr[i,j] = u_xr[i,j] - 0.5*dtdy*vu_y - 0.5*dt*gradp_x[i,j]

            # v dv/dy is the transverse term for the v states on x-interfaces
            vv_y = vbar*(v_yint[i,j+1] - v_yint[i,j])

            v_xl[i+1,j] = v_xl[i+1,j] - 0.5*dtdy*vv_y - 0.5*dt*gradp_y[i,j] + 0.5*dt*source[i,j]
            v_xr[i,j] = v_xr[i,j] - 0.5*dtdy*vv_y - 0.5*dt*gradp_y[i,j] + 0.5*dt*source[i,j]

            # u dv/dx is the transverse term for the v states on y-interfaces
            uv_x = ubar*(v_xint[i,j+1] - v_xint[i,j])

            v_yl[i,j+1] = v_yl[i,j+1] - 0.5*dtdx*uv_x - 0.5*dt*gradp_y[i,j] + 0.5*dt*source[i,j]
            v_yr[i,j] = v_yr[i,j] - 0.5*dtdx*uv_x - 0.5*dt*gradp_y[i,j] + 0.5*dt*source[i,j]

            # u du/dx is the transverse term for the u states on y-interfaces
            uu_x = ubar*(u_xint[i+1,j] - u_xint[i,j])

            u_yl[i,j+1] = u_yl[i,j+1] - 0.5*dtdx*uu_x - 0.5*dt*gradp_x[i,j]
            u_yr[i,j] = u_yr[i,j] - 0.5*dtdx*uu_x - 0.5*dt*gradp_x[i,j]
    """

    return u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr








def D_states(myg, dt, D, u_MAC, v_MAC, ldelta_rx, ldelta_ry):
    """
    this predicts D to the interfaces.  We use the MAC velocities to do
    the upwinding
    """

    #intialise
    D_xl = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    D_xr = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    D_yl = np.zeros((myg.qx,myg.qy), dtype=np.float64)
    D_yr = np.zeros((myg.qx,myg.qy), dtype=np.float64)

    dtdx = dt/myg.dx
    dtdy = dt/myg.dy
    """
    for j in range(myg.jlo-2, myg.jhi+2):
        for i in range(myg.ilo-2, myg.ihi+2):

            # D on x-edges
            D_xl[i+1,j] = D[i,j] + 0.5*(1. - dtdx*u_MAC[i+1,j])*ldelta_rx[i,j]
            D_xr[i  ,j] = D[i,j] - 0.5*(1. + dtdx*u_MAC[i,j])*ldelta_rx[i,j]

            # D on y-edges
            D_yl[i,j+1] = D[i,j] + 0.5*(1. - dtdy*v_MAC[i,j+1])*ldelta_ry[i,j]
            D_yr[i,j] = D[i,j] - 0.5*(1. + dtdy*v_MAC[i,j])*ldelta_ry[i,j]
    """
    # D on x-edges
    D_xl[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3] = \
        D[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] + 0.5 * (1. - dtdx * \
        u_MAC[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3]) * \
        ldelta_rx[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    D_xr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
        D[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] - 0.5 * (1. + dtdx * \
        u_MAC[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) * \
        ldelta_rx[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    # D on y-edges
    D_yl[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4] = \
        D[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] + 0.5 * (1. - dtdy * \
        v_MAC[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4]) * \
        ldelta_ry[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    D_yr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] = \
        D[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] - 0.5 * (1. + dtdy * \
        v_MAC[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) * \
        ldelta_ry[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]

    # we upwind based on the MAC velocities
    D_xint = upwind(myg, D_xl, D_xr, u_MAC)
    D_yint = upwind(myg, D_yl, D_yr, v_MAC)


    # now add the transverse term and the non-advective part of the normal
    # divergence

    u_x = (u_MAC[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3] - \
        u_MAC[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) / myg.dx
    v_y = (v_MAC[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4] - \
        v_MAC[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) / myg.dy

    #    (D v)_y is the transverse term for the x-interfaces
    # D u_x is the non-advective piece for the x-interfaces
    Dv_y = (D_yint[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4] * \
        v_MAC[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4] - \
        D_yint[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] * \
         v_MAC[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) / myg.dy

    D_xl[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3] -= 0.5 * dt * \
        (Dv_y[:,:] + D[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] * u_x[:,:])
    D_xr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] -= 0.5 * dt * \
        (Dv_y[:,:] + D[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] * u_x[:,:])

    #    (D u)_x is the transverse term for the y-interfaces
    # D v_y is the non-advective piece for the y-interfaces
    Du_x = (D_xint[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3] * \
        u_MAC[myg.ilo-1:myg.ihi+4,myg.jlo-2:myg.jhi+3] - \
        D_xint[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] * \
         u_MAC[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3]) / myg.dx

    D_yl[myg.ilo-2:myg.ihi+3,myg.jlo-1:myg.jhi+4] -= 0.5 * dt * \
        (Du_x[:,:] + D[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] * v_y[:,:])
    D_yr[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] -= 0.5 * dt * \
        (Du_x[:,:] + D[myg.ilo-2:myg.ihi+3,myg.jlo-2:myg.jhi+3] * v_y[:,:])

    """
    for j in range(myg.jlo-2, myg.jhi+2):
        for i in range(myg.ilo-2, myg.ihi+2):

            u_x = (u_MAC[i+1,j] - u_MAC[i,j])/myg.dx
            v_y = (v_MAC[i,j+1] - v_MAC[i,j])/myg.dy

            #    (D v)_y is the transverse term for the x-interfaces
            # D u_x is the non-advective piece for the x-interfaces
            Dv_y = (D_yint[i,j+1]*v_MAC[i,j+1] - D_yint[i,j]*v_MAC[i,j])/myg.dy

            D_xl[i+1,j] = D_xl[i+1,j] - 0.5*dt*(Dv_y + D[i,j]*u_x)
            D_xr[i,j] = D_xr[i,j] - 0.5*dt*(Dv_y + D[i,j]*u_x)

            # (D u)_x is the transverse term for the y-interfaces
            # D v_y is the non-advective piece for the y-interfaces
            Du_x = (D_xint[i+1,j]*u_MAC[i+1,j] - D_xint[i,j]*u_MAC[i,j])/myg.dx

            D_yl[i,j+1] = D_yl[i,j+1] - 0.5*dt*(Du_x + D[i,j]*v_y)
            D_yr[i,j] = D_yr[i,j] - 0.5*dt*(Du_x + D[i,j]*v_y)

    """

    # finally upwind the full states
    D_xint = upwind(myg, D_xl, D_xr, u_MAC)
    D_yint = upwind(myg, D_yl, D_yr, v_MAC)

    return D_xint, D_yint



def upwind(myg, q_l, q_r, s):

    """
      ! upwind the left and right states based on the specified input
      ! velocity, s.  The resulting interface state is q_int

    """

    q_int = np.zeros((myg.qx,myg.qy), dtype=np.float64)

    for j in range(myg.jlo-1, myg.jhi+1):
        for i in range(myg.ilo-1, myg.ihi+1):

            if (s[i,j] > 0.0):
                q_int[i,j] = q_l[i,j]
            elif (s[i,j] == 0.0):
                q_int[i,j] = 0.5*(q_l[i,j] + q_r[i,j])
            else:
                q_int[i,j] = q_r[i,j]

    return q_int







def riemann(myg, q_l, q_r):

    """
    ! Solve the Burger's Riemann problem given the input left and right
    ! states and return the state on the interface.
    !
    ! This uses the expressions from Almgren, Bell, and Szymczak 1996.
    """

    s = np.zeros((myg.qx,myg.qy), dtype=np.float64)

    for j in range(myg.jlo-1, myg.jhi+1):
        for i in range(myg.ilo-1, myg.ihi+1):

            if (q_l[i,j] > 0.0 and q_l[i,j] + q_r[i,j] > 0.0):
                s[i,j] = q_l[i,j]
            elif (q_l[i,j] <= 0.0 and q_r[i,j] >= 0.0):
                s[i,j] = 0.
            else:
                s[i,j] = q_r[i,j]

    return s





def riemann_and_upwind(myg, q_l, q_r):
    """
    First solve the Riemann problem given q_l and q_r to give the
    velocity on the interface and then use this velocity to upwind to
    determine the state (q_l, q_r, or a mix) on the interface).

    This differs from upwind, above, in that we don't take in a
    velocity to upwind with).
    """

    s = riemann(myg, q_l, q_r)
    q_int = upwind(myg, q_l, q_r, s)

    return q_int
