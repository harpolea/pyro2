from __future__ import print_function

def mac_vels(myg, dt, u, v, ldelta_ux, ldelta_vx, ldelta_uy,
    ldelta_vy, gradp_x, gradp_y, coeff, source):
    """
    Calculates the MAC velocities

    Parameters
    ----------
    myg : Grid2d object
        grid on which data lives
    dt : float
        timestep
    u : float array
        horizonal velocity
    v : float array
        vertical velocity
    ldelta_ux : float array
        x-limited u velocities
    ldelta_vx : float array
        x-limitied v velocities
    ldelta_uy : float array
        y-limitied u velocities
    ldelta_vy : float array
        y-limited v velocities
    gradp_x : float array
        gradient of the pressure(?) in x-direction
    gradp_y : float array
        gradient of the pressure(?) in y-direction
    source : float array
        source terms

    Returns
    -------
    u_MAC : float array
        u MAC velocity
    v_MAC : float array
        v MAC velocity
    """

    # get the full u and v left and right states (including transverse terms) on
    # both the x- and y-interfaces

    u_xl, u_xr, _, _, _, _, v_yl, v_yr = get_interface_states(myg,
        dt, u, v, ldelta_ux, ldelta_vx, ldelta_uy, ldelta_vy, gradp_x, gradp_y,
                            coeff, source)

    # Riemann problem -- this follows Burger's equation.  We don't use
    # any input velocity for the upwinding.  Also, we only care about
    # the normal states here (u on x and v on y)
    u_MAC = riemann_and_upwind(myg, u_xl, u_xr)
    v_MAC = riemann_and_upwind(myg, v_yl, v_yr)

    return u_MAC, v_MAC



def states(myg, dt, u, v, ldelta_ux, ldelta_vx, ldelta_uy, ldelta_vy,
                  gradp_x, gradp_y, coeff, source, u_MAC, v_MAC):
    """
    This is similar to mac_vels, but it predicts the interface states
    of both u and v on both interfaces, using the MAC velocities to
    do the upwinding.

    Parameters
    ----------
    myg : Grid2d object
        grid on which data lives
    dt : float
        timestep
    u : float array
        horizonal velocity
    v : float array
        vertical velocity
    ldelta_ux : float array
        x-limited u velocities
    ldelta_vx : float array
        x-limitied v velocities
    ldelta_uy : float array
        y-limitied u velocities
    ldelta_vy : float array
        y-limited v velocities
    gradp_x : float array
        gradient of the pressure(?) in x-direction
    gradp_y : float array
        gradient of the pressure(?) in y-direction
    source : float array
        source terms
    u_MAC : float array
        horizontal MAC velocities
    v_MAC : float array
        vertical MAC velcities

    Returns
    -------
    u_xint, v_xint, u_yint, v_yint : float array
        u- and v-velocities at the interfaces
    """

    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr = get_interface_states(myg,
            dt, u, v, ldelta_ux, ldelta_vx,ldelta_uy, ldelta_vy, gradp_x,
            gradp_y, coeff, source)

    u_xint = upwind(myg, u_xl, u_xr, u_MAC)
    v_xint = upwind(myg, v_xl, v_xr, u_MAC)
    u_yint = upwind(myg, u_yl, u_yr, v_MAC)
    v_yint = upwind(myg, v_yl, v_yr, v_MAC)

    return u_xint, v_xint, u_yint, v_yint





def get_interface_states(myg, dt, u, v, ldelta_ux, ldelta_vx, ldelta_uy,
                        ldelta_vy, gradp_x, gradp_y, coeff, source):

    """
    Compute the unsplit predictions of u and v on both the x- and
    y-interfaces.  This includes the transverse terms.

    Note that the gradp_x, gradp_y should have any coefficients
    already included (e.g. zeta/Dh)

    Parameters
    ----------
    myg : Grid2d object
        grid on which data lives
    dt : float
        timestep
    u : float array
        horizonal velocity
    v : float array
        vertical velocity
    ldelta_ux : float array
        x-limited u velocities
    ldelta_vx : float array
        x-limitied v velocities
    ldelta_uy : float array
        y-limitied u velocities
    ldelta_vy : float array
        y-limited v velocities
    gradp_x : float array
        gradient of the pressure(?) in x-direction
    gradp_y : float array
        gradient of the pressure(?) in y-direction
    source : float array
        source terms

    Returns
    -------
    u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr : float array
        predicts u- and v-velocities to interfaces
    """

    #intialise some stuff
    u_xl = myg.scratch_array()
    u_xr = myg.scratch_array()
    u_yl = myg.scratch_array()
    u_yr = myg.scratch_array()
    v_xl = myg.scratch_array()
    v_xr = myg.scratch_array()
    v_yl = myg.scratch_array()
    v_yr = myg.scratch_array()


    # first predict u and v to both interfaces, considering only the normal
    # part of the predictor.  These are the 'hat' states.

    dtdx = dt / myg.dx
    dtdy = dt / myg.dy

    # u on x-edges
    u_xl.ip(1, buf=2)[:,:] = u.v(buf=2) + \
        0.5 * (1. - dtdx * u.v(buf=2)) * ldelta_ux.v(buf=2)

    u_xr.v(buf=2)[:,:] = u.v(buf=2) - \
        0.5 * (1. + dtdx * u.v(buf=2)) * ldelta_ux.v(buf=2)

    # v on x-edges
    v_xl.ip(1, buf=2)[:,:] = v.v(buf=2) + \
        0.5 * (1. - dtdx * u.v(buf=2)) * ldelta_vx.v(buf=2)

    v_xr.v(buf=2)[:,:] = v.v(buf=2) - \
        0.5 * (1. + dtdx * u.v(buf=2)) * ldelta_vx.v(buf=2)

    # u on y-edges
    u_yl.jp(1, buf=2)[:,:] = u.v(buf=2) + \
        0.5 * (1. - dtdy * v.v(buf=2)) * ldelta_uy.v(buf=2)

    u_yr.v(buf=2)[:,:] = u.v(buf=2) - \
        0.5 * (1. + dtdy * v.v(buf=2)) * ldelta_uy.v(buf=2)

    # v on y-edges
    v_yl.jp(1, buf=2)[:,:] = v.v(buf=2) + \
        0.5 * (1. - dtdy * v.v(buf=2)) * ldelta_vy.v(buf=2)

    v_yr.v(buf=2)[:,:] = v.v(buf=2) - \
        0.5 * (1. + dtdy * v.v(buf=2)) * ldelta_vy.v(buf=2)



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

    # CHANGED: Fixed a load of sign errors in transverse terms

    # add the transverse flux differences to the preliminary interface states
    ubar = 0.5 * (uhat_adv.v(buf=1) + uhat_adv.ip(1, buf=1))
    vbar = 0.5 * (vhat_adv.v(buf=1) + vhat_adv.jp(1, buf=1))

    # v du/dy is the transverse term for the u states on x-interfaces
    vu_y = vbar[:,:] * (u_yint.jp(1, buf=1) - u_yint.v(buf=1))

    u_xl.ip(1, buf=1)[:,:] += -0.5 * dtdy * \
        vu_y[:,:] - 0.5 * dt * gradp_x.v(buf=1) * coeff.v(buf=1)
    u_xr.v(buf=1)[:,:] += -0.5 * dtdy * \
        vu_y[:,:] - 0.5 * dt * gradp_x.v(buf=1) * coeff.v(buf=1)

    # v dv/dy is the transverse term for the v states on x-interfaces
    vv_y = vbar[:,:] * (v_yint.jp(1, buf=1) - v_yint.v(buf=1))

    v_xl.ip(1, buf=1)[:,:] += -0.5 * dtdy * \
        vv_y[:,:] - 0.5 * dt * gradp_y.v(buf=1) * coeff.v(buf=1) \
        + 0.5 * dt * source.v(buf=1)
    v_xr.v(buf=1)[:,:] += -0.5 * dtdy * \
        vv_y[:,:] - 0.5 * dt * gradp_y.v(buf=1) * coeff.v(buf=1) \
        + 0.5 * dt * source.v(buf=1)

    # u dv/dx is the transverse term for the v states on y-interfaces
    uv_x = ubar[:,:] * (v_xint.ip(1, buf=1) - v_xint.v(buf=1))

    v_yl.jp(1, buf=1)[:,:] += -0.5 * dtdx * \
        uv_x[:,:] - 0.5 * dt * gradp_y.v(buf=1) * coeff.v(buf=1) \
        + 0.5 * dt * source.v(buf=1)
    v_yr.v(buf=1)[:,:] += -0.5 * dtdx * \
        uv_x[:,:] - 0.5 * dt * gradp_y.v(buf=1) * coeff.v(buf=1) \
        + 0.5 * dt * source.v(buf=1)

    # u du/dx is the transverse term for the u states on y-interfaces
    uu_x = ubar[:,:] * (u_xint.ip(1, buf=1) - u_xint.v(buf=1))

    u_yl.jp(1, buf=1)[:,:] += -0.5 * dtdx * \
        uu_x[:,:] - 0.5 * dt * gradp_x.v(buf=1) * coeff.v(buf=1)
    u_yr.v(buf=1)[:,:] += -0.5 * dtdx * \
        uu_x[:,:] - 0.5 * dt * gradp_x.v(buf=1) * coeff.v(buf=1)


    return u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr





def D_states(myg, dt, D, u_MAC, v_MAC, ldelta_rx, ldelta_ry):
    """
    This predicts D to the interfaces.  We use the MAC velocities to do
    the upwinding.

    Parameters
    ----------
    myg : Grid2d object
        grid on which data lives
    dt : float
        timestep
    D : float array
        'density'
    u_MAC : float array
        horizontal MAC velocities
    v_MAC : float array
        vertical MAC velcities
    ldelta_rx : float array
        x-limited density
    ldelta_ry : float array
        y-limitied density

    Returns
    -------
    D_xint, D_yint : float array
        D predicted to x- and y-interfaces
    """

    #intialise
    D_xl = myg.scratch_array()
    D_xr = myg.scratch_array()
    D_yl = myg.scratch_array()
    D_yr = myg.scratch_array()

    dtdx = dt / myg.dx
    dtdy = dt / myg.dy

    # D on x-edges
    D_xl.ip(1, buf=2)[:,:] = D.v(buf=2) + \
        0.5 * (1. - dtdx * u_MAC.ip(1, buf=2)) * ldelta_rx.v(buf=2)

    D_xr.v(buf=2)[:,:] = D.v(buf=2) - \
        0.5 * (1. + dtdx * u_MAC.v(buf=2)) * ldelta_rx.v(buf=2)

    # D on y-edges
    D_yl.jp(1, buf=2)[:,:] = D.v(buf=2) + \
        0.5 * (1. - dtdy * v_MAC.jp(1, buf=2)) * ldelta_ry.v(buf=2)

    D_yr.v(buf=2)[:,:] = D.v(buf=2) - \
        0.5 * (1. + dtdy * v_MAC.v(buf=2)) * ldelta_ry.v(buf=2)

    # we upwind based on the MAC velocities
    D_xint = upwind(myg, D_xl, D_xr, u_MAC)
    D_yint = upwind(myg, D_yl, D_yr, v_MAC)


    # now add the transverse term and the non-advective part of the normal
    # divergence

    u_x = (u_MAC.ip(1, buf=2) - u_MAC.v(buf=2)) / myg.dx
    v_y = (v_MAC.jp(1, buf=2) - v_MAC.v(buf=2)) / myg.dy

    #    (D v)_y is the transverse term for the x-interfaces
    # D u_x is the non-advective piece for the x-interfaces
    Dv_y = (D_yint.jp(1, buf=2) * v_MAC.jp(1, buf=2) - \
        D_yint.v(buf=2) * v_MAC.v(buf=2)) / myg.dy

    D_xl.ip(1, buf=2)[:,:] -= 0.5 * dt * (Dv_y[:,:] + D.v(buf=2) * u_x[:,:])
    D_xr.v(buf=2)[:,:] -= 0.5 * dt * (Dv_y[:,:] + D.v(buf=2) * u_x[:,:])

    #    (D u)_x is the transverse term for the y-interfaces
    # D v_y is the non-advective piece for the y-interfaces
    Du_x = (D_xint.ip(1, buf=2) * u_MAC.ip(1, buf=2) - \
        D_xint.v(buf=2) * u_MAC.v(buf=2)) / myg.dx

    D_yl.jp(1, buf=2)[:,:] -= 0.5 * dt * (Du_x[:,:] + D.v(buf=2) * v_y[:,:])
    D_yr.v(buf=2)[:,:] -= 0.5 * dt * (Du_x[:,:] + D.v(buf=2) * v_y[:,:])

    # finally upwind the full states
    D_xint = upwind(myg, D_xl, D_xr, u_MAC)
    D_yint = upwind(myg, D_yl, D_yr, v_MAC)

    return D_xint, D_yint




def upwind(myg, q_l, q_r, s):

    """
    Upwind the left and right states based on the specified input
    velocity, s.  The resulting interface state is q_int

    Parameters
    ----------
    myg : Grid2d object
        grid on which data lives
    q_l : float array
        left state
    q_r : float array
        right state
    s : float array
        specified input velocity

    Returns
    -------
    q_int : float array
        State predicted to interface
    """

    q_int = myg.scratch_array()

    for j in range(myg.jlo-1, myg.jhi+3):
        for i in range(myg.ilo-1, myg.ihi+3):

            if (s.d[i,j] > 0.0):
                q_int.d[i,j] = q_l.d[i,j]
            elif (s.d[i,j] == 0.0):
                q_int.d[i,j] = 0.5*(q_l.d[i,j] + q_r.d[i,j])
            else:
                q_int.d[i,j] = q_r.d[i,j]

    return q_int





def riemann(myg, q_l, q_r):
    """
    Solve the Burger's Riemann problem given the input left and right
    states and return the state on the interface.

    This uses the expressions from Almgren, Bell, and Szymczak 1996.

    Parameters
    ----------
    myg : Grid2d object
        grid on which data lives
    q_l : float array
        left state
    q_r : float array
        right state

    Returns
    -------
    s : float array
        state found at interface by solving Riemann problem
    """

    s = myg.scratch_array()

    for j in range(myg.jlo-1, myg.jhi+3):
        for i in range(myg.ilo-1, myg.ihi+3):

            if (q_l.d[i,j] > 0.0 and q_l.d[i,j] + q_r.d[i,j] > 0.0):
                s.d[i,j] = q_l.d[i,j]
            elif (q_l.d[i,j] <= 0.0 and q_r.d[i,j] >= 0.0):
                s.d[i,j] = 0.
            else:
                s.d[i,j] = q_r.d[i,j]

    return s





def riemann_and_upwind(myg, q_l, q_r):
    """
    First solve the Riemann problem given q_l and q_r to give the
    velocity on the interface and then use this velocity to upwind to
    determine the state (q_l, q_r, or a mix) on the interface).

    This differs from upwind, above, in that we don't take in a
    velocity to upwind with.

    Parameters
    ----------
    myg : Grid2d object
        grid on which data lives
    q_l : float array
        left state
    q_r : float array
        right state

    Returns
    -------
    q_int : float array
        state predicted to interface
    """

    s = riemann(myg, q_l, q_r)
    q_int = upwind(myg, q_l, q_r, s)

    return q_int
