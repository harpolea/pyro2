"""
Implementation of the Colella 2nd order unsplit Godunov scheme.  This
is a 2-dimensional implementation only.  We assume that the grid is
uniform, but it is relatively straightforward to relax this
assumption.

There are several different options for this solver (they are all
discussed in the Colella paper).

  limiter          = 0 to use no limiting
                   = 1 to use the 2nd order MC limiter
                   = 2 to use the 4th order MC limiter

  riemann          = HLLC to use the HLLC solver
                   = CGF to use the Colella, Glaz, and Ferguson solver

  use_flattening   = 1 to use the multidimensional flattening
                     algorithm at shocks

  delta, z0, z1      these are the flattening parameters.  The default
                     are the values listed in Colella 1990.

   j+3/2--+---------+---------+---------+
          |         |         |         |
     j+1 _|         |         |         |
          |         |         |         |
          |         |         |         |
   j+1/2--+---------XXXXXXXXXXX---------+
          |         X         X         |
       j _|         X         X         |
          |         X         X         |
          |         X         X         |
   j-1/2--+---------XXXXXXXXXXX---------+
          |         |         |         |
     j-1 _|         |         |         |
          |         |         |         |
          |         |         |         |
   j-3/2--+---------+---------+---------+
          |    |    |    |    |    |    |
              i-1        i        i+1
        i-3/2     i-1/2     i+1/2     i+3/2

We wish to solve

  U_t + F^x_x + F^y_y = H

we want U_{i+1/2}^{n+1/2} -- the interface values that are input to
the Riemann problem through the faces for each zone.

Taylor expanding yields

   n+1/2                     dU           dU
  U          = U   + 0.5 dx  --  + 0.5 dt --
   i+1/2,j,L    i,j          dx           dt


                             dU             dF^x   dF^y
             = U   + 0.5 dx  --  - 0.5 dt ( ---- + ---- - H )
                i,j          dx              dx     dy


                              dU      dF^x            dF^y
             = U   + 0.5 ( dx -- - dt ---- ) - 0.5 dt ---- + 0.5 dt H
                i,j           dx       dx              dy


                                  dt       dU           dF^y
             = U   + 0.5 dx ( 1 - -- A^x ) --  - 0.5 dt ---- + 0.5 dt H
                i,j               dx       dx            dy


                                dt       _            dF^y
             = U   + 0.5  ( 1 - -- A^x ) DU  - 0.5 dt ---- + 0.5 dt H
                i,j             dx                     dy

                     +----------+-----------+  +----+----+   +---+---+
                                |                   |            |

                    this is the monotonized   this is the   source term
                    central difference term   transverse
                                              flux term

There are two components, the central difference in the normal to the
interface, and the transverse flux difference.  This is done for the
left and right sides of all 4 interfaces in a zone, which are then
used as input to the Riemann problem, yielding the 1/2 time interface
values,

     n+1/2
    U
     i+1/2,j

Then, the zone average values are updated in the usual finite-volume
way:

    n+1    n     dt    x  n+1/2       x  n+1/2
   U    = U    + -- { F (U       ) - F (U       ) }
    i,j    i,j   dx       i-1/2,j        i+1/2,j


                 dt    y  n+1/2       y  n+1/2
               + -- { F (U       ) - F (U       ) }
                 dy       i,j-1/2        i,j+1/2

Updating U_{i,j}:

  -- We want to find the state to the left and right (or top and
     bottom) of each interface, ex. U_{i+1/2,j,[lr]}^{n+1/2}, and use
     them to solve a Riemann problem across each of the four
     interfaces.

  -- U_{i+1/2,j,[lr]}^{n+1/2} is comprised of two parts, the
     computation of the monotonized central differences in the normal
     direction (eqs. 2.8, 2.10) and the computation of the transverse
     derivatives, which requires the solution of a Riemann problem in
     the transverse direction (eqs. 2.9, 2.14).

       -- the monotonized central difference part is computed using
          the primitive variables.

       -- We compute the central difference part in both directions
          before doing the transverse flux differencing, since for the
          high-order transverse flux implementation, we use these as
          the input to the transverse Riemann problem.
"""

import compressible_gr.eos as eos
import compressible_gr.interface_f as interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch
from scipy.optimize import brentq
import numpy as np
import math

from util import msg

def unsplitFluxes(my_data, rp, vars, tc, dt):
    """
    unsplitFluxes returns the fluxes through the x and y interfaces by
    doing an unsplit reconstruction of the interface values and then
    solving the Riemann problem through all the interfaces at once

    currently we assume a gamma-law EOS

    The runtime parameter grav is assumed to be the gravitational
    acceleration in the y-direction

    Parameters
    ----------
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    vars : Variables object
        The Variables object that tells us which indices refer to which
        variables
    tc : TimerCollection object
        The timers we are using to profile
    dt : float
        The timestep we are advancing through.

    Returns
    -------
    out : ndarray, ndarray
        The fluxes on the x- and y-interfaces

    """

    tm_flux = tc.timer("unsplitFluxes")
    tm_flux.begin()

    myg = my_data.grid

    gamma = rp.get_param("eos.gamma")
    c = rp.get_param("eos.c")
    K = rp.get_param("eos.k_poly")


    #=========================================================================
    # compute the primitive variables
    #=========================================================================
    # Qp = (rho, u, v, h, p)

    D = my_data.get_var("D")
    Sx = my_data.get_var("Sx")
    Sy = my_data.get_var("Sy")
    tau = my_data.get_var("tau")
    r = myg.scratch_array()
    u = myg.scratch_array()
    v = myg.scratch_array()
    h = myg.scratch_array()
    p = myg.scratch_array()

    for i in range(myg.qx):
        for j in range(myg.qy):
            Q = (D.d[i,j], Sx.d[i,j], Sy.d[i,j], tau.d[i,j])
            Qp, c_s = cons_to_prim(Q, c, gamma)
            (r.d[i,j], u.d[i,j], v.d[i,j], h.d[i,j], p.d[i,j]) = Qp
    #print(p.d)
    smallp = 1.e-10
    p.d = p.d.clip(smallp)   # apply a floor to the pressure
    #print(p.d)

    #=========================================================================
    # compute the flattening coefficients
    #=========================================================================

    # there is a single flattening coefficient (xi) for all directions
    use_flattening = rp.get_param("compressible-gr.use_flattening")

    if use_flattening:
        delta = rp.get_param("compressible-gr.delta")
        z0 = rp.get_param("compressible-gr.z0")
        z1 = rp.get_param("compressible-gr.z1")

        xi_x = reconstruction_f.flatten(1, p.d, u.d, myg.qx, myg.qy, myg.ng, smallp, delta, z0, z1)
        xi_y = reconstruction_f.flatten(2, p.d, v.d, myg.qx, myg.qy, myg.ng, smallp, delta, z0, z1)

        xi = reconstruction_f.flatten_multid(xi_x, xi_y, p.d, myg.qx, myg.qy, myg.ng)
    else:
        xi = 1.0



    # monotonized central differences in x-direction
    tm_limit = tc.timer("limiting")
    tm_limit.begin()

    limiter = rp.get_param("compressible-gr.limiter")
    if limiter == 0:
        limitFunc = reconstruction_f.nolimit
    elif limiter == 1:
        limitFunc = reconstruction_f.limit2
    else:
        limitFunc = reconstruction_f.limit4

    ldelta_Dx = xi * limitFunc(1, D.d, myg.qx, myg.qy, myg.ng)
    ldelta_Sxx = xi * limitFunc(1, Sx.d, myg.qx, myg.qy, myg.ng)
    ldelta_Syx = xi * limitFunc(1, Sy.d, myg.qx, myg.qy, myg.ng)
    ldelta_taux = xi * limitFunc(1, tau.d, myg.qx, myg.qy, myg.ng)

    # monotonized central differences in y-direction
    ldelta_Dy = xi * limitFunc(2, D.d, myg.qx, myg.qy, myg.ng)
    ldelta_Sxy = xi * limitFunc(2, Sx.d, myg.qx, myg.qy, myg.ng)
    ldelta_Syy = xi * limitFunc(2, Sy.d, myg.qx, myg.qy, myg.ng)
    ldelta_tauy = xi * limitFunc(2, tau.d, myg.qx, myg.qy, myg.ng)

    tm_limit.end()



    #=========================================================================
    # x-direction
    #=========================================================================

    # left and right conservative variable states
    tm_states = tc.timer("interfaceStates")
    tm_states.begin()

    # ERROR: somehow, this makes v in V_l, V_r = 1.0 in all but the ghost cells (irrespective of the speed of light).
    _U_xl, _U_xr = interface_f.states(1, myg.qx, myg.qy, myg.ng, myg.dx, dt,
                                  vars.nvar,
                                  gamma, c,
                                  r.d, u.d, v.d, p.d,
                                  D.d, Sx.d, Sy.d, tau.d,
                                  ldelta_Dx, ldelta_Sxx, ldelta_Syx, ldelta_taux)

    tm_states.end()

    U_xl = myg.scratch_array(vars.nvar)
    U_xr = myg.scratch_array(vars.nvar)

    U_xl.d[:,:] = _U_xl
    U_xr.d[:,:] = _U_xr

    # stop the nans
    smallr = 1.e-10
    U_xl.d[:,:,vars.iD] = U_xl.d[:,:,vars.iD].clip(smallr)
    U_xr.d[:,:,vars.iD] = U_xr.d[:,:,vars.iD].clip(smallr)

    # transform interface states back into primitive variables
    V_xl = myg.scratch_array(vars.nvar)
    V_xr = myg.scratch_array(vars.nvar)


    #=========================================================================
    # y-direction
    #=========================================================================


    # left and right conservative variable states
    tm_states.begin()

    _U_yl, _U_yr = interface_f.states(2, myg.qx, myg.qy, myg.ng, myg.dy, dt,
                                  vars.nvar,
                                  gamma, c,
                                  r.d, u.d, v.d, p.d,
                                  D.d, Sx.d, Sy.d, tau.d,
                                  ldelta_Dy, ldelta_Sxy, ldelta_Syy, ldelta_tauy)

    U_yl = myg.scratch_array(vars.nvar)
    U_yr = myg.scratch_array(vars.nvar)

    U_yl.d[:,:] = _U_yl
    U_yr.d[:,:] = _U_yr

    tm_states.end()

    # stop the nans
    U_yl.d[:,:,vars.iD] = U_yl.d[:,:,vars.iD].clip(smallr)
    U_yr.d[:,:,vars.iD] = U_yr.d[:,:,vars.iD].clip(smallr)

    # transform interface states back into conserved variables
    V_yl = myg.scratch_array(vars.nvar)
    V_yr = myg.scratch_array(vars.nvar)

    """

    h_yl = h_from_eos(V_yl[:,:,vars.irho], gamma, K)
    h_yr = h_from_eos(V_yr[:,:,vars.irho], gamma, K)

    for i in range(myg.qx):
        for j in range(myg.qy):
            Qp_l = (V_yl[i,j,vars.irho], V_yl[i,j,vars.iu], V_yl[i,j,vars.iv], h_yl[i,j], V_yl[i,j,vars.ip])
            Qp_r = (V_yr[i,j,vars.irho], V_yr[i,j,vars.iu], V_yr[i,j,vars.iv], h_yr[i,j], V_yr[i,j,vars.ip])

            (U_yl.d[i,j,vars.iD], U_yl.d[i,j,vars.iSx], U_yl.d[i,j,vars.iSy], U_yl.d[i,j,vars.itau]) = prim_to_cons(Qp_l, c, gamma)

            (U_yr.d[i,j,vars.iD], U_yr.d[i,j,vars.iSx], U_yr.d[i,j,vars.iSy], U_yr.d[i,j,vars.itau]) = prim_to_cons(Qp_r, c, gamma)
    """

    # construct primitives
    for i in range(myg.qx):
        for j in range(myg.qy):
            Qc_xl = (U_xl.d[i,j,vars.iD], U_xl.d[i,j,vars.iSx], U_xl.d[i,j,vars.iSy], U_xl.d[i,j,vars.itau])

            Qc_xr = (U_xr.d[i,j,vars.iD], U_xr.d[i,j,vars.iSx], U_xr.d[i,j,vars.iSy], U_xr.d[i,j,vars.itau])

            Qc_yl = (U_yl.d[i,j,vars.iD], U_yl.d[i,j,vars.iSx], U_yl.d[i,j,vars.iSy], U_yl.d[i,j,vars.itau])

            Qc_yr = (U_yr.d[i,j,vars.iD], U_yr.d[i,j,vars.iSx], U_yr.d[i,j,vars.iSy], U_yr.d[i,j,vars.itau])



            Qp_xl, _ = cons_to_prim(Qc_xl, c, gamma)
            (V_xl.d[i,j,vars.irho], V_xl.d[i,j,vars.iu], V_xl.d[i,j,vars.iv], _, V_xl.d[i,j,vars.ip]) = Qp_xl
            Qp_yl,_ = cons_to_prim(Qc_yl, c, gamma)
            (V_yl.d[i,j,vars.irho], V_yl.d[i,j,vars.iu], V_yl.d[i,j,vars.iv],_, V_yl.d[i,j,vars.ip]) = Qp_yl

            Qp_yr,_ = cons_to_prim(Qc_yr, c, gamma)
            (V_yr.d[i,j,vars.irho], V_yr.d[i,j,vars.iu], V_yr.d[i,j,vars.iv],_, V_yr.d[i,j,vars.ip]) = Qp_yr
            Qp_xr,_ = cons_to_prim(Qc_xr, c, gamma)
            (V_xr.d[i,j,vars.irho], V_xr.d[i,j,vars.iu], V_xr.d[i,j,vars.iv],_, V_xr.d[i,j,vars.ip]) = Qp_xr

    #=========================================================================
    # apply source terms (zero for now)
    #=========================================================================
    #grav = rp.get_param("compressible-gr.grav")

    # Sy_xl[i,j] += 0.5*dt*D[i-1,j]*grav
    #U_xl.v(buf=1, n=vars.iSy)[:,:] += 0.5 * dt * D.ip(-1, buf=1) * grav
    #U_xl.v(buf=1, n=vars.itau)[:,:] += 0.5*dt*Sy.ip(-1, buf=1)*grav

    # Sy_xr[i,j] += 0.5*dt*D[i,j]*grav
    #U_xr.v(buf=1, n=vars.iSy)[:,:] += 0.5*dt*D.v(buf=1)*grav
    #U_xr.v(buf=1, n=vars.itau)[:,:] += 0.5*dt*Sy.v(buf=1)*grav

    # Sy_yl[i,j] += 0.5*dt*D[i,j-1]*grav
    #U_yl.v(buf=1, n=vars.iSy)[:,:] += 0.5*dt*D.jp(-1, buf=1)*grav
    #U_yl.v(buf=1, n=vars.itau)[:,:] += 0.5*dt*Sy.jp(-1, buf=1)*grav

    # Sy_yr[i,j] += 0.5*dt*D[i,j]*grav
    #U_yr.v(buf=1, n=vars.iSy)[:,:] += 0.5*dt*D.v(buf=1)*grav
    #U_yr.v(buf=1, n=vars.itau)[:,:] += 0.5*dt*Sy.v(buf=1)*grav


    #=========================================================================
    # compute transverse fluxes
    #=========================================================================
    tm_riem = tc.timer("riemann")
    tm_riem.begin()

    riemann = rp.get_param("compressible-gr.riemann")

    if riemann == "RHLLC":
        riemannFunc = interface_f.riemann_rhllc
    elif riemann == "RHLLE":
        riemannFunc = interface_f.riemann_rhlle
    else:
        msg.fail("ERROR: Riemann solver undefined")

    _fx = riemannFunc(1, myg.qx, myg.qy, myg.ng,
                      vars.nvar, vars.iD, vars.iSx, vars.iSy, vars.itau,
                      gamma, c, U_xl.d, U_xr.d, V_xl.d, V_xr.d)

    _fy = riemannFunc(2, myg.qx, myg.qy, myg.ng,
                      vars.nvar, vars.iD, vars.iSx, vars.iSy, vars.itau,
                      gamma, c, U_yl.d, U_yr.d, V_yl.d, V_yr.d)

    F_x = patch.ArrayIndexer(d=_fx, grid=myg)
    F_y = patch.ArrayIndexer(d=_fy, grid=myg)

    # FIXME: turning off transverse for now
    #F_x.d[:,:] = 0.
    #F_y.d[:,:] = 0.

    tm_riem.end()

    #=========================================================================
    # construct the interface values of U now
    #=========================================================================

    """
    finally, we can construct the state perpendicular to the interface
    by adding the central difference part to the trasverse flux
    difference.

    The states that we represent by indices i,j are shown below
    (1,2,3,4):


      j+3/2--+----------+----------+----------+
             |          |          |          |
             |          |          |          |
        j+1 -+          |          |          |
             |          |          |          |
             |          |          |          |    1: U_xl[i,j,:] = U
      j+1/2--+----------XXXXXXXXXXXX----------+                      i-1/2,j,L
             |          X          X          |
             |          X          X          |
          j -+        1 X 2        X          |    2: U_xr[i,j,:] = U
             |          X          X          |                      i-1/2,j,R
             |          X    4     X          |
      j-1/2--+----------XXXXXXXXXXXX----------+
             |          |    3     |          |    3: U_yl[i,j,:] = U
             |          |          |          |                      i,j-1/2,L
        j-1 -+          |          |          |
             |          |          |          |
             |          |          |          |    4: U_yr[i,j,:] = U
      j-3/2--+----------+----------+----------+                      i,j-1/2,R
             |    |     |    |     |    |     |
                 i-1         i         i+1
           i-3/2      i-1/2      i+1/2      i+3/2


    remember that the fluxes are stored on the left edge, so

    F_x[i,j,:] = F_x
                    i-1/2, j

    F_y[i,j,:] = F_y
                    i, j-1/2

    """

    tm_transverse = tc.timer("transverse flux addition")
    tm_transverse.begin()

    dtdx = dt/myg.dx
    dtdy = dt/myg.dy

    b = (2,1)

    for n in range(vars.nvar):

        # U_xl[i,j,:] = U_xl[i,j,:] - 0.5*dt/dy * (F_y[i-1,j+1,:] - F_y[i-1,j,:])
        U_xl.v(buf=b, n=n)[:,:] += \
            - 0.5 * dtdy * (F_y.ip_jp(-1, 1, buf=b, n=n) - F_y.ip(-1, buf=b, n=n))

        # U_xr[i,j,:] = U_xr[i,j,:] - 0.5*dt/dy * (F_y[i,j+1,:] - F_y[i,j,:])
        U_xr.v(buf=b, n=n)[:,:] += \
            - 0.5 * dtdy * (F_y.jp(1, buf=b, n=n) - F_y.v(buf=b, n=n))

        # U_yl[i,j,:] = U_yl[i,j,:] - 0.5*dt/dx * (F_x[i+1,j-1,:] - F_x[i,j-1,:])
        U_yl.v(buf=b, n=n)[:,:] += \
            - 0.5 * dtdx * (F_x.ip_jp(1, -1, buf=b, n=n) - F_x.jp(-1, buf=b, n=n))

        # U_yr[i,j,:] = U_yr[i,j,:] - 0.5*dt/dx * (F_x[i+1,j,:] - F_x[i,j,:])
        U_yr.v(buf=b, n=n)[:,:] += \
            - 0.5 * dtdx * (F_x.ip(1, buf=b, n=n) - F_x.v(buf=b, n=n))

    tm_transverse.end()


    #=========================================================================
    # construct the fluxes normal to the interfaces
    #=========================================================================

    # up until now, F_x and F_y stored the transverse fluxes, now we
    # overwrite with the fluxes normal to the interfaces

    # construct primitives
    for i in range(myg.qx):
        for j in range(myg.qy):
            Qc_xl = (U_xl.d[i,j,vars.iD], U_xl.d[i,j,vars.iSx], U_xl.d[i,j,vars.iSy], U_xl.d[i,j,vars.itau])

            Qc_xr = (U_xr.d[i,j,vars.iD], U_xr.d[i,j,vars.iSx], U_xr.d[i,j,vars.iSy], U_xr.d[i,j,vars.itau])

            Qc_yl = (U_yl.d[i,j,vars.iD], U_yl.d[i,j,vars.iSx], U_yl.d[i,j,vars.iSy], U_yl.d[i,j,vars.itau])

            Qc_yr = (U_yr.d[i,j,vars.iD], U_yr.d[i,j,vars.iSx], U_yr.d[i,j,vars.iSy], U_yr.d[i,j,vars.itau])

            Qp_xl, _ = cons_to_prim(Qc_xl, c, gamma)
            (V_xl.d[i,j,vars.irho], V_xl.d[i,j,vars.iu], V_xl.d[i,j,vars.iv], _, V_xl.d[i,j,vars.ip]) = Qp_xl
            Qp_yl,_ = cons_to_prim(Qc_yl, c, gamma)
            (V_yl.d[i,j,vars.irho], V_yl.d[i,j,vars.iu], V_yl.d[i,j,vars.iv],_, V_yl.d[i,j,vars.ip]) = Qp_yl

            Qp_yr,_ = cons_to_prim(Qc_yr, c, gamma)
            (V_yr.d[i,j,vars.irho], V_yr.d[i,j,vars.iu], V_yr.d[i,j,vars.iv],_, V_yr.d[i,j,vars.ip]) = Qp_yr
            Qp_xr,_ = cons_to_prim(Qc_xr, c, gamma)
            (V_xr.d[i,j,vars.irho], V_xr.d[i,j,vars.iu], V_xr.d[i,j,vars.iv],_, V_xr.d[i,j,vars.ip]) = Qp_xr

    tm_riem.begin()

    _fx = riemannFunc(1, myg.qx, myg.qy, myg.ng,
                      vars.nvar, vars.iD, vars.iSx, vars.iSy, vars.itau,
                      gamma, c, U_xl.d, U_xr.d, V_xl.d, V_xr.d)

    _fy = riemannFunc(2, myg.qx, myg.qy, myg.ng,
                      vars.nvar, vars.iD, vars.iSx, vars.iSy, vars.itau,
                      gamma, c, U_yl.d, U_yr.d, V_yl.d, V_yr.d)

    F_x = patch.ArrayIndexer(d=_fx, grid=myg)
    F_y = patch.ArrayIndexer(d=_fy, grid=myg)

    tm_riem.end()

    #=========================================================================
    # apply artificial viscosity
    #=========================================================================

    cvisc = rp.get_param("compressible-gr.cvisc")

    _ax, _ay = interface_f.artificial_viscosity(
        myg.qx, myg.qy, myg.ng, myg.dx, myg.dy,
        cvisc, u.d, v.d)

    avisco_x = patch.ArrayIndexer(d=_ax, grid=myg)
    avisco_y = patch.ArrayIndexer(d=_ay, grid=myg)


    b = (2,1)

    # F_x = F_x + avisco_x * (U(i-1,j) - U(i,j))
    F_x.v(buf=b, n=vars.iD)[:,:] += \
        avisco_x.v(buf=b) * (D.ip(-1, buf=b) - D.v(buf=b))

    F_x.v(buf=b, n=vars.iSx)[:,:] += \
        avisco_x.v(buf=b) * (Sx.ip(-1, buf=b) - Sx.v(buf=b))

    F_x.v(buf=b, n=vars.iSy)[:,:] += \
        avisco_x.v(buf=b) * (Sy.ip(-1, buf=b) - Sy.v(buf=b))

    F_x.v(buf=b, n=vars.itau)[:,:] += \
        avisco_x.v(buf=b) * (tau.ip(-1, buf=b) - tau.v(buf=b))

    # F_y = F_y + avisco_y * (U(i,j-1) - U(i,j))
    F_y.v(buf=b, n=vars.iD)[:,:] += \
        avisco_y.v(buf=b) * (D.jp(-1, buf=b) - D.v(buf=b))

    F_y.v(buf=b, n=vars.iSx)[:,:] += \
        avisco_y.v(buf=b) * (Sx.jp(-1, buf=b) - Sx.v(buf=b))

    F_y.v(buf=b, n=vars.iSy)[:,:] += \
        avisco_y.v(buf=b) * (Sy.jp(-1, buf=b) - Sy.v(buf=b))

    F_y.v(buf=b, n=vars.itau)[:,:] += \
        avisco_y.v(buf=b) * (tau.jp(-1, buf=b) - tau.v(buf=b))


    tm_flux.end()

    return F_x, F_y

def cons_to_prim(Q, c, gamma):
    """
    Converts the given set Q of conservative variables into a set Qp of primitive variables.
    """
    names = ['D', 'Sx', 'Sy', 'tau']
    nan_check(Q, names)
    (D, Sx, Sy, tau) = Q

    pmin = (Sx**2 + Sy**2)/c**2 - tau - D
    pmax = (gamma - 1.) * tau

    if pmax < 0.:
        pmax = np.fabs(pmax)

    if pmin > pmax:
        pmin = abs(np.sqrt(Sx**2 + Sy**2)/c - tau - D)

    if pmin < 0. or root_find_on_me1(pmin, Q, c, gamma) < 0. or math.isnan(pmin):
        pmin = 0.

    if math.isnan(pmax) or pmax == 0.:
        pmax = D**2 *c

    pbar = 0.5 * (pmin + pmax)
    # try catch block on root finder in case pmin, pmax unsuitable
    try:
        p = brentq(root_find_on_me1, pmin, pmax, args=(Q, c, gamma))
    except ValueError:
        print('pmin: ', pmin, '    f(pmin): ', root_find_on_me1(pmin, Q, c, gamma),'   pmax: ', pmax, '   f(pmax): ', root_find_on_me1(pmax, Q, c, gamma))
        pmin = pbar
        pmax *= 2.
        p = brentq(root_find_on_me1, pmin, pmax, args=(Q, c, gamma))

    u = Sx / (tau + D + p)
    v = Sy / (tau + D + p)
    w = W(u, v, c)
    v2 = (u**2 + v**2) / c**2

    if v2 > 1.:
        print('something is wrong here?')

    rho = D / w
    eps = (tau + D * (1. - w) + p * v2 / (v2 - 1.)) / (w**2 * rho)
    h = 1. + eps + p / rho
    c_s = np.sqrt(gamma * (gamma - 1.) * eps / (1. + gamma * eps))

    Qp = (rho, u, v, h, p)

    return Qp, c_s

def prim_to_cons(Q, c, gamma):
    """
    Converts the given set Q of primitive variables into a set Qc of conservative variables.
    """

    rho, u, v, h, p = Q

    w = W(u, v, c)

    D = rho * w
    Sx = rho * h * w**2 * u
    Sy = rho * h * w**2 * v
    tau = rho * h * w**2 - p - D

    Qc = (D, Sx, Sy, tau)

    return Qc

def W(u, v, c):
    """
    Lorentz factor
    """
    W = 1. - (u**2 + v**2)/c**2

    if W <= 0.:
        print("Oops, Lorentz factor is imaginary, with denominator squared of {}".format(W))
        print("u: {},  v: {}".format(u, v))
        raise ValueError
        return 1.e-10
    return 1. / np.sqrt(W)

def root_find_on_me1(p, Q, c, gamma):
    """
    Equation to root find on in order to find the primitive pressure.
    """

    D, Sx, Sy, tau = Q

    pbar  = p
    if pbar > 0.:
        pressure_local = pbar
        u_local = Sx / (tau + D + pbar)
        v_local = Sy / (tau + D + pbar)

        v2 = (u_local**2 + v_local**2) / c**2
        w = W(u_local, v_local, c)
        epsrho = (tau + D * (1. - w) + pbar * v2 / (v2 - 1.)) / w**2

        p_error = (gamma - 1.) * epsrho - pbar
    else:
        p_error = 1.e6

    return p_error


def root_find_on_me2(p, Q, c, gamma):
    """
    Equation to root find on in order to find the primitive pressure.
    """

    D, Sx, Sy, tau = Q
    p_bar = p

    if pbar > 0.:
        p_local = p_bar
        u_local = Sx / (tau + D + pbar)
        v_local = Sy / (tau + D + pbar)

        v2 = (u_local**2 + v_local**2) / c**2

        if v2 < 1.:
            W = W(u_local, v_local, c)
            epsrho = (tau + D * (1. - W) + pbar * v2 / (v2 - 1.)) / W**2

            rho = D / W
            h = 1. + pbar / (rho * (gamma - 1.)) + pbar / rho

            f = (gamma - 1.) * epsrho - pbar
            #df = v2 * ((gamma * pbar) / (rho * h)) - 1.

        else:
            f = 1.e2
            #df = 1.
    else:
        f = 1.e2
        #df = 1.

    return f

def h_from_eos(rho, gamma, K):
    """
    return h using the equation of state, given the density, ratio of specific heats, gamma, and the polytropic index, K.
    """
    p = p_from_eos(rho, gamma, K)
    e = p / (gamma - 1.) / rho
    return 1. + e + p / rho

def p_from_eos(rho, gamma, K):
    """
    return p using the equation of state, given the density, ratio of specific heats, gamma, and the polytropic index, K.
    """
    return K * rho**gamma

def rel_add_velocity(ux, uy, vx, vy, c):
    """
    Relativistic addition of velocities.
    """
    Wu = W(ux, uy, c)

    denom = (1. + (ux * vx + uy * vy) / c**2)

    upv_x = (ux + vx/Wu + (Wu * (ux * vx + uy * vy) * ux)/ (c**2 *(1. + Wu))) / denom
    upv_y = (uy + vy/Wu + (Wu * (ux * vx + uy * vy) * uy)/ (c**2 *(1. + Wu))) / denom

    return upv_x, upv_y

def nan_check(Q, names):
    for (q, n) in zip(Q, names):
        if math.isnan(q):
            print("NAN FOUND!!! {} is {}".format(n, q))
