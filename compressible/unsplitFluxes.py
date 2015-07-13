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

import compressible.eos as eos
import compressible.interface_f as interface_f
import mesh.reconstruction_f as reconstruction_f
import mesh.patch as patch

from util import msg
import pylsmlib


def burning(my_data, rp, vars, dt):
    """
    Carry out burning reactions at front over timestep dt.

    Parameters
    ----------
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    vars : Variables object
        The Variables object that tells us which indices refer to which
    dt : float
        The timestep we are advancing through.
    """
    pass


def calcSL(my_data, rp, vars):
    """
    Find the laminar burning speed

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

    Returns
    -------
    sL : ndarray
        The laminar flame speed
    """
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")
    phi  = my_data.get_var("phi")

    sL = np.zeros_like(xmom)
    return sL



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

    #=========================================================================
    # compute the primitive variables
    #=========================================================================
    # Q = (rho, u, v, p, phi)

    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")
    phi = my_data.get_var("phi")

    r = dens

    # get the velocities
    u = xmom/dens
    v = ymom/dens

    # get the pressure
    e = (ener - 0.5*(xmom**2 + ymom**2)/dens)/dens

    p = eos.pres(gamma, dens, e)

    smallp = 1.e-10
    p.d = p.d.clip(smallp)   # apply a floor to the pressure


    #=========================================================================
    # compute the flattening coefficients
    #=========================================================================

    # there is a single flattening coefficient (xi) for all directions
    use_flattening = rp.get_param("compressible.use_flattening")

    if use_flattening:
        delta = rp.get_param("compressible.delta")
        z0 = rp.get_param("compressible.z0")
        z1 = rp.get_param("compressible.z1")

        xi_x = reconstruction_f.flatten(1, p.d, u.d, myg.qx, myg.qy, myg.ng, smallp, delta, z0, z1)
        xi_y = reconstruction_f.flatten(2, p.d, v.d, myg.qx, myg.qy, myg.ng, smallp, delta, z0, z1)

        xi = reconstruction_f.flatten_multid(xi_x, xi_y, p.d, myg.qx, myg.qy, myg.ng)
    else:
        xi = 1.0



    # monotonized central differences in x-direction
    tm_limit = tc.timer("limiting")
    tm_limit.begin()

    limiter = rp.get_param("compressible.limiter")
    if limiter == 0:
        limitFunc = reconstruction_f.nolimit
    elif limiter == 1:
        limitFunc = reconstruction_f.limit2
    else:
        limitFunc = reconstruction_f.limit4

    ldelta_rx = xi*limitFunc(1, r.d, myg.qx, myg.qy, myg.ng)
    ldelta_ux = xi*limitFunc(1, u.d, myg.qx, myg.qy, myg.ng)
    ldelta_vx = xi*limitFunc(1, v.d, myg.qx, myg.qy, myg.ng)
    ldelta_px = xi*limitFunc(1, p.d, myg.qx, myg.qy, myg.ng)
    ldelta_phix = xi*limitFunc(1, phi.d, myg.qx, myg.qy, myg.ng)

    # monotonized central differences in y-direction
    ldelta_ry = xi*limitFunc(2, r.d, myg.qx, myg.qy, myg.ng)
    ldelta_uy = xi*limitFunc(2, u.d, myg.qx, myg.qy, myg.ng)
    ldelta_vy = xi*limitFunc(2, v.d, myg.qx, myg.qy, myg.ng)
    ldelta_py = xi*limitFunc(2, p.d, myg.qx, myg.qy, myg.ng)
    ldelta_phiy = xi*limitFunc(2, phi.d, myg.qx, myg.qy, myg.ng)

    tm_limit.end()



    #=========================================================================
    # x-direction
    #=========================================================================


    # left and right primitive variable states
    tm_states = tc.timer("interfaceStates")
    tm_states.begin()

    V_l, V_r = interface_f.states(1, myg.qx, myg.qy, myg.ng, myg.dx, dt,
                                  vars.nvar,
                                  gamma,
                                  r.d, u.d, v.d, p.d, phi.d,
                                  ldelta_rx, ldelta_ux, ldelta_vx, ldelta_px,
                                  ldelta_phix)

    tm_states.end()


    # transform interface states back into conserved variables
    U_xl = myg.scratch_array(vars.nvar)
    U_xr = myg.scratch_array(vars.nvar)

    U_xl.d[:,:,vars.idens] = V_l[:,:,vars.irho]
    U_xl.d[:,:,vars.ixmom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iu]
    U_xl.d[:,:,vars.iymom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iv]
    U_xl.d[:,:,vars.iener] = eos.rhoe(gamma, V_l[:,:,vars.ip]) + \
        0.5*V_l[:,:,vars.irho]*(V_l[:,:,vars.iu]**2 + V_l[:,:,vars.iv]**2)
    U_xl[:,:,vars.iphi]  = V_l[:,:,vars.iphi]

    U_xr.d[:,:,vars.idens] = V_r[:,:,vars.irho]
    U_xr.d[:,:,vars.ixmom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iu]
    U_xr.d[:,:,vars.iymom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iv]
    U_xr.d[:,:,vars.iener] = eos.rhoe(gamma, V_r[:,:,vars.ip]) + \
        0.5*V_r[:,:,vars.irho]*(V_r[:,:,vars.iu]**2 + V_r[:,:,vars.iv]**2)
    U_xr[:,:,vars.iphi]  = V_r[:,:,vars.iphi]



    #=========================================================================
    # y-direction
    #=========================================================================


    # left and right primitive variable states
    tm_states.begin()

    V_l, V_r = interface_f.states(2, myg.qx, myg.qy, myg.ng, myg.dy, dt,
                                  vars.nvar,
                                  gamma,
                                  r.d, u.d, v.d, p.d, phi.d,
                                  ldelta_ry, ldelta_uy, ldelta_vy, ldelta_py,
                                  ldelta_phiy)

    tm_states.end()


    # transform interface states back into conserved variables
    U_yl = myg.scratch_array(vars.nvar)
    U_yr = myg.scratch_array(vars.nvar)

    U_yl.d[:,:,vars.idens] = V_l[:,:,vars.irho]
    U_yl.d[:,:,vars.ixmom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iu]
    U_yl.d[:,:,vars.iymom] = V_l[:,:,vars.irho]*V_l[:,:,vars.iv]
    U_yl.d[:,:,vars.iener] = eos.rhoe(gamma, V_l[:,:,vars.ip]) + \
        0.5*V_l[:,:,vars.irho]*(V_l[:,:,vars.iu]**2 + V_l[:,:,vars.iv]**2)
    U_yl[:,:,vars.iphi]  = V_l[:,:,vars.iphi]

    U_yr.d[:,:,vars.idens] = V_r[:,:,vars.irho]
    U_yr.d[:,:,vars.ixmom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iu]
    U_yr.d[:,:,vars.iymom] = V_r[:,:,vars.irho]*V_r[:,:,vars.iv]
    U_yr.d[:,:,vars.iener] = eos.rhoe(gamma, V_r[:,:,vars.ip]) + \
        0.5*V_r[:,:,vars.irho]*(V_r[:,:,vars.iu]**2 + V_r[:,:,vars.iv]**2)
    U_yr[:,:,vars.iphi]  = V_r[:,:,vars.iphi]


    #=========================================================================
    # apply source terms
    #=========================================================================
    grav = rp.get_param("compressible.grav")

    # ymom_xl[i,j] += 0.5*dt*dens[i-1,j]*grav
    U_xl.v(buf=1, n=vars.iymom)[:,:] += 0.5*dt*dens.ip(-1, buf=1)*grav
    U_xl.v(buf=1, n=vars.iener)[:,:] += 0.5*dt*ymom.ip(-1, buf=1)*grav

    # ymom_xr[i,j] += 0.5*dt*dens[i,j]*grav
    U_xr.v(buf=1, n=vars.iymom)[:,:] += 0.5*dt*dens.v(buf=1)*grav
    U_xr.v(buf=1, n=vars.iener)[:,:] += 0.5*dt*ymom.v(buf=1)*grav

    # ymom_yl[i,j] += 0.5*dt*dens[i,j-1]*grav
    U_yl.v(buf=1, n=vars.iymom)[:,:] += 0.5*dt*dens.jp(-1, buf=1)*grav
    U_yl.v(buf=1, n=vars.iener)[:,:] += 0.5*dt*ymom.jp(-1, buf=1)*grav

    # ymom_yr[i,j] += 0.5*dt*dens[i,j]*grav
    U_yr.v(buf=1, n=vars.iymom)[:,:] += 0.5*dt*dens.v(buf=1)*grav
    U_yr.v(buf=1, n=vars.iener)[:,:] += 0.5*dt*ymom.v(buf=1)*grav


    #=========================================================================
    # compute transverse fluxes
    #=========================================================================
    tm_riem = tc.timer("riemann")
    tm_riem.begin()

    riemann = rp.get_param("compressible.riemann")

    if riemann == "HLLC":
        riemannFunc = interface_f.riemann_hllc
    elif riemann == "CGF":
        riemannFunc = interface_f.riemann_cgf
    else:
        msg.fail("ERROR: Riemann solver undefined")


    _fx = riemannFunc(1, myg.qx, myg.qy, myg.ng,
                      vars.nvar, vars.idens, vars.ixmom, vars.iymom, vars.iener,
                      vars.iphi, gamma, U_xl.d, U_xr.d)

    _fy = riemannFunc(2, myg.qx, myg.qy, myg.ng,
                      vars.nvar, vars.idens, vars.ixmom, vars.iymom, vars.iener,
                      vars.iphi, gamma, U_yl.d, U_yr.d)

    F_x = patch.ArrayIndexer(d=_fx, grid=myg)
    F_y = patch.ArrayIndexer(d=_fy, grid=myg)

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
            - 0.5*dtdy*(F_y.ip_jp(-1, 1, buf=b, n=n) - F_y.ip(-1, buf=b, n=n))

        # U_xr[i,j,:] = U_xr[i,j,:] - 0.5*dt/dy * (F_y[i,j+1,:] - F_y[i,j,:])
        U_xr.v(buf=b, n=n)[:,:] += \
            - 0.5*dtdy*(F_y.jp(1, buf=b, n=n) - F_y.v(buf=b, n=n))

        # U_yl[i,j,:] = U_yl[i,j,:] - 0.5*dt/dx * (F_x[i+1,j-1,:] - F_x[i,j-1,:])
        U_yl.v(buf=b, n=n)[:,:] += \
            - 0.5*dtdx*(F_x.ip_jp(1, -1, buf=b, n=n) - F_x.jp(-1, buf=b, n=n))

        # U_yr[i,j,:] = U_yr[i,j,:] - 0.5*dt/dx * (F_x[i+1,j,:] - F_x[i,j,:])
        U_yr.v(buf=b, n=n)[:,:] += \
            - 0.5*dtdx*(F_x.ip(1, buf=b, n=n) - F_x.v(buf=b, n=n))

    tm_transverse.end()


    #=========================================================================
    # construct the fluxes normal to the interfaces
    #=========================================================================

    # up until now, F_x and F_y stored the transverse fluxes, now we
    # overwrite with the fluxes normal to the interfaces

    tm_riem.begin()

    _fx = riemannFunc(1, myg.qx, myg.qy, myg.ng,
                      vars.nvar, vars.idens, vars.ixmom, vars.iymom, vars.iener,
                      vars.iphi, gamma, U_xl.d, U_xr.d)

    _fy = riemannFunc(2, myg.qx, myg.qy, myg.ng,
                      vars.nvar, vars.idens, vars.ixmom, vars.iymom, vars.iener,
                      vars.iphi, gamma, U_yl.d, U_yr.d)

    F_x = patch.ArrayIndexer(d=_fx, grid=myg)
    F_y = patch.ArrayIndexer(d=_fy, grid=myg)

    tm_riem.end()

    #=========================================================================
    # apply artificial viscosity
    #=========================================================================
    cvisc = rp.get_param("compressible.cvisc")

    _ax, _ay = interface_f.artificial_viscosity(
        myg.qx, myg.qy, myg.ng, myg.dx, myg.dy,
        cvisc, u.d, v.d)

    avisco_x = patch.ArrayIndexer(d=_ax, grid=myg)
    avisco_y = patch.ArrayIndexer(d=_ay, grid=myg)


    b = (2,1)

    # F_x = F_x + avisco_x * (U(i-1,j) - U(i,j))
    F_x.v(buf=b, n=vars.idens)[:,:] += \
        avisco_x.v(buf=b)*(dens.ip(-1, buf=b) - dens.v(buf=b))

    F_x.v(buf=b, n=vars.ixmom)[:,:] += \
        avisco_x.v(buf=b)*(xmom.ip(-1, buf=b) - xmom.v(buf=b))

    F_x.v(buf=b, n=vars.iymom)[:,:] += \
        avisco_x.v(buf=b)*(ymom.ip(-1, buf=b) - ymom.v(buf=b))

    F_x.v(buf=b, n=vars.iener)[:,:] += \
        avisco_x.v(buf=b)*(ener.ip(-1, buf=b) - ener.v(buf=b))

    F_x.v(buf=b, n=vars.iphi)[:,:] += \
        avisco_x.v(buf=b)*(phi.ip(-1, buf=b) - phi.v(buf=b))

    # F_y = F_y + avisco_y * (U(i,j-1) - U(i,j))
    F_y.v(buf=b, n=vars.idens)[:,:] += \
        avisco_y.v(buf=b)*(dens.jp(-1, buf=b) - dens.v(buf=b))

    F_y.v(buf=b, n=vars.ixmom)[:,:] += \
        avisco_y.v(buf=b)*(xmom.jp(-1, buf=b) - xmom.v(buf=b))

    F_y.v(buf=b, n=vars.iymom)[:,:] += \
        avisco_y.v(buf=b)*(ymom.jp(-1, buf=b) - ymom.v(buf=b))

    F_y.v(buf=b, n=vars.iener)[:,:] += \
        avisco_y.v(buf=b)*(ener.jp(-1, buf=b) - ener.v(buf=b))

    F_y.v(buf=b, n=vars.iphi)[:,:] += \
        avisco_y.v(buf=b)*(phi.jp(-1, buf=b) - phi.v(buf=b)) 

    tm_flux.end()

    return F_x, F_y
