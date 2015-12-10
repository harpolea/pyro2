subroutine states(idir, qx, qy, ng, dx, dt, &
                  nvar, &
                  gamma, c, &
                  r, u, v, p, &
                  ldelta_r, ldelta_u, ldelta_v, ldelta_p, &
                  q_l, q_r)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dt
  integer, intent(in) :: nvar
  double precision, intent(in) :: gamma, c

  ! 0-based indexing to match python
  double precision, intent(inout) :: r(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: u(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: v(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: p(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: ldelta_r(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_u(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_v(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_p(0:qx-1, 0:qy-1)

  double precision, intent(  out) :: q_l(0:qx-1, 0:qy-1, 0:nvar-1)
  double precision, intent(  out) :: q_r(0:qx-1, 0:qy-1, 0:nvar-1)

!f2py depend(qx, qy) :: r, u, v, p
!f2py depend(qx, qy) :: ldelta_r, ldelta_u, ldelta_v, ldelta_p
!f2py depend(qx, qy, nvar) :: q_l, q_r
!f2py intent(in) :: r, u, v, p
!f2py intent(in) :: ldelta_r, ldelta_u, ldelta_v, ldelta_p
!f2py intent(out) :: q_l, q_r



  ! predict the cell-centered state to the edges in one-dimension
  ! using the reconstructed, limited slopes.
  !
  ! We follow the convention here that V_l[i] is the left state at the
  ! i-1/2 interface and V_l[i+1] is the left state at the i+1/2
  ! interface.
  !
  !
  ! We need the left and right eigenvectors and the eigenvalues for
  ! the system projected along the x-direction
  !
  ! Taking our state vector as Q = (D, Sx, Sy, tau)^T, the eigenvalues
  ! are l0, l+, l-
  !
  ! We look at the equations of hydrodynamics in a split fashion --
  ! i.e., we only consider one dimension at a time.
  !
  ! Considering advection in the x-direction, the Jacobian matrix for
  ! the primitive variable formulation of the Euler equations
  ! projected in the x-direction is horrid.
  !
  ! The right eigenvectors are also horrid.
  !
  ! The left eigenvectors are again, horrid.

  ! The fluxes are going to be defined on the left edge of the
  ! computational zones
  !
  !           |             |             |             |
  !           |             |             |             |
  !          -+------+------+------+------+------+------+--
  !           |     i-1     |      i      |     i+1     |
  !                        ^ ^           ^
  !                    q_l,i q_r,i  q_l,i+1
  !
  ! q_r,i and q_l,i+1 are computed using the information in zone i,j.

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j, m

  double precision :: dq(0:nvar-1), q(0:nvar-1)
  double precision :: lvec(0:nvar-1,0:nvar-1), rvec(0:nvar-1,0:nvar-1)
  double precision :: eval(0:nvar-1)
  double precision :: betal(0:nvar-1), betar(0:nvar-1)

  double precision :: dtdx, dtdx4
  double precision :: cs, eps, v2, a_minus, a_plus, Kappa, h2OverDelta, W
  double precision :: h, uu, vv

  double precision :: sum, sum_l, sum_r, factor

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  dtdx = dt/dx
  dtdx4 = 0.25d0 * dtdx

  ! this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
  do j = jlo-2, jhi+2
     do i = ilo-2, ihi+2

        dq(:) = [ldelta_r(i,j), &
                 ldelta_u(i,j), &
                 ldelta_v(i,j), &
                 ldelta_p(i,j)]

        q(:) = [r(i,j), u(i,j), v(i,j), p(i,j)]

        eps = p(i,j) / (r(i,j) * (gamma - 1.0d0))
        h = 1.0d0 + eps + p(i,j)/r(i,j)

        cs = sqrt(gamma*(gamma - 1.0d0)*eps/(1.0d0 + gamma*eps))
        ! divide stuff by c to convert to natural units
        v2 = (v(i,j)**2 + u(i,j)**2)/c**2
        W = 1.0d0 / sqrt(1.0d0 - v2)
        uu = u(i,j)/c
        vv = v(i,j)/c
        cs = cs / c

        ! compute the eigenvalues and eigenvectors
        if (idir == 1) then
           eval(0) = (uu * (1.0d0 - cs**2) - cs * &
                sqrt((1.0d0 - v2)*(1.0d0 - uu**2 - &
                vv**2*cs**2))) / (1.0d0 - v2*cs**2)
           eval(1:2) = uu
           eval(3) = (uu * (1.0d0 - cs**2) + cs * &
                sqrt((1.0d0 - v2)*(1.0d0 - uu**2 - &
                vv**2*cs**2))) / (1.0d0 - v2*cs**2)

           a_minus = (1.0d0 - uu**2) / (1.0d0 - uu * eval(0))
           a_plus  = (1.0d0 - uu**2) / (1.0d0 - uu * eval(3))
           Kappa = h ! ideal gas: K = h
           h2OverDelta = 1.0d0 / &
                (W * (Kappa - 1.0d0) * &
                  (1.0d0 - uu**2) * &
                  (a_plus * eval(3) - a_minus * eval(0)))

           lvec(0,0) = h2OverDelta * &
                (h*W*a_plus*(uu - eval(3)) - &
                  uu - &
                  W**2*vv**2*(2.0d0 * Kappa - 1.0d0) * &
                                (uu - a_plus * eval(3)) + &
                  Kappa * a_plus * eval(3) )
           lvec(0, 1) = h2OverDelta * &
               ( 1.0d0 - Kappa * a_plus + &
                 W**2 * vv**2 * (2.0d0 * Kappa - 1.0d0) * &
                               (1.0d0 - a_plus) )
           lvec(0, 2) = h2OverDelta * &
               (W**2 * vv * (2.0d0 * Kappa - 1.0d0) * &
                               a_plus * (uu - eval(3)) )
           lvec(0, 3) = h2OverDelta * &
               ( -uu + Kappa * a_plus * eval(3) - &
                 W**2 * vv**2 * (2.0d0 * Kappa - 1.0d0) * &
                               (uu - a_plus * eval(3)) )

           lvec(3, 0) =-h2OverDelta * &
                ( h * W * a_minus * (uu - eval(0)) - &
                  uu - &
                  W**2 * vv**2 * (2.0d0 * Kappa - 1.0d0) * &
                                (uu - a_minus * eval(0)) + &
                  Kappa * a_minus * eval(0) )
           lvec(3, 1) =-h2OverDelta * &
                ( 1.0d0 - Kappa * a_minus + &
                  W**2 * vv**2 * (2.0d0 * Kappa - 1.0d0) * &
                                (1.0d0 - a_minus) )
           lvec(3, 1) =-h2OverDelta * &
                ( W**2 * vv * (2.0d0 * Kappa - 1.0d0) * &
                                a_minus * (uu - eval(0)) )
           lvec(3, 2) =-h2OverDelta * &
                ( -uu + Kappa * a_minus * eval(0) - &
                  W**2 * vv**2 * (2.0d0 * Kappa - 1.0d0) * &
                                (uu - a_minus * eval(0)) )

           lvec(1, 0) = W * ( h - W ) / (Kappa - 1.0d0)
           lvec(1, 1) = W * ( W * uu ) / (Kappa - 1.0d0)
           lvec(1, 2) = W * ( W * vv ) / (Kappa - 1.0d0)
           lvec(1, 3) = -W**2 / (Kappa - 1.0d0)

           lvec(2, 0) = -vv / (h * (1.0d0 - uu**2))
           lvec(2, 1) = uu * vv / (h * (1.0d0 - uu**2))
           lvec(2, 2) = 1.0d0 / h
           lvec(2, 3) = lvec(2,0)

           rvec(0,0) = 1.0d0
           rvec(0,1) = h * W * eval(0) * a_minus
           rvec(0,2) = h * W * vv
           rvec(0,3) = h * W * a_minus - 1.0d0

           rvec(3,0) = 1.0d0
           rvec(3,1) = h * W * eval(3) * a_plus
           rvec(3,2) = h * W * vv
           rvec(3,3) = h * W * a_plus - 1.0d0

           rvec(1,0) = Kappa / (h * W)
           rvec(1,1) = uu
           rvec(1,2) = vv
           rvec(1,3) = 1.0d0 - rvec(1,0)

           rvec(2,0) = W * vv
           rvec(2,1) = 2.0d0 * h * W**2 * uu * vv
           rvec(2,2) = h * (1.0d0 + 2.0d0 * W**2 * vv**2)
           rvec(2,3) = 2.0d0 * h * W**2 * vv - W * vv

        else
            eval(0) = (vv * (1.0d0 - cs**2) - cs * &
                 sqrt((1.0d0 - v2)*(1.0d0 - vv**2 - &
                 uu**2 * cs**2))) / (1.0d0 - v2*cs**2)
            eval(1:2) = vv
            eval(3) = (vv * (1.0d0 - cs**2) + cs * &
                 sqrt((1.0d0 - v2)*(1.0d0 - vv**2 - &
                 uu**2 * cs**2))) / (1.0d0 - v2*cs**2)

            a_minus = (1.0d0 - vv**2) / (1.0d0 - vv * eval(0))
            a_plus  = (1.0d0- vv**2) / (1.0d0 - vv * eval(3))
            Kappa = h ! ideal gas: K = h
            h2OverDelta = 1.0d0 / &
                 (W * (Kappa - 1.0d0) * &
                   (1.0d0 - vv**2) * &
                   (a_plus * eval(3) - a_minus * eval(0)))

            lvec(0,0) = h2OverDelta * &
                 (h * W * a_plus * (vv - eval(3)) - vv - &
                   W**2 * uu**2 * (2.0d0 * Kappa - 1.0d0) * &
                                 (vv - a_plus * eval(3)) + &
                   Kappa*a_plus*eval(3) )
            lvec(0, 1) = h2OverDelta * &
                (W**2 * uu*(2.0d0 * Kappa - 1.0d0) * &
                                a_plus * (vv - eval(3)) )
            lvec(0, 2) = h2OverDelta * &
                ( 1.0d0 - Kappa * a_plus + &
                  W**2 * uu**2 * (2.0d0 * Kappa - 1.0d0)*&
                                (1.0d0 - a_plus) )
            lvec(0, 3) = h2OverDelta * &
                ( -vv + Kappa * a_plus * eval(3) - &
                  W**2 * uu**2 * (2.0d0 * Kappa - 1.0d0)*&
                                (vv - a_plus * eval(3)) )

            lvec(3, 0) =-h2OverDelta * &
                 ( h * W * a_minus * (vv - eval(0)) - &
                   vv - &
                   W**2 * uu**2 * (2.0d0 * Kappa - 1.0d0)*&
                                 (vv - a_minus * eval(0)) + &
                   Kappa * a_minus * eval(0) )
            lvec(3, 1) =-h2OverDelta * &
                 ( W**2 * uu * (2.0d0 * Kappa - 1.0d0)*&
                                 a_minus*(vv - eval(0)) )
            lvec(3, 1) =-h2OverDelta * &
                 ( 1.0d0 - Kappa * a_minus + &
                   W**2 *uu**2 * (2.0d0 * Kappa - 1.0d0) * &
                                 (1.0d0 - a_minus) )
            lvec(3, 2) =-h2OverDelta * &
                 ( -vv + Kappa * a_minus * eval(0) - &
                   W**2 * uu**2 * (2.0d0 * Kappa - 1.0d0) * &
                                 (vv - a_minus * eval(0)) )

            lvec(1, 0) = -uu / (h * (1.0d0 - vv**2))
            lvec(1, 1) = 1.0d0 / h / (1.0d0 - vv**2) * (1.0d0 - vv**2)
            lvec(1, 2) = (uu * vv) / (h * (1.0d0 - vv**2))
            lvec(1, 3) = -uu / (h * (1.0d0 - vv**2))

            lvec(2, 0) = W * ( h - W ) / (Kappa - 1.0d0)
            lvec(2, 1) = W * ( W * uu ) / (Kappa - 1.0d0)
            lvec(2, 2) = W * ( W * vv ) / (Kappa - 1.0d0)
            lvec(2, 3) = -W**2 / (Kappa - 1.0d0)


            rvec(0,0) = 1.0d0
            rvec(0,1) = h * W * uu
            rvec(0,2) = h * W * eval(0) * a_minus
            rvec(0,3) = h * W * a_minus - 1.0d0

            rvec(3,0) = 1.0d0
            rvec(3,1) = h * W * uu
            rvec(3,2) = h * W * eval(3) * a_plus
            rvec(3,3) = h * W * a_plus - 1.0d0

            rvec(1,0) = W * uu
            rvec(1,1) = h * (1.0d0 + 2.0d0 * W**2 * uu**2)
            rvec(1,2) = 2.0d0 * h * W**2 * uu * vv
            rvec(1,3) = 2.0d0 * h * W**2 * vv - W * vv

            rvec(2,0) = Kappa / (h * W)
            rvec(2,1) = uu
            rvec(2,2) = vv
            rvec(2,3) = 1.0d0 - Kappa / (h * W)

        endif


        ! define the reference states
        if (idir == 1) then
           ! this is one the right face of the current zone,
           ! so the fastest moving eigenvalue is eval[3] = u + c
           factor = 0.5d0 * (1.0d0 - dtdx * max(eval(3), 0.0d0))
           q_l(i+1,j,:) = q(:) + factor * dq(:)

           ! left face of the current zone, so the fastest moving
           ! eigenvalue is eval[0] = u - c
           factor = 0.5d0 * (1.0d0 + dtdx * min(eval(0), 0.0d0))
           q_r(i,  j,:) = q(:) - factor * dq(:)

        else

           factor = 0.5d0 * (1.0d0 - dtdx * max(eval(3), 0.0d0))
           q_l(i,j+1,:) = q(:) + factor * dq(:)

           factor = 0.5d0 * (1.0d0 + dtdx * min(eval(0), 0.0d0))
           q_r(i,j,  :) = q(:) - factor * dq(:)

        endif

        ! compute the Vhat functions
        betal(:) = 0.0d0
        betar(:) = 0.0d0
        do m = 0, 3
           sum = dot_product(lvec(m,:),dq(:))

           betal(m) = dtdx4 * (eval(3) - eval(m)) * (sign(1.0d0,eval(m)) + 1.0d0) * sum
           betar(m) = dtdx4 * (eval(0) - eval(m)) * (1.0d0 - sign(1.0d0,eval(m)))*sum
        enddo

        ! put factors of c back in
        eval(:) = eval(:) * c
        lvec(:,1:2) = lvec(:,1:2) * c
        rvec(:,1:2) = rvec(:,1:2) * c
        lvec(1:2,:) = lvec(1:2,:) * c
        rvec(1:2,:) = rvec(1:2,:) * c
        betal(1:2) = betal(1:2) * c
        betar(1:2) = betar(1:2) * c

        ! construct the states
        do m = 0, 3
           sum_l = dot_product(betal(:),rvec(:,m))
           sum_r = dot_product(betar(:),rvec(:,m))

           if (idir == 1) then
              q_l(i+1,j,m) = q_l(i+1,j,m) + sum_l
              q_r(i,  j,m) = q_r(i,  j,m) + sum_r
           else
              q_l(i,j+1,m) = q_l(i,j+1,m) + sum_l
              q_r(i,j,  m) = q_r(i,j,  m) + sum_r
           endif
        enddo

     enddo
  enddo

end subroutine states


subroutine riemann_cgf(idir, qx, qy, ng, &
                       nvar, iD, iSx, iSy, itau, &
                       gamma, U_l, U_r, V_l, V_r, c, F)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  integer, intent(in) :: nvar, iD, iSx, iSy, itau
  double precision, intent(in) :: gamma, c

  ! 0-based indexing to match python
  double precision, intent(inout) :: U_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: U_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: V_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: V_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(  out) :: F(0:qx-1,0:qy-1,0:nvar-1)

!f2py depend(qx, qy, nvar) :: U_l, U_r, V_l, V_r
!f2py intent(in) :: U_l, U_r, V_l, V_r
!f2py intent(out) :: F

  ! Solve riemann shock tube problem for a general equation of
  ! state using the method of Colella, Glaz, and Ferguson.  See
  ! Almgren et al. 2010 (the CASTRO paper) for details.
  !
  ! The Riemann problem for the Euler's equation produces 4 regions,
  ! separated by the three characteristics (u - cs, u, u + cs):
  !
  !
  !        u - cs    t    u      u + cs
  !          \       ^   .       /
  !           \  *L  |   . *R   /
  !            \     |  .     /
  !             \    |  .    /
  !         L    \   | .   /    R
  !               \  | .  /
  !                \ |. /
  !                 \|./
  !        ----------+----------------> x
  !
  ! We care about the solution on the axis.  The basic idea is to use
  ! estimates of the wave speeds to figure out which region we are in,
  ! and then use jump conditions to evaluate the state there.
  !
  ! Only density jumps across the u characteristic.  All primitive
  ! variables jump across the other two.  Special attention is needed
  ! if a rarefaction spans the axis.

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision, parameter :: smallc = 1.e-10
  double precision, parameter :: smallrho = 1.e-10
  double precision, parameter :: smallp = 1.e-10

  double precision :: rho_l, un_l, ut_l, h_l, p_l
  double precision :: rho_r, un_r, ut_r, h_r, p_r

  double precision :: rhostar_l, rhostar_r, hstar_l, hstar_r
  double precision :: ustar, pstar, cstar_l, cstar_r
  double precision :: lambda_l, lambdastar_l, lambda_r, lambdastar_r
  double precision :: W_l, W_r, c_l, c_r, sigma
  double precision :: alpha, eps_l, eps_r

  double precision :: rho_state, un_state, ut_state, p_state, h_state, W


  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! primitive variable states
        rho_l  = V_l(i,j,iD)

        ! un = normal velocity; ut = transverse velocity
        if (idir == 1) then
           un_l    = V_l(i,j,iSx)
           ut_l    = V_l(i,j,iSy)
        else
           un_l    = V_l(i,j,iSy)
           ut_l    = V_l(i,j,iSx)
        endif

        h_l = V_l(i,j,itau)

        p_l   = (h_l - 1.0d0) * (gamma - 1.0d0) * rho_l / gamma
        p_l = max(p_l, smallp)
        eps_l = h_l - 1.0d0 - p_l / rho_l

        rho_r  = V_r(i,j,iD)

        if (idir == 1) then
            un_r    = V_r(i,j,iSx)
            ut_r    = V_r(i,j,iSy)
        else
            un_r    = V_r(i,j,iSy)
            ut_r    = V_r(i,j,iSx)
        endif

        h_r = V_r(i,j,itau)

        p_r   = (h_r - 1.0d0) * (gamma - 1.0d0) * rho_r / gamma
        p_r = max(p_r, smallp)
        eps_r = h_r - 1.0d0 - p_r / rho_r

        ! define the Lagrangian sound speed
        W_l = max(smallrho*smallc, rho_l * sqrt(gamma * (gamma - 1.0d0) * eps_l / (1.0d0 + gamma * eps_l)))
        W_r = max(smallrho*smallc, rho_r * sqrt(gamma * (gamma - 1.0d0) * eps_r / (1.0d0 + gamma * eps_r)))

        ! and the regular sound speeds
        c_l = max(smallc, sqrt(gamma * (gamma - 1.0d0) * eps_l / (1.0d0 + gamma * eps_l)))
        c_r = max(smallc, sqrt(gamma * (gamma - 1.0d0) * eps_r / (1.0d0 + gamma * eps_r)))

        ! define the star states
        pstar = (W_l*p_r + W_r*p_l + W_l*W_r*(un_l - un_r))/(W_l + W_r)
        pstar = max(pstar, smallp)
        ustar = (W_l*un_l + W_r*un_r + (p_l - p_r))/(W_l + W_r)

        ! now compute the remaining state to the left and right
        ! of the contact (in the star region)
        rhostar_l = rho_l + (pstar - p_l)/c_l**2
        rhostar_r = rho_r + (pstar - p_r)/c_r**2

        hstar_l = h_l + &
             (pstar - p_l)*(h_l/rho_l + p_l/rho_l)/c_l**2
        hstar_r = h_r + &
             (pstar - p_r)*(h_r/rho_r + p_r/rho_r)/c_r**2

        cstar_l = max(smallc,sqrt(gamma*pstar/rhostar_l))
        cstar_r = max(smallc,sqrt(gamma*pstar/rhostar_r))

        ! figure out which state we are in, based on the location of
        ! the waves
        if (ustar > 0.0d0) then

           ! contact is moving to the right, we need to understand
           ! the L and *L states

           ! Note: transverse velocity only jumps across contact
           ut_state = ut_l

           ! define eigenvalues
           lambda_l = un_l - c_l
           lambdastar_l = ustar - cstar_l

           if (pstar > p_l) then
              ! the wave is a shock -- find the shock speed
              sigma = (lambda_l + lambdastar_l)/2.0d0

              if (sigma > 0.0d0) then
                 ! shock is moving to the right -- solution is L state
                 rho_state = rho_l
                 un_state = un_l
                 p_state = p_l
                 h_state = h_l

              else
                 ! solution is *L state
                 rho_state = rhostar_l
                 un_state = ustar
                 p_state = pstar
                 h_state = hstar_l
              endif

           else
              ! the wave is a rarefaction
              if (lambda_l < 0.0d0 .and. lambdastar_l < 0.0d0) then
                 ! rarefaction fan is moving to the left -- solution is
                 ! *L state
                 rho_state = rhostar_l
                 un_state = ustar
                 p_state = pstar
                 h_state = hstar_l

              else if (lambda_l > 0.0d0 .and. lambdastar_l > 0.0d0) then
                 ! rarefaction fan is moving to the right -- solution is
                 ! L state
                 rho_state = rho_l
                 un_state = un_l
                 p_state = p_l
                 h_state = h_l

              else
                 ! rarefaction spans x/t = 0 -- interpolate
                 alpha = lambda_l/(lambda_l - lambdastar_l)

                 rho_state  = alpha*rhostar_l  + (1.0d0 - alpha)*rho_l
                 un_state   = alpha*ustar      + (1.0d0 - alpha)*un_l
                 p_state    = alpha*pstar      + (1.0d0 - alpha)*p_l
                 h_state = alpha*hstar_l + (1.0d0 - alpha)*h_l
              endif

           endif

        else if (ustar < 0) then

           ! contact moving left, we need to understand the R and *R
           ! states

           ! Note: transverse velocity only jumps across contact
           ut_state = ut_r

           ! define eigenvalues
           lambda_r = un_r + c_r
           lambdastar_r = ustar + cstar_r

           if (pstar > p_r) then
              ! the wave if a shock -- find the shock speed
              sigma = (lambda_r + lambdastar_r)/2.0d0

              if (sigma > 0.0d0) then
                 ! shock is moving to the right -- solution is *R state
                 rho_state = rhostar_r
                 un_state = ustar
                 p_state = pstar
                 h_state = hstar_r

              else
                 ! solution is R state
                 rho_state = rho_r
                 un_state = un_r
                 p_state = p_r
                 h_state = h_r
              endif

           else
              ! the wave is a rarefaction
              if (lambda_r < 0.0d0 .and. lambdastar_r < 0.0d0) then
                 ! rarefaction fan is moving to the left -- solution is
                 ! R state
                 rho_state = rho_r
                 un_state = un_r
                 p_state = p_r
                 h_state = h_r

              else if (lambda_r > 0.0d0 .and. lambdastar_r > 0.0d0) then
                 ! rarefaction fan is moving to the right -- solution is
                 ! *R state
                 rho_state = rhostar_r
                 un_state = ustar
                 p_state = pstar
                 h_state = hstar_r

              else
                 ! rarefaction spans x/t = 0 -- interpolate
                 alpha = lambda_r/(lambda_r - lambdastar_r)

                 rho_state  = alpha*rhostar_r  + (1.0d0 - alpha)*rho_r
                 un_state   = alpha*ustar      + (1.0d0 - alpha)*un_r
                 p_state    = alpha*pstar      + (1.0d0 - alpha)*p_r
                 h_state = alpha*hstar_r + (1.0d0 - alpha)*h_r

              endif

           endif

        else  ! ustar == 0

           rho_state = 0.5*(rhostar_l + rhostar_r)
           un_state = ustar
           ut_state = 0.5*(ut_l + ut_r)
           p_state = pstar
           h_state = 0.5*(hstar_l + hstar_r)

        endif

        ! compute the fluxes
        W = 1.0d0 / sqrt(1.0d0 - (un_state**2 + ut_state**2)/c**2)

        F(i,j,iD) = rho_state * W * un_state

        if (idir == 1) then
           F(i,j,iSx) = rho_state * h_state * W**2 * un_state**2 + p_state
           F(i,j,iSy) = rho_state * h_state * W**2 * ut_state * un_state
        else
           F(i,j,iSx) = rho_state * h_state * W**2 * ut_state * un_state
           F(i,j,iSy) = rho_state * h_state * W**2 * un_state**2 + p_state
        endif

        F(i,j,itau) = (rho_state * h_state * W**2 - rho_state * W) * un_state

     enddo
  enddo

end subroutine riemann_cgf


subroutine riemann_RHLLE(idir, qx, qy, ng, &
                        nvar, iD, iSx, iSy, itau, &
                        gamma, c, U_l, U_r, V_l, V_r, F)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  integer, intent(in) :: nvar, iD, iSx, iSy, itau
  double precision, intent(in) :: gamma, c

  ! 0-based indexing to match python
  double precision, intent(inout) :: U_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: U_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: V_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: V_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(  out) :: F(0:qx-1,0:qy-1,0:nvar-1)

!f2py depend(qx, qy, nvar) :: U_l, U_r, V_l, V_r
!f2py intent(in) :: U_l, U_r, V_l, V_r
!f2py intent(out) :: F

  ! this is the HLLE Riemann solver.

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision, parameter :: smallc = 1.e-10
  double precision, parameter :: smallrho = 1.e-10
  double precision, parameter :: smallp = 1.e-10

  double precision :: rho_l, un_l, ut_l, h_l, p_l, eps_l
  double precision :: rho_r, un_r, ut_r, h_r, p_r, eps_r
  double precision :: cs_l, cs_r
  double precision :: cs_bar, v_bar, a_l, a_r, a_lm, a_rp

  double precision :: F_l(0:nvar-1), F_r(0:nvar-1)


  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! primitive variable states
        rho_l  = V_l(i,j,iD)

        ! un = normal velocity; ut = transverse velocity
        if (idir == 1) then
            un_l    = V_l(i,j,iSx)
            ut_l    = V_l(i,j,iSy)
        else
            un_l    = V_l(i,j,iSy)
            ut_l    = V_l(i,j,iSx)
        endif

        p_l = V_l(i,j,itau)
        p_l = max(p_l, smallp)
        eps_l = p_l / (rho_l * (gamma - 1.0d0))
        h_l = 1.0d0 + eps_l + p_l/rho_l

        rho_r  = V_r(i,j,iD)

        if (idir == 1) then
            un_r    = V_r(i,j,iSx)
            ut_r    = V_r(i,j,iSy)
        else
            un_r    = V_r(i,j,iSy)
            ut_r    = V_r(i,j,iSx)
        endif

        p_r = V_r(i,j,itau)
        p_r = max(p_r, smallp)
        eps_r = p_r / (rho_r * (gamma - 1.0d0))
        h_r = 1.0d0 + eps_r + p_r/rho_r

        ! compute the sound speeds
        cs_l = max(smallc, sqrt(gamma * (gamma - 1.0d0) * eps_l / (1.0d0 + gamma * eps_l)))
        cs_r = max(smallc, sqrt(gamma * (gamma - 1.0d0) * eps_r / (1.0d0 + gamma * eps_r)))

        F_l(iD) = U_l(i,j,iD) * un_l
        F_r(iD) = U_r(i,j,iD) * un_r

        if (idir == 1) then
            F_l(iSx) = U_l(i,j,iSx) * un_l + p_l
            F_l(iSy) = U_l(i,j,iSy) * un_l
            F_r(iSx) = U_r(i,j,iSx) * un_r + p_r
            F_r(iSy) = U_r(i,j,iSy) * un_r
        else
            F_l(iSx) = U_l(i,j,iSx) * un_l
            F_l(iSy) = U_l(i,j,iSy) * un_l + p_l
            F_r(iSx) = U_r(i,j,iSx) * un_r
            F_r(iSy) = U_r(i,j,iSy) * un_r + p_r
        endif

        F_l(itau) = (U_l(i,j,itau) + p_l) * un_l
        F_r(itau) = (U_r(i,j,itau) + p_r) * un_r

        cs_bar = 0.5d0 * (cs_l + cs_r)
        if (idir == 1) then
            v_bar = 0.5d0 * (V_l(i,j,iSx) + V_r(i,j,iSx))
        else
            v_bar = 0.5d0 * (V_l(i,j,iSy) + V_r(i,j,iSy))
        endif

        a_r = (v_bar + cs_bar) / (c + (v_bar * cs_bar) / c)
        a_rp = max(0.0d0, a_r)
        a_l = (v_bar - cs_bar) / (c - (v_bar * cs_bar) / c)
        a_lm = min(0.0d0, a_l)

        if (a_l > 0.0d0) then
            F(i,j,:) = F_l(:)
        elseif (a_r > 0.0d0) then
            F(i,j,:) = (a_rp * F_l(:) - a_lm * F_r(:) + a_rp * a_lm * &
                (V_r(i,j,:) - V_l(i,j,:))) / (a_rp - a_lm)
        else
            F(i,j,:) = F_r(:)
        endif

     enddo
  enddo
end subroutine riemann_RHLLE


subroutine riemann_RHLLC(idir, qx, qy, ng, &
                        nvar, iD, iSx, iSy, itau, &
                        gamma, c, U_l, U_r, V_l, V_r, F)
  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  integer, intent(in) :: nvar, iD, iSx, iSy, itau
  double precision, intent(in) :: gamma, c

  ! 0-based indexing to match python
  double precision, intent(inout) :: U_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: U_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: V_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: V_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(  out) :: F(0:qx-1,0:qy-1,0:nvar-1)

!f2py depend(qx, qy, nvar) :: U_l, U_r, V_l, V_r
!f2py intent(in) :: U_l, U_r, V_l, V_r
!f2py intent(out) :: F

  ! this is the HLLC Riemann solver.

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision, parameter :: smallc = 1.e-10
  double precision, parameter :: smallrho = 1.e-10
  double precision, parameter :: smallp = 1.e-10

  double precision :: rho_l, un_l, ut_l, h_l, p_l, eps_l
  double precision :: rho_r, un_r, ut_r, h_r, p_r, eps_r
  double precision :: cs_l, cs_r, p_lstar, p_rstar
  double precision :: cs_bar, v_bar, a_l, a_r, a_lm, a_rp, a_c

  double precision :: U_lstar(0:nvar-1), U_rstar(0:nvar-1)
  double precision :: Q(0:nvar-1)
  double precision :: F_l(0:nvar-1), F_r(0:nvar-1)


  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! primitive variable states
        rho_l  = V_l(i,j,iD)

        ! un = normal velocity; ut = transverse velocity
        if (idir == 1) then
            un_l    = V_l(i,j,iSx)
            ut_l    = V_l(i,j,iSy)
        else
            un_l    = V_l(i,j,iSy)
            ut_l    = V_l(i,j,iSx)
        endif

        p_l = V_l(i,j,itau)
        p_l = max(p_l, smallp)
        eps_l = p_l / (rho_l * (gamma - 1.0d0))
        h_l = 1.0d0 + eps_l + p_l/rho_l

        rho_r  = V_r(i,j,iD)

        if (idir == 1) then
            un_r    = V_r(i,j,iSx)
            ut_r    = V_r(i,j,iSy)
        else
            un_r    = V_r(i,j,iSy)
            ut_r    = V_r(i,j,iSx)
        endif

        p_r = V_r(i,j,itau)
        p_r = max(p_r, smallp)
        eps_r = p_r / (rho_r * (gamma - 1.0d0))
        h_r = 1.0d0 + eps_r + p_r/rho_r

        ! compute the sound speeds
        cs_l = max(smallc, sqrt(gamma * (gamma - 1.0d0) * eps_l / (1.0d0 + gamma * eps_l)))
        cs_r = max(smallc, sqrt(gamma * (gamma - 1.0d0) * eps_r / (1.0d0 + gamma * eps_r)))

        F_l(iD) = U_l(i,j,iD) * un_l
        F_r(iD) = U_r(i,j,iD) * un_r

        if (idir == 1) then
            F_l(iSx) = U_l(i,j,iSx) * un_l + p_l
            F_l(iSy) = U_l(i,j,iSy) * un_l
            F_r(iSx) = U_r(i,j,iSx) * un_r + p_r
            F_r(iSy) = U_r(i,j,iSy) * un_r
        else
            F_l(iSx) = U_l(i,j,iSx) * un_l
            F_l(iSy) = U_l(i,j,iSy) * un_l + p_l
            F_r(iSx) = U_r(i,j,iSx) * un_r
            F_r(iSy) = U_r(i,j,iSy) * un_r + p_r
        endif

        F_l(itau) = (U_l(i,j,itau) + p_l) * un_l
        F_r(itau) = (U_r(i,j,itau) + p_r) * un_r

        cs_bar = 0.5d0 * (cs_l + cs_r)
        if (idir == 1) then
            v_bar = 0.5d0 * (V_l(i,j,iSx) + V_r(i,j,iSx))
        else
            v_bar = 0.5d0 * (V_l(i,j,iSy) + V_r(i,j,iSy))
        endif

        a_r = (v_bar + cs_bar) / (1.0d0 + v_bar * cs_bar)
        a_rp = max(0.0d0, a_r)
        a_l = (v_bar - cs_bar) / (1.0d0 - v_bar * cs_bar)
        a_lm = min(0.0d0, a_l)
        !!!!! how do you calculate a_c??????
        a_c = 0.5d0 * (a_r + a_l)

        ! left states

        Q(:) = a_l * U_l(i,j,:) - F_l(:)
        U_lstar(iD) = Q(iD) / (a_l - a_c)
        if (idir == 1) then
            p_lstar = (Q(iSx) - a_c * (Q(itau) + Q(iD))) / (a_c * a_l - 1.0d0)
            U_lstar(iSx) = (Q(iSx) + p_lstar)/(a_l - a_c)
            U_lstar(iSy) = Q(iSy) / (a_l - a_c)
        else
            p_lstar = (Q(iSy) - a_c * (Q(itau) + Q(iD))) / (a_c * a_l - 1.0d0)
            U_lstar(iSx) = Q(iSx)/(a_l - a_c)
            U_lstar(iSy) = (Q(iSy) + p_lstar) / (a_l - a_c)
        endif
        U_lstar(itau) = (Q(itau) + p_lstar*a_c) / (a_l - a_c)

        ! right states
        Q(:) = a_r * U_r(i,j,:) - F_r(:)
        U_rstar(iD) = Q(iD) / (a_r - a_c)
        if (idir == 1) then
            p_rstar = (Q(iSx) - a_c * (Q(itau) + Q(iD))) / (a_c * a_r - 1.0d0)
            U_rstar(iSx) = (Q(iSx) + p_rstar)/(a_r - a_c)
            U_rstar(iSy) = Q(iSy) / (a_r - a_c)
        else
            p_rstar = (Q(iSy) - a_c * (Q(itau) + Q(iD))) / (a_c * a_r - 1.0d0)
            U_rstar(iSx) = Q(iSx)/(a_r - a_c)
            U_rstar(iSy) = (Q(iSy) + p_rstar) / (a_r - a_c)
        endif
        U_rstar(itau) = (Q(itau) + p_rstar*a_c) / (a_r - a_c)


        if (a_l > 0.0d0) then
            F(i,j,:) = F_l(:)
        elseif (a_c > 0.0d0) then
            F(i,j,:) = F_l(:) + a_l * (U_lstar(:) - U_l(i,j,:))
        elseif (a_r > 0.0d0) then
            F(i,j,:) = F_r(:) + a_r * (U_rstar(:) - U_r(i,j,:))
        else
            F(i,j,:) = F_r(:)
        endif
     enddo
  enddo
end subroutine riemann_RHLLC



subroutine consFlux(idir, iD, iSx, iSy, itau, nvar, U_state, F, u, v, p)

  integer, intent(in) :: idir
  double precision, intent(in) :: u, v, p
  integer, intent(in) :: iD, iSx, iSy, itau, nvar
  double precision, intent(in) :: U_state(0:nvar-1)
  double precision, intent(out) :: F(0:nvar-1)

  if (idir == 1) then
     F(iD) = U_state(iD) * u
     F(iSx) = U_state(iSx) * u + p
     F(iSy) = U_state(iSy) * u
     F(itau) = (U_state(itau) + p) * u
  else
     F(iD) = U_state(iD) * v
     F(iSx) = U_state(iSx) * v
     F(iSy) = U_state(iSy) * v + p
     F(itau) = (U_state(itau) + p) * v
  endif

end subroutine consFlux

subroutine numFlux(idir, iD, iSx, iSy, itau, nvar, V_l, V_r, cs_l, cs_r, F_l, F_r, F)

    integer, intent(in) :: idir, iD, iSx, iSy, itau, nvar
    double precision, intent(in) :: V_l(0:nvar-1), V_r(0:nvar-1)
    double precision, intent(in) :: cs_l, cs_r
    double precision, intent(in) :: F_l(0:nvar-1), F_r(0:nvar-1)
    double precision, intent(out) :: F(0:nvar-1)

!f2py depend(qx, qy, nvar) :: V_l, V_r, F_l, F_r
!f2py intent(in) :: V_l, V_r, F_l, F_r
!f2py intent(out) :: F


    double precision :: cs_bar, v_bar, a_l, a_r, a_lm, a_rp

    cs_bar = 0.5d0 * (cs_l + cs_r)
    if (idir == 1) then
        v_bar = 0.5d0 * (V_l(iSx) + V_r(iSx))
    else
        v_bar = 0.5d0 * (V_l(iSy) + V_r(iSy))
    endif

    a_r = (v_bar + cs_bar) / (1.0d0 + v_bar * cs_bar)
    a_rp = max(0.0d0, a_r)
    a_l = (v_bar - cs_bar) / (1.0d0 - v_bar * cs_bar)
    a_lm = min(0.0d0, a_l)

    if (a_l > 0.0d0) then
        F(:) = F_l(:)
    elseif (a_r >= 0.0d0) then
        F(:) = (a_rp * F_l(:) - a_lm * F_r(:) + a_rp * a_lm * &
            (V_r(:) - V_l(:))) / (a_rp - a_lm)
    else
        F(:) = F_r(:)
    endif

end subroutine numFlux


subroutine artificial_viscosity(qx, qy, ng, dx, dy, &
                                cvisc, u, v, avisco_x, avisco_y)

  implicit none
  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dy
  double precision, intent(in) :: cvisc

  ! 0-based indexing to match python
  double precision, intent(in) :: u(0:qx-1, 0:qy-1)
  double precision, intent(in) :: v(0:qx-1, 0:qy-1)
  double precision, intent(out) :: avisco_x(0:qx-1, 0:qy-1)
  double precision, intent(out) :: avisco_y(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: u, v
!f2py depend(qx, qy) :: avisco_x, avisco_y
!f2py intent(in) :: u, v
!f2py intent(out) :: avisco_x, avisco_y

  ! compute the artifical viscosity.  Here, we compute edge-centered
  ! approximations to the divergence of the velocity.  This follows
  ! directly Colella & Woodward (1984) Eq. 4.5
  !
  ! data locations:
  !
  !   j+3/2--+---------+---------+---------+
  !          |         |         |         |
  !     j+1  +         |         |         |
  !          |         |         |         |
  !   j+1/2--+---------+---------+---------+
  !          |         |         |         |
  !        j +         X         |         |
  !          |         |         |         |
  !   j-1/2--+---------+----Y----+---------+
  !          |         |         |         |
  !      j-1 +         |         |         |
  !          |         |         |         |
  !   j-3/2--+---------+---------+---------+
  !          |    |    |    |    |    |    |
  !              i-1        i        i+1
  !        i-3/2     i-1/2     i+1/2     i+3/2
  !
  ! X is the location of avisco_x(i,j)
  ! Y is the location of avisco_y(i,j)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny

  integer :: i, j

  double precision :: divU_x, divU_y

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! start by computing the divergence on the x-interface.  The
        ! x-difference is simply the difference of the cell-centered
        ! x-velocities on either side of the x-interface.  For the
        ! y-difference, first average the four cells to the node on
        ! each end of the edge, and then difference these to find the
        ! edge centered y difference.
        divU_x = (u(i,j) - u(i-1,j))/dx + &
             0.25d0*(v(i,j+1) + v(i-1,j+1) - v(i,j-1) - v(i-1,j-1))/dy

        avisco_x(i,j) = cvisc*max(-divU_x*dx, 0.0d0)

        ! now the y-interface value
        divU_y = 0.25d0*(u(i+1,j) + u(i+1,j-1) - u(i-1,j) - u(i-1,j-1))/dx + &
             (v(i,j) - v(i,j-1))/dy

        avisco_y(i,j) = cvisc*max(-divU_y*dy, 0.0d0)

     enddo
  enddo

end subroutine artificial_viscosity
