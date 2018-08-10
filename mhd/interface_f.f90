subroutine states(idir, qx, qy, ng, dx, dt, &
                  irho, iu, iv, ibx, iby, ip, ix, nvar, nspec, &
                  gamma, &
                  qv, dqv, &
                  q_l, q_r)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dt
  integer, intent(in) :: irho, iu, iv, ibx, iby, ip, ix, nvar, nspec
  double precision, intent(in) :: gamma

  ! 0-based indexing to match python
  double precision, intent(inout) :: qv(0:qx-1, 0:qy-1, 0:nvar-1)
  double precision, intent(inout) :: dqv(0:qx-1, 0:qy-1, 0:nvar-1)

  double precision, intent(  out) :: q_l(0:qx-1, 0:qy-1, 0:nvar-1)
  double precision, intent(  out) :: q_r(0:qx-1, 0:qy-1, 0:nvar-1)

!f2py depend(qx, qy, nvar) :: qv, dqv
!f2py depend(qx, qy, nvar) :: q_l, q_r
!f2py intent(in) :: qv, dqv
!f2py intent(out) :: q_l, q_r

  ! predict the cell-centered state to the edges in one-dimension
  ! using the reconstructed, limited slopes.
  !
  ! We follow the convection here that V_l[i] is the left state at the
  ! i-1/2 interface and V_l[i+1] is the left state at the i+1/2
  ! interface.
  !
  !
  ! We need the left and right eigenvectors and the eigenvalues for
  ! the system projected along the x-direction
  !
  ! Taking our state vector 0
  ! are u - c, u, u + c.
  !
  ! We look at the equations of hydrodynamics in a split fashion --
  ! i.e., we only consider one dimension at a time.
  !
  ! Considering advection in the x-direction, the Jacobian matrix for
  ! the primitive variable formulation of the Euler equations
  ! projected in the x-direction is:
  !
  !        / u   r   0   0 \
  !        | 0   u   0  1/r |
  !    A = | 0   0   u   0  |
  !        \ 0  rc^2 0   u  /
  !
  ! The right eigenvectors are
  !
  !        /  1  \        / 1 \        / 0 \        /  1  \
  !        |-c/r |        | 0 |        | 0 |        | c/r |
  !   r1 = |  0  |   r2 = | 0 |   r3 = | 1 |   r4 = |  0  |
  !        \ c^2 /        \ 0 /        \ 0 /        \ c^2 /
  !
  ! In particular, we see from r3 that the transverse velocity (v in
  ! this case) is simply advected at a speed u in the x-direction.
  !
  ! The left eigenvectors are
  !
  !    l1 =     ( 0,  -r/(2c),  0, 1/(2c^2) )
  !    l2 =     ( 1,     0,     0,  -1/c^2  )
  !    l3 =     ( 0,     0,     1,     0    )
  !    l4 =     ( 0,   r/(2c),  0, 1/(2c^2) )
  !
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
  integer :: i, j, n, m

  double precision :: dq(0:nvar-1), q(0:nvar-1)
  double precision :: lvec(0:nvar-1+2,0:nvar-1+2), rvec(0:nvar-1+2,0:nvar-1+2)
  double precision :: eval(0:nvar-1+2)
  double precision :: betal(0:nvar-1+2), betar(0:nvar-1+2)

  double precision :: dtdx, dtdx4
  double precision :: cf, cs, ca, a
  double precision :: af, as, phi, betax, betay, betaz

  double precision :: sum, sum_l, sum_r, factor

  integer :: ns

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  ns = nvar - nspec

  dtdx = dt/dx
  dtdx4 = 0.25d0*dtdx

  ! this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
  do j = jlo-2, jhi+2
     do i = ilo-2, ihi+2

        ! dq(:) = dq(i,j,:)0.0d0
        ! dq(0:2) = dqv(i,j,0:2)
        ! dq(4:5) = dqv(i,j,3:4)
        ! dq(7) = dqv(i,j,5)
        ! q(:) = q(i,j,:)0.0d0
        ! q(0:2) = qv(i,j,0:2)
        ! q(4:5) = qv(i,j,3:4)
        ! q(7) = qv(i,j,5)
        dq(:) = dqv(i,j,:)
        q(:) = qv(i,j,:)

        a = sqrt(gamma*q(ip)/q(irho))

        write(*,*) "a = ", a

        lvec(:,:) = 0.0d0
        rvec(:,:) = 0.0d0
        eval(:) = 0.0d0

        ! compute the eigenvalues and eigenvectors
        if (idir == 1) then

            cf = sqrt(0.5d0*((q(ibx)**2+q(iby)**2)/q(irho) + a**2) + &
                sqrt(0.25d0*((q(ibx)**2+q(iby)**2)/q(irho) + a**2)**2 - a**2*q(ibx)**2/q(irho)))
            cs = sqrt(0.5d0*((q(ibx)**2+q(iby)**2)/q(irho) + a**2) - &
                sqrt(0.25d0*((q(ibx)**2+q(iby)**2)/q(irho) + a**2)**2 - a**2*q(ibx)**2/q(irho)))
            ca = sqrt(q(ibx)**2/q(irho))

           eval(:) = [q(iu) - cf, q(iu) - ca, q(iu) - cs, q(iu), q(iu), q(iu) + cs, q(iu)+ca, q(iu)+cf]

           phi = atan((sqrt((q(ibx)**2+q(iby)**2)/q(irho))-ca)/(abs(q(ibx))-a))

           if (cf**2-cs**2 == 0.0d0) then
               af = sin(phi)
               as = cos(phi)
           elseif (a**2 - cs**2 == 0.0d0) then
               af = 0.0d0
               as = sqrt((cf**2 - a**2)/(cf**2-cs**2))
           elseif (cf**2-a**2 == 0) then
               af = sqrt((a**2-cs**2)/(cf**2-cs**2))
               as = 0.0d0
           else
               af = sqrt((a**2-cs**2)/(cf**2-cs**2))
               as = sqrt((cf**2-a**2)/(cf**2-cs**2))
           endif

           betax = sign(1.0d0, q(ibx))
           if (q(iby) == 0) then
               betay = 1/sqrt(2.0d0)
               betaz = 1/sqrt(2.0d0)
           else
               betay = 1.0d0
               betaz = 0.0d0
           endif

           ! - fast
           rvec(0,0:ns-1) = [q(irho)*af, q(irho)*af*(q(iu)-cf), &
            q(irho)*(af*q(iv)+as*cs*betax*betay), q(irho)*as*cs*betax*betaz, 0.0d0, &
            sqrt(q(irho))*as*a*betay, sqrt(q(irho))*as*a*betaz, &
            q(irho)*af*(0.5d0*(q(iu)**2+q(iv)**2)-q(iu)*cf+a**2/(gamma-1.0d0))+as*betay*q(iv)*(sqrt(q(irho))*a-q(irho)*cs*betax) ]

            lvec(0,0:ns-1) = [ 0.0d0, -0.5d0*af*cf/a**2, &
                0.5d0*as/a**2 * cs*betay*betax, 0.5d0*as/a**2*cs*betaz*betax, &
                0.0d0, 0.5d0*as/(sqrt(q(irho))*a)*betay, &
                0.5d0*as/(sqrt(q(irho))*a)*betaz, 0.5d0*af/(q(irho)*a**2)]
           ! -alfven
           rvec(1,0:ns-1) = [0.0d0, 0.0d0, -betaz*sqrt(q(irho)**2/2.0d0), &
                betay*sqrt(q(irho)**2/2.0d0), 0.0d0, -betaz*sqrt(q(irho)**2/2.0d0), &
                betay*sqrt(q(irho)**2/2.0d0), 0.0d0]

           lvec(1,0:ns-1) = [0.0d0, 0.0d0, -betaz*sqrt(q(irho)**2/2.0d0), &
                betay*sqrt(q(irho)**2/2.0d0), 0.0d0, -betaz*sqrt(0.5d0/q(irho)**2), &
                betay*sqrt(0.5d0/q(irho)**2), 0.0d0]

           ! - slow
          rvec(2,0:ns-1) = [q(irho)*as, q(irho)*as*(q(iu)-cs), &
           q(irho)*(as*q(iv)+af*cf*betax*betay), q(irho)*af*cf*betax*betaz, 0.0d0, &
           -sqrt(q(irho))*af*a*betay, -sqrt(q(irho))*af*a*betaz, &
           q(irho)*as*(0.5d0*(q(iu)**2+q(iv)**2)-q(iu)*cs+a**2/(gamma-1.0d0))+af*betay*q(iv)*(sqrt(q(irho))*a-q(irho)*cf*betax) ]

           lvec(2,0:ns-1) = [ 0.0d0, -0.5d0*as*cs/a**2, &
               0.5d0*af/a**2 * cf*betay*betax, 0.5d0*af/a**2*cf*betaz*betax, &
               0.0d0, -0.5d0*af/(sqrt(q(irho))*a)*betay, &
               -0.5d0*af/(sqrt(q(irho))*a)*betaz, 0.5d0*as/(q(irho)*a**2)]

            ! entropy
           rvec(3,0:ns-1) = [1.0d0, q(iu), q(iv), 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.5d0*(q(iu)**2+q(iv)**2) ]
           lvec(3,0:ns-1) = [1.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0 ]

           ! magnetic flux
           rvec(4, 0:ns-1) = [0.0d0, 0.0d0, 0.0d0, 0.0d0, 1.0d0, 0.0d0, 0.0d0, q(ibx)]
           lvec(4, 0:ns-1) = [0.0d0, 0.0d0, 0.0d0, 0.0d0, 1.0d0, 0.0d0, 0.0d0, q(ibx)]

           ! + slow

           rvec(5,0:ns-1) = [q(irho)*as, q(irho)*as*(q(iu)+cs), &
            q(irho)*(as*q(iv)-af*cf*betax*betay), q(irho)*af*cf*betax*betaz, 0.0d0, &
            -sqrt(q(irho))*af*a*betay, -sqrt(q(irho))*af*a*betaz, &
            q(irho)*as*(0.5d0*(q(iu)**2+q(iv)**2)+q(iu)*cs+a**2/(gamma-1.0d0))+af*betay*q(iv)*(sqrt(q(irho))*a+q(irho)*cf*betax) ]

            lvec(5,0:ns-1) = [ 0.0d0, 0.5d0*as*cs/a**2, &
                -0.5d0*af/a**2 * cf*betay*betax, -0.5d0*af/a**2*cf*betaz*betax, &
                0.0d0, -0.5d0*af/(sqrt(q(irho))*a)*betay, &
                -0.5d0*af/(sqrt(q(irho))*a)*betaz, 0.5d0*as/(q(irho)*a**2)]

            ! + alfven
            rvec(6,0:ns-1) = [0.0d0, 0.0d0, -betaz*sqrt(q(irho)**2/2.0d0), &
                betay*sqrt(q(irho)**2/2.0d0), 0.0d0, betaz*sqrt(q(irho)**2/2.0d0), &
                 -betay*sqrt(q(irho)**2/2.0d0), 0.0d0]

            lvec(6,0:ns-1) = [0.0d0, 0.0d0, -betaz*sqrt(q(irho)**2/2.0d0), &
                betay*sqrt(q(irho)**2/2.0d0), 0.0d0, betaz*sqrt(0.5d0/q(irho)**2), &
                -betay*sqrt(0.5d0/q(irho)**2), 0.0d0]

            ! + fast
            rvec(7,0:ns-1) = [q(irho)*af, q(irho)*af*(q(iu)+cf), &
             q(irho)*(af*q(iv)-as*cs*betax*betay), q(irho)*as*cs*betax*betaz, 0.0d0, &
             sqrt(q(irho))*as*a*betay, sqrt(q(irho))*as*a*betaz, &
             q(irho)*af*(0.5d0*(q(iu)**2+q(iv)**2)+q(iu)*cf+a**2/(gamma-1.0d0))+as*betay*q(iv)*(sqrt(q(irho))*a+q(irho)*cs*betax) ]

             lvec(7,0:ns-1) = [ 0.0d0, 0.5d0*af*cf/a**2, &
                 -0.5d0*as/a**2 * cs*betay*betax, -0.5d0*as/a**2*cs*betaz*betax, &
                 0.0d0, 0.5d0*as/(sqrt(q(irho))*a)*betay, &
                 0.5d0*as/(sqrt(q(irho))*a)*betaz, 0.5d0*af/(q(irho)*a**2)]

           ! now the species -- they only have a 1 in their corresponding slot
           eval(ns:) = q(iu)
           do n = ix, ix-1+nspec
              lvec(n,n) = 1.0
              rvec(n,n) = 1.0
           enddo
        else
            cf = sqrt(0.5d0*((q(ibx)**2+q(iby)**2)/q(irho) + a**2) + &
                sqrt(0.25d0*((q(ibx)**2+q(iby)**2)/q(irho) + a**2)**2 - a**2*q(iby)**2/q(irho)))
            cs = sqrt(0.5d0*((q(ibx)**2+q(iby)**2)/q(irho) + a**2) - &
                sqrt(0.25d0*((q(ibx)**2+q(iby)**2)/q(irho) + a**2)**2 - a**2*q(iby)**2/q(irho)))
            ca = sqrt(q(iby)**2/q(irho))

           eval = [q(iv) - cf, q(iv)-ca, q(iv)-cs, q(iv), q(iv), q(iv) + cs, q(iv)+ca, q(iv)+cf]

           phi = atan((sqrt((q(ibx)**2+q(iby)**2)/q(irho))-ca)/(abs(q(iby))-a))

           if (cf**2-cs**2 == 0.0d0) then
               af = sin(phi)
               as = cos(phi)
           elseif (a**2 - cs**2 == 0.0d0) then
               af = 0.0d0
               as = sqrt((cf**2 - a**2)/(cf**2-cs**2))
           elseif (cf**2-a**2 == 0) then
               af = sqrt((a**2-cs**2)/(cf**2-cs**2))
               as = 0.0d0
           else
               af = sqrt((a**2-cs**2)/(cf**2-cs**2))
               as = sqrt((cf**2-a**2)/(cf**2-cs**2))
           endif

           betay = sign(1.0d0, q(iby))
           if (q(iby) == 0) then
               betax = 1/sqrt(2.0d0)
               betaz = 1/sqrt(2.0d0)
           else
               betax = 1.0d0
               betaz = 0.0d0
           endif

           ! - fast
           rvec(0,0:ns-1) = [q(irho)*af, q(irho)*af*(q(iv)-cf), &
            q(irho)*(af*q(iu)+as*cs*betax*betay), q(irho)*as*cs*betay*betaz, &
            sqrt(q(irho))*as*a*betax, 0.0d0, sqrt(q(irho))*as*a*betaz, &
            q(irho)*af*(0.5d0*(q(iu)**2+q(iv)**2)-q(iv)*cf+a**2/(gamma-1.0d0))+as*betax*q(iu)*(sqrt(q(irho))*a-q(irho)*cs*betay) ]

            lvec(0,0:ns-1) = [ 0.0d0, -0.5d0*af*cf/a**2, &
                0.5d0*as/a**2 * cs*betay*betax, 0.5d0*as/a**2*cs*betaz*betay, &
                0.5d0*as/(sqrt(q(irho))*a)*betax, 0.0d0, &
                0.5d0*as/(sqrt(q(irho))*a)*betaz, 0.5d0*af/(q(irho)*a**2)]

           ! -alfven
           rvec(1,0:ns-1) = [0.0d0, -betay*sqrt(q(irho)**2/2.0d0), 0.0d0, &
                betax*sqrt(q(irho)**2/2.0d0), -betay*sqrt(q(irho)**2/2.0d0), 0.0d0, &
                betax*sqrt(q(irho)**2/2.0d0), 0.0d0]

           lvec(1,0:ns-1) = [0.0d0, -betay*sqrt(q(irho)**2/2.0d0), 0.0d0, &
                betax*sqrt(q(irho)**2/2.0d0), -betay*sqrt(0.5d0/q(irho)**2), 0.0d0, &
                betax*sqrt(0.5d0/q(irho)**2), 0.0d0]

           ! - slow
          rvec(2,0:ns-1) = [q(irho)*as, q(irho)*as*(q(iv)-cs), &
           q(irho)*(as*q(iu)+af*cf*betax*betay), q(irho)*af*cf*betay*betaz, &
           -sqrt(q(irho))*af*a*betax, 0.0d0, -sqrt(q(irho))*af*a*betaz, &
           q(irho)*as*(0.5d0*(q(iu)**2+q(iv)**2)-q(iv)*cs+a**2/(gamma-1.0d0))+af*betax*q(iu)*(sqrt(q(irho))*a-q(irho)*cf*betay) ]

           lvec(2,0:ns-1) = [ 0.0d0, -0.5d0*as*cs/a**2, &
               0.5d0*af/a**2 * cf*betay*betax, 0.5d0*af/a**2*cf*betaz*betay, &
               -0.5d0*af/(sqrt(q(irho))*a)*betax, 0.0d0, &
               -0.5d0*af/(sqrt(q(irho))*a)*betaz, 0.5d0*as/(q(irho)*a**2)]

            ! entropy
           rvec(3,0:ns-1) = [1.0d0, q(iu), q(iv), 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.5d0*(q(iu)**2+q(iv)**2) ]
           lvec(3,0:ns-1) = [1.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0 ]

           ! magnetic flux
           rvec(4, 0:ns-1) = [0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 1.0d0, 0.0d0, q(iby)]
           lvec(4, 0:ns-1) = [0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 1.0d0, 0.0d0, q(iby)]

           ! + slow

           rvec(5,0:ns-1) = [q(irho)*as, q(irho)*as*(q(iv)+cs), &
            q(irho)*(as*q(iu)-af*cf*betax*betay), -q(irho)*af*cf*betay*betaz, &
            -sqrt(q(irho))*af*a*betax, 0.0d0, -sqrt(q(irho))*af*a*betaz, &
            q(irho)*as*(0.5d0*(q(iu)**2+q(iv)**2)+q(iv)*cs+a**2/(gamma-1.0d0))+af*betax*q(iu)*(sqrt(q(irho))*a+q(irho)*cf*betay) ]

            lvec(5,0:ns-1) = [ 0.0d0, -0.5d0*as*cs/a**2, &
                0.5d0*af/a**2 * cf*betay*betax, 0.5d0*af/a**2*cf*betaz*betay, &
                -0.5d0*af/(sqrt(q(irho))*a)*betax, 0.0d0, &
                -0.5d0*af/(sqrt(q(irho))*a)*betaz, 0.5d0*as/(q(irho)*a**2)]

            ! + alfven
            rvec(6,0:ns-1) = [0.0d0, -betay*sqrt(q(irho)**2/2.0d0), 0.0d0, &
                 betax*sqrt(q(irho)**2/2.0d0), betay*sqrt(q(irho)**2/2.0d0), 0.0d0, &
                 -betax*sqrt(q(irho)**2/2.0d0), 0.0d0]

            lvec(6,0:ns-1) = [0.0d0, -betay*sqrt(q(irho)**2/2.0d0), 0.0d0, &
                 betax*sqrt(q(irho)**2/2.0d0), betay*sqrt(0.5d0/q(irho)**2), 0.0d0, &
                 -betax*sqrt(0.5d0/q(irho)**2), 0.0d0]

            ! + fast
            rvec(7,0:ns-1) = [q(irho)*af, q(irho)*af*(q(iv)+cf), &
             q(irho)*(af*q(iu)-as*cs*betax*betay),-q(irho)*as*cs*betay*betaz, &
             sqrt(q(irho))*as*a*betax, 0.0d0, sqrt(q(irho))*as*a*betaz, &
             q(irho)*af*(0.5d0*(q(iu)**2+q(iv)**2)+q(iv)*cf+a**2/(gamma-1.0d0))+as*betax*q(iu)*(sqrt(q(irho))*a+q(irho)*cs*betay) ]

             lvec(7,0:ns-1) = [ 0.0d0, -0.5d0*af*cf/a**2, &
                 0.5d0*as/a**2 * cs*betay*betax, 0.5d0*as/a**2*cs*betaz*betay, &
                 0.5d0*as/(sqrt(q(irho))*a)*betax, 0.0d0, &
                 0.5d0*as/(sqrt(q(irho))*a)*betaz, 0.5d0*af/(q(irho)*a**2)]

           ! now the species -- they only have a 1 in their corresponding slot
           eval(ns:) = q(iv)
           do n = ix, ix-1+nspec
              lvec(n,n) = 1.0
              rvec(n,n) = 1.0
           enddo

        endif

        ! compute the Vhat functions
        do m = 0, nvar-1+2
           sum = dot_product(lvec(m,:),dq(:))

           betal(m) = dtdx4*(eval(7) - eval(m))*(sign(1.0d0,eval(m)) + 1.0d0)*sum
           betar(m) = dtdx4*(eval(0) - eval(m))*(1.0d0 - sign(1.0d0,eval(m)))*sum
        enddo

        ! shift stuff along to get rid of z-direction stuff
        ! q(3:-2) = q(4:-1)
        ! q(5) = q(7)
        ! dq(3:-2) = dq(4:-1)
        ! dq(5) = dq(7)
        rvec(:,3:-2) = rvec(:,4:-1)
        rvec(:,5) = rvec(:,7)

        ! define the reference states
        if (idir == 1) then
           ! this is one the right face of the current zone,
           ! so the fastest moving eigenvalue is eval[3] = u + c
           factor = 0.5d0*(1.0d0 - dtdx*max(eval(7), 0.0d0))
           q_l(i+1,j,:) = q(:nvar-1) + factor*dq(:nvar-1)

           ! left face of the current zone, so the fastest moving
           ! eigenvalue is eval[3] = u - c
           factor = 0.5d0*(1.0d0 + dtdx*min(eval(0), 0.0d0))
           q_r(i,  j,:) = q(:nvar-1) - factor*dq(:nvar-1)

        else

           factor = 0.5d0*(1.0d0 - dtdx*max(eval(7), 0.0d0))
           q_l(i,j+1,:) = q(:nvar-1) + factor*dq(:nvar-1)

           factor = 0.5d0*(1.0d0 + dtdx*min(eval(0), 0.0d0))
           q_r(i,j,  :) = q(:nvar-1) - factor*dq(:nvar-1)

        endif

        ! construct the states
        do m = 0, nvar-1
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


subroutine riemann_HLLC(idir, qx, qy, ng, &
                        nvar, idens, ixmom, iymom, ibx, iby, iener, irhoX, nspec, &
                        lower_solid, upper_solid, &
                        gamma, U_l, U_r, F)


  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  integer, intent(in) :: nvar, idens, ixmom, iymom, ibx, iby, iener, irhoX, nspec
  integer, intent(in) :: lower_solid, upper_solid
  double precision, intent(in) :: gamma

  ! 0-based indexing to match python
  double precision, intent(inout) :: U_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: U_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(  out) :: F(0:qx-1,0:qy-1,0:nvar-1)

!f2py depend(qx, qy, nvar) :: U_l, U_r
!f2py intent(in) :: U_l, U_r
!f2py intent(out) :: F

  ! this is the HLLC Riemann solver.  The implementation follows
  ! directly out of Toro's book.  Note: this does not handle the
  ! transonic rarefaction.

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision, parameter :: smallc = 1.e-10
  double precision, parameter :: smallrho = 1.e-10
  double precision, parameter :: smallp = 1.e-10

  double precision :: rho_l, un_l, ut_l, rhoe_l, p_l, bx_l, by_l
  double precision :: rho_r, un_r, ut_r, rhoe_r, p_r, bx_r, by_r
  double precision :: xn(nspec)

  double precision :: cf_l, cf_r, cs_l, cs_r, ca_l, ca_r, a_l, a_r
  double precision :: S_l, S_r
  double precision :: U_state(0:nvar-1), F_l(0:nvar-1), F_r(0:nvar-1)

 ! NOTE: this is just HLL for now

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! primitive variable states
        rho_l  = U_l(i,j,idens)

        bx_l = U_l(i,j,ibx)
        bx_r = U_r(i,j,ibx)
        by_l = U_l(i,j,iby)
        by_r = U_r(i,j,iby)

        ! un = normal velocity; ut = transverse velocity
        if (idir == 1) then
           un_l    = U_l(i,j,ixmom)/rho_l
           ut_l    = U_l(i,j,iymom)/rho_l
        else
           un_l    = U_l(i,j,iymom)/rho_l
           ut_l    = U_l(i,j,ixmom)/rho_l
        endif

        rhoe_l = U_l(i,j,iener) - 0.5d0*rho_l*(un_l**2 + ut_l**2) - 0.5d0*(bx_l**2+by_l**2)

        p_l   = rhoe_l*(gamma - 1.0d0)
        p_l = max(p_l, smallp)

        rho_r  = U_r(i,j,idens)

        if (idir == 1) then
           un_r    = U_r(i,j,ixmom)/rho_r
           ut_r    = U_r(i,j,iymom)/rho_r
        else
           un_r    = U_r(i,j,iymom)/rho_r
           ut_r    = U_r(i,j,ixmom)/rho_r
        endif

        rhoe_r = U_r(i,j,iener) - 0.5d0*rho_r*(un_r**2 + ut_r**2) - 0.5d0*(bx_r**2+by_r**2)

        p_r   = rhoe_r*(gamma - 1.0d0)
        p_r = max(p_r, smallp)
        ! compute the sound speeds
        a_l = max(smallc, sqrt(gamma*p_l/rho_l))
        a_r = max(smallc, sqrt(gamma*p_r/rho_r))
        if (idir == 1) then
            cf_l = sqrt(0.5d0*((bx_l**2+by_l**2)/rho_l + a_l**2) + &
                sqrt(0.25d0*((bx_l**2+by_l**2)/rho_l + a_l**2)**2 - a_l**2*bx_l**2/rho_l))
            cf_r = sqrt(0.5d0*((bx_r**2+by_r**2)/rho_r + a_r**2) + &
                sqrt(0.25d0*((bx_r**2+by_r**2)/rho_r + a_r**2)**2 - a_r**2*bx_r**2/rho_r))
            cs_l = sqrt(0.5d0*((bx_l**2+by_l**2)/rho_l + a_l**2) - &
                sqrt(0.25d0*((bx_l**2+by_l**2)/rho_l + a_l**2)**2 - a_l**2*bx_l**2/rho_l))
            cs_r = sqrt(0.5d0*((bx_r**2+by_r**2)/rho_r + a_r**2) - &
                sqrt(0.25d0*((bx_r*2+by_r**2)/rho_r + a_r**2)**2 - a_r**2*bx_r**2/rho_r))
            ca_l = sqrt(bx_l**2/rho_l)
            ca_r = sqrt(bx_r**2/rho_r)

           S_l = min(un_l - cf_l, un_l - ca_l, un_l - cs_l, un_r - cf_r, un_r - ca_r, un_r - cs_r)
           S_r = max(un_l + cs_l, un_l+ca_l, un_l+cf_l, un_r + cs_r, un_r+ca_r, un_r+cf_r)
       else
           cf_l = sqrt(0.5d0*((bx_l**2+by_l**2)/rho_l + a_l**2) + &
               sqrt(0.25d0*((bx_l**2+by_l**2)/rho_l + a_l**2)**2 - a_l**2*by_l**2/rho_l))
           cf_r = sqrt(0.5d0*((bx_r**2+by_r**2)/rho_r + a_r**2) + &
               sqrt(0.25d0*((bx_r**2+by_r**2)/rho_r + a_r**2)**2 - a_r**2*by_r**2/rho_r))
           cs_l = sqrt(0.5d0*((bx_l**2+by_l**2)/rho_l + a_l**2) - &
               sqrt(0.25d0*((bx_l**2+by_l**2)/rho_l + a_l**2)**2 - a_l**2*by_l**2/rho_l))
           cs_r = sqrt(0.5d0*((bx_r**2+by_r**2)/rho_r + a_r**2) - &
               sqrt(0.25d0*((bx_r*2+by_r**2)/rho_r + a_r**2)**2 - a_r**2*by_r**2/rho_r))
           ca_l = sqrt(by_l**2/rho_l)
           ca_r = sqrt(by_r**2/rho_r)

           S_l = min(un_l - cf_l, un_l - ca_l, un_l - cs_l, un_r - cf_r, un_r - ca_r, un_r - cs_r)
           S_r = max(un_l + cs_l, un_l+ca_l, un_l+cf_l, un_r + cs_r, un_r+ca_r, un_r+cf_r)
       endif

       call consFlux(idir, gamma, idens, ixmom, iymom, ibx, iby, iener, irhoX, nvar, nspec, &
                     U_l, F_l)
        call consFlux(idir, gamma, idens, ixmom, iymom, ibx, iby, iener, irhoX, nvar, nspec, &
                   U_r, F_r)

        ! looks like we don't actually need this for HLL
       U_state(:) = (S_r*U_r(i,j,:) - S_l*U_l(i,j,:) - F_r + F_l) / (S_r - S_l)

       if (S_l .gt. 0.0d0) then
           F(i,j,:) = F_l
       elseif ( (S_l .le. 0.0d0) .and. (S_r .ge. 0.0d0) ) then
           F(i,j,:) = (S_r*F_l - S_l*F_r + S_r*S_l*(U_r(i,j,:)-U_l(i,j,:))) / (S_r-S_l)
       else
           F(i,j,:) = F_r
       endif

        ! we should deal with solid boundaries somehow here

     enddo
  enddo
end subroutine riemann_HLLC

subroutine consFlux(idir, gamma, idens, ixmom, iymom, ibx, iby, iener, irhoX, nvar, nspec, U_state, F)

  integer, intent(in) :: idir
  double precision, intent(in) :: gamma
  integer, intent(in) :: idens, ixmom, iymom, iener, ibx, iby, irhoX, nvar, nspec
  double precision, intent(in) :: U_state(0:nvar-1)
  double precision, intent(out) :: F(0:nvar-1)

  double precision :: p, u, v, bx, by

  u = U_state(ixmom)/U_state(idens)
  v = U_state(iymom)/U_state(idens)
  bx = U_state(ibx)
  by = U_state(iby)

  p = (U_state(iener) - 0.5d0*U_state(idens)*(u*u + v*v))*(gamma - 1.0d0) + 0.5d0 * (bx**2 + by**2)

  if (idir == 1) then
     F(idens) = U_state(idens)*u
     F(ixmom) = U_state(ixmom)*u + p - bx**2
     F(iymom) = U_state(iymom)*u - bx*by
     F(ibx) = 0.0d0
     F(iby) = u*by - bx*v
     F(iener) = (U_state(iener) + p)*u - bx * (u*bx + v*by)
     if (nspec > 0) then
        F(irhoX:irhoX-1+nspec) = U_state(irhoX:irhoX-1+nspec)*u
     endif
  else
     F(idens) = U_state(idens)*v
     F(ixmom) = U_state(ixmom)*v - by*bx
     F(iymom) = U_state(iymom)*v + p - by**2
     F(ibx) = v*bx - by*u
     F(iby) = 0.0d0
     F(iener) = (U_state(iener) + p)*v - by * (u*bx + v*by)
     if (nspec > 0) then
        F(irhoX:irhoX-1+nspec) = U_state(irhoX:irhoX-1+nspec)*v
     endif
  endif

end subroutine consFlux


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
