function f(p, D, W, rho, gamma) result (root)
    ! function for doing root finding

    implicit none

    double precision :: root
    double precision :: p, D, W, rho, gamma

    double precision :: h

    h = 1.0d0 + gamma * p / (gamma - 1.0d0) / rho

    root = (gamma - 1.0d0) * (D * h - p - rho) / W - p

end function f

function brentq(x1, b, D, W, rho, gamma) result (p)
    ! route finder using brent's method
    implicit none

    double precision :: p
    double precision  :: D, W, rho, x1, gamma
    double precision :: b

    double precision, parameter :: TOL = 1.0d-6
    integer, parameter :: ITMAX = 100

    double precision :: a, c, dd, fa, fb, fc, fs, s
    logical :: mflag, con1, con2, con3, con4, con5
    integer :: i

    double precision :: f

    a = x1
    c = 0.0d0
    dd = 0.0d0
    fa = f(a, D, W, rho, gamma)
    fb = f(b, D, W, rho, gamma)
    fc = 0.0d0

    if (fa * fb >= 0.0d0) then
        p = x1
        return
    end if

    if (abs(fa) < abs(fb)) then
        dd = a
        a = b
        b = dd

        dd = fa
        fa = fb
        fb = dd
    end if

    c = a
    fc = fa

    mflag = .true.

    do i = 1, ITMAX
        if (fa /= fc .and. fb /= fc) then
            s = a*fb*fc / ((fa-fb) * (fa-fc)) + b*fa*fc / ((fb-fa)*(fb-fc)) +&
                c*fa*fb / ((fc-fa)*(fc-fb))
        else
            s = b - fb * (b-a) / (fb-fa)
        end if

        con1 = .false.

        if (0.25d0 * (3.0d0 * a + b) < b) then
            if ( s < 0.25d0 * (3.0d0 * a + b) .or. s > b) then
                con1 = .true.
            end if
        else if (s < b .or. s > 0.25d0  * (3.0d0 * a + b)) then
            con1 = .true.
        end if

        con2 = mflag .and. abs(s - b) >= 0.5d0 * abs(b-c)

        con3 = (.not. mflag) .and. abs(s-b) >= 0.5d0 * abs(c-d)

        con4 = mflag .and. abs(b-c) < TOL

        con5 = (.not. mflag) .and. abs(c-d) < TOL

        if (con1 .or. con2 .or. con3 .or. con4 .or. con5) then
            s = 0.5d0 * (a + b)
            mflag = .true.
        else
            mflag = .false.
        end if

        fs = f(s, D, W, rho, gamma)

        if (abs(fa) < abs(fb)) then
            dd = a
            a = b
            b = dd

            dd = fa
            fa = fb
            fb = dd
        end if

        dd = c
        c = b
        fc = fb

        if (fa * fs < 0.0d0) then
            b = s
            fb = fs
        else
            a = s
            fa = fs
        end if

        if (fb == 0.0d0 .or. fs == 0.0d0 .or. abs(b-a) < TOL) then
            p = b
            return
        end if

    end do

    p = x1

end function brentq

subroutine cons_to_prim(D, Ux, Uy, p0, qx, qy, gamma, rho)
    ! convert an input vector of conserved variables to primitive variables

    implicit none

    integer, intent(in) :: qx, qy
    double precision, intent(in) :: gamma
    double precision, intent(in) :: D(0:qx-1, 0:qy-1)
    double precision, intent(in) :: Ux(0:qx-1, 0:qy-1)
    double precision, intent(in) :: Uy(0:qx-1, 0:qy-1)
    double precision, intent(in) :: p0(0:qx-1)
    double precision, intent(out) :: rho(0:qx-1, 0:qy-1)
    ! double precision, intent(out) :: p(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: D, Ux, Uy, rho
!f2py depend(qx) :: p0
!f2py intent(in) :: D, Ux, Uy, p0
!f2py intent(out) :: rho

    integer :: i, j
    double precision :: pmin, pmax, fmin, fmax, W, tau, h, u, v
    double precision :: U2(0:qx-1, 0:qy-1)
    double precision :: p(0:qx-1, 0:qy-1)

    double precision :: f, brentq

    double precision, parameter :: smallp = 1.0e-6

    U2 = Ux**2 + Uy**2

    do j = 0, qy-1
        do i = 0, qx-1
            if (U2(i, j) < 1.0d-15) then
                W = 1.0d0
            else
                W = sqrt(0.5d0 / U2(i, j) + sqrt(0.25 / U2(i, j)**2 + 1.0d0))
            endif

            rho(i, j) = D(i, j) / W
            h = 1.0d0 + gamma * p0(i) / (gamma - 1.0d0) / rho(i, j)
            tau = D(i, j) * h * W - p0(i) - D(i, j)

            pmax = max((gamma-1.0d0)*tau, 1.0d-2)

            pmin = max(1.0d-6*pmax, D(i,j) * h * W**2 * sqrt(U2(i, j)) - tau - D(i, j))

            fmin = f(pmin, D(i, j), W, rho(i, j), gamma)
            fmax = f(pmax, D(i, j), W, rho(i, j), gamma)

            if (fmin * fmax > 0.0d0) then
                pmin = pmin * 1.0d-2
                fmin = f(pmin, D(i, j), W, rho(i, j), gamma)
            endif

            if (fmin * fmax > 0.0d0) then
                pmax = min(pmax*1.0d2, 1.0d0)
            endif

            ! try:
            p(i, j) = brentq(pmin, pmax, D(i, j), W, rho(i, j), gamma)

            if ((p(i, j) /= p(i, j)) .or. &
                (p(i, j)-1 == p(i, j)) .or. &
                (abs(p(i, j)) > 1.0e10)) then ! nan or infty alert
                p(i, j) = max((gamma-1.0d0)*tau, smallp)
            endif

            tau = D(i, j) * h * W - p(i, j) - D(i, j)

            ! except ValueError:
            !     q(i, j, ip) = max((gamma-1.0d0)*U(i, j, iener), 0.0d0)

            if (abs(tau + D(i, j) + p(i, j)) < 1.0d-5) then
                u = Ux(i, j)
                v = Uy(i, j)
            else
                u = Ux(i, j)/(tau + D(i, j) + p(i, j))
                v = Uy(i, j)/(tau + D(i, j) + p(i, j))
            endif

            ! nan check
            if (u /= u) then
                u = 0.0d0
            endif
            if (v /= v) then
                v = 0.0d0
            endif

            W = 1.0d0 / sqrt(1.0d0 - u**2 - v**2)

            rho(i, j) = D(i, j) / W

        enddo
    enddo


end subroutine cons_to_prim
