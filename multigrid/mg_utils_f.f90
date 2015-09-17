! build with
! python setup.py build_ext --inplace

subroutine smooth_f(qx, qy, ng, nsmooth, v, f, bcs, eta_x, eta_y, v_out)

    implicit none

    integer, intent(in) :: qx, qy, ng, nsmooth
    double precision, intent(inout) :: v(0:qx-1, 0:qy-1)
    double precision, intent(inout) :: f(0:qx-1, 0:qy-1)
    integer, intent(inout) :: bcs(0:3)
    double precision, intent(inout) :: eta_x(0:qx-1, 0:qy-1)
    double precision, intent(inout) :: eta_y(0:qx-1, 0:qy-1)
    double precision, intent(out) :: v_out(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: v, f
!f2py depend(qx, qy) :: eta_x, eta_y
!f2py depend(qx, qy) :: v_out
!f2py intent(in) :: v
!f2py intent(in) :: f, bcs, eta_x, eta_y
!f2py intent(out) :: v_out

    double precision :: denom(0:qx-1, 0:qy-1)
    double precision :: v_temp(0:qx-1, 0:qy-1)

    integer :: i, j, ix, iy
    integer :: nx, ny, ilo, ihi, jlo, jhi
    integer :: ixs(0:3), iys(0:3)

    nx = qx - 2*ng; ny = qy - 2*ng
    ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

    call fill_BCS_f(qx, qy, ng, v, bcs, v_out)
    ixs = (/ 0, 1, 1, 0 /)
    iys = (/ 0, 1, 0, 1 /)

    v_out(:,:) = v(:,:)

    do i = 0, nsmooth-1
        do j = 0, 3
            ix = ixs(j)
            iy = iys(j)

            denom(ilo+ix:ihi:2, jlo+iy:jhi:2) = &
                eta_x(ilo+1+ix:ihi+1:2, jlo+iy:jhi:2) + &
                eta_x(ilo+ix:ihi:2, jlo+iy:jhi:2) + &
                eta_y(ilo+ix:ihi:2, jlo+1+iy:jhi+1:2) + &
                eta_y(ilo+ix:ihi:2, jlo+iy:jhi:2)

            v_out(ilo+ix:ihi:2, jlo+iy:jhi:2) = &
                (-f(ilo+ix:ihi:2, jlo+iy:jhi:2) + &
                eta_x(ilo+1+ix:ihi+1:2, jlo+iy:jhi:2) * &
                v_out(ilo+1+ix:ihi+1:2, jlo+iy:jhi:2) + &
                eta_x(ilo+ix:ihi:2, jlo+iy:jhi:2) * &
                v_out(ilo-1+ix:ihi-1:2, jlo+iy:jhi:2) + &
                eta_y(ilo+ix:ihi:2, jlo+1+iy:jhi+1:2) * &
                v_out(ilo+ix:ihi:2, jlo+1+iy:jhi+1:2) + &
                eta_y(ilo+ix:ihi:2, jlo+iy:jhi:2) * &
                v_out(ilo+ix:ihi:2, jlo-1+iy:jhi-1:2)) / &
                denom(ilo+ix:ihi:2, jlo+iy:jhi:2)

            if (j == 1 .or. j == 3) then
                call fill_BCS_f(qx, qy, ng, v_out, bcs, v_temp)
                v_out(:,:) = v_temp(:,:)
            end if

        end do
    end do

end subroutine smooth_f

subroutine fill_BCs_f(qx, qy, ng, g, bcs, g_out)

    implicit none

    integer, intent(in) :: qx, qy, ng
    double precision, intent(inout) :: g(0:qx-1, 0:qy-1)
    integer, intent(in) :: bcs(0:3)
    double precision, intent(out) :: g_out(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: g
!f2py depend(qx, qy) :: g_out
!f2py intent(inout) :: g
!f2py intent(in) :: bcs
!f2py intent(out) :: bcs

    integer :: i, j
    integer :: nx, ny, ilo, ihi, jlo, jhi

    nx = qx - 2*ng; ny = qy - 2*ng
    ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

    g_out(:,:) = g(:,:)

    ! -x boundary
    ! outflow, neumann
    if (bcs(0) == 0) then
        do  i = 0, ilo-1
            g_out(i,:) = g_out(ilo,:)
        end do

    ! reflect-even
    else if (bcs(0) == 1) then
        do i = 0, ilo-1
            g_out(i,:) = g_out(2*ng-i-1,:)
        end do

    ! reflect-odd, dirichlet
    else if (bcs(0) == 2) then
        do i = 0, ilo-1
            g_out(i,:) = -g_out(2*ng-i-1,:)
        end do

    ! periodic
    else
        do i = 0, ilo-1
            g_out(i,:) = g_out(ihi-ng+i+1,:)
        end do
    end if

    ! +x boundary
    ! outflow, neumann
    if (bcs(1) == 0) then
        do  i = ihi+1, 2*ng+nx-1
            g_out(i,:) = g_out(ihi,:)
        end do

    ! reflect-even
    else if (bcs(1) == 1) then
        do i = 0, ng-1
            g_out(ihi+1+i,:) = g_out(ihi-i,:)
        end do

    ! reflect-odd, dirichlet
    else if (bcs(1) == 2) then
        do i = 0, ng-1
            g_out(ihi+1+i,:) = -g_out(ihi-i,:)
        end do

    ! periodic
    else
        do i = ihi+1, qx-1
            g_out(i,:) = g_out(i-ihi-1+ng,:)
        end do
    end if

    ! -y boundary
    ! outflow, neumann
    if (bcs(2) == 0) then
        do j = 0, jlo-1
            g_out(:,j) = g_out(:,jlo)
        end do

    ! reflect-even
    else if (bcs(2) == 1) then
        do j = 0, jlo-1
            g_out(:,j) = g_out(:,2*ng-j-1)
        end do

    ! reflect-odd, dirichlet
    else if (bcs(2) == 2) then
        do j = 0, jlo-1
            g_out(:,j) = -g_out(:,2*ng-j-1)
        end do

    ! periodic
    else
        do j = 0, jlo-1
            g_out(:,j) = g_out(:,jhi-ng+j+1)
        end do
    end if

    ! +y boundary
    ! outflow, neumann
    if (bcs(3) == 0) then
        do j = jhi+1, ny+2*ng-1
            g_out(:,j) = g_out(:,jhi)
        end do

    ! reflect-even
    else if (bcs(3) == 1) then
        do j = 0, ng-1
            g_out(:,jhi+1+j) = g_out(:,jhi-j)
        end do

    ! reflect-odd, dirichlet
    else if (bcs(3) == 2) then
        do j = 0, ng-1
            g_out(:,jhi+1+j) = -g_out(:,jhi-j)
        end do

    ! periodic
    else
        do j = jhi+1, qy-1
            g_out(:,j) = g_out(:,j-jhi-1+ng)
        end do
    end if

end subroutine fill_BCs_f
