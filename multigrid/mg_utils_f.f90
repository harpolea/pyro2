function smooth_f(qx, qy, ng, level, nsmooth, v, f, bcs, eta_x, eta_y, vout)

    implicit none

    integer, intent(in) :: qx, qy, ng, level, nsmooth
    double precision, intent(inout) :: v(0:qx-1, 0:qy-1)
    double precision, intent(inout) :: f(0:qx-1, 0:qy-1)
    integer, intent(inout) :: bcs(0:3)
    double precision, intent(inout) :: eta_x(0:qx-1, 0:qy-1)
    double precision, intent(inout) :: eta_y(0:qx-1, 0:qy-1)
    double precision, intent(out) :: vout(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: v, f
!f2py depend(qx, qy) :: eta_x, eta_y
!f2py depend(qx, qy) :: vout
!f2py intent(in) :: v, f, bcs, eta_x, eta_y
!f2py intent(out) :: vout

    double precision :: denom(0:qx-1, 0:qy-1)

    integer :: i, j, ix, iy
    integer :: nx, ny, ilo, ihi, jlo, jhi
    integer :: ixs(0:4), iys(0:4)

    nx = qx - 2*ng; ny = qy - 2*ng
    ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

    vout = fill_BCS(qx, qy, ng, v, bcs)
    ixs = (/ 0, 1, 1, 0 /)
    iys = (/0, 1, 0, 1/)

    vout(:,:) = v(:,:)

    do i = 0, nsmooth-1
        do j = 0, 3
            ix = ixs(j)
            iy = iys(j)

            denom(myg.ilo+ix:myg.ihi:2,myg.jlo+iy:myg.jhi:2) = eta_x(myg.ilo+1+ix:myg.ihi+1:2,myg.jlo+iy:myg.jhi:2) + eta_x(myg.ilo+ix:myg.ihi:2,myg.jlo+iy:myg.jhi:2) + eta_y(myg.ilo+ix:myg.ihi:2,myg.jlo+1+iy:myg.jhi+1:2) + eta_y(myg.ilo+ix:myg.ihi:2,myg.jlo+iy:myg.jhi:2)

            vout(myg.ilo+ix:myg.ihi:2,myg.jlo+iy:myg.jhi:2) = (-f(myg.ilo+ix:myg.ihi:2,myg.jlo+iy:myg.jhi:2) + eta_x(myg.ilo+1+ix:myg.ihi+1:2,myg.jlo+iy:myg.jhi:2) * vout(myg.ilo+1+ix:myg.ihi+1:2,myg.jlo+iy:myg.jhi:2) + eta_x(myg.ilo+ix:myg.ihi:2,myg.jlo+iy:myg.jhi:2) * vout(myg.ilo-1+ix:myg.ihi-1:2,myg.jlo+iy:myg.jhi:2) + eta_y(myg.ilo+ix:myg.ihi:2,myg.jlo+1+iy:myg.jhi+1:2) vout(myg.ilo+ix:myg.ihi:2,myg.jlo+1+iy:myg.jhi+1:2) + eta_y(myg.ilo+ix:myg.ihi:2,myg.jlo+iy:myg.jhi:2) * vout(myg.ilo+ix:myg.ihi:2,myg.jlo-1+iy:myg.jhi-1:2)) / denom(myg.ilo+ix:myg.ihi:2,myg.jlo+iy:myg.jhi:2)

            if (n == 1 .or. n == 3) then
                vout = fill_BCS(qx, qy, ng, vout, bcs)
            end if

        end do
    end do

end function smooth_f

function fill_BCs_f(qx, qy, ng, g, bcs, newg)

    implicit none

    integer, intent(in) :: qx, qy, ng
    double precision, intent(inout) :: g(0:qx-1, 0:qy-1)
    integer, intent(in) :: bcs(0:3)
    double precision, intent(out) :: newg(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: g
!f2py depend(qx, qy) :: newg
!f2py intent(in) :: g, bcs
!f2py intent(out) :: newg

    integer :: i, j
    integer :: nx, ny, ilo, ihi, jlo, jhi

    nx = qx - 2*ng; ny = qy - 2*ng
    ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

    newg(:,:) = g(:,:)

    ! -x boundary
    ! outflow, neumann
    if (bcs(0) == 0) then
        do  i = 0, ilo-1
            do j = 0, qy-1
                newg(i,j) = g(ilo,j)
            end do
        end do

    ! reflect-even
    else if (bcs(0) == 1) then
        do i = 0, ilo-1
            do j = 0, qy-1
                newg(i,j) = g(2*ng-i-1,j)
            end do
        end do

    ! reflect-odd, dirichlet
    else if (bcs(0) == 2) then
        do i = 0, ilo-1
            do j = 0, qy-1
                newg(i,j) = -g(2*ng-i-1,j)
            end do
        end do

    ! periodic
    else
        do i = 0, ilo-1
            do j = 0, qy-1
                newg(i,j) = g(ihi-ng+i+1,j)
            end do
        end do
    end if

    ! +x boundary
    ! outflow, neumann
    if (bcs(1) == 0) then
        do  i = ihi+1, qx-1
            do j = 0, qy-1
                newg(i,j) = g(ihi,j)
            end do
        end do

    ! reflect-even
    else if (bcs(1) == 1) then
        do i = 0, ng-1
            do j = 0, qy-1
                newg(ihi+1+i,j) = g(ihi-i,j)
            end do
        end do

    ! reflect-odd, dirichlet
    else if (bcs(1) == 2) then
        do i = 0, ng-1
            do j = 0, qy-1
                newg(ihi+1+i,j) = -g(ihi-i,j)
            end do
        end do

    ! periodic
    else
        do i = ihi+1, qx-1
            do j = 0, qy-1
                newg(i,j) = g(i-ihi-1+ng,j)
            end do
        end do
    end if

    ! -y boundary
    ! outflow, neumann
    if (bcs(2) == 0) then
        do  i = 0, qx-1
            do j = 0, jlo-1
                newg(i,j) = g(i,jlo)
            end do
        end do

    ! reflect-even
    else if (bcs(2) == 1) then
        do i = 0, qx-1
            do j = 0, jlo-1
                newg(i,j) = g(i,2*ng-j-1)
            end do
        end do

    ! reflect-odd, dirichlet
    else if (bcs(2) == 2) then
        do i = 0, qx-1
            do j = 0, jlo-1
                newg(i,j) = -g(i,2*ng-j-1)
            end do
        end do

    ! periodic
    else
        do i = 0, qx-1
            do j = 0, jlo-1
                newg(i,j) = g(i,jhi-ng+j+1)
            end do
        end do
    end if

    ! +y boundary
    ! outflow, neumann
    if (bcs(3) == 0) then
        do  i = 0, qx-1
            do j = jhi+1, qy-1
                newg(i,j) = g(i,jhi)
            end do
        end do

    ! reflect-even
    else if (bcs(3) == 1) then
        do i = 0, qx-1
            do j = 0, ng-1
                newg(i,jhi+1+j) = g(i,jhi-j)
            end do
        end do

    ! reflect-odd, dirichlet
    else if (bcs(3) == 2) then
        do i = 0, qx-1
            do j = 0, ng-1
                g(i,jhi+1+j) = -g(i,jhi-j)
            end do
        end do

    ! periodic
    else
        do i = 0, qx-1
            do j = jhi+1, qy-1
                g(i,j) = g(i,j-jhi-1+ng)
            end do
        end do
    end if


end function fill_BCs_f
