! Functions for accessing 4D arrays using unstructured lists of ijk

! Retrieve data from the 4D array x at the given list of ijk
! Return in an unstructured list
subroutine get_by_ijk(x, xu, ijk, ni, nj, nk, nv, npt)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nv
    integer, intent (in)  :: npt

    real, intent (inout) :: x(ni, nj, nk, nv)
    real, intent (out) :: xu(npt*nv)
    integer*2, intent (in) :: ijk(3, npt)

    integer :: ipt
    integer :: i
    integer :: iv
    integer :: j
    integer :: k

    ! If we have some points
    if (npt > 0) then
        ! Loop over all points
        do ipt = 1,npt

            ! Extract indices
            i = ijk(1, ipt)
            j = ijk(2, ipt)
            k = ijk(3, ipt)

            ! Loop over vars
            do iv = 1,nv
                xu(nv*(ipt-1)+iv) = x(i, j, k, iv)
            end do

        end do
    end if

end subroutine

! Given two 4D arrays and lists of ijk indexes into each,
! average the variables at corresponding indexes and assign
! back to both the original arrays
subroutine average_by_ijk(x1, x2, ijk1, ijk2, ni1, nj1, nk1, ni2, nj2, nk2, npt, nv)

    integer, intent (in)  :: npt
    integer, intent (in)  :: ni1
    integer, intent (in)  :: nj1
    integer, intent (in)  :: nk1
    integer, intent (in)  :: ni2
    integer, intent (in)  :: nj2
    integer, intent (in)  :: nk2
    integer, intent (in) :: nv

    real, intent (inout) :: x1(ni1, nj1, nk1, nv)
    real, intent (inout) :: x2(ni2, nj2, nk2, nv)
    integer*2, intent (in) :: ijk1(3, npt)
    integer*2, intent (in) :: ijk2(3, npt)

    integer :: ipt
    real :: avg(nv)


    integer :: i1
    integer :: j1
    integer :: k1

    integer :: i2
    integer :: j2
    integer :: k2

    ! If we have some points
    if (npt > 0) then
        ! Loop over all points
        do ipt = 1,npt

            ! Extract indices
            i1 = ijk1(1, ipt)
            j1 = ijk1(2, ipt)
            k1 = ijk1(3, ipt)
            i2 = ijk2(1, ipt)
            j2 = ijk2(2, ipt)
            k2 = ijk2(3, ipt)

            ! Get average
            avg = 0.5e0*(x1(i1, j1, k1, :) + x2(i2, j2, k2, :))
            x1(i1, j1, k1, :) = avg
            x2(i2, j2, k2, :) = avg

        end do
    end if

end subroutine

subroutine set_by_ijk(x, xu, ijk, ni, nj, nk, nv, npt, nb)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nv
    integer, intent (in)  :: nb
    integer, intent (in)  :: npt

    real, intent (inout) :: x(ni, nj, nk, nv)
    real, intent (inout) :: xu(nb)
    integer*2, intent (inout) :: ijk(3, npt)

    integer :: ipt
    integer :: i
    integer :: iv
    integer :: j
    integer :: k

    ! If we have some points
    if (npt > 0) then
        ! Loop over all points
        do ipt = 1,npt

            ! Extract indices
            i = ijk(1, ipt)
            j = ijk(2, ipt)
            k = ijk(3, ipt)

            ! Loop over vars
            do iv = 1,nv
                x(i, j, k, iv) = xu(nv*(ipt-1)+iv)
            end do

        end do
    end if

end subroutine
