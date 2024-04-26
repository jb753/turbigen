! Functions for accessing 4D arrays using unstructured lists of ijk

! Retrieve data from the 4D array x at the given list of ijk
! Return in an unstructured list
subroutine get_by_ijk(x, xu, ijk, ni, nj, nk, nv, npt)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nv
    integer, intent (in)  :: npt

    real*4, intent (inout) :: x(ni, nj, nk, nv)
    real*4, intent (out) :: xu(npt*nv)
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
subroutine average_by_ijk(x1, x2, ijk1, ijk2, npt)

    integer, intent (in)  :: npt

    real*4, intent (inout) :: x1(:, :, :, :)
    real*4, intent (inout) :: x2(:, :, :, :)
    integer*2, intent (in) :: ijk1(3, npt)
    integer*2, intent (in) :: ijk2(3, npt)

    integer :: ipt
    real*4 :: avg(5)

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

    real*4, intent (inout) :: x(ni, nj, nk, nv)
    real*4, intent (inout) :: xu(nb)
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


! Given two 4D arrays and lists of nodal ijk indexes into each,
! average the variables at the faces that correspond to these
! nodal indexes and assign back to both the original arrays
subroutine face_average_by_ijk(x1, x2, ijk1, ijk2, d1, d2, npt)

    integer, intent (in)  :: npt

    real*4, intent (inout) :: x1(:, :, :, :)
    real*4, intent (inout) :: x2(:, :, :, :)
    integer*2, intent (in) :: ijk1(3, npt)
    integer*2, intent (in) :: ijk2(3, npt)
    integer, intent (in) :: d1
    integer, intent (in) :: d2

    integer :: ipt
    real*4 :: avg(5)

    integer :: i1
    integer :: j1
    integer :: k1

    integer :: i2
    integer :: j2
    integer :: k2

    integer :: s1(4)
    integer :: s2(4)

    integer*2 :: ijkf1(3, npt)
    integer*2 :: ijkf2(3, npt)

    ! Convert nodal to face indices
    ! Setting invalid points to -1
    s1 = size(x1)
    s2 = size(x2)
    call node_index_to_face(ijk1, ijkf1, s1, d1, npt)
    call node_index_to_face(ijk2, ijkf2, s2, d2, npt)

    ! If we have some points
    if (npt > 0) then
        ! Loop over all points
        do ipt = 1,npt

            ! Extract indices on first block
            i1 = ijk1(1, ipt)
            j1 = ijk1(2, ipt)
            k1 = ijk1(3, ipt)
            i2 = ijk2(1, ipt)
            j2 = ijk2(2, ipt)
            k2 = ijk2(3, ipt)

            ! Skip invalid points
            if ((i1.lt.0).or.(i2.lt.0)) then
                continue
            end if

            ! Get average
            avg = 0.5e0*(x1(i1, j1, k1, :) + x2(i2, j2, k2, :))
            x1(i1, j1, k1, :) = avg
            x2(i2, j2, k2, :) = avg

        end do
    end if

end subroutine


subroutine node_index_to_face(ijk, ijkf, s, d, npt)

    integer*2, intent (in) :: ijk(3, npt)
    integer*2, intent (inout) :: ijkf(3, npt)

    integer, intent (in) :: d
    integer, intent (in) :: s(4)
    integer, intent (in) :: npt

    integer :: i
    integer :: j
    integer :: k

    integer :: ipt

    integer :: ni
    integer :: nj
    integer :: nk

    ni = s(1)
    nj = s(2)
    nk = s(3)

    ! If we have some points
    if (npt > 0) then
        ! Loop over all points
        do ipt = 1,npt

            ! Extract indices
            i = ijk(1, ipt)
            j = ijk(2, ipt)
            k = ijk(3, ipt)

            ! Convert the nodal indices to face indices
            if (d.eq.1) then
                ! i-faces: same i, discard j=nj and k=nk
                if ((j.eq.nj).or.(k.eq.nk)) then
                    ijkf(:, ipt) = -1
                    continue
                end if
            else if (d.eq.2) then
                ! j-faces: same j, discard i=ni and k=nk
                if ((i.eq.ni).or.(k.eq.nk)) then
                    ijkf(:, ipt) = -1
                    continue
                end if
            else if (d.eq.3) then
                ! k-faces: same k, discard i=ni and j=nj
                if ((i.eq.ni).or.(j.eq.nj)) then
                    ijkf(:, ipt) = -1
                    continue
                end if
            end if
            ijkf(:,ipt) = ijk(:, ipt)
        end do
    end if

end subroutine
