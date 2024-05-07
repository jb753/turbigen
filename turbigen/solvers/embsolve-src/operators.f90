! Divergence and gradient operators

subroutine div(x, divx, vol, dAi, dAj, dAk, ni, nj, nk)
    ! Divergence at cell center by summing fluxes

    real*4, intent (inout)  :: x(ni, nj, nk, 3)

    real*4, intent (inout)  :: dAi(ni, nj-1, nk-1, 3)
    real*4, intent (inout)  :: dAj(ni-1, nj, nk-1, 3)
    real*4, intent (inout)  :: dAk(ni-1, nj-1, nk, 3)
    real*4, intent (inout)  :: vol(ni-1, nj-1, nk-1)

    real*4, intent (inout)  :: divx(ni-1, nj-1, nk-1)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*4 :: xi(ni, nj-1, nk-1, 3)
    real*4 :: xj(ni-1, nj, nk-1, 3)
    real*4 :: xk(ni-1, nj-1, nk, 3)

    call node_to_face( x, xi, xj, xk, ni, nj, nk, 3 )

    call sum_fluxes(xi, xj, xk, dAi, dAj, dAk, -vol, divx, ni, nj, nk, 1)

end subroutine


subroutine grad(x, gradx, vol, dAi, dAj, dAk, r, rc, ni, nj, nk)
    ! Gradient at cell centers
    !
    ! We construct a vector field u from the scalar of
    ! interest phi, such that div u = dphi/dx say.
    ! Summing the fluxes of u into each cell and dividing
    ! by volume gives, by Gauss' Theorem, the volume-averaged
    ! dphi/dx. Repeat for the three coordinate directions.

    real*4, intent (inout)  :: x(ni, nj, nk)

    real*4, intent (inout)  :: dAi(ni, nj-1, nk-1, 3)
    real*4, intent (inout)  :: dAj(ni-1, nj, nk-1, 3)
    real*4, intent (inout)  :: dAk(ni-1, nj-1, nk, 3)
    real*4, intent (inout)  :: vol(ni-1, nj-1, nk-1)

    real*4, intent (inout)  :: gradx(ni-1, nj-1, nk-1, 3)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer :: ii

    real*4 :: xi(ni, nj-1, nk-1, 3)
    real*4 :: xj(ni-1, nj, nk-1, 3)
    real*4 :: xk(ni-1, nj-1, nk, 3)
    real*4 :: xv(ni, nj, nk, 3)

    real*4, intent (inout)  :: r(ni, nj, nk)
    real*4, intent (inout) :: rc(ni-1, nj-1, nk-1)

    ! Initialise vector to hold scalar in each direction
    xv = 0e0

    ! Loop over coordinate directions
    do ii = 1,3

        ! Set the current coordinate direction of the
        ! storge vector to the scalar we want to take
        ! gradient of
        xv(:,:,:,ii) = x

        ! Special case for theta direction
        if (ii.eq.2) then
            xv(:,:,:,ii) = xv(:,:,:,ii)/r
        end if

        ! Find values on faces
        call node_to_face( xv, xi, xj, xk, ni, nj, nk, 3 )

        ! Apply Gauss' theorem to get volume-averaged spatial derivative
        call sum_fluxes(xi, xj, xk, dAi, dAj, dAk, -vol, gradx(:,:,:,ii), ni, nj, nk, 1)

        ! Special case the theta direction
        if (ii.eq.2) then
            gradx(:,:,:,ii) = gradx(:,:,:,ii)*rc
        end if

        ! Reset the storage vector to zero
        xv(:,:,:,ii) = 0e0

    end do

end subroutine
