! Divergence and gradient operators

subroutine div(x, divx, vol, dAi, dAj, dAk, ni, nj, nk)
    ! Divergence at cell center by summing fluxes

    real, intent (in)  :: x(ni, nj, nk, 3)

    real, intent (in)  :: dAi(ni, nj-1, nk-1, 3)
    real, intent (in)  :: dAj(ni-1, nj, nk-1, 3)
    real, intent (in)  :: dAk(ni-1, nj-1, nk, 3)
    real, intent (in)  :: vol(ni-1, nj-1, nk-1)

    real, intent (inout)  :: divx(ni-1, nj-1, nk-1)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real :: xi(ni, nj-1, nk-1, 3)
    real :: xj(ni-1, nj, nk-1, 3)
    real :: xk(ni-1, nj-1, nk, 3)

    call node_to_face( x, xi, xj, xk, ni, nj, nk, 3 )

    call sum_fluxes(xi, xj, xk, dAi, dAj, dAk, divx, ni, nj, nk, 1)
    divx = -divx/vol

end subroutine


subroutine grad(x, gradx, vol, dAi, dAj, dAk, r, rc, ni, nj, nk)
    ! Gradient at cell centers
    !
    ! We construct a vector field u from the scalar of
    ! interest phi, such that div u = dphi/dx say.
    ! Summing the fluxes of u into each cell and dividing
    ! by volume gives, by Gauss' Theorem, the volume-averaged
    ! dphi/dx. Repeat for the three coordinate directions.

    real, intent (in)  :: x(ni, nj, nk)

    real, intent (in)  :: dAi(ni, nj-1, nk-1, 3)
    real, intent (in)  :: dAj(ni-1, nj, nk-1, 3)
    real, intent (in)  :: dAk(ni-1, nj-1, nk, 3)
    real, intent (in)  :: vol(ni-1, nj-1, nk-1)

    real, intent (inout)  :: gradx(ni-1, nj-1, nk-1, 3)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer :: ii

    real :: xi(ni, nj-1, nk-1, 3)
    real :: xj(ni-1, nj, nk-1, 3)
    real :: xk(ni-1, nj-1, nk, 3)
    real :: xv(ni, nj, nk, 3)

    real, intent (in)  :: r(ni, nj, nk)
    real, intent (in) :: rc(ni-1, nj-1, nk-1)

    ! Initialise vector to hold scalar in each direction
    xv = 0e0

    ! Loop over coordinate directions
    do ii = 1,3

        !$omp parallel

        ! Set the current coordinate direction of the
        ! storge vector to the scalar we want to take
        ! gradient of
        !$omp workshare
        xv(:,:,:,ii) = x
        !$omp end workshare

        ! Special case for theta direction
        if (ii.eq.2) then
            !$omp workshare
            xv(:,:,:,ii) = xv(:,:,:,ii)/r
            !$omp end workshare
        end if

        ! Find values on faces
        call node_to_face( xv, xi, xj, xk, ni, nj, nk, 3 )
        !$omp end parallel

        ! Apply Gauss' theorem to get volume-averaged spatial derivative
        call sum_fluxes(xi, xj, xk, dAi, dAj, dAk, gradx(:,:,:,ii), ni, nj, nk, 1)
        gradx(:,:,:,ii) = - gradx(:,:,:,ii) / vol

        !$omp parallel

        ! Special case the theta direction
        if (ii.eq.2) then
            !$omp workshare
            gradx(:,:,:,ii) = gradx(:,:,:,ii)*rc
            !$omp end workshare
        end if

        ! Reset the storage vector to zero
        !$omp workshare
        xv(:,:,:,ii) = 0e0
        !$omp end workshare
        !$omp end parallel

    end do

end subroutine
