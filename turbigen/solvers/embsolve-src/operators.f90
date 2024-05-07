subroutine div(x, divx, vol, dAi, dAj, dAk, ni, nj, nk)

    implicit none

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

    implicit none

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

    xv = 0e0

    do ii = 1,3
        xv(:,:,:,ii) = x
        if (ii.eq.2) then
            xv(:,:,:,ii) = xv(:,:,:,ii)/r
        end if
        call node_to_face( xv, xi, xj, xk, ni, nj, nk, 3 )
        call sum_fluxes(xi, xj, xk, dAi, dAj, dAk, -vol, gradx(:,:,:,ii), ni, nj, nk, 1)
        if (ii.eq.2) then
            gradx(:,:,:,ii) = gradx(:,:,:,ii)*rc
        end if
        xv = 0e0
    end do

end subroutine
