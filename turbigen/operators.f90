! Operators for div/grad/smoothing on the grid

subroutine smooth(x, ssf, sf2, sf4, ni, nj, nk, np)
    ! Smooth the 4D array

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real*4, intent (in)  :: sf2
    real*4, intent (in)  :: sf4

    real*4, intent (inout)  :: x(ni, nj, nk, np)
    real*4, intent (inout)  :: ssf(ni, nj, nk, 3)
    real*4 :: xs2(ni, nj, nk, np, 3)
    real*4 :: xs4(ni, nj, nk, np, 3)

    integer :: ip

    ! ! Initialise to zero
    ! xs2 = 0e0
    ! xs4 = 0e0

    ! Accumulate 2nd-order smoothed values for each direcion in turn
    ! We will divide by three later

    ! i interior
    xs2(2:ni-1, :, :, :, 1) = ( &
        x(1:ni-2, :, :, :) + x(3:ni, :, :, :) &
    )/2e0

    ! i start
    xs2(1, :, :, :, 1) =  ( &
        2e0*x(2, :, :, :) - x(3, :, :, :) &
    )

    ! i end
    xs2(ni, :, :, :, 1) = ( &
        2e0*x(ni-1, :, :, :) - x(ni-2, :, :, :) &
    )

    ! j interior
    xs2(:, 2:nj-1, :, :, 2) = ( &
        x(:, 1:nj-2, :, :) + x(:, 3:nj,   :, :) &
    )/2e0

    ! j start
    xs2(:, 1, :, :, 2) =  ( &
        2e0*x(:, 2, :, :) - x(:, 3,   :, :) &
    )

    ! j end
    xs2(:, nj, :, :, 2) = ( &
        2e0*x(:, nj-1, :, :) - x(:, nj-2, :, :) &
    )

    ! k interior
    xs2(:, :, 2:nk-1, :, 3) = ( &
        x(:, :, 1:nk-2, :) + x(:, :,   3:nk, :) &
    )/2e0

    ! k start
    xs2(:, :, 1, :, 3) = ( &
        2e0*x(:, :, 2, :) - x(:, :,   3, :) &
    )

    ! k end
    xs2(:, :, nk, :, 3) = ( &
        2e0*x(:, :, nk-1, :) - x(:, :,   nk-2, :) &
    )

    ! Accumulate 4th-order smoothed values for each direcion in turn
    ! We will divide by three later

    ! i interior
    xs4(3:ni-2, :, :, :, 1) = ( &
        -     x(1:ni-4, :, :, :) + 4e0*x(2:ni-3, :, :, :) &
        + 4e0*x(4:ni-1, :, :, :) -     x(5:ni,   :, :, :) &
    )/6e0

    ! i=1
    xs4(1, :, :, :, 1) =  ( &
        4e0*x(2, :, :, :) - 6e0*x(3, :, :, :) &
        + 4e0*x(4, :, :, :) -     x(5, :, :, :) &
    )

    ! i=2
    xs4(2, :, :, :, 1) = ( &
            x(1, :, :, :) + 6e0*x(3, :, :, :) &
        - 4e0*x(4, :, :, :) +     x(5, :, :, :) &
    )/4e0

    ! i=ni-1
    xs4(ni-1, :, :, :, 1) = ( &
            x(ni-4, :, :, :) - 4e0*x(ni-3, :, :, :) &
        + 6e0*x(ni-2, :, :, :) +     x(ni, :, :, :) &
    )/4e0

    ! i=ni
    xs4(ni, :, :, :, 1) = ( &
        -     x(ni-4, :, :, :) + 4e0*x(ni-3, :, :, :) &
        - 6e0*x(ni-2, :, :, :) + 4e0*x(ni-1, :, :, :) &
    )


    ! j interior
    xs4(:, 3:nj-2, :, :, 2) = ( &
        -     x(:, 1:nj-4, :, :) + 4e0*x(:, 2:nj-3, :, :) &
        + 4e0*x(:, 4:nj-1, :, :) -     x(:,   5:nj, :, :) &
    )/6e0

    ! j=1
    xs4(:, 1, :, :, 2) = ( &
        4e0*x(:, 2, :, :) - 6e0*x(:, 3, :, :) &
        + 4e0*x(:, 4, :, :) -     x(:, 5, :, :) &
    )

    ! j=2
    xs4(:, 2, :, :, 2) = ( &
            x(:, 1, :, :) + 6e0*x(:, 3, :, :) &
        - 4e0*x(:, 4, :, :) +     x(:, 5, :, :) &
    )/4e0

    ! j=nj-1
    xs4(:, nj-1, :, :, 2) = ( &
            x(:, nj-4, :, :) - 4e0*x(:, nj-3, :, :) &
        + 6e0*x(:, nj-2, :, :) +     x(:, nj, :, :) &
    )/4e0

    ! j=nj
    xs4(:, nj, :, :, 2) = ( &
        -     x(:, nj-4, :, :) + 4e0*x(:, nj-3, :, :) &
        - 6e0*x(:, nj-2, :, :) + 4e0*x(:, nj-1, :, :) &
    )


    ! k interior
    xs4(:, :, 3:nk-2, :, 3) = ( &
        -     x(:, :, 1:nk-4, :) + 4e0*x(:, :, 2:nk-3, :) &
        + 4e0*x(:, :, 4:nk-1, :) -     x(:,   :, 5:nk, :) &
    )/6e0

    ! k=1
    xs4(:, :, 1, :, 3) = ( &
        4e0*x(:, :, 2, :) - 6e0*x(:, :, 3, :) &
        + 4e0*x(:, :, 4, :) -     x(:, :, 5, :) &
    )

    ! k=2
    xs4(:, :, 2, :, 3) = ( &
            x(:, :, 1, :) + 6e0*x(:, :, 3, :) &
        - 4e0*x(:, :, 4, :) +     x(:, :, 5, :) &
    )/4e0

    ! k=nk-1
    xs4(:, :, nk-1, :, 3) = ( &
            x(:, :, nk-4, :) - 4e0*x(:, :, nk-3, :) &
        + 6e0*x(:, :, nk-2, :) +     x(:, :, nk, :) &
    )/4e0

    ! k=nk
    xs4(:, :, nk, :, 3) = ( &
        -     x(:, :, nk-4, :) + 4e0*x(:, :, nk-3, :) &
        - 6e0*x(:, :, nk-2, :) + 4e0*x(:, :, nk-1, :) &
    )

    ! Apply the scale factors for each direction
    do ip = 1,np
        xs2(:, :, :, ip, :) = xs2(:, :, :, ip, :) * ssf(:, :, :, :)
        xs4(:, :, :, ip, :) = xs4(:, :, :, ip, :) * ssf(:, :, :, :)
    end do

    ! now smooth
    x = (1e0-sf2-sf4)*x + (sf2*sum(xs2,5) + sf4*sum(xs4,5))

end subroutine


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
