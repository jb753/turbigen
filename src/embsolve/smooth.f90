! Blended 4th and 2nd order smoothing on a 4D array
!
! Smooths x towards linear and cubic fits, which cause
! 2nd- and 4th-order errors respectively. The 4th-order
! term is constant throughout the flow and provides
! background dissipation to suppress odd-even decoupling.
! The 2nd-order term adapts to the flow, being proportional
! to the second derivative of pressure and switching off
! the 4th-order term: usually it is only active in
! non-smooth regions such as shock waves. However, a floor
! can be set to the 2nd-order term to provide constant
! smoothing. The effect of smoothing in each grid direction
! is scaled proportional to the grid spacing via L.
!
subroutine smooth( &
        x, P, L, &  ! Array to smooth
        sf4, sf2, sf2min, &  ! Smoothing factors
        ni, nj, nk, np &  ! Array sizes
    )

    ! Array sizes
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    ! Smoothing factors
    real, intent (in)  :: sf4
    real, intent (in)  :: sf2
    real, intent (in)  :: sf2min

    ! Array to smooth
    real, intent (inout)  :: x(ni, nj, nk, np)

    ! Pressure for adaptive term
    real, intent (in)  :: P(ni, nj, nk)

    ! Side length scale factors
    real, intent (in)  :: L(ni, nj, nk, 3)

    ! Working variables
    real :: nu(ni, nj, nk, 3)
    real :: xs2(ni, nj, nk, np, 3)
    real :: xs4(ni, nj, nk, np, 3)
    real :: sfx2(ni, nj, nk)
    real :: sfx4(ni, nj, nk)
    real :: sf2n(ni, nj, nk, 3)
    real :: sf4n(ni, nj, nk, 3)
    real :: sftn(ni, nj, nk)
    integer :: ip

    ! 2nd-order smoothed values for each direcion

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

    ! 4th-order smoothed values for each direcion
    xs4 = xs2

    ! i interior
    xs4(3:ni-2, :, :, :, 1) = ( &
        -     x(1:ni-4, :, :, :) + 4e0*x(2:ni-3, :, :, :) &
        + 4e0*x(4:ni-1, :, :, :) -     x(5:ni,   :, :, :) &
    )/6e0

    ! j interior
    xs4(:, 3:nj-2, :, :, 2) = ( &
        -     x(:, 1:nj-4, :, :) + 4e0*x(:, 2:nj-3, :, :) &
        + 4e0*x(:, 4:nj-1, :, :) -     x(:,   5:nj, :, :) &
    )/6e0

    ! k interior
    xs4(:, :, 3:nk-2, :, 3) = ( &
        -     x(:, :, 1:nk-4, :) + 4e0*x(:, :, 2:nk-3, :) &
        + 4e0*x(:, :, 4:nk-1, :) -     x(:,   :, 5:nk, :) &
    )/6e0

    ! Calculate the pressure sensor (Jameson et al. 1981)

    ! interior i
    nu(2:ni-1, :, :, 1) = &
        abs(P(1:ni-2, :, :) - 2e0*P(2:ni-1, :, :) + P(3:ni, :, :)) &
        /  (P(1:ni-2, :, :) + 2e0*P(2:ni-1, :, :) + P(3:ni, :, :))

    ! start/end i
    nu(1, :, :, 1) = &
        abs(P(1, :, :) - 2e0*P(2, :, :) + P(3, :, :)) &
        /  (P(1, :, :) + 2e0*P(2, :, :) + P(3, :, :))
    nu(ni, :, :, 1) = &
        abs(P(ni, :, :) - 2e0*P(ni-1, :, :) + P(ni-2, :, :)) &
        /  (P(ni, :, :) + 2e0*P(ni-1, :, :) + P(ni-2, :, :))

    ! interior j
    nu(:, 2:nj-1, :, 2) = &
        abs(P(:, 1:nj-2, :) - 2e0*P(:, 2:nj-1, :) + P(:, 3:nj, :)) &
        /  (P(:, 1:nj-2, :) + 2e0*P(:, 2:nj-1, :) + P(:, 3:nj, :))

    ! start/end j
    nu(:, 1, :, 2) = &
        abs(P(:, 1, :) - 2e0*P(:, 2, :) + P(:, 3, :)) &
        /  (P(:, 1, :) + 2e0*P(:, 2, :) + P(:, 3, :))
    nu(:, nj, :, 2) = &
        abs(P(:, nj, :) - 2e0*P(:, nj-1, :) + P(:, nj-2, :)) &
        /  (P(:, nj, :) + 2e0*P(:, nj-1, :) + P(:, nj-2, :))

    ! interior k
    nu(:, :, 2:nk-1, 3) = &
        abs(P(:, :, 1:nk-2) - 2e0*P(:, :, 2:nk-1) + P(:, :, 3:nk)) &
        /  (P(:, :, 1:nk-2) + 2e0*P(:, :, 2:nk-1) + P(:, :, 3:nk))

    ! start/end k
    nu(:, :, 1, 3) = &
        abs(P(:, :, 1) - 2e0*P(:, :, 2) + P(:, :, 3)) &
        /  (P(:, :, 1) + 2e0*P(:, :, 2) + P(:, :, 3))
    nu(:, :, nk, 3) = &
        abs(P(:, :, nk) - 2e0*P(:, :, nk-1) + P(:, :, nk-2)) &
        /  (P(:, :, nk) + 2e0*P(:, :, nk-1) + P(:, :, nk-2))

    ! Calculate nodal smoothing factors for each direction

    ! 2nd-order
    sf2n = sf2*nu
    where (sf2n.lt.sf2min)
        sf2n = sf2min
    end where

    ! 4th-order
    sf4n = sf4-sf2n
    where (sf4n.lt.0e0)
        sf4n = 0e0
    end where

    ! Apply the scale factors for cell side length
    sf2n = sf2n * L
    sf4n = sf4n * L

    ! Loop over properties
    do ip=1,np

        ! Products of local smoothing factors and flow property
        ! Summed over all grid directions
        sfx2 = sum(sf2n*xs2(:,:,:,ip,:),4)
        sfx4 = sum(sf4n*xs4(:,:,:,ip,:),4)

        ! Total smoothing factor for all grid directions
        sftn = sum(sf2n + sf4n,4)

        ! Do the smoothing
        x(:,:,:,ip) = (1e0-sftn)*x(:,:,:,ip)  + sfx2 + sfx4

    end do



end subroutine
