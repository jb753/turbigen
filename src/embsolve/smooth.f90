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
    implicit none

    ! Inputs
    integer, intent(in) :: ni, nj, nk, np
    real, intent(in) :: sf4, sf2, sf2min
    real, intent(inout) :: x(ni, nj, nk, np)
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: L(ni, nj, nk, 3)

    ! Locals
    integer :: i, j, k, ip, d
    real :: xs2(3), xs4(3), nu(3)
    real :: sf2n(3), sf4n(3), sfx2, sfx4, sftn
    real :: p1, p2, p3, denom
    real :: xnew(ni, nj, nk, np)

    do ip = 1, np
        do k = 1, nk
            do j = 1, nj
                do i = 1, ni

                    ! Loop over directions: 1=i, 2=j, 3=k
                    do d = 1, 3
                        ! Compute smoothed xs2 and xs4 (2nd and 4th order)

                        select case (d)
                        case (1)  ! i-direction
                            if (i == 1) then
                                xs2(d) = 2e0 * x(2,j,k,ip) - x(3,j,k,ip)
                            else if (i == ni) then
                                xs2(d) = 2e0 * x(ni-1,j,k,ip) - x(ni-2,j,k,ip)
                            else
                                xs2(d) = 0.5e0 * (x(i-1,j,k,ip) + x(i+1,j,k,ip))
                            end if

                            if (i >= 3 .and. i <= ni-2) then
                                xs4(d) = (-x(i-2,j,k,ip) + 4e0*x(i-1,j,k,ip) + 4e0*x(i+1,j,k,ip) - x(i+2,j,k,ip)) / 6e0
                            else
                                xs4(d) = xs2(d)
                            end if

                            ! Pressure sensor
                            if (i == 1) then
                                p1 = P(1,j,k); p2 = P(2,j,k); p3 = P(3,j,k)
                            else if (i == ni) then
                                p1 = P(ni-2,j,k); p2 = P(ni-1,j,k); p3 = P(ni,j,k)
                            else
                                p1 = P(i-1,j,k); p2 = P(i,j,k); p3 = P(i+1,j,k)
                            end if

                        case (2)  ! j-direction
                            if (j == 1) then
                                xs2(d) = 2e0 * x(i,2,k,ip) - x(i,3,k,ip)
                            else if (j == nj) then
                                xs2(d) = 2e0 * x(i,nj-1,k,ip) - x(i,nj-2,k,ip)
                            else
                                xs2(d) = 0.5e0 * (x(i,j-1,k,ip) + x(i,j+1,k,ip))
                            end if

                            if (j >= 3 .and. j <= nj-2) then
                                xs4(d) = (-x(i,j-2,k,ip) + 4e0*x(i,j-1,k,ip) + 4e0*x(i,j+1,k,ip) - x(i,j+2,k,ip)) / 6e0
                            else
                                xs4(d) = xs2(d)
                            end if

                            if (j == 1) then
                                p1 = P(i,1,k); p2 = P(i,2,k); p3 = P(i,3,k)
                            else if (j == nj) then
                                p1 = P(i,nj-2,k); p2 = P(i,nj-1,k); p3 = P(i,nj,k)
                            else
                                p1 = P(i,j-1,k); p2 = P(i,j,k); p3 = P(i,j+1,k)
                            end if

                        case (3)  ! k-direction
                            if (k == 1) then
                                xs2(d) = 2e0 * x(i,j,2,ip) - x(i,j,3,ip)
                            else if (k == nk) then
                                xs2(d) = 2e0 * x(i,j,nk-1,ip) - x(i,j,nk-2,ip)
                            else
                                xs2(d) = 0.5e0 * (x(i,j,k-1,ip) + x(i,j,k+1,ip))
                            end if

                            if (k >= 3 .and. k <= nk-2) then
                                xs4(d) = (-x(i,j,k-2,ip) + 4e0*x(i,j,k-1,ip) + 4e0*x(i,j,k+1,ip) - x(i,j,k+2,ip)) / 6e0
                            else
                                xs4(d) = xs2(d)
                            end if

                            if (k == 1) then
                                p1 = P(i,j,1); p2 = P(i,j,2); p3 = P(i,j,3)
                            else if (k == nk) then
                                p1 = P(i,j,nk-2); p2 = P(i,j,nk-1); p3 = P(i,j,nk)
                            else
                                p1 = P(i,j,k-1); p2 = P(i,j,k); p3 = P(i,j,k+1)
                            end if
                        end select

                        ! Calculate pressure limiter
                        nu(d) = abs(p1 - 2e0*p2 + p3) / (p1 + 2e0*p2 + p3)

                        ! Compute smoothing factors
                        sf2n(d) = max(sf2 * nu(d), sf2min)
                        sf2n(d) = sf2n(d) * L(i,j,k,d)
                        sf4n(d) = max(sf4 - sf2n(d), 0e0)
                        sf4n(d) = sf4n(d) * L(i,j,k,d)
                    end do

                    ! Combine smoothing directions
                    sfx2 = 0e0
                    sfx4 = 0e0
                    sftn = 0e0
                    do d = 1, 3
                        sfx2 = sfx2 + sf2n(d) * xs2(d)
                        sfx4 = sfx4 + sf4n(d) * xs4(d)
                        sftn = sftn + sf2n(d) + sf4n(d)
                    end do

                    ! Final update
                    xnew(i,j,k,ip) = (1e0 - sftn) * x(i,j,k,ip) + sfx2 + sfx4

                end do
            end do
        end do
    end do

    ! Copy smoothed array back to original
    do ip = 1, np
        do k = 1, nk
            do j = 1, nj
                do i = 1, ni
                    x(i,j,k,ip) = xnew(i,j,k,ip)
                end do
            end do
        end do
    end do

end subroutine
