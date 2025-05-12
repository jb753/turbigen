! Evaluate and summing fluxes

subroutine set_fluxes( &
    cons, Vxrt, P, ho, &                  ! Flow properties
    Omega, &                              ! Reference frame angular velocity
    r, ri, rj, rk, &                      ! Node and face-centered radii
    ijk_iwall, ijk_jwall, ijk_kwall, &    ! Wall locations
    fluxi, fluxj, fluxk, &                ! Fluxes out
    ni, nj, nk, niwall, njwall, nkwall &  ! Numbers of points dummy args
    )

    ! Flow properties and body force
    ! Nodal conserved quantities: rho, rhoVx, rhoVr, rhorVt, rhoe
    real, intent (in) :: cons(ni, nj, nk, 5)
    real, intent (in) :: Vxrt(ni, nj, nk, 3)
    real, intent (in) :: P   (ni, nj, nk)
    real, intent (in) :: ho  (ni, nj, nk)

    ! Reference frame angular velocity
    real, intent (in) :: Omega

    ! Radii at nodes and face centers
    real, intent(in) :: r( ni, nj, nk)
    real, intent(in) :: ri( ni, nj-1, nk-1)
    real, intent(in) :: rj( ni-1, nj, nk-1)
    real, intent(in) :: rk( ni-1, nj-1, nk)

    ! Wall locations
    integer*2, intent (in) :: ijk_iwall(3, niwall)
    integer*2, intent (in) :: ijk_jwall(3, njwall)
    integer*2, intent (in) :: ijk_kwall(3, nkwall)

    ! Fluxes out
    real, intent (inout) :: fluxi(ni, nj-1, nk-1, 3, 5)
    real, intent (inout) :: fluxj(ni-1, nj, nk-1, 3, 5)
    real, intent (inout) :: fluxk(ni-1, nj-1, nk, 3, 5)

    ! Numbers of points dummy args
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: niwall
    integer, intent (in)  :: njwall
    integer, intent (in)  :: nkwall

    ! End of input declarations

    ! Declare working variables

    ! Face pressures
    real :: Pi( ni, nj-1, nk-1)
    real :: Pj( ni-1, nj, nk-1)
    real :: Pk( ni-1, nj-1, nk)

    ! Fluxes per unit mass
    real :: fmass( ni, nj, nk, 4)
    real :: fmassi( ni, nj-1, nk-1, 4)
    real :: fmassj( ni-1, nj, nk-1, 4)
    real :: fmassk( ni-1, nj-1, nk, 4)

    ! Mass fluxes
    real :: rhoV(ni, nj, nk, 3)
    real :: rhoVi(ni, nj-1, nk-1, 3)
    real :: rhoVj(ni-1, nj, nk-1, 3)
    real :: rhoVk(ni-1, nj-1, nk, 3)

    ! Misc
    integer :: id
    integer :: ip

    ! Extract the quantities we will need to get fluxes
    rhoV = cons(:, :, :, 2:4)
    rhoV(:, :, :, 3) = cons(:,:,:,1)*(Vxrt(:, :, :, 3) - Omega*r)


    ! Calculate face-centered pressure
    call node_to_face( P, Pi, Pj, Pk, ni, nj, nk, 1)

    ! Evaluate the mass flux at face centers
    call node_to_face( rhoV, rhoVi, rhoVj, rhoVk, ni, nj, nk, 3)

    ! zero mass fluxes on the wall
    call zero_wall_fluxes(rhoVi, ijk_iwall, ni, nj-1, nk-1, 3, niwall)
    call zero_wall_fluxes(rhoVj, ijk_jwall, ni-1, nj, nk-1, 3, njwall)
    call zero_wall_fluxes(rhoVk, ijk_kwall, ni-1, nj-1, nk, 3, nkwall)

    ! Mass fluxes through ijk faces
    fluxi(:, :, :, :, 1) = rhoVi
    fluxj(:, :, :, :, 1) = rhoVj
    fluxk(:, :, :, :, 1) = rhoVk

    fmass(:, :, :, 1) = Vxrt(:,:,:,1)  ! axial momentum per unit mass
    fmass(:, :, :, 2) = Vxrt(:,:,:,2)  ! radial momentum per unit mass
    fmass(:, :, :, 3) = Vxrt(:,:,:,3)*r ! angular momentum per unit mass
    fmass(:, :, :, 4) = ho  ! energy per unit mass

    ! Distribute to the faces
    call node_to_face( fmass, fmassi, fmassj, fmassk, ni, nj, nk, 4)

    ! Now multiply fmass and rhoV for fluxes of other quantites
    do ip = 1,4
        do id = 1,3
            fluxi(:, :, :, id, ip+1) = rhoVi(:, :, :, id) * fmassi(:, :, :, ip)
            fluxj(:, :, :, id, ip+1) = rhoVj(:, :, :, id) * fmassj(:, :, :, ip)
            fluxk(:, :, :, id, ip+1) = rhoVk(:, :, :, id) * fmassk(:, :, :, ip)
        end do
    end do

    ! Add pressure fluxes
    call add_pressure_fluxes(fluxi, Pi, ri, Omega, ni, nj-1, nk-1)
    call add_pressure_fluxes(fluxj, Pj, rj, Omega, ni-1, nj, nk-1)
    call add_pressure_fluxes(fluxk, Pk, rk, Omega, ni-1, nj-1, nk)

end subroutine

subroutine add_pressure_fluxes(flux, P, r, Omega, ni, nj, nk)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    real, intent (in)  :: r(ni, nj, nk)
    real, intent (in)  :: Omega
    real, intent (out) :: flux(ni, nj, nk, 3, 5)
    real, intent (in)  :: P(ni, nj, nk)

    ! pressure fluxes
    ! x-mom in x-dirn
    flux(:, :, :, 1, 2) = flux(:, :, :, 1, 2) + P
    ! r-mom in r-dirn
    flux(:, :, :, 2, 3) = flux(:, :, :, 2, 3) + P
    ! rt-mom in t-dirn
    flux(:, :, :, 3, 4) = flux(:, :, :, 3, 4) + r*P
    ! ho in t-dirn
    flux(:, :, :, 3, 5) = flux(:, :, :, 3, 5) + r*Omega*P


end subroutine


subroutine sum_fluxes(fi, fj, fk, dAi, dAj, dAk, fsum, ni, nj, nk, np)
    implicit none

    integer, intent (in)  :: ni, nj, nk, np
    real, intent (in)     :: dAi(ni, nj-1, nk-1, 3)
    real, intent (in)     :: dAj(ni-1, nj, nk-1, 3)
    real, intent (in)     :: dAk(ni-1, nj-1, nk, 3)
    real, intent (in)     :: fi(ni, nj-1, nk-1, 3, np)
    real, intent (in)     :: fj(ni-1, nj, nk-1, 3, np)
    real, intent (in)     :: fk(ni-1, nj-1, nk, 3, np)
    real, intent (out)    :: fsum(ni-1, nj-1, nk-1, np)

    integer :: i, j, k, ip, d
    real :: fisum, fjsum, fksum

    ! Warning, this version seems to have more numerical error
    ! than using intrinsic sum() for the dot product
    ! but is faster.
    do ip = 1, np
        do k = 1, nk-1
            do j = 1, nj-1
                do i = 1, ni-1
                    fisum = 0.0
                    fjsum = 0.0
                    fksum = 0.0
                    do d = 1, 3
                        fisum = fisum + dAi(i, j, k, d) * fi(i, j, k, d, ip)
                        fisum = fisum - dAi(i+1, j, k, d) * fi(i+1, j, k, d, ip)

                        fjsum = fjsum + dAj(i, j, k, d) * fj(i, j, k, d, ip)
                        fjsum = fjsum - dAj(i, j+1, k, d) * fj(i, j+1, k, d, ip)

                        fksum = fksum + dAk(i, j, k, d) * fk(i, j, k, d, ip)
                        fksum = fksum - dAk(i, j, k+1, d) * fk(i, j, k+1, d, ip)
                    end do
                    fsum(i, j, k, ip) = fisum + fjsum + fksum
                end do
            end do
        end do
    end do
end subroutine

subroutine zero_wall_fluxes(x, ijk, ni, nj, nk, nc, npt)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nc
    integer, intent (in)  :: npt

    real, intent (inout) :: x(ni, nj, nk, nc)
    integer*2, intent (in) :: ijk(3, npt)

    integer :: ipt

    ! If we have some points
    if (npt > 0) then
        ! Loop over all points
        do ipt = 1,npt
            ! Set to zero
            x(ijk(1,ipt) , ijk(2,ipt), ijk(3,ipt), :) = 0e0
        end do
    end if

end subroutine
