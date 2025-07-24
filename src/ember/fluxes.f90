! Evaluate and summing fluxes


subroutine add_polar_source( &
    cons, Vxrt, P, Pref, r, vol, fsum, &
    ni, nj, nk &
    )

    implicit none

    ! Flow properties and body force
    ! Nodal conserved quantities: rho, rhoVx, rhoVr, rhorVt, rhoe
    real, intent (in) :: cons(ni, nj, nk, 5)
    real, intent (in) :: Vxrt(ni, nj, nk, 3)
    real, intent (in) :: P(ni, nj, nk)
    real, intent (in) :: Pref
    real, intent (in) :: r(ni, nj, nk)
    real, intent (in) :: vol(ni-1, nj-1, nk-1)
    real, intent (inout) :: fsum(ni-1, nj-1, nk-1)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real :: Sn(ni, nj, nk)
    real :: S(ni-1, nj-1, nk-1)
    real :: rho(ni, nj, nk)
    real :: Vt(ni, nj, nk)

    rho = cons(:, :, :, 1)
    Vt = Vxrt(:, :, :, 3)
    Sn = (rho*Vt*Vt + (P-Pref))/r
    call node_to_cell(Sn, S, ni, nj, nk, 1)
    S = S * vol
    fsum = fsum + S

end subroutine

subroutine set_fluxes( &
    cons, Vxrt, h, r, Omega, &                  ! Flow properties
    ijk_iwall, ijk_jwall, ijk_kwall, &    ! Wall locations
    fluxi, fluxj, fluxk, &                ! Fluxes out
    ni, nj, nk, niwall, njwall, nkwall &  ! Numbers of points dummy args
    )

    ! Flow properties and body force
    ! Nodal conserved quantities: rho, rhoVx, rhoVr, rhorVt, rhoe
    real, intent (in) :: cons(ni, nj, nk, 5)
    real, intent (in) :: Vxrt(ni, nj, nk, 3)
    real, intent (in) :: h  (ni, nj, nk)
    real, intent (in) :: r  (ni, nj, nk)
    real, intent (in) :: Omega

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

    ! Stagnation enthalpy
    real :: ho( ni, nj, nk)

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
    ho = h + 0.5e0*sum(Vxrt*Vxrt, 4)


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

end subroutine

subroutine add_pressure_fluxes_all(fluxi, fluxj, fluxk, P, Pref, ri, rj, rk, Omega, ni, nj, nk)

    ! Reference frame angular velocity
    real, intent (in) :: Omega

    ! Radii at nodes and face centers
    real, intent(in) :: P( ni, nj, nk)

    ! Reference pressure
    real, intent (in) :: Pref

    ! Fluxes out
    real, intent (inout) :: fluxi(ni, nj-1, nk-1, 3, 5)
    real, intent (inout) :: fluxj(ni-1, nj, nk-1, 3, 5)
    real, intent (inout) :: fluxk(ni-1, nj-1, nk, 3, 5)

    integer, intent (in) :: ni
    integer, intent (in) :: nj
    integer, intent (in) :: nk

    ! Working vars
    real, intent(in) :: ri( ni, nj-1, nk-1)
    real, intent(in) :: rj( ni-1, nj, nk-1)
    real, intent(in) :: rk( ni-1, nj-1, nk)

    real :: Pm( ni, nj, nk)
    real :: Pi( ni, nj-1, nk-1)
    real :: Pj( ni-1, nj, nk-1)
    real :: Pk( ni-1, nj-1, nk)


    ! Calculate face-centered pressure
    Pm = P - Pref
    call node_to_face( Pm, Pi, Pj, Pk, ni, nj, nk, 1)

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
    real, intent (inout)    :: fsum(ni-1, nj-1, nk-1, np)

    integer :: ip
    real :: fisum(ni, nj-1, nk-1)
    real :: fjsum(ni-1, nj, nk-1)
    real :: fksum(ni-1, nj-1, nk)

    do ip = 1, np

        ! Face fluxes
        fisum = sum(dAi*fi(:, :, :, :, ip), dim=4)
        fjsum = sum(dAj*fj(:, :, :, :, ip), dim=4)
        fksum = sum(dAk*fk(:, :, :, :, ip), dim=4)

        ! Net cell flux
        fsum(:,:,:,ip) = ( &
            fisum(1:ni-1,:,:) - fisum(2:ni,:,:) & ! i faces
            + fjsum(:,1:nj-1,:) - fjsum(:,2:nj,:) & ! j faces
            + fksum(:,:,1:nk-1) - fksum(:,:,2:nk) & ! k faces
        )


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
