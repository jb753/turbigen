! Routines for adding viscous effects

subroutine viscous_force(cons, fvisc, mu, mu_turb, xlength, vol, dAi, dAj, dAk, r, rc, ri, rj, rk, ni, nj, nk)

    implicit none

    real*4, intent (inout)  :: cons(ni, nj, nk, 5)
    real*4, intent (inout)  :: fvisc(ni-1, nj-1, nk-1, 5)
    real*4 :: fvisc_new(ni-1, nj-1, nk-1, 5)

    real*4, intent (inout)  :: dAi(ni, nj-1, nk-1, 3)
    real*4, intent (inout)  :: dAj(ni-1, nj, nk-1, 3)
    real*4, intent (inout)  :: dAk(ni-1, nj-1, nk, 3)
    real*4, intent (inout)  :: vol(ni-1, nj-1, nk-1)
    real*4, intent (inout)  :: xlength(ni-1, nj-1, nk-1)
    real*4, intent (inout)  :: r(ni, nj, nk)
    real*4, intent (inout)  :: rc(ni-1, nj-1, nk-1)
    real*4, intent (inout)  :: ri(ni, nj-1, nk-1)
    real*4, intent (inout)  :: rj(ni-1, nj, nk-1)
    real*4, intent (inout)  :: rk(ni-1, nj-1, nk)

    real*4, intent (inout)  :: mu

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*4 :: tauc(ni-1, nj-1, nk-1, 6)
    real*4 :: taui(ni, nj-1, nk-1, 6)
    real*4 :: tauj(ni-1, nj, nk-1, 6)
    real*4 :: tauk(ni-1, nj-1, nk, 6)

    real*4 :: visc_lim

    real*4 :: fi(ni, nj-1, nk-1, 3, 5)
    real*4 :: fj(ni-1, nj, nk-1, 3, 5)
    real*4 :: fk(ni-1, nj-1, nk, 3, 5)


    real*4 :: V(ni, nj, nk, 3)
    real*4 :: Vc(ni-1, nj-1, nk-1, 3)
    real*4 :: roc(ni-1, nj-1, nk-1)
    real*4 :: gradV(ni-1, nj-1, nk-1, 3, 3)
    real*4 :: divV(ni-1, nj-1, nk-1)
    real*4 :: vort(ni-1, nj-1, nk-1, 3)
    real*4 :: vort_mag(ni-1, nj-1, nk-1)
    real*4, intent (inout)  :: mu_turb(ni-1, nj-1, nk-1)
    integer :: i


    ! Evaluate velocities
    do i = 1,3
        V(:,:,:, i) = cons(:,:,:,i+1)/cons(:,:,:,1)
    end do
    V(:,:,:,3) = V(:,:,:,3)/r


    ! Cell-centered vars
    call node_to_cell(V, Vc, ni, nj, nk, 3)
    call node_to_cell(cons(:,:,:,1), roc, ni, nj, nk, 1)

    ! Calculate grad V
    do i = 1,3
        call grad(V(:,:,:,i), gradV(:,:,:,:,i), vol, dAi, dAj, dAk, r, rc, ni, nj, nk)
    end do
    ! gradV is indexed (..., which dirn, which velocity)

    ! Calculate divergence of V
    call div(V, divV, vol, dAi, dAj, dAk, ni, nj, nk)
    divV = divV*2e0/3e0

    ! tau contains the six unique terms in the tensor
    ! divV and gradV are cell-centered

    ! tau_xx = 2*dVx_dx - 2/3*divV
    tauc(:,:,:,1) = 2e0*gradV(:,:,:,1,1) - divV

    ! tau_rr = 2*dVr_dr - 2/3*divV
    tauc(:,:,:,2) = 2e0*gradV(:,:,:,2,2) - divV

    ! tau_tt = 2*(dVt_dt/r + Vr/r) - 2/3*divV
    tauc(:,:,:,3) = 2e0*(gradV(:,:,:,3,3)+ Vc(:,:,:,2))/rc - divV

    ! tau_xr = tau_rx = dVx_dr + dVr_dx
    tauc(:,:,:,4) = gradV(:,:,:,2,1) + gradV(:,:,:,1,2)

    ! tau_xt = tau_tx = dVx_dt/r + dVt_dx
    tauc(:,:,:,5) = gradV(:,:,:,3,1)/rc + gradV(:,:,:,1,3)

    ! tau_rt = tau_tr = dVr_dt/r + dVt_dr - Vt/r
    tauc(:,:,:,6) = gradV(:,:,:,3,2)/rc + gradV(:,:,:,2,3) - Vc(:,:,:,3)/rc


    ! Calculate vorticity
    ! From multall
    ! omega_x = dVr/dt - dVt/dr - Vt/r
    ! omega_r = dVt/dx - dVx/dt
    ! omega_t = dVx/dr - dVr/dx
    ! From databook curl V
    ! omega_x = (1/r)[d(rVt)/dr - dVr/dt]
    ! omega_r = (1/r)[d(Vx)/dt - d(rVt)/dz]
    vort = 0e0
    vort(:,:,:,1) = gradV(:,:,:, 3, 2) - gradV(:,:,:,2,3) - Vc(:,:,:,3)/rc
    vort(:,:,:,2) = gradV(:,:,:, 1, 3) - gradV(:,:,:,3,1)
    vort(:,:,:,3) = gradV(:,:,:, 2, 1) - gradV(:,:,:,1,2)
    vort_mag = sqrt(sum(vort*vort,4))

    mu_turb = roc*xlength*vort_mag
    visc_lim = 1000e0*mu
    where (mu_turb.ge.visc_lim)
        mu_turb = visc_lim
    end where

    do i = 1,6
        tauc(:,:,:,i) = -tauc(:,:,:,i) *( mu + mu_turb)
    end do

    ! Now distribute cell values to faces
    call cell_to_face(tauc, taui, tauj, tauk, ni, nj, nk, 6)

    ! We need to assemble the viscous fluxes from the stress tensor components
    call viscous_flux(fi, taui, ri, ni, nj-1, nk-1)
    call viscous_flux(fj, tauj, rj, ni-1, nj, nk-1)
    call viscous_flux(fk, tauk, rk, ni-1, nj-1, nk)

    ! Get the net flux into each cell
    call sum_fluxes(fi, fj, fk, dAi, dAj, dAk, vol, fvisc_new, ni, nj, nk, 5)

    ! Apply relaxation
    fvisc = 0.1e0*fvisc_new + 0.9e0*fvisc

end subroutine

subroutine viscous_flux(f, tau, r, ni, nj, nk)

    implicit none
    real*4, intent (inout) :: tau(ni, nj, nk, 6)
    real*4, intent (inout) :: f(ni, nj, nk, 3, 5)
    real*4, intent (inout) :: r(ni, nj, nk)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    ! 1 tau_xx
    ! 2 tau_rr
    ! 3 tau_tt
    ! 4 tau_xr
    ! 5 tau_xt
    ! 6 tau_rt

    ! mass
    f(:, :, :, :, 1) = 0e0

    ! x-momentum
    f(:, :, :, 1, 2) = tau(:, :, :, 1)  ! tau_xx
    f(:, :, :, 2, 2) = tau(:, :, :, 4)  ! tau_xr
    f(:, :, :, 3, 2) = tau(:, :, :, 5)  ! tau_xt

    ! r-momentum
    f(:, :, :, 1, 3) = tau(:, :, :, 4)  ! tau_rx
    f(:, :, :, 2, 3) = tau(:, :, :, 2)  ! tau_rr
    f(:, :, :, 3, 3) = tau(:, :, :, 6)  ! tau_rt

    ! rt-momentum
    f(:, :, :, 1, 4) = tau(:, :, :, 5) * r  ! tau_tx
    f(:, :, :, 2, 4) = tau(:, :, :, 6) * r  ! tau_tr
    f(:, :, :, 3, 4) = tau(:, :, :, 3) * r  ! tau_tt

    ! energy
    f(:, :, :, :, 5) = 0e0

end subroutine


! Add on cell forces due to wall functions
subroutine wall_function(f, ijk, dirn, cons, r, vol, dw, dA, mu, ni, nj, nk, nwall)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nwall

    real*4, intent (inout) :: f(ni-1, nj-1, nk-1, 5)
    integer*2, intent (in) :: ijk(3, nwall)
    integer*2 :: dirn
    real*4, intent (inout) :: cons(ni, nj, nk, 5)
    real*4, intent (in) :: r(ni, nj, nk)
    real*4, intent (inout) :: vol(ni-1,nj-1,nk-1)

    real*4, intent (in) :: dw(nwall)
    real*4, intent (in) :: dA(nwall)
    real*4, intent (in) :: mu

    real*4 :: rw
    real*4 :: rw0
    real*4 :: Rew

    real*4 :: roVxrtw(4)
    real*4 :: Vxrtw(3)
    real*4 :: vec(3)
    real*4 :: roVxrtw0(4)
    real*4 :: Vxrtw0(3)
    real*4 :: row0
    real*4 :: row
    real*4 :: Vw
    real*4 :: Vw0
    integer :: iwall
    integer :: i
    integer :: j
    integer :: k
    integer :: i1
    integer :: j1
    integer :: k1
    integer :: ic
    integer :: jc
    integer :: kc

    real*4 :: a1
    real*4 :: a2
    real*4 :: a3
    real*4 :: lnRew
    real*4 :: cf
    real*4 :: tauw
    real*4 :: rc
    real*4 :: yplus
    real*4 :: vtau

    a1 = 1.767e-3
    a2 = 3.177e-2
    a3 = 2.5614-1

    roVxrtw = 0e0
    row = 0e0
    rw = 0e0

    roVxrtw0 = 0e0
    row0 = 0e0
    rw0 = 0e0

    ! If we have at least one wall
    if (nwall > 0) then
        ! Loop over all points
        do iwall = 1,nwall

            ! Extract indices
            i = ijk(1, iwall)
            j = ijk(2, iwall)
            k = ijk(3, iwall)

            ! Choose wall direction
            if (dirn.eq.1) then

                ! These are i-faces

                ! Face-centered density and velocity on wall
                roVxrtw0 = ( &
                    cons(i, j  , k   ,1:4) &
                    + cons(i, j+1, k   ,1:4) &
                    + cons(i, j  , k+1 ,1:4) &
                    + cons(i, j+1, k+1 ,1:4) &
                )/4e0
                rw0 = ( &
                    r(i, j  , k  ) &
                    + r(i, j+1, k  ) &
                    + r(i, j  , k+1) &
                    + r(i, j+1, k+1) &
                )/4e0

                ! Choose the i index of one node off wall
                if (i.eq.1) then
                    i1 = i + 1
                else
                    i1 = i - 1
                end if

                ! Face-centered density and velocity
                roVxrtw = ( &
                    cons(i1, j  , k   ,1:4) &
                    + cons(i1, j+1, k   ,1:4) &
                    + cons(i1, j  , k+1 ,1:4) &
                    + cons(i1, j+1, k+1 ,1:4) &
                )/4e0
                rw = ( &
                    r(i1, j  , k  ) &
                    + r(i1, j+1, k  ) &
                    + r(i1, j  , k+1) &
                    + r(i1, j+1, k+1) &
                )/4e0

            else if (dirn.eq.2) then

                ! These are j-faces

                ! Face-centered density and velocity on wall
                roVxrtw0 = ( &
                    cons(i  , j, k  , 1:4) &
                    + cons(i+1, j, k  , 1:4) &
                    + cons(i  , j, k+1, 1:4) &
                    + cons(i+1, j, k+1, 1:4) &
                )/4e0
                rw0 = ( &
                    r(i  , j, k  ) &
                    + r(i+1, j, k  ) &
                    + r(i  , j, k+1) &
                    + r(i+1, j, k+1) &
                )/4e0

                ! Choose the j index of one node off wall
                if (j.eq.1) then
                    j1 = j + 1
                else
                    j1 = j- 1
                end if

                ! Face-centered density and velocity
                roVxrtw = ( &
                    cons(i  , j1, k  , 1:4) &
                    + cons(i+1, j1, k  , 1:4) &
                    + cons(i  , j1, k+1, 1:4) &
                    + cons(i+1, j1, k+1, 1:4) &
                )/4e0
                rw = ( &
                    r(i  , j1, k  ) &
                    + r(i+1, j1, k  ) &
                    + r(i  , j1, k+1) &
                    + r(i+1, j1, k+1) &
                )/4e0

            else if (dirn.eq.3) then


                ! These are k faces
                ! Face-centered density and velocity
                roVxrtw0 = ( &
                    cons(i  , j  , k, 1:4) &
                    + cons(i+1, j  , k, 1:4) &
                    + cons(i  , j+1, k, 1:4) &
                    + cons(i+1, j+1, k, 1:4) &
                )/4e0
                rw0 = ( &
                    r(i  , j  , k) &
                    + r(i+1, j  , k) &
                    + r(i  , j+1, k) &
                    + r(i+1, j+1, k) &
                )/4e0

                ! Choose index for one node off wall
                if (k.eq.1) then
                    k1 = k + 1
                else
                    k1 = k - 1
                end if

                ! Face-centered density and velocity
                roVxrtw = ( &
                    cons(i  , j  , k1, 1:4) &
                    + cons(i+1, j  , k1, 1:4) &
                    + cons(i  , j+1, k1, 1:4) &
                    + cons(i+1, j+1, k1, 1:4) &
                )/4e0
                rw = ( &
                    r(i  , j  , k1) &
                    + r(i+1, j  , k1) &
                    + r(i  , j+1, k1) &
                    + r(i+1, j+1, k1) &
                )/4e0

            end if

            roVxrtw(4) = roVxrtw(4)/rw
            row = roVxrtw(1)
            Vxrtw = roVxrtw(2:4)/row

            roVxrtw0(4) = roVxrtw0(4)/rw0
            row0 = roVxrtw0(1)
            Vxrtw0 = roVxrtw0(2:4)/row0

            ! Form the cell Reynolds
            Vw = sqrt(sum(Vxrtw*Vxrtw, 1))
            Vw0 = sqrt(sum(Vxrtw0*Vxrtw0, 1))
            Rew = row * Vw * dw(iwall)/mu
            lnRew = alog(Rew)
            if (Rew.lt.125e0) then
                cf = 1e0/Rew
            else
                cf = a1 + a2/lnRew + a3/lnRew/lnRew
            end if
            tauw = cf * 0.5e0 * row *Vw*Vw

            ! Get indices into the cell for this face
            if (i.eq.ni) then
                ic = ni-1
            else
                ic = i
            end if
            if (j.eq.nj) then
                jc = nj-1
            else
                jc = j
            end if
            if (k.eq.nk) then
                kc = nk-1
            else
                kc = k
            end if

            ! multiply by face area magnitude
            ! direction is opposite to cell velocity
            vec = -Vxrtw0*dA(iwall)/vol(ic, jc, kc)
            if (Vw.gt.0e0) then
                vec = vec/Vw0
            else
                vec = 0e0
            end if
            ! print*, ic, jc, kc

            vtau = sqrt(tauw/row)
            yplus = row*vtau*dw(iwall)/mu

            ! print*, yplus
            ! print*, Vxrtw, Vw
            ! print*, vec

            rc = ( &
                r(ic, jc, kc) &
                + r(ic+1, jc, kc) &
                + r(ic, jc+1, kc) &
                + r(ic+1, jc+1, kc) &
                + r(ic, jc, kc+1) &
                + r(ic+1, jc, kc+1) &
                + r(ic, jc+1, kc+1) &
                + r(ic+1, jc+1, kc+1) &
            )/8e0

            ! print*, i, j, k
            ! print*, ic, jc, kc
            ! print*, ni, nj, nk
            ! print*, tauw, vec
            ! print*, mu, row, Vw, dw(iwall)
            ! print*, Rew, tauw, dA(iwall), vol(ic, jc, kc)


            f(ic, jc, kc, 2) = f(ic, jc, kc, 2) + vec(1)*tauw
            f(ic, jc, kc, 3) = f(ic, jc, kc, 3) + vec(2)*tauw
            f(ic, jc, kc, 4) = f(ic, jc, kc, 4) + rc*vec(3)*tauw

        end do
    end if


end subroutine
