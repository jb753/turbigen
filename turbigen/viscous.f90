! Routines for adding viscous effects

subroutine shear_stress(cons, mu, xlength, taui, tauj, tauk, vol, dAi, dAj, dAk, r, rc, ni, nj, nk)

    implicit none

    real*4, intent (inout)  :: cons(ni, nj, nk, 5)

    real*4, intent (inout)  :: dAi(ni, nj-1, nk-1, 3)
    real*4, intent (inout)  :: dAj(ni-1, nj, nk-1, 3)
    real*4, intent (inout)  :: dAk(ni-1, nj-1, nk, 3)
    real*4, intent (inout)  :: vol(ni-1, nj-1, nk-1)
    real*4, intent (inout)  :: xlength(ni-1, nj-1, nk-1)
    real*4, intent (inout)  :: r(ni, nj, nk)
    real*4, intent (inout)  :: rc(ni-1, nj-1, nk-1)

    real*4, intent (inout)  :: mu

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*4 :: tauc(ni-1, nj-1, nk-1, 6)
    real*4, intent (inout) :: taui(ni, nj-1, nk-1, 6)
    real*4, intent (inout) :: tauj(ni-1, nj, nk-1, 6)
    real*4, intent (inout) :: tauk(ni-1, nj-1, nk, 6)

    real*4 :: visc_lim

    real*4 :: V(ni, nj, nk, 3)
    real*4 :: Vc(ni-1, nj-1, nk-1, 3)
    real*4 :: roc(ni-1, nj-1, nk-1)
    real*4 :: gradV(ni-1, nj-1, nk-1, 3, 3)
    real*4 :: divV(ni-1, nj-1, nk-1)
    real*4 :: vort(ni-1, nj-1, nk-1, 3)
    real*4 :: vort_mag(ni-1, nj-1, nk-1)
    real*4 :: mu_turb(ni-1, nj-1, nk-1)
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
    vort = 0e0
    vort(:,:,:,1) = gradV(:,:,:, 3, 2) - gradV(:,:,:,2,3) - Vc(:,:,:,3)/rc
    vort(:,:,:,2) = gradV(:,:,:, 1, 3) - gradV(:,:,:,3,1)
    vort(:,:,:,3) = gradV(:,:,:, 2, 1) - gradV(:,:,:,1,2)
    vort_mag = sqrt(sum(vort*vort,4))

    ! Set turbulent viscosity using mixing length
    mu_turb = roc*xlength*vort_mag

    ! Apply a limiting turbulent viscosity ratio
    visc_lim = 1000e0*mu
    where (mu_turb.ge.visc_lim)
        mu_turb = visc_lim
    end where

    ! Get shear stress for a Newtonian fluid
    do i = 1,6
        tauc(:,:,:,i) = -tauc(:,:,:,i) *( mu + mu_turb)
    end do

    ! Now distribute cell values to faces
    call cell_to_face(tauc, taui, tauj, tauk, ni, nj, nk, 6)

    ! Before evaluating the fluxes, we need to average across periodics
    ! To make shear stress continuous

end subroutine

subroutine viscous_force( &
        fvisc, cons, taui, tauj, tauk, vol, dAi, dAj, dAk, r, ri, rj, rk, &
        ijk_iwall, ijk_jwall, ijk_kwall, &
        dw_iwall, dw_jwall, dw_kwall, &
        dA_iwall, dA_jwall, dA_kwall, &
        mu, &
        ni, nj, nk, niwall, njwall, nkwall)

    implicit none

    real*4, intent (inout)  :: fvisc(ni-1, nj-1, nk-1, 5)

    real*4, intent (inout) :: taui(ni, nj-1, nk-1, 6)
    real*4, intent (inout) :: tauj(ni-1, nj, nk-1, 6)
    real*4, intent (inout) :: tauk(ni-1, nj-1, nk, 6)

    real*4, intent (in)  :: vol(ni-1, nj-1, nk-1)
    real*4, intent (in)  :: dAi(ni, nj-1, nk-1, 3)
    real*4, intent (in)  :: dAj(ni-1, nj, nk-1, 3)
    real*4, intent (in)  :: dAk(ni-1, nj-1, nk, 3)

    real*4, intent (in)  :: r(ni, nj, nk)
    real*4, intent (in)  :: ri(ni, nj-1, nk-1)
    real*4, intent (in)  :: rj(ni-1, nj, nk-1)
    real*4, intent (in)  :: rk(ni-1, nj-1, nk)

    real*4, intent (in) :: cons(ni, nj, nk, 5)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: niwall
    integer, intent (in)  :: njwall
    integer, intent (in)  :: nkwall

    real*4, intent (in) :: mu

    ! Wall locations
    integer*2, intent (in) :: ijk_iwall(3, niwall)
    integer*2, intent (in) :: ijk_jwall(3, njwall)
    integer*2, intent (in) :: ijk_kwall(3, nkwall)

    real*4, intent (in) :: dw_iwall(niwall)
    real*4, intent (in) :: dw_jwall(njwall)
    real*4, intent (in) :: dw_kwall(nkwall)

    real*4, intent (in) :: dA_iwall(niwall)
    real*4, intent (in) :: dA_jwall(njwall)
    real*4, intent (in) :: dA_kwall(nkwall)

    real*4 :: fi(ni, nj-1, nk-1, 3, 5)
    real*4 :: fj(ni-1, nj, nk-1, 3, 5)
    real*4 :: fk(ni-1, nj-1, nk, 3, 5)

    real*4 :: fvisc_new(ni-1, nj-1, nk-1, 5)

    real*4 :: rfvisc
    rfvisc = 0.2e0

    ! No shear stress at wall
    ! We add back using wall functions later
    call zero_wall_stress(taui, ijk_iwall, 1, ni, nj-1, nk-1, niwall)
    call zero_wall_stress(tauj, ijk_jwall, 2, ni-1, nj, nk-1, njwall)
    call zero_wall_stress(tauk, ijk_kwall, 3, ni-1, nj-1, nk, nkwall)

    ! Assemble the viscous fluxes from the stress tensor components
    call viscous_flux(fi, taui, ri, ni, nj-1, nk-1)
    call viscous_flux(fj, tauj, rj, ni-1, nj, nk-1)
    call viscous_flux(fk, tauk, rk, ni-1, nj-1, nk)

    ! Get the net flux into each cell
    call sum_fluxes(fi, fj, fk, dAi, dAj, dAk, vol, fvisc_new, ni, nj, nk, 5)

    ! ! Zero the forces on cells at the wall
    ! ! Because with wall functions the velocity gradient at the wall should
    ! ! have no effect on the solution. We only care about the shear stress
    ! ! that is transmitted up to the next cell off the wall.
    ! call zero_wall_forces(fvisc_new, ijk_iwall, ni, nj, nk, niwall)
    ! call zero_wall_forces(fvisc_new, ijk_jwall, ni, nj, nk, njwall)
    ! call zero_wall_forces(fvisc_new, ijk_kwall, ni, nj, nk, nkwall)

    ! ! Add on forces one cell off wall due to stress from wall function
    call wall_function( &
        fvisc_new, ijk_iwall, 1, cons, r, vol, dw_iwall, dA_iwall, mu, ni, nj, nk, niwall &
    )
    call wall_function( &
        fvisc_new, ijk_jwall, 2, cons, r, vol, dw_jwall, dA_jwall, mu, ni, nj, nk, njwall &
    )
    call wall_function( &
        fvisc_new, ijk_kwall, 3, cons, r, vol, dw_kwall, dA_kwall, mu, ni, nj, nk, nkwall &
    )

    ! Apply relaxation
    fvisc = rfvisc*fvisc_new + (1e0-rfvisc)*fvisc
    ! fvisc = fvisc_new

end subroutine

subroutine viscous_flux(f, tau, r, ni, nj, nk)

    implicit none
    real*4, intent (in) :: tau(ni, nj, nk, 6)
    real*4, intent (out) :: f(ni, nj, nk, 3, 5)
    real*4, intent (in) :: r(ni, nj, nk)

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
    integer, intent(in) :: dirn
    real*4, intent (in) :: cons(ni, nj, nk, 5)
    real*4, intent (in) :: r(ni, nj, nk)
    real*4, intent (in) :: vol(ni-1,nj-1,nk-1)

    real*4, intent (in) :: dw(nwall)
    real*4, intent (in) :: dA(nwall)
    real*4, intent (in) :: mu

    real*4 :: rw
    real*4 :: Rew

    real*4 :: roVxrtw(4)
    real*4 :: Vxrtw(3)
    real*4 :: vec(3)
    real*4 :: row
    real*4 :: Vw
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

    a1 = -1.767e-3
    a2 = 3.177e-2
    a3 = 2.5614-1

    roVxrtw = 0e0
    row = 0e0
    rw = 0e0

    ! If we have at least one wall
    if (nwall > 0) then
        ! Loop over all points
        do iwall = 1,nwall

            ! Extract indices
            i = ijk(1, iwall)
            j = ijk(2, iwall)
            k = ijk(3, iwall)

            ! Skip dummy points
            if (i.lt.0) then
                cycle
            end if

            ! Choose wall direction
            if (dirn.eq.1) then

                ! These are i-faces

                ! Choose the i index of one node off wall
                if (i.eq.1) then
                    i1 = i + 1
                else
                    i1 = i - 1
                end if
                j1 = j
                k1 = k

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

                ! Choose the j index of one node off wall
                if (j.eq.1) then
                    j1 = j + 1
                else
                    j1 = j- 1
                end if
                i1 = i
                k1 = k

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


                ! Choose index for one node off wall
                if (k.eq.1) then
                    k1 = k + 1
                else
                    k1 = k - 1
                end if
                i1 = i
                j1 = j

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

            ! Form the cell Reynolds
            Vw = sqrt(sum(Vxrtw*Vxrtw, 1))
            Rew = row * Vw * dw(iwall)/mu
            lnRew = alog(Rew)
            if (Rew.lt.125e0) then
                cf = 2e0/Rew
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

            ! ic = i1
            ! jc = j1
            ! kc = k1

            ! multiply by face area magnitude
            ! direction is opposite to cell velocity
            vec = -Vxrtw*dA(iwall)/vol(ic, jc, kc)
            if (Vw.gt.0e0) then
                vec = vec/Vw
            else
                vec = 0e0
            end if
            ! print*, ic, jc, kc

            vtau = sqrt(tauw/row)
            yplus = row*vtau*dw(iwall)/mu
            ! print*,yplus

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

            f(ic, jc, kc, 2) = f(ic, jc, kc, 2) + vec(1)*tauw
            f(ic, jc, kc, 3) = f(ic, jc, kc, 3) + vec(2)*tauw
            f(ic, jc, kc, 4) = f(ic, jc, kc, 4) + rc*vec(3)*tauw

            ! f(ic, jc, kc, 2) =  vec(1)*tauw
            ! f(ic, jc, kc, 3) =  vec(2)*tauw
            ! f(ic, jc, kc, 4) =  rc*vec(3)*tauw

        end do
    end if


end subroutine


subroutine zero_wall_forces(f, ijk, ni, nj, nk, nwall)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nwall

    real*4, intent (inout) :: f(ni-1, nj-1, nk-1, 5)
    integer*2, intent (in) :: ijk(3, nwall)

    integer :: i
    integer :: j
    integer :: k
    integer :: iwall
    integer :: ic
    integer :: jc
    integer :: kc

    ! If we have at least one wall
    if (nwall > 0) then

        ! Loop over all points
        do iwall = 1,nwall

            ! Extract indices
            i = ijk(1, iwall)
            j = ijk(2, iwall)
            k = ijk(3, iwall)

            ! Skip dummy points
            if (i.lt.0) then
                cycle
            end if

            ! Get indices into the cells for this face
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

            ! Set momentum forcings to zero
            f(ic, jc, kc, 2:4) = 0e0

        end do
    end if


end subroutine


subroutine zero_wall_stress(tau, ijk, dirn, ni, nj, nk, nwall)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nwall

    integer, intent (in)  :: dirn

    ! Warning: depending on which direction faces we are setting,
    ! tau will have smaller dimension, e.g.
    !   taui(ni, nj-1, nk-1, 6)
    real*4, intent (inout) :: tau(ni, nj, nk, 6)
    integer*2, intent (in) :: ijk(3, nwall)

    integer :: i
    integer :: j
    integer :: k
    integer :: iwall
    integer :: i1 = 0
    integer :: j1 = 0
    integer :: k1 = 0

    ! If we have at least one wall
    if (nwall > 0) then

        ! Loop over all points
        do iwall = 1,nwall

            ! Extract indices
            i = ijk(1, iwall)
            j = ijk(2, iwall)
            k = ijk(3, iwall)

            ! Skip dummy points
            if (i.lt.0) then
                cycle
            end if

            ! ! Choose wall direction
            ! if (dirn.eq.1) then

            !     ! These are i-faces

            !     ! Choose the i index of one node off wall
            !     if (i.eq.1) then
            !         i1 = i + 1
            !     else
            !         i1 = i - 1
            !     end if
            !     j1 = j
            !     k1 = k

            ! else if (dirn.eq.2) then

            !     ! These are j-faces

            !     ! Choose the j index of one node off wall
            !     if (j.eq.1) then
            !         j1 = j + 1
            !     else
            !         j1 = j- 1
            !     end if
            !     i1 = i
            !     k1 = k

            ! else if (dirn.eq.3) then

            !     ! This is a k-face

            !     ! Choose index for one node off wall
            !     if (k.eq.1) then
            !         k1 = k + 1
            !     else
            !         k1 = k - 1
            !     end if
            !     i1 = i
            !     j1 = j

            ! end if

            ! ! Now set shear stress to zero one face off wall
            tau(i, j, k, :) = 0e0

        end do

    end if

end subroutine
