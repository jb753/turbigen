! Routines for adding viscous effects

subroutine shear_stress(&
    cons, V, T, mu, cp, Pr_turb, xlength, vol, dAi, dAj, dAk, &
    Omega, r, rc, ri, rj, rk, ijk_iwall, ijk_jwall, ijk_kwall, dw_iwall, dw_jwall, dw_kwall, &
    dA_iwall, dA_jwall, dA_kwall, fvisc, ni, nj, nk, niwall, njwall, nkwall)

    implicit none

    real, intent (in)  :: cons(ni, nj, nk, 5)

    real, intent (in)  :: dAi(ni, nj-1, nk-1, 3)
    real, intent (in)  :: dAj(ni-1, nj, nk-1, 3)
    real, intent (in)  :: dAk(ni-1, nj-1, nk, 3)
    real, intent (in)  :: vol(ni-1, nj-1, nk-1)
    real, intent (in)  :: xlength(ni-1, nj-1, nk-1)
    real, intent (in)  :: r(ni, nj, nk)
    real, intent (in)  :: rc(ni-1, nj-1, nk-1)
    real, intent (in)  :: ri(ni, nj-1, nk-1)
    real, intent (in)  :: rj(ni-1, nj, nk-1)
    real, intent (in)  :: rk(ni-1, nj-1, nk)

    real, intent (in)  :: mu
    real, intent (in)  :: Pr_turb
    real, intent (in)  :: Omega
    real, intent (in)  :: cp

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: niwall
    integer, intent (in)  :: njwall
    integer, intent (in)  :: nkwall

    real :: tauc(ni-1, nj-1, nk-1, 6)
    real :: taui(ni, nj-1, nk-1, 6)
    real :: tauj(ni-1, nj, nk-1, 6)
    real :: tauk(ni-1, nj-1, nk, 6)

    real, intent (in) :: dw_iwall(niwall)
    real, intent (in) :: dw_jwall(njwall)
    real, intent (in) :: dw_kwall(nkwall)

    real, intent (in) :: dA_iwall(niwall)
    real, intent (in) :: dA_jwall(njwall)
    real, intent (in) :: dA_kwall(nkwall)
    real :: visc_lim

    real, intent (in) :: V(ni, nj, nk, 3)
    real :: T(ni, nj, nk)
    real :: Vc(ni-1, nj-1, nk-1, 3)
    real :: roc(ni-1, nj-1, nk-1)
    real :: gradV(ni-1, nj-1, nk-1, 3, 3)
    real :: gradT(ni-1, nj-1, nk-1, 3)
    real :: divV(ni-1, nj-1, nk-1)
    real :: vort(ni-1, nj-1, nk-1, 3)
    real :: vort_mag(ni-1, nj-1, nk-1)
    real :: mu_turb(ni-1, nj-1, nk-1)

    real :: Vi(ni, nj-1, nk-1, 3)
    real :: Vj(ni-1, nj, nk-1, 3)
    real :: Vk(ni-1, nj-1, nk, 3)

    integer*2, intent (in) :: ijk_iwall(3, niwall)
    integer*2, intent (in) :: ijk_jwall(3, njwall)
    integer*2, intent (in) :: ijk_kwall(3, nkwall)

    real, intent(inout) :: fvisc(ni-1, nj-1, nk-1, 5)
    real :: fvisc_new(ni-1, nj-1, nk-1, 5)
    integer :: i

    real :: fi(ni, nj-1, nk-1, 3, 5)
    real :: fj(ni-1, nj, nk-1, 3, 5)
    real :: fk(ni-1, nj-1, nk, 3, 5)

    real :: rfvisc
    real :: rfvisc1
    real :: k(ni-1, nj-1, nk-1)
    real :: qc(ni-1, nj-1, nk-1, 3)
    real :: qi(ni, nj-1, nk-1, 3)
    real :: qj(ni-1, nj, nk-1, 3)
    real :: qk(ni-1, nj-1, nk, 3)
    real :: Vrel(ni, nj, nk, 3)

    rfvisc = 0.2e0
    rfvisc1 = 1e0-rfvisc


    ! Relative velocity
    Vrel = V
    Vrel(:,:,:,3) = Vrel(:,:,:,3)-Omega*r

    !$omp parallel
    ! Cell-centered vars
    call node_to_cell(Vrel, Vc, ni, nj, nk, 3)
    call node_to_cell(cons(:,:,:,1), roc, ni, nj, nk, 1)
    !$omp end parallel

    call node_to_face(V, Vi, Vj, Vk, ni, nj, nk, 3)

    ! Calculate grad V
    do i = 1,3
        call grad(Vrel(:,:,:,i), gradV(:,:,:,:,i), vol, dAi, dAj, dAk, r, rc, ni, nj, nk)
    end do
    ! gradV is indexed (..., which dirn, which velocity)

    ! Temperature gradients
    call grad(T, gradT, vol, dAi, dAj, dAk, r, rc, ni, nj, nk)

    ! Calculate divergence of V
    call div(Vrel, divV, vol, dAi, dAj, dAk, ni, nj, nk)
    divV = divV*2e0/3e0

    ! tau contains the six unique terms in the tensor
    ! divV and gradV are cell-centered
    !$omp parallel
    !$omp workshare

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
    vort(:,:,:,1) = gradV(:,:,:, 3, 2) - gradV(:,:,:,2,3) - Vc(:,:,:,3)/rc
    vort(:,:,:,2) = gradV(:,:,:, 1, 3) - gradV(:,:,:,3,1)
    vort(:,:,:,3) = gradV(:,:,:, 2, 1) - gradV(:,:,:,1,2)

    vort_mag = sqrt(sum(vort*vort,4))

    ! Set turbulent viscosity using mixing length
    mu_turb = roc*xlength*vort_mag

    ! Apply a limiting turbulent viscosity ratio
    visc_lim = 3000e0*mu
    where (mu_turb.ge.visc_lim)
        mu_turb = visc_lim
    end where
    !$omp end workshare

    ! Thermal conductivity
    k = (mu + mu_turb) * cp / Pr_turb

    ! Cell heat flux
    do i = 1,3
        qc(:,:,:,i) = -k * gradT(:,:,:,i)
    end do

    ! Get shear stress for a Newtonian fluid
    do i = 1,6
        !$omp workshare
        tauc(:,:,:,i) = -tauc(:,:,:,i) * (mu + mu_turb)
        !$omp end workshare
    end do

    ! Now distribute cell values to faces
    call cell_to_face(tauc, taui, tauj, tauk, ni, nj, nk, 6)
    call cell_to_face(qc, qi, qj, qk, ni, nj, nk, 3)

    ! At this point, we could average across periodic patches
    ! to make the shear stress continous

    ! No shear stress at wall
    ! We add back using wall functions later
    call zero_wall_stress(taui, ijk_iwall, ni, nj-1, nk-1, niwall, 6)
    call zero_wall_stress(tauj, ijk_jwall, ni-1, nj, nk-1, njwall, 6)
    call zero_wall_stress(tauk, ijk_kwall, ni-1, nj-1, nk, nkwall, 6)

    ! No heat flow through walls
    call zero_wall_stress(qi, ijk_iwall, ni, nj-1, nk-1, niwall, 3)
    call zero_wall_stress(qj, ijk_jwall, ni-1, nj, nk-1, njwall, 3)
    call zero_wall_stress(qk, ijk_kwall, ni-1, nj-1, nk, nkwall, 3)

    ! Assemble the viscous fluxes from the stress tensor components
    call viscous_flux(fi, taui, qi, Vi, ri, ni, nj-1, nk-1)
    call viscous_flux(fj, tauj, qj, Vj, rj, ni-1, nj, nk-1)
    call viscous_flux(fk, tauk, qk, Vk, rk, ni-1, nj-1, nk)
    !$omp end parallel

    ! Get the net viscous force on each cell
    call sum_fluxes(fi, fj, fk, dAi, dAj, dAk, fvisc_new, ni, nj, nk, 5)

    ! ! Add on wall cell forces due to stress from wall function
    call wall_function( &
        fvisc_new, ijk_iwall, 1, cons, Omega, r, dw_iwall, dA_iwall, mu, ni, nj, nk, niwall &
    )
    call wall_function( &
        fvisc_new, ijk_jwall, 2, cons, Omega, r, dw_jwall, dA_jwall, mu, ni, nj, nk, njwall &
    )
    call wall_function( &
        fvisc_new, ijk_kwall, 3, cons, Omega, r, dw_kwall, dA_kwall, mu, ni, nj, nk, nkwall &
    )

    ! Apply relaxation
    !$omp parallel workshare
    fvisc = rfvisc*fvisc_new + rfvisc1*fvisc
    !$omp end parallel workshare

end subroutine

subroutine viscous_flux(f, tau, q, V, r, ni, nj, nk)

    implicit none
    real, intent (in) :: tau(ni, nj, nk, 6)
    real, intent (in) :: q(ni, nj, nk, 3)
    real, intent (out) :: f(ni, nj, nk, 3, 5)
    real, intent (in) :: r(ni, nj, nk)
    real, intent (in) :: V(ni, nj, nk,3)

    real :: wvisc(ni, nj, nk, 3)
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    ! 1 tau_xx
    ! 2 tau_rr
    ! 3 tau_tt
    ! 4 tau_xr
    ! 5 tau_xt
    ! 6 tau_rt

    !$omp workshare
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

    ! Tensor dot for shear work fluxes
    ! x-direction: V dot tau_?x = Vx*tau_xx + Vr*tau_rx + Vt*tau_tx
    wvisc(:,:,:,1) = &
        V(:,:,:,1)*tau(:,:,:,1) + V(:,:,:,2)*tau(:,:,:,4) + V(:,:,:,3)*tau(:,:,:,5)
    ! r-direction: V dot tau_?r = Vx*tau_xr + Vr*tau_rr + Vt*tau_tr
    wvisc(:,:,:,2) = &
        V(:,:,:,1)*tau(:,:,:,4) + V(:,:,:,2)*tau(:,:,:,2) + V(:,:,:,3)*tau(:,:,:,6)
    ! t-direction: V dot tau_?t = Vx*tau_xt + Vr*tau_rt + Vt*tau_tt
    wvisc(:,:,:,3) = &
        V(:,:,:,1)*tau(:,:,:,5) + V(:,:,:,2)*tau(:,:,:,6) + V(:,:,:,3)*tau(:,:,:,3)

    ! energy fluxes are heat fluxes plus shear work fluxes
    f(:, :, :, 1, 5) = q(:,:,:,1) + wvisc(:,:,:,1)
    f(:, :, :, 2, 5) = q(:,:,:,2) + wvisc(:,:,:,2)
    f(:, :, :, 3, 5) = q(:,:,:,3) + wvisc(:,:,:,3)

    !$omp end workshare

end subroutine


! Add on cell forces due to wall functions
subroutine wall_function(f, ijk, dirn, cons, &
        Omega, r, dw, dA, mu, &
        ni, nj, nk, nwall)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nwall

    real, intent (inout) :: f(ni-1, nj-1, nk-1, 5)
    integer*2, intent (in) :: ijk(3, nwall)
    integer, intent(in) :: dirn
    real, intent (in) :: cons(ni, nj, nk, 5)
    real, intent (in) :: r(ni, nj, nk)

    real, intent (in) :: dw(nwall)
    real, intent (in) :: dA(nwall)
    real, intent (in) :: mu
    real, intent (in) :: Omega


    real :: rw
    real :: Rew

    real :: roVxrtw(4)
    real :: Vxrtw(3)
    real :: vec(3)
    real :: row
    real :: Vw
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

    real :: a1
    real :: a2
    real :: a3
    real :: lnRew
    real :: cf
    real :: tauw
    real :: rc
    real :: Rew_lim = 127.53373025e0

    ! real :: yplus
    ! real :: vtau

    a1 = -1.767e-3
    a2 = 3.177e-2
    a3 = 2.5614e-1

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

            ! Put velocity in relative frame
            Vxrtw(3) = Vxrtw(3) - rw*Omega

            ! Form the cell Reynolds
            Vw = sqrt(sum(Vxrtw*Vxrtw, 1))
            Rew = row * Vw * dw(iwall)/mu
            ! lnRew = alog(Rew)
            lnRew = log(Rew)
            ! if (Rew.lt.125e0) then
            if (Rew.lt.Rew_lim) then
                ! Note: the TS user manual is off by factor of 2
                ! The below is correct and as in MULTALL
                cf = 2e0/Rew
            else
                cf = (a1 + a2/lnRew + a3/lnRew/lnRew)
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
            vec = -Vxrtw*dA(iwall)
            if (Vw.gt.0e0) then
                vec = vec/Vw
            else
                vec = 0e0
            end if

            ! vtau = sqrt(tauw/row)
            ! yplus = row*vtau*dw(iwall)/mu

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
            f(ic, jc, kc, 5) = f(ic, jc, kc, 5) + Omega*rc*vec(3)*tauw

        end do
    end if


end subroutine


subroutine zero_wall_stress(tau, ijk, ni, nj, nk, nwall, nd)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nwall
    integer, intent (in)  :: nd

    ! Warning: depending on which direction faces we are setting,
    ! tau will have smaller dimension, e.g.
    !   taui(ni, nj-1, nk-1, 6)
    real, intent (inout) :: tau(ni, nj, nk, nd)
    integer*2, intent (in) :: ijk(3, nwall)

    integer :: i
    integer :: j
    integer :: k
    integer :: iwall

    ! If we have at least one wall
    if (nwall > 0) then

        ! Loop over all points
        !$omp do private(i,j,k)
        do iwall = 1,nwall

            ! Extract indices
            i = ijk(1, iwall)
            j = ijk(2, iwall)
            k = ijk(3, iwall)

            ! Skip dummy points
            if (i.gt.0) then
                tau(i, j, k, :) = 0e0
            end if

        end do
        !$omp end do

    end if

end subroutine
