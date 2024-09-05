! Compiled subroutines for the Enhanced Multi-Block Solve
! These are compiled with Python interfaces using f2py
! and then called from the main program which deals with
! fiddly bits like block patching and boundary conditions
module embsolve

    implicit none

contains

    include 'indexing.f90'
    include 'distribute.f90'
    include 'operators.f90'
    include 'smooth.f90'
    include 'viscous.f90'
    include 'fluxes.f90'
    include 'multigrid.f90'

    ! Using the current flow field, calculate changes to conserved
    ! variables that advance each cell in time. Specifically,
    ! 1) Evaluate fluxes of conserved quantities across each face
    ! 2) Sum fluxes into each cell
    ! 3) Add on body forces and source terms (defined per unit vol)
    ! 4) The net flow and body force for each cell is the 'residual' of
    !    the conservation equations:
    !       resid = vol * d(cons)/dt = flux_net + source * vol
    !    The change in the conserved variables is given by integrating
    !    forward in time:
    !       d(cons) = resid * dt / vol = (flux_net/vol + source) * dt
    !
    subroutine residual(&
        cons, Vxrt, P, Pref, ho, fb, &              ! Flow properties and body force
        Omega, &                              ! Reference frame angular velocity
        r, ri, rj, rk, &                      ! Node and face-centered radii
        dAi, dAj, dAk, vol, dt, &             ! Cell areas, volumes, time step
        ijk_iwall, ijk_jwall, ijk_kwall, &    ! Wall locations
        ijk_mg, fmgrid, &                     ! Multigrid indexing and factor
        fdamp, &                              ! Damping factor
        sf2, sf4, sf2min, L, &                              ! Damping factor
        resid, &                              ! Cell residual out
        resid_last, &                          ! Previous residual out
        ischeme, &
        ni, nj, nk, niwall, njwall, nkwall, nmg &  ! Numbers of points dummy args
        )

        ! Number of conserved variables nv = 5
        ! Number of coordinate directions nc = 3
        ! Have to use magic numbers or f2py thinks they should be dummy args

        ! Flow properties and body force
        ! Nodal conserved quantities: rho, rhoVx, rhoVr, rhorVt, rhoe
        real*4, intent (inout) :: cons(ni, nj, nk, 5)
        real*4, intent (in) :: Vxrt(ni, nj, nk, 3)
        real*4, intent (in) :: P   (ni, nj, nk)
        real*4, intent (in) :: ho  (ni, nj, nk)
        ! Cell body force per unit volume (and potential mass/energy sources)
        real*4, intent (in) :: fb   (ni-1, nj-1, nk-1, 5)

        ! Reference frame angular velocity
        real*4, intent (in)  :: Omega
        real*4, intent (in)  :: Pref

        ! Radii at nodes and face centers
        real*4, intent(in) :: r( ni, nj, nk)
        real*4, intent(in) :: ri( ni, nj-1, nk-1)
        real*4, intent(in) :: rj( ni-1, nj, nk-1)
        real*4, intent(in) :: rk( ni-1, nj-1, nk)

        ! Cell areas, volumes, time steps
        real*4, intent (in)  :: dAi(ni, nj-1, nk-1, 3)
        real*4, intent (in)  :: dAj(ni-1, nj, nk-1, 3)
        real*4, intent (in)  :: dAk(ni-1, nj-1, nk, 3)
        real*4, intent (in)  :: vol(ni-1, nj-1, nk-1, nmg)
        real*4, intent (in)  :: dt(ni-1, nj-1, nk-1, nmg)

        ! Wall locations
        integer*2, intent (in) :: ijk_iwall(3, niwall)
        integer*2, intent (in) :: ijk_jwall(3, njwall)
        integer*2, intent (in) :: ijk_kwall(3, nkwall)

        ! Multigrid indices
        integer*2, intent (in) :: ijk_mg(3, ni-1, nj-1, nk-1, nmg-1)

        ! Cell residual out
        real*4, intent (inout) :: resid(ni-1, nj-1, nk-1, 5)
        real*4, intent (inout) :: resid_last(ni-1, nj-1, nk-1, 5)

        real*4, intent(in) :: L( ni, nj, nk, 3)
        real*4, intent(in) :: sf2
        real*4, intent(in) :: sf4
        real*4, intent(in) :: sf2min

        ! Numbers of points dummy args
        integer, intent (in)  :: ni
        integer, intent (in)  :: nj
        integer, intent (in)  :: nk
        integer, intent (in)  :: niwall
        integer, intent (in)  :: njwall
        integer, intent (in)  :: nkwall
        integer, intent (in)  :: nmg

        ! Scalar settings
        integer, intent (in)  :: ischeme
        real*4, intent (in)  :: fmgrid
        real*4, intent (in)  :: fdamp

        ! End of argument declarations
        ! Begin working variables

        ! Fluxes on cell faces for each dirn and eqn
        real*4 :: fluxi(ni, nj-1, nk-1, 3, 5)
        real*4 :: fluxj(ni-1, nj, nk-1, 3, 5)
        real*4 :: fluxk(ni-1, nj-1, nk, 3, 5)

        ! For the centrifugal source term
        real*4 :: S(ni, nj, nk)
        real*4 :: Sc(ni-1, nj-1, nk-1)
        real*4 :: rho(ni, nj, nk)
        real*4 :: Vt(ni, nj, nk)

        ! Net fluxes for each cell
        real*4 :: fsum(ni-1, nj-1, nk-1, 5)
        ! Cell-centered residual

        real*4 :: Pm (ni, nj, nk)

        integer :: iv


        ! End of working variable declarations

        Pm = P - Pref

        ! Calculate the convective fluxes
        call set_fluxes( &
            cons, Vxrt, Pm, ho, &              ! Flow properties and body force
            Omega, &                              ! Reference frame angular velocity
            r, ri, rj, rk, &                      ! Node and face-centered radii
            ijk_iwall, ijk_jwall, ijk_kwall, &    ! Wall locations
            fluxi, fluxj, fluxk, &                ! Fluxes out
            ni, nj, nk, niwall, njwall, nkwall &  ! Numbers of points dummy args
            )

        ! Evaluate source term at nodes, average to cell center
        rho = cons(:, :, :, 1)
        Vt = Vxrt(:, :, :, 3)
        S = (rho*Vt*Vt + Pm)/r
        call node_to_cell(S, Sc, ni, nj, nk, 1)
        Sc = Sc * vol(:,:,:,1)

        ! Sum fluxes to get the net flux into each cell
        call sum_fluxes( &
            fluxi, fluxj, fluxk, &  ! Fluxes on the faces
            dAi, dAj, dAk, &        ! Cell geometry
            fsum, &                 ! Net flux
            ni, nj, nk, 5 &         ! Numbers of points for dummy args
        )

        ! Add on source term to the radial momentum eqn
        fsum(:,:,:,3) = fsum(:,:,:,3) + Sc

        ! Add on body forces
        fsum = fsum + fb

        ! fsum now contains the sum of fluxes for all cells
        call multigrid_integrate(fsum, resid, ijk_mg, dt, vol, fmgrid, ni-1, nj-1, nk-1, 5, nmg-1)

        ! Damp out the cell changes
        if (fdamp.gt.0) then
            call damp(resid, fdamp, ni-1, nj-1, nk-1)
        end if

        ! Time march and distribute to nodes
        call step(cons, resid, resid_last, ischeme, ni, nj, nk)

        call smooth( cons, P, L, sf4, sf2, sf2min, ni, nj, nk, 5)

    end subroutine

    subroutine step(cons, R1, R2, ischeme, ni, nj, nk)

        real*4, intent (inout)  :: cons(ni, nj, nk, 5)
        real*4, intent (inout) :: R1(ni-1, nj-1, nk-1, 5)
        real*4, intent (inout) :: R2(ni-1, nj-1, nk-1, 5)
        integer, intent (in) :: ischeme
        integer, intent (in)  :: ni
        integer, intent (in)  :: nj
        integer, intent (in)  :: nk

        real*4 :: Rcell(ni-1, nj-1, nk-1, 5)
        real*4 :: Rnode(ni, nj, nk, 5)

        if (ischeme.eq.-1) then
            ! At the start, we have no previous time level available
            ! So just apply the one residual we have
            Rcell = R1
            R2 = R1
        else if (ischeme.eq.0) then
        ! Otherwise, combine current and previous time level
        ! According to the selected time marching scheme
            Rcell = 2e0*R1 - R2
            R2 = R1
        else if (ischeme.eq.1) then
            Rcell = 2e0*R1 - 1.65e0*R2
            R2 = R1 - 0.65e0*R2
        end if

        ! Distribute cell residual to nodes and add on
        call cell_to_node(Rcell, Rnode, ni, nj, nk, 5)
        cons = cons + Rnode

    end subroutine

    subroutine secondary(r, cons, Vxrt, halfVsq, u, ni, nj, nk)

        implicit none

        integer, intent (in)  :: ni
        integer, intent (in)  :: nj
        integer, intent (in)  :: nk

        real*4, intent (inout)  :: cons(ni, nj, nk, 5)
        real*4, intent (inout)  :: Vxrt(ni, nj, nk, 3)
        real*4, intent (inout)  :: halfVsq(ni, nj, nk)
        real*4, intent (inout)  :: u(ni, nj, nk)
        real*4, intent (inout)  :: r(ni, nj, nk)

        integer :: ic

        do ic = 1,3
            Vxrt(:,:,:, ic) = cons(:,:,:,ic+1)/cons(:,:,:,1)
        end do
        Vxrt(:,:,:,3) = Vxrt(:,:,:,3)/r
        halfVsq = 0.5e0*sum(Vxrt*Vxrt, 4)

        u = cons(:,:,:,5)/cons(:,:,:,1) - halfVsq

    end subroutine


    ! Apply negative feedback to damp down large changes, Denton (2017)
    subroutine damp(R, fdamp, ni, nj, nk)


        integer, intent (in) :: ni
        integer, intent (in) :: nj
        integer, intent (in) :: nk

        integer :: ip

        real*4, intent (inout) :: R(ni-1, nj-1, nk-1, 5)
        real*4, intent (in) :: fdamp
        real*4 :: R_abs(ni-1, nj-1, nk-1, 5)
        real*4 :: R_avg(5)

        ! Calculate absolute and average values over all cells
        R_abs = abs(R)
        R_avg = sum(sum(sum(R_abs,1),1),1)/float((ni-1)*(nj-1)*(nk-1))

        ! Apply damping to all cons Ruals
        where (R_avg.eq.0)
            R_avg = 1e-9
        end where
        do ip = 1, 5
            R(:,:,:,ip) = R(:,:,:,ip) &
                / (1e0 + R_abs(:,:,:,ip)/R_avg(ip)/fdamp)
        end do

    end subroutine

end module embsolve
