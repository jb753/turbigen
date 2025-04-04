! Apply Denton (2017) multigrid to cell residuals to accelerate convergence
subroutine multigrid_integrate( &
        fsum, &  ! Cell net fluxes
        dU, &    ! Residuals
        ijkmg, &  ! Multigrid block indices
        dt_vol, &  ! Multigrid block indices
        fmgrid, &  ! Scaling factor on multigrid time step
        ni, nj, nk, np, nlev &  ! Array sizes
    )

    ! Array sizes
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np
    integer, intent (in)  :: nlev

    ! Fine cell net fluxes
    real, intent (inout)  :: fsum(ni, nj, nk, np)

    ! Fine cell residuals
    real, intent (inout)  :: dU(ni, nj, nk, np)

    ! Multigrid factor
    real, intent (in) :: fmgrid

    ! Block indices
    integer*2, intent (in) :: ijkmg(3, ni, nj, nk, nlev)

    ! Multigrid vol and timesteps
    real, intent (in) :: dt_vol(ni, nj, nk, nlev+1)

    ! Working variables
    real :: fsum_mg(ni, nj, nk, np, nlev)
    integer :: i
    integer :: j
    integer :: k
    integer :: ilev
    integer :: ip
    integer :: ib
    integer :: jb
    integer :: kb

    fsum_mg = 0e0

    ! First we will loop over fine points and use the multigrid
    ! indices to add on the changes to the correct coarse block
    ! Once we have visited all nodes, the coarse block changes are
    ! correct, and we loop over fine points again and use the multigrid
    ! indices to extract the summed coarse block change for each fine
    ! point and add on multiplied by the safety factor fmgrid.

    !$omp parallel

    ! Loop over multigrid levels
    do ilev = 1,nlev

        ! Loop over fine cells in the block
        !$omp do private(j, i, ib, jb, kb)
        do k = 1,nk
            do j = 1,nj
                do i = 1,ni

                    ! Pull out the indices of the coarse
                    ! block that corresponds to current fine point
                    ib = ijkmg(1, i, j, k, ilev)
                    jb = ijkmg(2, i, j, k, ilev)
                    kb = ijkmg(3, i, j, k, ilev)

                    ! Accumulate sum from this fine cell
                    fsum_mg(ib, jb, kb, :, ilev) = &
                        fsum_mg(ib, jb, kb, :, ilev) + fsum(i, j, k, :)

                end do
            end do
        end do
        !$omp end do
    end do

    ! Intialise residual to fine value
    dU = 0e0
    do ip = 1, 5
        !$omp workshare
        dU(:,:,:,ip)  = fsum( :,:,:,ip) * dt_vol(:,:,:,1)
        !$omp end workshare
    end do


    ! Loop over multigrid levels
    do ilev = 1,nlev

        ! Loop over fine points in the block
        !$omp do private(j, i, ib, jb, kb)
        do k = 1,nk
            do j = 1,nj
                do i = 1,ni

                    ! Pull out the indices of the coarse
                    ! block that corresponds to current fine point
                    ib = ijkmg(1, i, j, k, ilev)
                    jb = ijkmg(2, i, j, k, ilev)
                    kb = ijkmg(3, i, j, k, ilev)

                    ! Add on residual from coarse block
                    do ip = 1, 5
                        dU(i, j, k, ip) = dU(i, j, k, ip) + &
                            fmgrid/(2**(ilev-1)) &
                            * fsum_mg(ib, jb, kb, ip, ilev) &
                            * dt_vol(ib, jb, kb, ilev+1)
                    end do


                end do
            end do
        end do
        !$omp end do

    end do
    !$omp end parallel

end subroutine


subroutine set_timesteps( dt_vol, vol, a, Vxrt, U, dlmin, ijkmg, CFL, ni, nj, nk, nlev )

    ! Array sizes
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nlev

    ! Multigrid cell volumes and timesteps
    real, intent (inout)  :: vol(ni-1, nj-1, nk-1, nlev+1)
    real, intent (inout)  :: dt_vol(ni-1, nj-1, nk-1, nlev+1)
    real, intent (inout)  :: dlmin(ni-1, nj-1, nk-1, nlev+1)

    ! Fine nodal velocities
    real, intent (inout)  :: Vxrt(ni, nj, nk, 3)
    real, intent (inout)  :: U(ni, nj, nk)
    real, intent (inout)  :: a(ni, nj, nk)

    ! Courant number
    real, intent (in) :: CFL
    real, parameter :: relax = 0.1

    ! Multigrid indices
    integer*2, intent (in) :: ijkmg(3, ni-1, nj-1, nk-1, nlev)

    ! Working vars
    real :: Vref_node(ni, nj, nk)
    real :: Vref_cell(ni-1, nj-1, nk-1)
    real :: Vref_mg(ni-1, nj-1, nk-1, nlev+1)
    real :: dt_new(ni-1, nj-1, nk-1, nlev+1)
    real :: vol_fac
    real :: Vxrt_rel(ni, nj, nk, 3)
    integer :: i
    integer :: j
    integer :: k
    integer :: ib
    integer :: jb
    integer :: kb
    integer :: ilev

    ! Calculate relative velocity at nodes
    ! We base time step on relative velocity magnitude because
    ! the control volumes can be rotating in theta direction
    Vxrt_rel = Vxrt
    Vxrt_rel(:,:,:,3) = Vxrt_rel(:,:,:,3)-U

    ! Get cell velocity magnitude plus speed of sound
    Vref_node = sqrt(sum(Vxrt_rel*Vxrt_rel,4)) + a
    call node_to_cell(Vref_node, Vref_cell, ni, nj, nk, 1)

    ! Trivial fine grid level
    Vref_mg = 0e0
    Vref_mg(:,:,:,1) = Vref_cell

    ! Loop over multigrid levels
    do ilev = 1,nlev

        ! Loop over fine cells in the block
        do i = 1,ni-1
            do j = 1,nj-1
                do k = 1,nk-1

                    ! Pull out the indices of the coarse
                    ! block that corresponds to current fine cell
                    ib = ijkmg(1, i, j, k, ilev)
                    jb = ijkmg(2, i, j, k, ilev)
                    kb = ijkmg(3, i, j, k, ilev)

                    ! Accumulate Vref from this fine cell
                    vol_fac = vol(i, j, k, 1)/vol(ib, jb, kb, ilev+1)
                    Vref_mg(ib, jb, kb, ilev+1) = Vref_mg(ib, jb, kb, ilev+1)+ vol_fac*Vref_cell(i, j, k)

                end do
            end do
        end do
    end do

    ! Now eval time step
    dt_new = CFL * dlmin / Vref_mg / vol
    dt_vol = relax * dt_new + (1e0 - relax)*dt_vol

end subroutine


subroutine multigrid_indices( &
        ijkmg, &  ! Multigrid block indices
        nb, &     ! Multigrid block sizes
        ni, nj, nk, nlev &  ! Array sizes
    )

    integer*2, intent (in)  :: ni
    integer*2, intent (in)  :: nj
    integer*2, intent (in)  :: nk
    integer*2, intent (in)  :: nlev
    integer*2, intent (inout) :: ijkmg(3, ni, nj, nk, nlev)
    integer*2, intent (inout) :: nb(nlev)

    integer*2 :: i
    integer*2 :: j
    integer*2 :: k
    integer*2 :: ilev
    integer*2 :: nbi

    do ilev = 1,nlev

        ! Number of cells along each side of this
        ! multigrid level is product of all previous
        nbi = product(nb(1:ilev))

        do i = 1,ni
            do j = 1,nj
                do k = 1,nk
                    ijkmg(1, i, j, k, ilev) = (i-1) / nbi
                    ijkmg(2, i, j, k, ilev) = (j-1) / nbi
                    ijkmg(3, i, j, k, ilev) = (k-1) / nbi
                end do
            end do
        end do
    end do

end subroutine



subroutine multigrid_volumes( &
        volmg, &  ! Multigrid block indices
        vol, &    ! Fine volumes
        ijkmg, &  ! Multigrid block indices
        ni, nj, nk, nlev &  ! Array sizes
    )

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nlev
    real, intent (inout) :: vol(ni, nj, nk)
    real, intent (inout) :: volmg(ni, nj, nk, nlev+1)
    integer*2, intent (inout) :: ijkmg(3, ni, nj, nk, nlev)

    integer :: i
    integer :: j
    integer :: k
    integer :: ib
    integer :: jb
    integer :: kb
    integer :: ilev

    ! Finest grid level is trivial
    volmg(:,:,:,1) = vol

    do ilev = 1,nlev
        do i = 1,ni
            do j = 1,nj
                do k = 1,nk
                    ! Get the coarse block index for this fine point
                    ib = ijkmg(1, i, j, k, ilev)
                    jb = ijkmg(2, i, j, k, ilev)
                    kb = ijkmg(3, i, j, k, ilev)

                    ! Accumulate fine volume onto the coarse volume
                    volmg(ib, jb, kb, ilev + 1) = volmg(ib, jb, kb, ilev + 1) + vol(i, j, k)
                end do
            end do
        end do
    end do

end subroutine
