! Apply Denton (2017) multigrid to cell residuals to accelerate convergence
subroutine multigrid_integrate( &
        fsum, &  ! Cell net fluxes
        dU, &    ! Residuals
        ijkmg, &  ! Multigrid block indices
        dt, &  ! Multigrid block indices
        vol, &  ! Multigrid block indices
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
    real*4, intent (inout)  :: fsum(ni, nj, nk, np)

    ! Fine cell residuals
    real*4, intent (inout)  :: dU(ni, nj, nk, np)

    ! Multigrid factor
    real*4, intent (in) :: fmgrid

    ! Block indices
    integer*2, intent (in) :: ijkmg(3, ni, nj, nk, nlev)

    ! Multigrid vol and timesteps
    real*4, intent (in) :: vol(ni, nj, nk, nlev+1)
    real*4, intent (in) :: dt(ni, nj, nk, nlev+1)

    ! Working variables
    real*4 :: fsum_mg(nlev, ni, nj, nk, np)
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

    ! Loop over multigrid levels
    do ilev = 1,nlev

        ! Loop over fine cells in the block
        do i = 1,ni
            do j = 1,nj
                do k = 1,nk

                    ! Pull out the indices of the coarse
                    ! block that corresponds to current fine point
                    ib = ijkmg(1, i, j, k, ilev)
                    jb = ijkmg(2, i, j, k, ilev)
                    kb = ijkmg(3, i, j, k, ilev)

                    ! Accumulate sum from this fine cell
                    fsum_mg(ilev, ib, jb, kb, :) = &
                        fsum_mg(ilev, ib, jb, kb, :) + fsum(i, j, k, :)

                end do
            end do
        end do
    end do

    ! Intialise residual to fine value
    dU = 0e0
    do ip = 1, 5
        dU(:,:,:,ip)  = fsum( :,:,:,ip) * dt(:,:,:,1)/vol(:,:,:,1)
    end do

    ! Loop over multigrid levels
    do ilev = 1,nlev

        ! Loop over fine points in the block
        do i = 1,ni
            do j = 1,nj
                do k = 1,nk

                    ! Pull out the indices of the coarse
                    ! block that corresponds to current fine point
                    ib = ijkmg(1, i, j, k, ilev)
                    jb = ijkmg(2, i, j, k, ilev)
                    kb = ijkmg(3, i, j, k, ilev)

                    ! Add on residual from coarse block
                    do ip = 1, 5
                        dU(i, j, k, ip) = dU(i, j, k, ip) + &
                            fmgrid/(2**(ilev-1)) &
                            * fsum_mg(ilev, ib, jb, kb, ip) &
                            * dt(ib, jb, kb, ilev+1) &
                            / vol(ib, jb, kb, ilev+1)
                    end do

                end do
            end do
        end do

    end do

end subroutine


subroutine set_timesteps( dt, vol, a, Vxrt, dlmin, ijkmg, CFL, relax, ni, nj, nk, nlev )

    ! Array sizes
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: nlev

    ! Multigrid cell volumes and timesteps
    real*4, intent (inout)  :: vol(ni-1, nj-1, nk-1, nlev+1)
    real*4, intent (inout)  :: dt(ni-1, nj-1, nk-1, nlev+1)
    real*4, intent (inout)  :: dlmin(ni-1, nj-1, nk-1, nlev+1)

    ! Fine nodal velocities
    real*4, intent (inout)  :: Vxrt(ni, nj, nk, 3)
    real*4, intent (inout)  :: a(ni, nj, nk)

    ! Courant number
    real*4, intent (in) :: CFL
    real*4, intent (in) :: relax

    ! Multigrid indices
    integer*2, intent (in) :: ijkmg(3, ni-1, nj-1, nk-1, nlev)

    ! Working vars
    real*4 :: Vref_node(ni, nj, nk)
    real*4 :: Vref_cell(ni-1, nj-1, nk-1)
    real*4 :: Vref_mg(ni-1, nj-1, nk-1, nlev+1)
    real*4 :: dt_new(ni-1, nj-1, nk-1, nlev+1)
    real*4 :: vol_fac
    integer :: i
    integer :: j
    integer :: k
    integer :: ib
    integer :: jb
    integer :: kb
    integer :: ilev

    ! Get cell velocity magnitude plus speed of sound
    Vref_node = sqrt(sum(Vxrt*Vxrt,4)) + a
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
    dt_new = CFL * dlmin / Vref_mg
    dt = relax * dt_new + (1e0 - relax)*dt

end subroutine
