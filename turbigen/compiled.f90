!! ! Compiled functions to speed up expensive calulations

subroutine step(conserved, P, ho, r, Omega, walli, wallj, wallk, dt, dAi, dAj, dAk, vol, halfVsq, u, &
        resid1,resid2, start_flag, ni, nj, nk)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*8, intent (inout)  :: conserved(ni, nj, nk, 5)
    real*8, intent (inout) :: resid1(ni, nj, nk, 5)
    real*8, intent (inout) :: resid2(ni, nj, nk, 5)
    real*8, intent (inout)  :: Omega
    logical*1, intent (inout)  :: walli(ni, nj-1, nk-1)
    logical*1, intent (inout)  :: wallj(ni-1, nj, nk-1)
    logical*1, intent (inout)  :: wallk(ni-1, nj-1, nk)
    real*8, intent (inout)  :: dAi(ni, nj-1, nk-1, 3)
    real*8, intent (inout)  :: dAj(ni-1, nj, nk-1, 3)
    real*8, intent (inout)  :: dAk(ni-1, nj-1, nk, 3)
    real*8, intent (inout)  :: vol(ni-1, nj-1, nk-1)
    real*8, intent (inout)  :: dt(ni-1, nj-1, nk-1)

    integer :: start_flag

    integer :: ip

    real*8 :: Sn(ni, nj, nk, 1)
    real*8 :: Sc(ni-1, nj-1, nk-1, 1)

    real*8 :: conservedi(ni, nj-1, nk-1, 5)
    real*8 :: conservedj(ni-1, nj, nk-1,5)
    real*8 :: conservedk(ni-1, nj-1, nk,5)

    real*8 :: fi(ni, nj-1, nk-1, 3, 5)
    real*8 :: fj(ni-1, nj, nk-1, 3, 5)
    real*8 :: fk(ni-1, nj-1, nk, 3, 5)

    real*8 :: fsum_vol(ni-1, nj-1, nk-1, 5)
    real*8 :: resc(ni-1, nj-1, nk-1, 5)

    real*8, intent(inout) :: P( ni, nj, nk)
    real*8, intent(inout) :: ho( ni, nj, nk)
    real*8, intent(inout) :: r( ni, nj, nk)

    real*8, intent(inout) :: u( ni, nj, nk)
    real*8, intent(inout) :: halfVsq(ni, nj, nk)

    real*8 :: Pi( ni, nj-1, nk-1)
    real*8 :: Pj( ni-1, nj, nk-1)
    real*8 :: Pk( ni-1, nj-1, nk)

    real*8 :: hoi( ni, nj-1, nk-1)
    real*8 :: hoj( ni-1, nj, nk-1)
    real*8 :: hok( ni-1, nj-1, nk)

    real*8 :: ri( ni, nj-1, nk-1)
    real*8 :: rj( ni-1, nj, nk-1)
    real*8 :: rk( ni-1, nj-1, nk)

    ! Calculate source term at nodes, average at cell centers
    Sn(:, :, :, 1) = (&
        P/r &  ! P/r
        + ( &
            conserved(:,:,:,4)*conserved(:,:,:,4) &  ! rhorVt**2
            /conserved(:,:,:,1) & ! over rho
            /r/r/r & ! over r**3
        ) &
    )
    call node_to_cell(Sn, Sc, ni, nj, nk, 1)

    ! Get face-centered vars
    call node_to_face( &
        conserved, conservedi, conservedj, conservedk, &
        ni, nj, nk, 5 &
    )

    call node_to_face( &
        P, Pi, Pj, Pk, &
         ni, nj, nk, 1 &
    )
    call node_to_face( &
        ho, hoi, hoj, hok, &
         ni, nj, nk, 1 &
    )
    call node_to_face( &
        r, ri, rj, rk, &
         ni, nj, nk, 1 &
    )


    ! Evaluate fluxes on each set of faces
    call get_fluxes_face(conservedi, Pi, hoi, ri, walli, Omega, fi, ni, nj-1, nk-1)
    call get_fluxes_face(conservedj, Pj, hoj, rj, wallj, Omega, fj, ni-1, nj, nk-1)
    call get_fluxes_face(conservedk, Pk, hok, rk, wallk, Omega, fk, ni-1, nj-1, nk)

    ! Get the net flux into each cell
    call sum_fluxes(fi, fj, fk, dAi, dAj, dAk, vol, fsum_vol, ni, nj, nk)

    ! Add on source term
    fsum_vol(:,:,:,3) = fsum_vol(:,:,:,3) + Sc(:,:,:,1)

    ! Integrate forward in time
    do ip = 1, 5
        resc(:,:,:,ip)  = fsum_vol( :,:,:,ip) * dt
    end do

    ! Distribute change to nodes
    call cell_to_node(resc, resid1, ni, nj, nk, 5)

    if (start_flag.eq.1) then
        conserved = conserved + resid1
    else
        conserved = conserved + 2d0*resid1 - resid2
    end if
    resid2 = resid1

    call calculate_secondary(r, conserved, halfVsq, u, ni, nj, nk)

end subroutine

subroutine calculate_secondary(r, conserved, halfVsq, u, ni, nj, nk)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*8, intent (inout)  :: conserved(ni, nj, nk, 5)
    real*8, intent (inout)  :: halfVsq(ni, nj, nk)
    real*8, intent (inout)  :: u(ni, nj, nk)
    real*8, intent (inout)  :: r(ni, nj, nk)
    real*8 :: Vxrt(ni, nj, nk, 3)

    integer :: ic

    
    do ic = 1,3
        Vxrt(:,:,:, ic) = conserved(:,:,:,ic+1)/conserved(:,:,:,1)
    end do
    Vxrt(:,:,:,3) = Vxrt(:,:,:,3)/r

    halfVsq = 0.5d0*sum(Vxrt*Vxrt, 4)

    u = conserved(:,:,:,5)/conserved(:,:,:,1) - halfVsq

end subroutine

subroutine get_fluxes_face(conserved, P, ho, r, wall, Omega, flux, ni, nj, nk)
    ! using face-centered properties, evaluate fluxes for one direction

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*8, intent (in)  :: conserved(ni, nj, nk, 5)
    real*8, intent (in)  :: P(ni, nj, nk)
    real*8, intent (in)  :: ho(ni, nj, nk)
    real*8, intent (in)  :: r(ni, nj, nk)
    real*8, intent (in)  :: Omega
    logical*1, intent (in)  :: wall(ni, nj, nk)

    real*8, intent (out) :: flux(ni, nj, nk, 3, 5)

    integer :: ic
    integer :: ip

    real*8 :: Vx(ni, nj, nk)
    real*8 :: Vr(ni, nj, nk)
    real*8 :: rVt(ni, nj, nk)

    ! Calculate velocities
    Vx = conserved(:, :, :,2)/conserved(:,:,:,1)
    Vr = conserved(:, :, :,3)/conserved(:,:,:,1)
    rVt = conserved(:, :, :,4)/conserved(:,:,:,1)

    ! mass fluxes in each direction
    flux(:,:,:,1,1) = conserved(:, :, :,2)  ! rhoVx
    flux(:,:,:,2,1) = conserved(:, :, :,3)  ! rhoVr
    flux(:,:,:,3,1) = conserved(:, :, :,4)/r  ! rhoVt=rhorVt/r

    ! x-mom flux for each coordinate direction
    do ic = 1,3
        flux(:,:,:,ic,2) = flux(:,:,:,ic,1) * Vx
    end do

    ! r-mom flux for each coordinate direction
    do ic = 1,3
        flux(:,:,:,ic,3) = flux(:,:,:,ic,1) * Vr
    end do

    ! rt-mom flux for each coordinate direction
    do ic = 1,3
        flux(:,:,:,ic,4) = flux(:,:,:,ic,1) * rVt
    end do

    ! ho flux for each coordinate direction
    do ic = 1,3
        flux(:,:,:,ic,5) = flux(:,:,:,ic,1) * ho
    end do

    ! zero convective fluxes on walls
    do ip = 1,5
        do ic = 1,3
            where (wall)
                flux(:, :, :, ic, ip) = 0d0
            end where
        end do
    end do

    ! pressure fluxes
    ! x-mom in x-dirn
    flux(:, :, :, 1, 2) = flux(:, :, :, 1, 2) + P
    ! r-mom in r-dirn
    flux(:, :, :, 2, 3) = flux(:, :, :, 2, 3) + P
    ! rt-mom in t-dirn
    flux(:, :, :, 3, 4) = flux(:, :, :, 3, 4) + r*P
    ! ho in t-dirn
    flux(:, :, :, 3, 5) = flux(:, :, :, 3, 5) + Omega*r*P


end subroutine


subroutine sum_fluxes(fi, fj, fk, dAi, dAj, dAk, vol, Fsum, ni, nj, nk)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    integer :: ip

    real*8, intent (in)  :: dAi(ni, nj-1, nk-1, 3)
    real*8, intent (in)  :: dAj(ni-1, nj, nk-1, 3)
    real*8, intent (in)  :: dAk(ni-1, nj-1, nk, 3)
    real*8, intent (in)  :: vol(ni-1, nj-1, nk-1)

    real*8, intent (in)  :: fi(ni, nj-1, nk-1, 3, 5)
    real*8, intent (in)  :: fj(ni-1, nj, nk-1, 3, 5)
    real*8, intent (in)  :: fk(ni-1, nj-1, nk, 3, 5)

    real*8 :: fisum(ni, nj-1, nk-1)
    real*8 :: fjsum(ni-1, nj, nk-1)
    real*8 :: fksum(ni-1, nj-1, nk)

    real*8, intent (out)  :: fsum(ni-1, nj-1, nk-1, 5)


    do ip = 1, 5
        ! Dot product areas with the fluxes
        fisum = sum(dAi*fi(:,:,:,:,ip),4)
        fjsum = sum(dAj*fj(:,:,:,:,ip),4)
        fksum = sum(dAk*fk(:,:,:,:,ip),4)
        ! Net flux per unit volume
        fsum(:, :, :, ip) = (&
              fisum(1:ni-1,:,:) - fisum(2:ni,:,:) & ! i faces
            + fjsum(:,1:nj-1,:) - fjsum(:,2:nj,:) & ! j faces
            + fksum(:,:,1:nk-1) - fksum(:,:,2:nk) & ! k faces
        )/vol
    end do

end subroutine


subroutine node_to_face(xn, xi, xj, xk, ni, nj, nk, np)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real*8, intent (inout)  :: xn(ni, nj, nk, np)
    real*8, intent (inout)  :: xi(ni, nj-1, nk-1, np)
    real*8, intent (inout)  :: xj(ni-1, nj, nk-1, np)
    real*8, intent (inout)  :: xk(ni-1, nj-1, nk, np)

    ! Values on i-faces are average over four bounding vertices
    xi = (&
          xn(:, 1:nj-1, 1:nk-1, :) & ! j, k
        + xn(:, 2:nj,   1:nk-1, :) & ! j+1, k
        + xn(:, 1:nj-1, 2:nk  , :) & ! j, k+1
        + xn(:, 2:nj,   2:nk  , :) & ! j+1, k+1
    )/4d0

    ! Values on j-faces are average over four bounding vertices
    xj = (&
          xn(1:ni-1, :, 1:nk-1, :) & ! i, k
        + xn(2:ni,   :, 1:nk-1, :) & ! i+1, k
        + xn(1:ni-1, :, 2:nk  , :) & ! i, k+1
        + xn(2:ni,   :, 2:nk  , :) & ! i+1, k+1
    )/4d0

    ! Values on k-faces are average over four bounding vertices
    xk = (&
          xn(1:ni-1, 1:nj-1, :, :) & ! i, j
        + xn(2:ni,   1:nj-1, :, :) & ! i+1, j
        + xn(1:ni-1, 2:nj,   :, :) & ! i, j+1
        + xn(2:ni,   2:nj,   :, :) & ! i+1, j+1
    )/4d0

end subroutine


subroutine node_to_cell(xn, xc, ni, nj, nk, np)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real*8, intent (inout)  :: xn(ni, nj, nk, np)
    real*8, intent (inout)  :: xc(ni-1, nj-1, nk-1, np)

    ! Cell values are the average of all eight hex vertices
    xc = (&
          xn(1:ni-1, 1:nj-1, 1:nk-1, :) & ! i,j,k
        + xn(2:ni,   1:nj-1, 1:nk-1, :) & ! i+1,j,k
        + xn(2:ni,   2:nj,   1:nk-1, :) & ! i+1,j+1,k
        + xn(1:ni-1, 2:nj,   1:nk-1, :) & ! i,j+1,k
        + xn(1:ni-1, 1:nj-1, 2:nk,   :) & ! i,j,k+1
        + xn(2:ni,   1:nj-1, 2:nk,   :) & ! i+1,j,k+1
        + xn(2:ni,   2:nj,   2:nk,   :) & ! i+1,j+1,k+1
        + xn(1:ni-1, 2:nj,   2:nk,   :) & ! i,j+1,k+1
    )/8d0


end subroutine

subroutine cell_to_node(xc, xn, ni, nj, nk, np)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real*8, intent (inout)  :: xc(ni-1, nj-1, nk-1, np)
    real*8, intent (inout)  :: xn(ni, nj, nk, np)

    ! Interior nodes take 1/8 from each adjacent cell
    xn(2:ni-1, 2:nj-1, 2:nk-1, :) = (&
          xc(1:ni-2, 1:nj-2, 1:nk-2, :) & ! i,j,k
        + xc(2:ni-1, 1:nj-2, 1:nk-2, :) & ! i+1,j,k
        + xc(2:ni-1, 2:nj-1, 1:nk-2, :) & ! i+1,j+1,k
        + xc(1:ni-2, 2:nj-1, 1:nk-2, :) & ! i,j+1,k
        + xc(1:ni-2, 1:nj-2, 2:nk-1, :) & ! i,j,k+1
        + xc(2:ni-1, 1:nj-2, 2:nk-1, :) & ! i+1,j,k+1
        + xc(2:ni-1, 2:nj-1, 2:nk-1, :) & ! i+1,j+1,k+1
        + xc(1:ni-2, 2:nj-1, 2:nk-1, :) & ! i,j+1,k+1
    )/8d0

    ! Face nodes take 1/4 from each adjacent cell

    ! i=1
    xn(1, 2:nj-1, 2:nk-1, :) = (&
          xc(1, 1:nj-2, 1:nk-2, :) & ! 1,j,k
        + xc(1, 2:nj-1, 1:nk-2, :) & ! 1,j+1,k
        + xc(1, 1:nj-2, 2:nk-1, :) & ! 1,j,k+1
        + xc(1, 2:nj-1, 2:nk-1, :) & ! 1,j+1,k+1
    )/4d0

    ! i=ni
    xn(ni, 2:nj-1, 2:nk-1, :) = (&
          xc(ni-1, 1:nj-2, 1:nk-2, :) & ! ni-1,j,k
        + xc(ni-1, 2:nj-1, 1:nk-2, :) & ! ni-1,j+1,k
        + xc(ni-1, 1:nj-2, 2:nk-1, :) & ! ni-1,j,k+1
        + xc(ni-1, 2:nj-1, 2:nk-1, :) & ! ni-1,j+1,k+1
    )/4d0

    ! j=1
    xn(2:ni-1, 1, 2:nk-1, :) = (&
          xc(1:ni-2, 1, 1:nk-2, :) & ! i,1,k
        + xc(2:ni-1, 1, 1:nk-2, :) & ! i+1,1,k
        + xc(1:ni-2, 1, 2:nk-1, :) & ! i,1,k+1
        + xc(2:ni-1, 1, 2:nk-1, :) & ! i+1,1,k+1
    )/4d0

    ! j=nj
    xn(2:ni-1, nj, 2:nk-1, :) = (&
          xc(1:ni-2, nj-1, 1:nk-2, :) & ! i,nj-1,k
        + xc(2:ni-1, nj-1, 1:nk-2, :) & ! i+1,nj-1,k
        + xc(1:ni-2, nj-1, 2:nk-1, :) & ! i,nj-1,k+1
        + xc(2:ni-1, nj-1, 2:nk-1, :) & ! i+1,nj-1,k+1
    )/4d0

    ! k=1
    xn(2:ni-1, 2:nj-1, 1, :) = (&
          xc(1:ni-2, 1:nj-2, 1, :) &
        + xc(2:ni-1, 1:nj-2, 1, :) &
        + xc(1:ni-2, 2:nj-1, 1, :) &
        + xc(2:ni-1, 2:nj-1, 1, :) &
    )/4d0

    ! k=nk
    xn(2:ni-1, 2:nj-1, nk, :) = (&
          xc(1:ni-2, 1:nj-2, nk-1, :) &
        + xc(2:ni-1, 1:nj-2, nk-1, :) &
        + xc(1:ni-2, 2:nj-1, nk-1, :) &
        + xc(2:ni-1, 2:nj-1, nk-1, :) &
    )/4d0

    ! Edges take 1/2 from each adjacent cell

    ! i=1, j=1
    xn(1, 1, 2:nk-1, :) = (&
          xc(1, 1, 1:nk-2, :) &
        + xc(1, 1, 2:nk-1, :) &
    )/2d0

    ! i=1, j=nj
    xn(1, nj, 2:nk-1, :) = (&
          xc(1, nj-1, 1:nk-2, :) &
        + xc(1, nj-1, 2:nk-1, :) &
    )/2d0

    ! i=ni, j=1
    xn(ni, 1, 2:nk-1, :) = (&
          xc(ni-1, 1, 1:nk-2, :) &
        + xc(ni-1, 1, 2:nk-1, :) &
    )/2d0

    ! i=ni, j=nj
    xn(ni, nj, 2:nk-1, :) = (&
          xc(ni-1, nj-1, 1:nk-2, :) &
        + xc(ni-1, nj-1, 2:nk-1, :) &
    )/2d0

    ! i=1, k=1
    xn(1, 2:nj-1, 1, :) = (&
          xc(1, 1:nj-2, 1, :) &
        + xc(1, 2:nj-1, 1, :) &
    )/2d0

    ! i=1, k=nk
    xn(1, 2:nj-1, nk, :) = (&
          xc(1, 1:nj-2, nk-1, :) &
        + xc(1, 2:nj-1, nk-1, :) &
    )/2d0

    ! i=ni, k=1
    xn(ni, 2:nj-1, 1, :) = (&
          xc(ni-1, 1:nj-2, 1, :) &
        + xc(ni-1, 2:nj-1, 1, :) &
    )/2d0

    ! i=ni, k=nk
    xn(ni, 2:nj-1, nk, :) = (&
          xc(ni-1, 1:nj-2, nk-1, :) &
        + xc(ni-1, 2:nj-1, nk-1, :) &
    )/2d0

    ! j=1, k=1
    xn(2:ni-1, 1, 1, :) = (&
          xc(1:ni-2, 1, 1, :) &
        + xc(2:ni-1, 1, 1, :) &
    )/2d0

    ! j=1, k=nk
    xn(2:ni-1, 1, nk, :) = (&
          xc(1:ni-2, 1, nk-1, :) &
        + xc(2:ni-1, 1, nk-1, :) &
    )/2d0

    ! j=nj, k=1
    xn(2:ni-1, nj, 1, :) = (&
          xc(1:ni-2, nj-1, 1, :) &
        + xc(2:ni-1, nj-1, 1, :) &
    )/2d0

    ! j=nj, k=nk
    xn(2:ni-1, nj, nk, :) = (&
          xc(1:ni-2, nj-1, nk-1, :) &
        + xc(2:ni-1, nj-1, nk-1, :) &
    )/2d0

    ! Corners take entirety from nearest cell
    xn(1,  1,  1, :) = xc(1,    1,    1, :)
    xn(1,  nj, 1, :) = xc(1,    nj-1, 1, :)
    xn(ni, nj, 1, :) = xc(ni-1, nj-1, 1, :)
    xn(ni, 1,  1, :) = xc(ni-1, 1,    1, :)
    xn(1,  1,  nk, :) = xc(1,    1,    nk-1, :)
    xn(1,  nj, nk, :) = xc(1,    nj-1, nk-1, :)
    xn(ni, nj, nk, :) = xc(ni-1, nj-1, nk-1, :)
    xn(ni, 1,  nk, :) = xc(ni-1, 1,    nk-1, :)


end subroutine

subroutine smooth(x, sf2, sf4, ni, nj, nk, np)
    ! Smooth the 4D array

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real*8, intent (in)  :: sf2
    real*8, intent (in)  :: sf4

    real*8, intent (inout)  :: x(ni, nj, nk, np)
    real*8 :: xs2(ni, nj, nk, np)
    real*8 :: xs4(ni, nj, nk, np)

    ! Initialise to zero
    xs2 = 0d0
    xs4 = 0d0

    ! Accumulate 2nd-order smoothed values for each direcion in turn
    ! We will divide by three later

    ! i interior
    xs2(2:ni-1, :, :, :) = xs2(2:ni-1, :, :, :) + ( &
          x(1:ni-2, :, :, :) + x(3:ni, :, :, :) &
    )/2d0

    ! i start
    xs2(1, :, :, :) = xs2(1, :, :, :) + ( &
          2d0*x(2, :, :, :) - x(3, :, :, :) &
    )

    ! i end
    xs2(ni, :, :, :) = xs2(ni, :, :, :) + ( &
          2d0*x(ni-1, :, :, :) - x(ni-2, :, :, :) &
    )

    ! j interior
    xs2(:, 2:nj-1, :, :) = xs2(:, 2:nj-1, :, :) + ( &
          x(:, 1:nj-2, :, :) + x(:, 3:nj,   :, :) &
    )/2d0

    ! j start
    xs2(:, 1, :, :) = xs2(:, 1, :, :) + ( &
          2d0*x(:, 2, :, :) - x(:, 3,   :, :) &
    )

    ! j end
    xs2(:, nj, :, :) = xs2(:, nj, :, :) + ( &
          2d0*x(:, nj-1, :, :) - x(:, nj-2, :, :) &
    )

    ! k interior
    xs2(:, :, 2:nk-1, :) = xs2(:, :, 2:nk-1, :) + ( &
          x(:, :, 1:nk-2, :) + x(:, :,   3:nk, :) &
    )/2d0

    ! k start
    xs2(:, :, 1, :) = xs2(:, :, 1, :) + ( &
          2d0*x(:, :, 2, :) - x(:, :,   3, :) &
    )

    ! k end
    xs2(:, :, nk, :) = xs2(:, :, nk, :) + ( &
          2d0*x(:, :, nk-1, :) - x(:, :,   nk-2, :) &
    )

    ! Accumulate 4th-order smoothed values for each direcion in turn
    ! We will divide by three later

    ! i interior
    xs4(3:ni-2, :, :, :) = xs4(3:ni-2, :, :, :) + ( &
        -     x(1:ni-4, :, :, :) + 4d0*x(2:ni-3, :, :, :) &
        + 4d0*x(4:ni-1, :, :, :) -     x(5:ni,   :, :, :) &
    )/6d0

    ! i=1
    xs4(1, :, :, :) = xs4(1, :, :, :) + ( &
          4d0*x(2, :, :, :) - 6d0*x(3, :, :, :) &
        + 4d0*x(4, :, :, :) -     x(5, :, :, :) &
    )

    ! i=2
    xs4(2, :, :, :) = xs4(2, :, :, :) + ( &
              x(1, :, :, :) + 6d0*x(3, :, :, :) &
        - 4d0*x(4, :, :, :) +     x(5, :, :, :) &
    )/4d0

    ! i=ni-1
    xs4(ni-1, :, :, :) = xs4(ni-1, :, :, :) + ( &
              x(ni-4, :, :, :) - 4d0*x(ni-3, :, :, :) &
        + 6d0*x(ni-2, :, :, :) +     x(ni, :, :, :) &
    )/4d0

    ! i=ni
    xs4(ni, :, :, :) = xs4(ni, :, :, :) + ( &
        -     x(ni-4, :, :, :) + 4d0*x(ni-3, :, :, :) &
        - 6d0*x(ni-2, :, :, :) + 4d0*x(ni-1, :, :, :) &
    )


    ! j interior
    xs4(:, 3:nj-2, :, :) = xs4(:, 3:nj-2, :, :) + ( &
        -     x(:, 1:nj-4, :, :) + 4d0*x(:, 2:nj-3, :, :) &
        + 4d0*x(:, 4:nj-1, :, :) -     x(:,   5:nj, :, :) &
    )/6d0

    ! j=1
    xs4(:, 1, :, :) = xs4(:, 1, :, :) + ( &
          4d0*x(:, 2, :, :) - 6d0*x(:, 3, :, :) &
        + 4d0*x(:, 4, :, :) -     x(:, 5, :, :) &
    )

    ! j=2
    xs4(:, 2, :, :) = xs4(:, 2, :, :) + ( &
              x(:, 1, :, :) + 6d0*x(:, 3, :, :) &
        - 4d0*x(:, 4, :, :) +     x(:, 5, :, :) &
    )/4d0

    ! j=nj-1
    xs4(:, nj-1, :, :) = xs4(:, nj-1, :, :) + ( &
              x(:, nj-4, :, :) - 4d0*x(:, nj-3, :, :) &
        + 6d0*x(:, nj-2, :, :) +     x(:, nj, :, :) &
    )/4d0

    ! j=nj
    xs4(:, nj, :, :) = xs4(:, nj, :, :) + ( &
        -     x(:, nj-4, :, :) + 4d0*x(:, nj-3, :, :) &
        - 6d0*x(:, nj-2, :, :) + 4d0*x(:, nj-1, :, :) &
    )


    ! k interior
    xs4(:, :, 3:nk-2, :) = xs4(:, :, 3:nk-2, :) + ( &
        -     x(:, :, 1:nk-4, :) + 4d0*x(:, :, 2:nk-3, :) &
        + 4d0*x(:, :, 4:nk-1, :) -     x(:,   :, 5:nk, :) &
    )/6d0

    ! k=1
    xs4(:, :, 1, :) = xs4(:, :, 1, :) + ( &
          4d0*x(:, :, 2, :) - 6d0*x(:, :, 3, :) &
        + 4d0*x(:, :, 4, :) -     x(:, :, 5, :) &
    )

    ! k=2
    xs4(:, :, 2, :) = xs4(:, :, 2, :) + ( &
              x(:, :, 1, :) + 6d0*x(:, :, 3, :) &
        - 4d0*x(:, :, 4, :) +     x(:, :, 5, :) &
    )/4d0

    ! k=nk-1
    xs4(:, :, nk-1, :) = xs4(:, :, nk-1, :) + ( &
              x(:, :, nk-4, :) - 4d0*x(:, :, nk-3, :) &
        + 6d0*x(:, :, nk-2, :) +     x(:, :, nk, :) &
    )/4d0

    ! k=nk
    xs4(:, :, nk, :) = xs4(:, :, nk, :) + ( &
        -     x(:, :, nk-4, :) + 4d0*x(:, :, nk-3, :) &
        - 6d0*x(:, :, nk-2, :) + 4d0*x(:, :, nk-1, :) &
    )

    ! now smooth
    x = (1d0-sf2-sf4)*x + (sf2*xs2 + sf4*xs4)/3d0

end subroutine
