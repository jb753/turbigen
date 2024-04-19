!! ! Compiled functions to speed up expensive calulations

subroutine residual(conserved, P, ho, r, f, Omega, walli, wallj, wallk, dt, dAi, dAj, dAk, vol, &
        resid, ni, nj, nk)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*4, intent (inout)  :: conserved(ni, nj, nk, 5)
    real*4, intent (inout) :: resid(ni, nj, nk, 5)
    real*4, intent (inout)  :: Omega
    logical*1, intent (inout)  :: walli(ni, nj-1, nk-1)
    logical*1, intent (inout)  :: wallj(ni-1, nj, nk-1)
    logical*1, intent (inout)  :: wallk(ni-1, nj-1, nk)
    real*4, intent (inout)  :: dAi(ni, nj-1, nk-1, 3)
    real*4, intent (inout)  :: dAj(ni-1, nj, nk-1, 3)
    real*4, intent (inout)  :: dAk(ni-1, nj-1, nk, 3)
    real*4, intent (inout)  :: vol(ni-1, nj-1, nk-1)
    real*4, intent (inout)  :: dt(ni-1, nj-1, nk-1)

    real*4, intent (inout)  :: f(ni-1, nj-1, nk-1, 5)

    integer :: ip

    real*4 :: Sn(ni, nj, nk)
    real*4 :: Sc(ni-1, nj-1, nk-1)

    real*4 :: fn(ni, nj, nk, 3, 5)
    real*4 :: fi(ni, nj-1, nk-1, 3, 5)
    real*4 :: fj(ni-1, nj, nk-1, 3, 5)
    real*4 :: fk(ni-1, nj-1, nk, 3, 5)

    real*4 :: fsum_vol(ni-1, nj-1, nk-1, 5)
    real*4 :: resc(ni-1, nj-1, nk-1, 5)

    real*4, intent(inout) :: P( ni, nj, nk)
    real*4, intent(inout) :: ho( ni, nj, nk)
    real*4, intent(inout) :: r( ni, nj, nk)

    real*4 :: Vt( ni, nj, nk)

    real*4 :: Pi( ni, nj-1, nk-1)
    real*4 :: Pj( ni-1, nj, nk-1)
    real*4 :: Pk( ni-1, nj-1, nk)

    real*4 :: ri( ni, nj-1, nk-1)
    real*4 :: rj( ni-1, nj, nk-1)
    real*4 :: rk( ni-1, nj-1, nk)

    ! integer, intent (in) :: nstep_avg
    ! real*8, intent (inout)  :: conserved_avg(ni, nj, nk, 5)

    ! Calculate source term at nodes, average at cell center
    Vt = conserved(:,:,:,4)/conserved(:,:,:,1)/r
    Sn(:, :, :) = (conserved(:,:,:,1) * Vt*Vt + P)/r
    call node_to_cell(Sn, Sc, ni, nj, nk, 1)

    call node_to_face( &
        P, Pi, Pj, Pk, &
         ni, nj, nk, 1 &
    )

    call node_to_face( &
        r, ri, rj, rk, &
         ni, nj, nk, 1 &
    )

    ! Evaluate convective fluxes at nodes
    call get_fluxes_node(conserved, ho, r, fn, ni, nj, nk)

    ! Distribute to faces
    do ip = 1,5
        call node_to_face( &
            fn(:,:,:,:,ip), fi(:,:,:,:,ip), fj(:,:,:,:,ip), fk(:,:,:,:,ip), &
            ni, nj, nk, 3 &
        )
    end do

    call get_fluxes_face(fi, Pi, ri, walli, Omega, ni, nj-1, nk-1)
    call get_fluxes_face(fj, Pj, rj, wallj, Omega, ni-1, nj, nk-1)
    call get_fluxes_face(fk, Pk, rk, wallk, Omega, ni-1, nj-1, nk)

    ! Get the net flux into each cell
    call sum_fluxes(fi, fj, fk, dAi, dAj, dAk, vol, fsum_vol, ni, nj, nk, 5)

    ! Add on source term
    fsum_vol(:,:,:,3) = fsum_vol(:,:,:,3) + Sc

    ! Add on body forces
    fsum_vol = fsum_vol + f

    ! Integrate forward in time
    do ip = 1, 5
        resc(:,:,:,ip)  = fsum_vol( :,:,:,ip) * dt
    end do

    ! Distribute change to nodes
    call cell_to_node(resc, resid, ni, nj, nk, 5)

end subroutine

subroutine damp(resid, fdamp, ni, nj, nk)
    ! Apply negative feedback to damp down large residuals
    ! Denton (2017)

    real*4, intent (inout) :: resid(ni-1, nj-1, nk-1, 5)
    real*4, intent (in) :: fdamp
    real*4 :: resid_abs(ni-1, nj-1, nk-1, 5)
    real*4 :: resid_avg(5)

    ! Calculate absolute and average values over all cells
    resid_abs = abs(resid)
    resid_avg = sum(sum(sum(resid_abs,1),1),1)/float((ni-1)*(nj-1)*(nk-1))

    ! Apply damping to all conserved residuals
    where (resid_avg.eq.0)
        resid_avg = 1e-9
    end where
    do ip = 1, 5
        resid(:,:,:,ip) = resid(:,:,:,ip) &
            / (1e0 + resid_abs(:,:,:,ip)/resid_avg(ip)/fdamp)
    end do

end subroutine

subroutine step(conserved, conserved_avg, resid1, resid2, istep, istep_avg, nstep_avg, ischeme, ni, nj, nk)

    real*4, intent (inout)  :: conserved(ni, nj, nk, 5)
    real*8, intent (inout)  :: conserved_avg(ni, nj, nk, 5)
    real*4, intent (inout) :: resid1(ni, nj, nk, 5)
    real*4, intent (inout) :: resid2(ni, nj, nk, 5)
    integer, intent (in) :: istep
    integer, intent (in) :: istep_avg
    integer, intent (in) :: nstep_avg
    integer, intent (in) :: ischeme
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    if (istep.eq.0) then
        conserved = conserved + resid1
        resid2 = resid1
    else
        if (ischeme.eq.0) then
            conserved = conserved + 2e0*resid1 - resid2
            resid2 = resid1
        else
            conserved = conserved + 2e0*resid1 - 1.65e0*resid2
            resid2 = resid1 - 0.65e0*resid2
        end if
    end if

    if (istep.ge.istep_avg) then
        conserved_avg = conserved_avg + conserved/float(nstep_avg)
    end if

end subroutine

subroutine calculate_secondary(r, conserved, halfVsq, u, ni, nj, nk)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*4, intent (inout)  :: conserved(ni, nj, nk, 5)
    real*4, intent (inout)  :: halfVsq(ni, nj, nk)
    real*4, intent (inout)  :: u(ni, nj, nk)
    real*4, intent (inout)  :: r(ni, nj, nk)
    real*4 :: Vxrt(ni, nj, nk, 3)

    integer :: ic


    do ic = 1,3
        Vxrt(:,:,:, ic) = conserved(:,:,:,ic+1)/conserved(:,:,:,1)
    end do
    Vxrt(:,:,:,3) = Vxrt(:,:,:,3)/r

    halfVsq = 0.5e0*sum(Vxrt*Vxrt, 4)

    u = conserved(:,:,:,5)/conserved(:,:,:,1) - halfVsq

end subroutine

subroutine get_fluxes_node(conserved, ho, r, flux, ni, nj, nk)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*4, intent (in)  :: conserved(ni, nj, nk, 5)
    real*4, intent (in)  :: ho(ni, nj, nk)
    real*4, intent (in)  :: r(ni, nj, nk)

    real*4, intent (out) :: flux(ni, nj, nk, 3, 5)

    integer :: ic

    real*4 :: Vx(ni, nj, nk)
    real*4 :: Vr(ni, nj, nk)
    real*4 :: rVt(ni, nj, nk)

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

end subroutine

subroutine get_fluxes_face(flux, P, r, wall, Omega, ni, nj, nk)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*4, intent (in)  :: r(ni, nj, nk)
    real*4, intent (in)  :: Omega

    real*4, intent (out) :: flux(ni, nj, nk, 3, 5)

    integer :: ic
    integer :: ip

    real*4, intent (in)  :: P(ni, nj, nk)
    logical*1, intent (in)  :: wall(ni, nj, nk)

    ! zero convective fluxes on walls
    do ip = 1,5
        do ic = 1,3
            where (wall)
                flux(:, :, :, ic, ip) = 0e0
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


subroutine sum_fluxes(fi, fj, fk, dAi, dAj, dAk, vol, Fsum, ni, nj, nk, np)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    integer :: ip

    real*4, intent (in)  :: dAi(ni, nj-1, nk-1, 3)
    real*4, intent (in)  :: dAj(ni-1, nj, nk-1, 3)
    real*4, intent (in)  :: dAk(ni-1, nj-1, nk, 3)
    real*4, intent (in)  :: vol(ni-1, nj-1, nk-1)

    real*4, intent (in)  :: fi(ni, nj-1, nk-1, 3, np)
    real*4, intent (in)  :: fj(ni-1, nj, nk-1, 3, np)
    real*4, intent (in)  :: fk(ni-1, nj-1, nk, 3, np)

    real*4 :: fisum(ni, nj-1, nk-1)
    real*4 :: fjsum(ni-1, nj, nk-1)
    real*4 :: fksum(ni-1, nj-1, nk)

    real*4, intent (out)  :: fsum(ni-1, nj-1, nk-1, np)

    fsum = 0e0
    do ip = 1, np
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

    real*4, intent (inout)  :: xn(ni, nj, nk, np)
    real*4, intent (inout)  :: xi(ni, nj-1, nk-1, np)
    real*4, intent (inout)  :: xj(ni-1, nj, nk-1, np)
    real*4, intent (inout)  :: xk(ni-1, nj-1, nk, np)

    ! Values on i-faces are average over four bounding vertices
    xi = (&
          xn(:, 1:nj-1, 1:nk-1, :) & ! j, k
        + xn(:, 2:nj,   1:nk-1, :) & ! j+1, k
        + xn(:, 1:nj-1, 2:nk  , :) & ! j, k+1
        + xn(:, 2:nj,   2:nk  , :) & ! j+1, k+1
    )/4e0

    ! Values on j-faces are average over four bounding vertices
    xj = (&
          xn(1:ni-1, :, 1:nk-1, :) & ! i, k
        + xn(2:ni,   :, 1:nk-1, :) & ! i+1, k
        + xn(1:ni-1, :, 2:nk  , :) & ! i, k+1
        + xn(2:ni,   :, 2:nk  , :) & ! i+1, k+1
    )/4e0

    ! Values on k-faces are average over four bounding vertices
    xk = (&
          xn(1:ni-1, 1:nj-1, :, :) & ! i, j
        + xn(2:ni,   1:nj-1, :, :) & ! i+1, j
        + xn(1:ni-1, 2:nj,   :, :) & ! i, j+1
        + xn(2:ni,   2:nj,   :, :) & ! i+1, j+1
    )/4e0

end subroutine


subroutine node_to_cell(xn, xc, ni, nj, nk, np)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real*4, intent (inout)  :: xn(ni, nj, nk, np)
    real*4, intent (inout)  :: xc(ni-1, nj-1, nk-1, np)

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
    )/8e0


end subroutine

subroutine cell_to_node(xc, xn, ni, nj, nk, np)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real*4, intent (inout)  :: xc(ni-1, nj-1, nk-1, np)
    real*4, intent (inout)  :: xn(ni, nj, nk, np)

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
    )/8e0

    ! Face nodes take 1/4 from each adjacent cell

    ! i=1
    xn(1, 2:nj-1, 2:nk-1, :) = (&
          xc(1, 1:nj-2, 1:nk-2, :) & ! 1,j,k
        + xc(1, 2:nj-1, 1:nk-2, :) & ! 1,j+1,k
        + xc(1, 1:nj-2, 2:nk-1, :) & ! 1,j,k+1
        + xc(1, 2:nj-1, 2:nk-1, :) & ! 1,j+1,k+1
    )/4e0

    ! i=ni
    xn(ni, 2:nj-1, 2:nk-1, :) = (&
          xc(ni-1, 1:nj-2, 1:nk-2, :) & ! ni-1,j,k
        + xc(ni-1, 2:nj-1, 1:nk-2, :) & ! ni-1,j+1,k
        + xc(ni-1, 1:nj-2, 2:nk-1, :) & ! ni-1,j,k+1
        + xc(ni-1, 2:nj-1, 2:nk-1, :) & ! ni-1,j+1,k+1
    )/4e0

    ! j=1
    xn(2:ni-1, 1, 2:nk-1, :) = (&
          xc(1:ni-2, 1, 1:nk-2, :) & ! i,1,k
        + xc(2:ni-1, 1, 1:nk-2, :) & ! i+1,1,k
        + xc(1:ni-2, 1, 2:nk-1, :) & ! i,1,k+1
        + xc(2:ni-1, 1, 2:nk-1, :) & ! i+1,1,k+1
    )/4e0

    ! j=nj
    xn(2:ni-1, nj, 2:nk-1, :) = (&
          xc(1:ni-2, nj-1, 1:nk-2, :) & ! i,nj-1,k
        + xc(2:ni-1, nj-1, 1:nk-2, :) & ! i+1,nj-1,k
        + xc(1:ni-2, nj-1, 2:nk-1, :) & ! i,nj-1,k+1
        + xc(2:ni-1, nj-1, 2:nk-1, :) & ! i+1,nj-1,k+1
    )/4e0

    ! k=1
    xn(2:ni-1, 2:nj-1, 1, :) = (&
          xc(1:ni-2, 1:nj-2, 1, :) &
        + xc(2:ni-1, 1:nj-2, 1, :) &
        + xc(1:ni-2, 2:nj-1, 1, :) &
        + xc(2:ni-1, 2:nj-1, 1, :) &
    )/4e0

    ! k=nk
    xn(2:ni-1, 2:nj-1, nk, :) = (&
          xc(1:ni-2, 1:nj-2, nk-1, :) &
        + xc(2:ni-1, 1:nj-2, nk-1, :) &
        + xc(1:ni-2, 2:nj-1, nk-1, :) &
        + xc(2:ni-1, 2:nj-1, nk-1, :) &
    )/4e0

    ! Edges take 1/2 from each adjacent cell

    ! i=1, j=1
    xn(1, 1, 2:nk-1, :) = (&
          xc(1, 1, 1:nk-2, :) &
        + xc(1, 1, 2:nk-1, :) &
    )/2e0

    ! i=1, j=nj
    xn(1, nj, 2:nk-1, :) = (&
          xc(1, nj-1, 1:nk-2, :) &
        + xc(1, nj-1, 2:nk-1, :) &
    )/2e0

    ! i=ni, j=1
    xn(ni, 1, 2:nk-1, :) = (&
          xc(ni-1, 1, 1:nk-2, :) &
        + xc(ni-1, 1, 2:nk-1, :) &
    )/2e0

    ! i=ni, j=nj
    xn(ni, nj, 2:nk-1, :) = (&
          xc(ni-1, nj-1, 1:nk-2, :) &
        + xc(ni-1, nj-1, 2:nk-1, :) &
    )/2e0

    ! i=1, k=1
    xn(1, 2:nj-1, 1, :) = (&
          xc(1, 1:nj-2, 1, :) &
        + xc(1, 2:nj-1, 1, :) &
    )/2e0

    ! i=1, k=nk
    xn(1, 2:nj-1, nk, :) = (&
          xc(1, 1:nj-2, nk-1, :) &
        + xc(1, 2:nj-1, nk-1, :) &
    )/2e0

    ! i=ni, k=1
    xn(ni, 2:nj-1, 1, :) = (&
          xc(ni-1, 1:nj-2, 1, :) &
        + xc(ni-1, 2:nj-1, 1, :) &
    )/2e0

    ! i=ni, k=nk
    xn(ni, 2:nj-1, nk, :) = (&
          xc(ni-1, 1:nj-2, nk-1, :) &
        + xc(ni-1, 2:nj-1, nk-1, :) &
    )/2e0

    ! j=1, k=1
    xn(2:ni-1, 1, 1, :) = (&
          xc(1:ni-2, 1, 1, :) &
        + xc(2:ni-1, 1, 1, :) &
    )/2e0

    ! j=1, k=nk
    xn(2:ni-1, 1, nk, :) = (&
          xc(1:ni-2, 1, nk-1, :) &
        + xc(2:ni-1, 1, nk-1, :) &
    )/2e0

    ! j=nj, k=1
    xn(2:ni-1, nj, 1, :) = (&
          xc(1:ni-2, nj-1, 1, :) &
        + xc(2:ni-1, nj-1, 1, :) &
    )/2e0

    ! j=nj, k=nk
    xn(2:ni-1, nj, nk, :) = (&
          xc(1:ni-2, nj-1, nk-1, :) &
        + xc(2:ni-1, nj-1, nk-1, :) &
    )/2e0

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

subroutine cell_to_face(xc, xi, xj, xk, ni, nj, nk, np)

    implicit none
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np
    real*4, intent (inout)  :: xc(ni-1, nj-1, nk-1, np)
    real*4, intent (inout)  :: xi(ni, nj-1, nk-1, np)
    real*4, intent (inout)  :: xj(ni-1, nj, nk-1, np)
    real*4, intent (inout)  :: xk(ni-1, nj-1, nk, np)

    ! interior i-faces are average of i and i+1
    xi(2:ni-1, :, :, :) = ( &
        xc(1:ni-2, :, :, :) &
        + xc(2:ni-1, :, :, :) &
    )/2e0

    ! i start and end
    xi(1, :, :, :) = xc(1, :, :, :)
    xi(ni, :, :, :) = xc(ni-1, :, :, :)

    ! interior j-faces are average of j and j+1
    xj(:, 2:nj-1, :, :) = ( &
        xc(:, 1:nj-2, :, :) &
        + xc(:, 2:nj-1, :, :) &
    )/2e0

    ! j start and end
    xj(:, 1, :, :) = xc(:, 1, :, :)
    xj(:, nj, :, :) = xc(:, nj-1, :, :)

    ! interior k-faces are average of k and k+1
    xk(:, :, 2:nk-1, :) = ( &
        xc(:, :, 1:nk-2, :) &
        + xc(:, :, 2:nk-1, :) &
    )/2e0

    ! k start and end
    xk(:, :, 1, :) = xc(:, :, 1, :)
    xk(:, :, nk, :) = xc(:, :, nk-1, :)

end subroutine

subroutine smooth(x, sf2, sf4, ni, nj, nk, np)
    ! Smooth the 4D array

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real*4, intent (in)  :: sf2
    real*4, intent (in)  :: sf4

    real*4, intent (inout)  :: x(ni, nj, nk, np)
    real*4 :: xs2(ni, nj, nk, np)
    real*4 :: xs4(ni, nj, nk, np)

    ! Initialise to zero
    xs2 = 0e0
    xs4 = 0e0

    ! Accumulate 2nd-order smoothed values for each direcion in turn
    ! We will divide by three later

    ! i interior
    xs2(2:ni-1, :, :, :) = xs2(2:ni-1, :, :, :) + ( &
          x(1:ni-2, :, :, :) + x(3:ni, :, :, :) &
    )/2e0

    ! i start
    xs2(1, :, :, :) = xs2(1, :, :, :) + ( &
          2e0*x(2, :, :, :) - x(3, :, :, :) &
    )

    ! i end
    xs2(ni, :, :, :) = xs2(ni, :, :, :) + ( &
          2e0*x(ni-1, :, :, :) - x(ni-2, :, :, :) &
    )

    ! j interior
    xs2(:, 2:nj-1, :, :) = xs2(:, 2:nj-1, :, :) + ( &
          x(:, 1:nj-2, :, :) + x(:, 3:nj,   :, :) &
    )/2e0

    ! j start
    xs2(:, 1, :, :) = xs2(:, 1, :, :) + ( &
          2e0*x(:, 2, :, :) - x(:, 3,   :, :) &
    )

    ! j end
    xs2(:, nj, :, :) = xs2(:, nj, :, :) + ( &
          2e0*x(:, nj-1, :, :) - x(:, nj-2, :, :) &
    )

    ! k interior
    xs2(:, :, 2:nk-1, :) = xs2(:, :, 2:nk-1, :) + ( &
          x(:, :, 1:nk-2, :) + x(:, :,   3:nk, :) &
    )/2e0

    ! k start
    xs2(:, :, 1, :) = xs2(:, :, 1, :) + ( &
          2e0*x(:, :, 2, :) - x(:, :,   3, :) &
    )

    ! k end
    xs2(:, :, nk, :) = xs2(:, :, nk, :) + ( &
          2e0*x(:, :, nk-1, :) - x(:, :,   nk-2, :) &
    )

    ! Accumulate 4th-order smoothed values for each direcion in turn
    ! We will divide by three later

    ! i interior
    xs4(3:ni-2, :, :, :) = xs4(3:ni-2, :, :, :) + ( &
        -     x(1:ni-4, :, :, :) + 4e0*x(2:ni-3, :, :, :) &
        + 4e0*x(4:ni-1, :, :, :) -     x(5:ni,   :, :, :) &
    )/6e0

    ! i=1
    xs4(1, :, :, :) = xs4(1, :, :, :) + ( &
          4e0*x(2, :, :, :) - 6e0*x(3, :, :, :) &
        + 4e0*x(4, :, :, :) -     x(5, :, :, :) &
    )

    ! i=2
    xs4(2, :, :, :) = xs4(2, :, :, :) + ( &
              x(1, :, :, :) + 6e0*x(3, :, :, :) &
        - 4e0*x(4, :, :, :) +     x(5, :, :, :) &
    )/4e0

    ! i=ni-1
    xs4(ni-1, :, :, :) = xs4(ni-1, :, :, :) + ( &
              x(ni-4, :, :, :) - 4e0*x(ni-3, :, :, :) &
        + 6e0*x(ni-2, :, :, :) +     x(ni, :, :, :) &
    )/4e0

    ! i=ni
    xs4(ni, :, :, :) = xs4(ni, :, :, :) + ( &
        -     x(ni-4, :, :, :) + 4e0*x(ni-3, :, :, :) &
        - 6e0*x(ni-2, :, :, :) + 4e0*x(ni-1, :, :, :) &
    )


    ! j interior
    xs4(:, 3:nj-2, :, :) = xs4(:, 3:nj-2, :, :) + ( &
        -     x(:, 1:nj-4, :, :) + 4e0*x(:, 2:nj-3, :, :) &
        + 4e0*x(:, 4:nj-1, :, :) -     x(:,   5:nj, :, :) &
    )/6e0

    ! j=1
    xs4(:, 1, :, :) = xs4(:, 1, :, :) + ( &
          4e0*x(:, 2, :, :) - 6e0*x(:, 3, :, :) &
        + 4e0*x(:, 4, :, :) -     x(:, 5, :, :) &
    )

    ! j=2
    xs4(:, 2, :, :) = xs4(:, 2, :, :) + ( &
              x(:, 1, :, :) + 6e0*x(:, 3, :, :) &
        - 4e0*x(:, 4, :, :) +     x(:, 5, :, :) &
    )/4e0

    ! j=nj-1
    xs4(:, nj-1, :, :) = xs4(:, nj-1, :, :) + ( &
              x(:, nj-4, :, :) - 4e0*x(:, nj-3, :, :) &
        + 6e0*x(:, nj-2, :, :) +     x(:, nj, :, :) &
    )/4e0

    ! j=nj
    xs4(:, nj, :, :) = xs4(:, nj, :, :) + ( &
        -     x(:, nj-4, :, :) + 4e0*x(:, nj-3, :, :) &
        - 6e0*x(:, nj-2, :, :) + 4e0*x(:, nj-1, :, :) &
    )


    ! k interior
    xs4(:, :, 3:nk-2, :) = xs4(:, :, 3:nk-2, :) + ( &
        -     x(:, :, 1:nk-4, :) + 4e0*x(:, :, 2:nk-3, :) &
        + 4e0*x(:, :, 4:nk-1, :) -     x(:,   :, 5:nk, :) &
    )/6e0

    ! k=1
    xs4(:, :, 1, :) = xs4(:, :, 1, :) + ( &
          4e0*x(:, :, 2, :) - 6e0*x(:, :, 3, :) &
        + 4e0*x(:, :, 4, :) -     x(:, :, 5, :) &
    )

    ! k=2
    xs4(:, :, 2, :) = xs4(:, :, 2, :) + ( &
              x(:, :, 1, :) + 6e0*x(:, :, 3, :) &
        - 4e0*x(:, :, 4, :) +     x(:, :, 5, :) &
    )/4e0

    ! k=nk-1
    xs4(:, :, nk-1, :) = xs4(:, :, nk-1, :) + ( &
              x(:, :, nk-4, :) - 4e0*x(:, :, nk-3, :) &
        + 6e0*x(:, :, nk-2, :) +     x(:, :, nk, :) &
    )/4e0

    ! k=nk
    xs4(:, :, nk, :) = xs4(:, :, nk, :) + ( &
        -     x(:, :, nk-4, :) + 4e0*x(:, :, nk-3, :) &
        - 6e0*x(:, :, nk-2, :) + 4e0*x(:, :, nk-1, :) &
    )

    ! now smooth
    x = (1e0-sf2-sf4)*x + (sf2*xs2 + sf4*xs4)/3e0

end subroutine


subroutine div(x, divx, vol, dAi, dAj, dAk, ni, nj, nk)

    implicit none

    real*4, intent (inout)  :: x(ni, nj, nk, 3)

    real*4, intent (inout)  :: dAi(ni, nj-1, nk-1, 3)
    real*4, intent (inout)  :: dAj(ni-1, nj, nk-1, 3)
    real*4, intent (inout)  :: dAk(ni-1, nj-1, nk, 3)
    real*4, intent (inout)  :: vol(ni-1, nj-1, nk-1)

    real*4, intent (inout)  :: divx(ni-1, nj-1, nk-1)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*4 :: xi(ni, nj-1, nk-1, 3)
    real*4 :: xj(ni-1, nj, nk-1, 3)
    real*4 :: xk(ni-1, nj-1, nk, 3)

    call node_to_face( x, xi, xj, xk, ni, nj, nk, 3 )

    call sum_fluxes(xi, xj, xk, dAi, dAj, dAk, -vol, divx, ni, nj, nk, 1)

end subroutine


subroutine grad(x, gradx, vol, dAi, dAj, dAk, r, ni, nj, nk)

    implicit none

    real*4, intent (inout)  :: x(ni, nj, nk)

    real*4, intent (inout)  :: dAi(ni, nj-1, nk-1, 3)
    real*4, intent (inout)  :: dAj(ni-1, nj, nk-1, 3)
    real*4, intent (inout)  :: dAk(ni-1, nj-1, nk, 3)
    real*4, intent (inout)  :: vol(ni-1, nj-1, nk-1)
    real*4, intent (inout)  :: r(ni, nj, nk)

    real*4, intent (inout)  :: gradx(ni-1, nj-1, nk-1, 3)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer :: ii

    real*4 :: xi(ni, nj-1, nk-1, 3)
    real*4 :: xj(ni-1, nj, nk-1, 3)
    real*4 :: xk(ni-1, nj-1, nk, 3)
    real*4 :: xv(ni, nj, nk, 3)

    real*4 :: rc(ni-1, nj-1, nk-1)

    ! Find radii at cell centers
    call node_to_cell(r, rc, ni, nj, nk, 1)

    xv = 0e0

    do ii = 1,3
        xv(:,:,:,ii) = x
        if (ii.eq.2) then
            xv(:,:,:,ii) = xv(:,:,:,ii)/r
        end if
        call node_to_face( xv, xi, xj, xk, ni, nj, nk, 3 )
        call sum_fluxes(xi, xj, xk, dAi, dAj, dAk, -vol, gradx(:,:,:,ii), ni, nj, nk, 1)
        if (ii.eq.2) then
            gradx(:,:,:,ii) = gradx(:,:,:,ii)*rc
        end if
        xv = 0e0
    end do

end subroutine

subroutine viscous_force(conserved, fvisc, mu, mu_turb, xlength, walli, wallj, wallk, vol, dAi, dAj, dAk, r, ni, nj, nk)

    implicit none

    real*4, intent (inout)  :: conserved(ni, nj, nk, 5)
    real*4, intent (inout)  :: fvisc(ni-1, nj-1, nk-1, 5)
    real*4 :: fvisc_new(ni-1, nj-1, nk-1, 5)

    real*4, intent (inout)  :: dAi(ni, nj-1, nk-1, 3)
    real*4, intent (inout)  :: dAj(ni-1, nj, nk-1, 3)
    real*4, intent (inout)  :: dAk(ni-1, nj-1, nk, 3)
    real*4, intent (inout)  :: vol(ni-1, nj-1, nk-1)
    real*4, intent (inout)  :: xlength(ni-1, nj-1, nk-1)
    real*4, intent (inout)  :: r(ni, nj, nk)

    logical*1, intent (inout)  :: walli(ni, nj-1, nk-1)
    logical*1, intent (inout)  :: wallj(ni-1, nj, nk-1)
    logical*1, intent (inout)  :: wallk(ni-1, nj-1, nk)

    real*4, intent (inout)  :: mu

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*4 :: tauc(ni-1, nj-1, nk-1, 6)
    real*4 :: taui(ni, nj-1, nk-1, 6)
    real*4 :: tauj(ni-1, nj, nk-1, 6)
    real*4 :: tauk(ni-1, nj-1, nk, 6)

    real*4 :: ri(ni, nj-1, nk-1)
    real*4 :: rj(ni-1, nj, nk-1)
    real*4 :: rk(ni-1, nj-1, nk)

    real*4 :: visc_lim

    real*4 :: fi(ni, nj-1, nk-1, 3, 5)
    real*4 :: fj(ni-1, nj, nk-1, 3, 5)
    real*4 :: fk(ni-1, nj-1, nk, 3, 5)


    real*4 :: rc(ni-1, nj-1, nk-1)
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
        V(:,:,:, i) = conserved(:,:,:,i+1)/conserved(:,:,:,1)
    end do
    V(:,:,:,3) = V(:,:,:,3)/r


    ! Cell-centered vars
    call node_to_cell(r, rc, ni, nj, nk, 1)
    call node_to_cell(V, Vc, ni, nj, nk, 3)
    call node_to_cell(conserved(:,:,:,1), roc, ni, nj, nk, 1)

    ! Face-centered vars
    call node_to_face(r, ri, rj, rk, ni, nj, nk, 1)

    ! Calculate grad V
    do i = 1,3
        call grad(V(:,:,:,i), gradV(:,:,:,:,i), vol, dAi, dAj, dAk, r, ni, nj, nk)
    end do
    ! gradV is indexed (..., which dirn, which velocity)

    ! Calculate divergence of V
    ! call div(V, divV, vol, dAi, dAj, dAk, ni, nj, nk)
    ! divV = divV*2e0/3e0

    ! tau contains the six unique terms in the tensor
    ! divV and gradV are cell-centered

    ! tau_xx = 2*dVx_dx - 2/3*divV
    tauc(:,:,:,1) = 2e0*gradV(:,:,:,1,1)! - divV

    ! tau_rr = 2*dVr_dr - 2/3*divV
    tauc(:,:,:,2) = 2e0*gradV(:,:,:,2,2)! - divV

    ! tau_tt = 2*(dVt_dt/r + Vr/r) - 2/3*divV
    tauc(:,:,:,3) = 2e0*(gradV(:,:,:,3,3)+ Vc(:,:,:,2))/rc! - divV

    ! tau_xr = tau_rx = dVx_dr + dVr_dx
    tauc(:,:,:,4) = gradV(:,:,:,2,1) + gradV(:,:,:,1,2)

    ! tau_xt = tau_tx = dVx_dt/r + dVt_dx
    tauc(:,:,:,5) = gradV(:,:,:,3,1)/rc + gradV(:,:,:,1,3)

    ! tau_rt = tau_tr = dVr_dt/r + dVt_dr - Vt/r
    tauc(:,:,:,6) = gradV(:,:,:,3,2)/rc + gradV(:,:,:,2,3) - Vc(:,:,:,3)/rc


    ! Calculate vorticity
    ! omega_x = dVr/dt - dVt/dr - Vt/r
    vort = 0e0
    vort(:,:,:,1) = gradV(:,:,:, 3, 2) - gradV(:,:,:,2,3) - Vc(:,:,:,3)/rc
    ! omega_r = dVt/dx - dVx/dt
    vort(:,:,:,2) = gradV(:,:,:, 1, 3) - gradV(:,:,:,3,1)
    ! omega_t = dVx/dr - dVr/dx
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
    fvisc = 0.2e0*fvisc_new + 0.8e0*fvisc
    ! fvisc = 0.5e0*fvisc_new + 0.5e0*fvisc
    ! fvisc = -fvisc_new

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
