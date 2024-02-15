! Compiled functions to speed up expensive calulations

subroutine step(conserved, Phor, Omega, walli, wallj, wallk, dt, dAi, dAj, dAk, vol, resid, ni, nj, nk)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*8, intent (inout)  :: conserved(5, ni, nj, nk)
    real*8, intent (out) :: resid(5,ni, nj, nk)
    real*8, intent (inout)  :: Phor(3, ni, nj, nk)
    real*8, intent (inout)  :: Omega
    logical*1, intent (inout)  :: walli(ni, nj-1, nk-1)
    logical*1, intent (inout)  :: wallj(ni-1, nj, nk-1)
    logical*1, intent (inout)  :: wallk(ni-1, nj-1, nk)
    real*8, intent (inout)  :: dAi(3, ni, nj-1, nk-1)
    real*8, intent (inout)  :: dAj(3, ni-1, nj, nk-1)
    real*8, intent (inout)  :: dAk(3, ni-1, nj-1, nk)
    real*8, intent (inout)  :: vol(ni-1, nj-1, nk-1)
    real*8, intent (inout)  :: dt(ni-1, nj-1, nk-1)

    integer :: ip

    real*8 :: Sn(1,ni, nj, nk)
    real*8 :: Sc(1,ni-1, nj-1, nk-1)

    real*8 :: conservedi(5,ni, nj-1, nk-1)
    real*8 :: conservedj(5,ni-1, nj, nk-1)
    real*8 :: conservedk(5,ni-1, nj-1, nk)

    real*8 :: Phori(3, ni, nj-1, nk-1)
    real*8 :: Phorj(3, ni-1, nj, nk-1)
    real*8 :: Phork(3, ni-1, nj-1, nk)

    real*8 :: fi(5,3,ni, nj-1, nk-1)
    real*8 :: fj(5,3,ni-1, nj, nk-1)
    real*8 :: fk(5,3,ni-1, nj-1, nk)

    real*8 :: fsum_vol(5,ni-1, nj-1, nk-1)
    real*8 :: resc(5,ni-1, nj-1, nk-1)


    ! Calculate source term at nodes, average at cell centers
    Sn(1, :, :, :) = Phor(1,:,:,:)/Phor(3,:,:,:) &
        + conserved(4,:,:,:)*conserved(4,:,:,:)/conserved(1,:,:,:)/Phor(3,:,:,:)/Phor(3,:,:,:)/Phor(3,:,:,:)
    call node_to_cell(Sn, Sc, 1, ni, nj, nk)

    ! Get face-centered vars
    call node_to_face( &
        conserved, conservedi, conservedj, conservedk, &
        5, ni, nj, nk &
    )
    call node_to_face( &
        Phor, Phori, Phorj, Phork, &
        3, ni, nj, nk &
    )

    ! Evaluate fluxes on each set of faces
    call get_fluxes_face(conservedi, Phori, walli, Omega, fi, ni, nj-1, nk-1)
    call get_fluxes_face(conservedj, Phorj, wallj, Omega, fj, ni-1, nj, nk-1)
    call get_fluxes_face(conservedk, Phork, wallk, Omega, fk, ni-1, nj-1, nk)

    ! Get the net flux into each cell
    call sum_fluxes(fi, fj, fk, dAi, dAj, dAk, vol, fsum_vol, 5, ni, nj, nk)


    ! Add on source term
    fsum_vol(3,:,:,:) = fsum_vol(3,:,:,:) + Sc(1,:,:,:)

    ! Integrate forward in time
    do ip = 1, 5
        resc(ip,:,:,:)  = fsum_vol(ip, :,:,:) * dt
    end do

    ! Distribute change to nodes
    call cell_to_node(resc, resid, 5, ni, nj, nk)

    ! ! Add change
    ! conserved = conserved + resn

    ! ! Smooth
    ! call smooth(conserved, conservedsmth, sf, 5, ni, nj, nk)
    ! conserved = conservedsmth

    ! print *,'SMOOTHED'

end subroutine


subroutine get_fluxes_face(conserved, Phor, wall, Omega, flux, ni, nj, nk)
    ! using face-centered properties, evaluate fluxes for one direction

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*8, intent (in)  :: conserved(5, ni, nj, nk)
    real*8, intent (in)  :: Phor(3, ni, nj, nk)
    real*8, intent (in)  :: Omega
    logical*1, intent (in)  :: wall(ni, nj, nk)

    real*8, intent (out) :: flux(5, 3, ni, nj, nk)

    integer :: ic
    integer :: ip

    real*8 :: Vx(ni, nj, nk)
    real*8 :: Vr(ni, nj, nk)
    real*8 :: rVt(ni, nj, nk)

    ! Calculate velocities
    Vx = conserved(2,:, :, :)/conserved(1,:,:,:)
    Vr = conserved(3,:, :, :)/conserved(1,:,:,:)
    rVt = conserved(4,:, :, :)/conserved(1,:,:,:)

    ! mass
    flux(1,1,:,:,:) = conserved(2,:, :, :)  ! rhoVx
    flux(1,2,:,:,:) = conserved(3,:, :, :)  ! rhoVr
    flux(1,3,:,:,:) = conserved(4,:, :, :)/Phor(3,:,:,:)  ! rhoVt=rhorVt/r

    ! x-mom
    do ic = 1,3
        flux(2,ic,:,:,:) = flux(1,ic,:,:,:) * Vx
    end do

    ! r-mom
    do ic = 1,3
        flux(3,ic,:,:,:) = flux(1,ic,:,:,:) * Vr
    end do

    ! rt-mom
    do ic = 1,3
        flux(4,ic,:,:,:) = flux(1,ic,:,:,:) * rVt
    end do

    ! ho
    do ic = 1,3
        flux(5,ic,:,:,:) = flux(1,ic,:,:,:) * Phor(2,:,:,:)
    end do

    ! zero convective fluxes on walls
    do ip = 1,5
        do ic = 1,3
            where (wall)
                flux(ip, ic, :, :, :) = 0.0
            end where
        end do
    end do

    ! pressure fluxes
    ! x-mom in x-dirn
    flux(2, 1, :, :, :) = flux(2, 1, :, :, :) + Phor(1,:,:,:)
    ! r-mom in r-dirn
    flux(3, 2, :, :, :) = flux(3, 2, :, :, :) + Phor(1,:,:,:)
    ! rt-mom in t-dirn
    flux(4, 3, :, :, :) = flux(4, 3, :, :, :) + Phor(3,:,:,:)*Phor(1,:,:,:)
    ! ho in t-dirn
    flux(5, 3, :, :, :) = flux(5, 3, :, :, :) + Omega*Phor(3,:,:,:)*Phor(1,:,:,:)


end subroutine


subroutine sum_fluxes(fi, fj, fk, dAi, dAj, dAk, vol, Fsum, np, ni, nj, nk)

    implicit none

    integer, intent (in)  :: np
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    integer :: ip

    real*8, intent (in)  :: dAi(3, ni, nj-1, nk-1)
    real*8, intent (in)  :: dAj(3, ni-1, nj, nk-1)
    real*8, intent (in)  :: dAk(3, ni-1, nj-1, nk)
    real*8, intent (in)  :: vol(ni-1, nj-1, nk-1)

    real*8, intent (in)  :: fi(np, 3, ni, nj-1, nk-1)
    real*8, intent (in)  :: fj(np, 3, ni-1, nj, nk-1)
    real*8, intent (in)  :: fk(np, 3, ni-1, nj-1, nk)

    real*8 :: fisum(ni, nj-1, nk-1)
    real*8 :: fjsum(ni-1, nj, nk-1)
    real*8 :: fksum(ni-1, nj-1, nk)

    real*8, intent (out)  :: fsum(np, ni-1, nj-1, nk-1)

    fsum = 0.

    do ip = 1,np
        ! Dot product areas with the fluxes
        fisum = sum(dAi*fi(ip,:,:,:,:),1)
        fjsum = sum(dAj*fj(ip,:,:,:,:),1)
        fksum = sum(dAk*fk(ip,:,:,:,:),1)
        ! Add on the differences
        fsum(ip, :, :, :) = fsum(ip, :, :, :) + (fisum(1:ni-1,:,:) - fisum(2:ni,:,:))
        fsum(ip, :, :, :) = fsum(ip, :, :, :) + (fjsum(:,1:nj-1,:) - fjsum(:,2:nj,:))
        fsum(ip, :, :, :) = fsum(ip, :, :, :) + (fksum(:,:,1:nk-1) - fksum(:,:,2:nk))
        ! Divide by volume
        fsum(ip, :, :, :) = fsum(ip, :, :, :)/vol
    end do


end subroutine


subroutine node_to_face(xn, xi, xj, xk, np, ni, nj, nk)

    implicit none

    integer, intent (in)  :: np
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*8, intent (inout)  :: xn(np, ni, nj, nk)
    real*8, intent (out)  :: xi(np, ni, nj-1, nk-1)
    real*8, intent (out)  :: xj(np, ni-1, nj, nk-1)
    real*8, intent (out)  :: xk(np, ni-1, nj-1, nk)

    ! Values on i-faces are average over four bounding vertices
    xi = (&
          xn(:, :, 1:nj-1, 1:nk-1) & ! j, k
        + xn(:, :, 2:nj,   1:nk-1) & ! j+1, k
        + xn(:, :, 1:nj-1, 2:nk  ) & ! j, k+1
        + xn(:, :, 2:nj,   2:nk  ) & ! j+1, k+1
    )/4.0

    ! Values on j-faces are average over four bounding vertices
    xj = (&
          xn(:, 1:ni-1, :, 1:nk-1) & ! i, k
        + xn(:, 2:ni,   :, 1:nk-1) & ! i+1, k
        + xn(:, 1:ni-1, :, 2:nk  ) & ! i, k+1
        + xn(:, 2:ni,   :, 2:nk  ) & ! i+1, k+1
    )/4.0

    ! Values on k-faces are average over four bounding vertices
    xk = (&
          xn(:, 1:ni-1, 1:nj-1, :) & ! i, j
        + xn(:, 2:ni,   1:nj-1, :) & ! i+1, j
        + xn(:, 1:ni-1, 2:nj,   :) & ! i, j+1
        + xn(:, 2:ni,   2:nj,   :) & ! i+1, j+1
    )/4.0

end subroutine


subroutine node_to_cell(xn, xc, np, ni, nj, nk)

    implicit none

    integer, intent (in)  :: np
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*8, intent (inout)  :: xn(np, ni, nj, nk)
    real*8, intent (out)  :: xc(np, ni-1, nj-1, nk-1)

    ! Cell values are the average of all eight hex vertices
    xc = (&
          xn(:, 1:ni-1, 1:nj-1, 1:nk-1) & ! i,j,k
        + xn(:, 2:ni,   1:nj-1, 1:nk-1) & ! i+1,j,k
        + xn(:, 2:ni,   2:nj,   1:nk-1) & ! i+1,j+1,k
        + xn(:, 1:ni-1, 2:nj,   1:nk-1) & ! i,j+1,k
        + xn(:, 1:ni-1, 1:nj-1, 2:nk) & ! i,j,k+1
        + xn(:, 2:ni,   1:nj-1, 2:nk) & ! i+1,j,k+1
        + xn(:, 2:ni,   2:nj,   2:nk) & ! i+1,j+1,k+1
        + xn(:, 1:ni-1, 2:nj,   2:nk) & ! i,j+1,k+1
    )/8.0


end subroutine

subroutine cell_to_node(xc, xn, np, ni, nj, nk)

    implicit none

    integer, intent (in)  :: np
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*8, intent (in)  :: xc(np, ni-1, nj-1, nk-1)
    real*8, intent (out)  :: xn(np, ni, nj, nk)

    ! Interior nodes take 1/8 from each adjacent cell
    xn(:, 2:ni-1, 2:nj-1, 2:nk-1) = (&
          xc(:, 1:ni-2, 1:nj-2, 1:nk-2) & ! i,j,k
        + xc(:, 2:ni-1, 1:nj-2, 1:nk-2) & ! i+1,j,k
        + xc(:, 2:ni-1, 2:nj-1, 1:nk-2) & ! i+1,j+1,k
        + xc(:, 1:ni-2, 2:nj-1, 1:nk-2) & ! i,j+1,k
        + xc(:, 1:ni-2, 1:nj-2, 2:nk-1) & ! i,j,k+1
        + xc(:, 2:ni-1, 1:nj-2, 2:nk-1) & ! i+1,j,k+1
        + xc(:, 2:ni-1, 2:nj-1, 2:nk-1) & ! i+1,j+1,k+1
        + xc(:, 1:ni-2, 2:nj-1, 2:nk-1) & ! i,j+1,k+1
    )/8.0

    ! Face nodes take 1/4 from each adjacent cell

    ! i=1
    xn(:, 1, 2:nj-1, 2:nk-1) = (&
          xc(:, 1, 1:nj-2, 1:nk-2) & ! 1,j,k
        + xc(:, 1, 2:nj-1, 1:nk-2) & ! 1,j+1,k
        + xc(:, 1, 1:nj-2, 2:nk-1) & ! 1,j,k+1
        + xc(:, 1, 2:nj-1, 2:nk-1) & ! 1,j+1,k+1
    )/4.0

    ! i=ni
    xn(:, ni, 2:nj-1, 2:nk-1) = (&
          xc(:, ni-1, 1:nj-2, 1:nk-2) & ! ni-1,j,k
        + xc(:, ni-1, 2:nj-1, 1:nk-2) & ! ni-1,j+1,k
        + xc(:, ni-1, 1:nj-2, 2:nk-1) & ! ni-1,j,k+1
        + xc(:, ni-1, 2:nj-1, 2:nk-1) & ! ni-1,j+1,k+1
    )/4.0

    ! j=1
    xn(:, 2:ni-1, 1, 2:nk-1) = (&
          xc(:, 1:ni-2, 1, 1:nk-2) & ! i,1,k
        + xc(:, 2:ni-1, 1, 1:nk-2) & ! i+1,1,k
        + xc(:, 1:ni-2, 1, 2:nk-1) & ! i,1,k+1
        + xc(:, 2:ni-1, 1, 2:nk-1) & ! i+1,1,k+1
    )/4.0

    ! j=nj
    xn(:, 2:ni-1, nj, 2:nk-1) = (&
          xc(:, 1:ni-2, nj-1, 1:nk-2) & ! i,nj-1,k
        + xc(:, 2:ni-1, nj-1, 1:nk-2) & ! i+1,nj-1,k
        + xc(:, 1:ni-2, nj-1, 2:nk-1) & ! i,nj-1,k+1
        + xc(:, 2:ni-1, nj-1, 2:nk-1) & ! i+1,nj-1,k+1
    )/4.0

    ! k=1
    xn(:, 2:ni-1, 2:nj-1, 1) = (&
          xc(:, 1:ni-2, 1:nj-2, 1) &
        + xc(:, 2:ni-1, 1:nj-2, 1) &
        + xc(:, 1:ni-2, 2:nj-1, 1) &
        + xc(:, 2:ni-1, 2:nj-1, 1) &
    )/4.0

    ! k=nk
    xn(:, 2:ni-1, 2:nj-1, nk) = (&
          xc(:, 1:ni-2, 1:nj-2, nk-1) &
        + xc(:, 2:ni-1, 1:nj-2, nk-1) &
        + xc(:, 1:ni-2, 2:nj-1, nk-1) &
        + xc(:, 2:ni-1, 2:nj-1, nk-1) &
    )/4.0

    ! Edges take 1/2 from each adjacent cell

    ! i=1, j=1
    xn(:, 1, 1, 2:nk-1) = (&
          xc(:, 1, 1, 1:nk-2) &
        + xc(:, 1, 1, 2:nk-1) &
    )/2.0

    ! i=1, j=nj
    xn(:, 1, nj, 2:nk-1) = (&
          xc(:, 1, nj-1, 1:nk-2) &
        + xc(:, 1, nj-1, 2:nk-1) &
    )/2.0

    ! i=ni, j=1
    xn(:, ni, 1, 2:nk-1) = (&
          xc(:, ni-1, 1, 1:nk-2) &
        + xc(:, ni-1, 1, 2:nk-1) &
    )/2.0

    ! i=ni, j=nj
    xn(:, ni, nj, 2:nk-1) = (&
          xc(:, ni-1, nj-1, 1:nk-2) &
        + xc(:, ni-1, nj-1, 2:nk-1) &
    )/2.0

    ! i=1, k=1
    xn(:, 1, 2:nj-1, 1) = (&
          xc(:, 1, 1:nj-2, 1) &
        + xc(:, 1, 2:nj-1, 1) &
    )/2.0

    ! i=1, k=nk
    xn(:, 1, 2:nj-1, nk) = (&
          xc(:, 1, 1:nj-2, nk-1) &
        + xc(:, 1, 2:nj-1, nk-1) &
    )/2.0

    ! i=ni, k=1
    xn(:, ni, 2:nj-1, 1) = (&
          xc(:, ni-1, 1:nj-2, 1) &
        + xc(:, ni-1, 2:nj-1, 1) &
    )/2.0

    ! i=ni, k=nk
    xn(:, ni, 2:nj-1, nk) = (&
          xc(:, ni-1, 1:nj-2, nk-1) &
        + xc(:, ni-1, 2:nj-1, nk-1) &
    )/2.0

    ! j=1, k=1
    xn(:, 2:ni-1, 1, 1) = (&
          xc(:, 1:ni-2, 1, 1) &
        + xc(:, 2:ni-1, 1, 1) &
    )/2.0

    ! j=1, k=nk
    xn(:, 2:ni-1, 1, nk) = (&
          xc(:, 1:ni-2, 1, nk-1) &
        + xc(:, 2:ni-1, 1, nk-1) &
    )/2.0

    ! j=nj, k=1
    xn(:, 2:ni-1, nj, 1) = (&
          xc(:, 1:ni-2, nj-1, 1) &
        + xc(:, 2:ni-1, nj-1, 1) &
    )/2.0

    ! j=nj, k=nk
    xn(:, 2:ni-1, nj, nk) = (&
          xc(:, 1:ni-2, nj-1, nk-1) &
        + xc(:, 2:ni-1, nj-1, nk-1) &
    )/2.0

    ! Corners take entirety from nearest cell
    xn(:, 1,  1,  1) = xc(:, 1,    1,    1)
    xn(:, 1,  nj, 1) = xc(:, 1,    nj-1, 1)
    xn(:, ni, nj, 1) = xc(:, ni-1, nj-1, 1)
    xn(:, ni, 1,  1) = xc(:, ni-1, 1,    1)
    xn(:, 1,  1,  nk) = xc(:, 1,    1,    nk-1)
    xn(:, 1,  nj, nk) = xc(:, 1,    nj-1, nk-1)
    xn(:, ni, nj, nk) = xc(:, ni-1, nj-1, nk-1)
    xn(:, ni, 1,  nk) = xc(:, ni-1, 1,    nk-1)


end subroutine

subroutine smooth(x, xs, sf, np, ni, nj, nk)
    ! Smooth the 4D array x towards a linear fitted value
    ! This yields a second-order error

    implicit none

    integer, intent (in)  :: np
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*8, intent (in)  :: sf

    real*8, intent (inout)  :: x(np, ni, nj, nk)
    real*8, intent (out)  :: xs(np, ni, nj, nk)

    real*8 :: sf1
    real*8 :: sfu3

    ! Shorthand smoothing factors
    sf1 = 1. - sf
    sfu3 = sf/3.0

    ! interior points
    xs(:, 2:ni-1, 2:nj-1, 2:nk-1) = sf1*x(:, 2:ni-1, 2:nj-1, 2:nk-1) + sfu3*( &
          0.5*x(:, 1:ni-2, 2:nj-1, 2:nk-1) +  0.5*x(:, 3:ni,   2:nj-1, 2:nk-1) &  ! i
        + 0.5*x(:, 2:ni-1, 1:nj-2, 2:nk-1) +  0.5*x(:, 2:ni-1, 3:nj,   2:nk-1) &  ! j
        + 0.5*x(:, 2:ni-1, 2:nj-1, 1:nk-2) +  0.5*x(:, 2:ni-1, 2:nj-1, 3:nk  ) &  ! k
    )

    ! six faces

    ! i=1
    xs(:, 1, 2:nj-1, 2:nk-1) = sf1*x(:, 1, 2:nj-1, 2:nk-1) + sfu3*( &
          2.0*x(:, 2, 2:nj-1, 2:nk-1) -      x(:, 3, 2:nj-1, 2:nk-1) & ! i
        + 0.5*x(:, 1, 1:nj-2, 2:nk-1) +  0.5*x(:, 1, 3:nj,   2:nk-1) & ! j
        + 0.5*x(:, 1, 2:nj-1, 1:nk-2) +  0.5*x(:, 1, 2:nj-1, 3:nk  ) & ! k
    )

    ! i=ni
    xs(:, ni, 2:nj-1, 2:nk-1) = sf1*x(:, ni, 2:nj-1, 2:nk-1) + sfu3*( &
          2.0*x(:, ni-1, 2:nj-1, 2:nk-1) -      x(:, ni-2, 2:nj-1, 2:nk-1) & ! i
        + 0.5*x(:, ni,   1:nj-2, 2:nk-1) +  0.5*x(:, ni,   3:nj,   2:nk-1) & ! j
        + 0.5*x(:, ni,   2:nj-1, 1:nk-2) +  0.5*x(:, ni,   2:nj-1, 3:nk  ) & ! k
    )

    ! j=1
    xs(:, 2:ni-1, 1, 2:nk-1) = sf1*x(:, 2:ni-1, 1, 2:nk-1) + sfu3*( &
          0.5*x(:, 1:ni-2, 1, 2:nk-1) +  0.5*x(:, 3:ni,   1, 2:nk-1) & ! i
        + 2.0*x(:, 2:ni-1, 2, 2:nk-1) -      x(:, 2:ni-1, 3, 2:nk-1) & ! j
        + 0.5*x(:, 2:ni-1, 1, 1:nk-2) +  0.5*x(:, 2:ni-1, 1, 3:nk  ) & ! k
    )

    ! j=nj
    xs(:, 2:ni-1, nj, 2:nk-1) = sf1*x(:, 2:ni-1, nj, 2:nk-1) + sfu3*( &
          0.5*x(:, 1:ni-2, nj,   2:nk-1) +  0.5*x(:, 3:ni,   nj,   2:nk-1) & ! i
        + 2.0*x(:, 2:ni-1, nj-1, 2:nk-1) -      x(:, 2:ni-1, nj-2, 2:nk-1) & ! j
        + 0.5*x(:, 2:ni-1, nj,   1:nk-2) +  0.5*x(:, 2:ni-1, nj,   3:nk  ) & ! k
    )

    ! k=1
    xs(:, 2:ni-1, 2:nj-1, 1) = sf1*x(:, 2:ni-1, 2:nj-1, 1) + sfu3*( &
          0.5*x(:, 1:ni-2, 2:nj-1, 1) +  0.5*x(:, 3:ni,   2:nj-1, 1) &  ! i
        + 0.5*x(:, 2:ni-1, 1:nj-2, 1) +  0.5*x(:, 2:ni-1, 3:nj,   1) &  ! j
        + 2.0*x(:, 2:ni-1, 2:nj-1, 2) -      x(:, 2:ni-1, 2:nj-1, 3) &  ! k
    )

    ! k=nk
    xs(:, 2:ni-1, 2:nj-1, nk) = sf1*x(:, 2:ni-1, 2:nj-1, nk) + sfu3*( &
          0.5*x(:, 1:ni-2, 2:nj-1, nk  ) +  0.5*x(:, 3:ni  , 2:nj-1, nk  ) & ! i
        + 0.5*x(:, 2:ni-1, 1:nj-2, nk  ) +  0.5*x(:, 2:ni-1, 3:nj,   nk  ) & ! j
        + 2.0*x(:, 2:ni-1, 2:nj-1, nk-1) -      x(:, 2:ni-1, 2:nj-1, nk-2) & ! k
    )

    ! twelve edges

    ! i=1,  j=1
    xs(:, 1, 1, 2:nk-1) = sf1*x(:, 1, 1, 2:nk-1) + sfu3*( &
          2.0*x(:, 2, 1, 2:nk-1) -      x(:, 3, 1, 2:nk-1) & ! i
        + 2.0*x(:, 1, 2, 2:nk-1) -      x(:, 1, 3, 2:nk-1) & ! j
        + 0.5*x(:, 1, 1, 1:nk-2) +  0.5*x(:, 1, 1, 3:nk  ) & ! k
    )

    ! i=1,  j=nj
    xs(:, 1, nj, 2:nk-1) = sf1*x(:, 1, nj, 2:nk-1) + sfu3*( &
          2.0*x(:, 2, nj,   2:nk-1) -      x(:, 3, nj,   2:nk-1) & ! i
        + 2.0*x(:, 1, nj-1, 2:nk-1) -      x(:, 1, nj-2, 2:nk-1) & ! j
        + 0.5*x(:, 1, nj,   1:nk-2) +  0.5*x(:, 1, nj,   3:nk  ) & ! k
    )

    ! i=ni,  j=1
    xs(:, ni, 1, 2:nk-1) = sf1*x(:, ni, 1, 2:nk-1) + sfu3*( &
          2.0*x(:, ni-1, 1, 2:nk-1) -      x(:, ni-2, 1, 2:nk-1) & ! i
        + 2.0*x(:, ni,   2, 2:nk-1) -      x(:, ni,   3, 2:nk-1) & ! j
        + 0.5*x(:, ni,   1, 1:nk-2) +  0.5*x(:, ni,   1, 3:nk  ) & ! k
    )

    ! i=ni,  j=nj
    xs(:, ni, nj, 2:nk-1) = sf1*x(:, ni, nj, 2:nk-1) + sfu3*( &
          2.0*x(:, ni-1, nj,   2:nk-1) -      x(:, ni-2, nj,   2:nk-1) & ! i
        + 2.0*x(:, ni,   nj-1, 2:nk-1) -      x(:, ni,   nj-2, 2:nk-1) & ! j
        + 0.5*x(:, ni,   nj,   1:nk-2) +  0.5*x(:, ni,   nj,   3:nk  ) & ! k
    )

    ! i=1, k=1
    xs(:, 1, 2:nj-1, 1) = sf1*x(:, 1, 2:nj-1, 1) + sfu3*( &
          2.0*x(:, 2, 2:nj-1, 1) -      x(:, 3, 2:nj-1, 1) &  ! i
        + 0.5*x(:, 1, 1:nj-2, 1) +  0.5*x(:, 1, 3:nj,   1) &  ! j
        + 2.0*x(:, 1, 2:nj-1, 2) -      x(:, 1, 2:nj-1, 3) &  ! k
    )

    ! i=1, k=nk
    xs(:, 1, 2:nj-1, nk) = sf1*x(:, 1, 2:nj-1, nk) + sfu3*( &
          2.0*x(:, 2, 2:nj-1, nk  ) -     x(:, 3, 2:nj-1, nk  ) &  ! i
        + 0.5*x(:, 1, 1:nj-2, nk  ) + 0.5*x(:, 1, 3:nj,   nk  ) &  ! j
        + 2.0*x(:, 1, 2:nj-1, nk-1) -     x(:, 1, 2:nj-1, nk-2) &  ! k
    )

    ! i=ni, k=1
    xs(:, ni, 2:nj-1, 1) = sf1*x(:, ni, 2:nj-1, 1) + sfu3*( &
          2.0*x(:, ni-1, 2:nj-1, 1) -     x(:, ni-2, 2:nj-1, 1) &  ! i
        + 0.5*x(:, ni,   1:nj-2, 1) + 0.5*x(:, ni,   3:nj,   1) &  ! j
        + 2.0*x(:, ni,   2:nj-1, 2) -     x(:, ni,   2:nj-1, 3) &  ! k
    )

    ! i=ni, k=nk
    xs(:, ni, 2:nj-1, nk) = sf1*x(:, ni, 2:nj-1, nk) + sfu3*( &
          2.0*x(:, ni-1, 2:nj-1, nk  ) -     x(:, ni-2, 2:nj-1, nk  ) &  ! i
        + 0.5*x(:, ni,   1:nj-2, nk  ) + 0.5*x(:, ni,   3:nj,   nk  ) &  ! j
        + 2.0*x(:, ni,   2:nj-1, nk-1) -     x(:, ni,   2:nj-1, nk-2) &  ! k
    )

    ! j=1, k=1
    xs(:, 2:ni-1, 1, 1) = sf1*x(:, 2:ni-1, 1, 1) + sfu3*( &
          0.5*x(:, 1:ni-2, 1, 1) + 0.5*x(:, 3:ni,   1, 1) &  ! i
        + 2.0*x(:, 2:ni-1, 2, 1) -     x(:, 2:ni-1, 3, 1) &  ! j
        + 2.0*x(:, 2:ni-1, 1, 2) -     x(:, 2:ni-1, 1, 3) &  ! k
    )

    ! j=1, k=nk
    xs(:, 2:ni-1, 1, nk) = sf1*x(:, 2:ni-1, 1, nk) + sfu3*( &
          0.5*x(:, 1:ni-2, 1, nk  ) + 0.5*x(:, 3:ni,   1, nk  ) &  ! i
        + 2.0*x(:, 2:ni-1, 2, nk  ) -     x(:, 2:ni-1, 3, nk  ) &  ! j
        + 2.0*x(:, 2:ni-1, 1, nk-1) -     x(:, 2:ni-1, 1, nk-2) &  ! k
    )

    ! j=nj, k=1
    xs(:, 2:ni-1, nj, 1) = sf1*x(:, 2:ni-1, nj, 1) + sfu3*( &
          0.5*x(:, 1:ni-2, nj,   1) + 0.5*x(:, 3:ni,   nj,   1) &  ! i
        + 2.0*x(:, 2:ni-1, nj-1, 1) -     x(:, 2:ni-1, nj-2, 1) &  ! j
        + 2.0*x(:, 2:ni-1, nj,   2) -     x(:, 2:ni-1, nj,   3) &  ! k
    )

    ! j=nj, k=nk
    xs(:, 2:ni-1, nj, nk) = sf1*x(:, 2:ni-1, nj, nk) + sfu3*( &
          0.5*x(:, 1:ni-2, nj,   nk  ) + 0.5*x(:, 3:ni,   nj,   nk  ) &  ! i
        + 2.0*x(:, 2:ni-1, nj-1, nk  ) -     x(:, 2:ni-1, nj-2, nk  ) &  ! j
        + 2.0*x(:, 2:ni-1, nj,   nk-1) -     x(:, 2:ni-1, nj,   nk-2) &  ! k
    )

    ! eight vertices

    ! i=1, j=1, k=1
    xs(:, 1, 1, 1) = sf1*x(:, 1, 1, 1) + sfu3*( &
          2.0*x(:, 2, 1, 1) - x(:, 3, 1, 1) &  ! i
        + 2.0*x(:, 1, 2, 1) - x(:, 1, 3, 1) &  ! j
        + 2.0*x(:, 1, 1, 2) - x(:, 1, 1, 3) &  ! k
    )

    ! i=1, j=1, k=nk
    xs(:, 1, 1, nk) = sf1*x(:, 1, 1, nk) + sfu3*( &
          2.0*x(:, 2, 1, nk  ) - x(:, 3, 1, nk  ) &  ! i
        + 2.0*x(:, 1, 2, nk  ) - x(:, 1, 3, nk  ) &  ! j
        + 2.0*x(:, 1, 1, nk-1) - x(:, 1, 1, nk-2) &  ! k
    )

    ! i=1, j=nj, k=1
    xs(:, 1, nj, 1) = sf1*x(:, 1, nj, 1) + sfu3*( &
          2.0*x(:, 2, nj,   1) - x(:, 3, nj,   1) &  ! i
        + 2.0*x(:, 1, nj-1, 1) - x(:, 1, nj-2, 1) &  ! j
        + 2.0*x(:, 1, nj,   2) - x(:, 1, nj,   3) &  ! k
    )

    ! i=1, j=nj, k=nk
    xs(:, 1, nj, nk) = sf1*x(:, 1, nj, nk) + sfu3*( &
          2.0*x(:, 2, nj,     nk) - x(:, 3, nj,   nk  ) &  ! i
        + 2.0*x(:, 1, nj-1,   nk) - x(:, 1, nj-2, nk  ) &  ! j
        + 2.0*x(:, 1, nj,   nk-1) - x(:, 1, nj,   nk-2) &  ! k
    )


    ! i=ni, j=1, k=1
    xs(:, ni, 1, 1) = sf1*x(:, ni, 1, 1) + sfu3*( &
          2.0*x(:, ni-1, 1, 1) - x(:, ni-2, 1, 1) &  ! i
        + 2.0*x(:, ni,   2, 1) - x(:, ni,   3, 1) &  ! j
        + 2.0*x(:, ni,   1, 2) - x(:, ni,   1, 3) &  ! k
    )

    ! i=ni, j=1, k=nk
    xs(:, ni, 1, nk) = sf1*x(:, ni, 1, nk) + sfu3*( &
          2.0*x(:, ni-1, 1, nk  ) - x(:, ni-2, 1, nk  ) &  ! i
        + 2.0*x(:, ni, 2,   nk  ) - x(:, ni,   3, nk  ) &  ! j
        + 2.0*x(:, ni, 1,   nk-1) - x(:, ni,   1, nk-2) &  ! k
    )

    ! i=ni, j=nj, k=1
    xs(:, ni, nj, 1) = sf1*x(:, ni, nj, 1) + sfu3*( &
          2.0*x(:, ni-1,  nj,   1) - x(:, ni-2,  nj,   1) &  ! i
        + 2.0*x(:, ni, nj-1, 1) - x(:, ni, nj-2, 1) &  ! j
        + 2.0*x(:, ni, nj,   2) - x(:, ni, nj,   3) &  ! k
    )

    ! i=ni, j=nj, k=nk
    xs(:, ni, nj, nk) = sf1*x(:, ni, nj, nk) + sfu3*( &
          2.0*x(:, ni-1, nj,   nk  ) - x(:, ni-2, nj,   nk  ) &  ! i
        + 2.0*x(:, ni,   nj-1, nk  ) - x(:, ni,   nj-2, nk  ) &  ! j
        + 2.0*x(:, ni,   nj,   nk-1) - x(:, ni,   nj,   nk-2) &  ! k
    )

end subroutine
