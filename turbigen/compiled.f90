! Compiled functions to speed up expensive calulations

subroutine node_to_face(xn, xi, xj, xk, np, ni, nj, nk)

    implicit none

    integer, intent (in)  :: np
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*8, intent (in)  :: xn(np, ni, nj, nk)
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

    real*8, intent (in)  :: xn(np, ni, nj, nk)
    real*8, intent (out)  :: xc(np, ni-1, nj-1, nk-1)

    ! Cell values are the average of all eight hex vertices
    xc = (&
          xn(:, 1:ni-1, 1:nj-1, 1:nk-1) & ! i,j,k
        + xn(:, 2:ni, 1:nj-1, 1:nk-1) & ! i+1,j,k
        + xn(:, 2:ni, 2:nj, 1:nk-1) & ! i+1,j+1,k
        + xn(:, 1:ni-1, 2:nj, 1:nk-1) & ! i,j+1,k
        + xn(:, 1:ni-1, 1:nj-1, 2:nk) & ! i,j,k+1
        + xn(:, 2:ni, 1:nj-1, 2:nk) & ! i+1,j,k+1
        + xn(:, 2:ni, 2:nj, 2:nk) & ! i+1,j+1,k+1
        + xn(:, 1:ni-1, 2:nj, 2:nk) & ! i,j+1,k+1
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
    xn(:, 2:ni-1, 1, nk) = (&
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

subroutine smooth(x, sf, np, ni, nj, nk)
    ! Smooth the 4D array x in-place towards a linear fitted value
    ! This yields a second-order error

    implicit none

    integer, intent (in)  :: np
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real*8, intent (in)  :: sf

    real*8, intent (inout)  :: x(np, ni, nj, nk)

    real*8 :: xc(np, ni, nj, nk)

    real*8 :: sf1
    real*8 :: sfu3

    ! Shorthand smoothing factors
    sf1 = 1. - sf
    sfu3 = sf/3.0

    ! Take a copy of the data so we can insert smoothed
    ! values in-place into the original array
    xc = x

    ! interior points
    x(:, 2:ni-1, 2:nj-1, 2:nk-1) = sf1*xc(:, 2:ni-1, 2:nj-1, 2:nk-1) + sfu3*( &
          0.5*xc(:, 1:ni-2, 2:nj-1, 2:nk-1) +  0.5*xc(:, 3:ni,   2:nj-1, 2:nk-1) &  ! i
        + 0.5*xc(:, 2:ni-1, 1:nj-2, 2:nk-1) +  0.5*xc(:, 2:ni-1, 3:nj,   2:nk-1) &  ! j
        + 0.5*xc(:, 2:ni-1, 2:nj-1, 1:nk-2) +  0.5*xc(:, 2:ni-1, 2:nj-1, 3:nk  ) &  ! k
    )

    ! six faces

    ! i=1
    x(:, 1, 2:nj-1, 2:nk-1) = sf1*xc(:, 1, 2:nj-1, 2:nk-1) + sfu3*( &
          2.0*xc(:, 2, 2:nj-1, 2:nk-1) -      xc(:, 3, 2:nj-1, 2:nk-1) & ! i
        + 0.5*xc(:, 1, 1:nj-2, 2:nk-1) +  0.5*xc(:, 1, 3:nj,   2:nk-1) & ! j
        + 0.5*xc(:, 1, 2:nj-1, 1:nk-2) +  0.5*xc(:, 1, 2:nj-1, 3:nk  ) & ! k
    )

    ! i=ni
    x(:, ni, 2:nj-1, 2:nk-1) = sf1*xc(:, ni, 2:nj-1, 2:nk-1) + sfu3*( &
          2.0*xc(:, ni-1, 2:nj-1, 2:nk-1) -      xc(:, ni-2, 2:nj-1, 2:nk-1) & ! i
        + 0.5*xc(:, ni,   1:nj-2, 2:nk-1) +  0.5*xc(:, ni,   3:nj,   2:nk-1) & ! j
        + 0.5*xc(:, ni,   2:nj-1, 1:nk-2) +  0.5*xc(:, ni,   2:nj-1, 3:nk  ) & ! k
    )

    ! j=1
    x(:, 2:ni-1, 1, 2:nk-1) = sf1*xc(:, 2:ni-1, 1, 2:nk-1) + sfu3*( &
          0.5*xc(:, 1:ni-2, 1, 2:nk-1) +  0.5*xc(:, 3:ni,   1, 2:nk-1) & ! i
        + 2.0*xc(:, 2:ni-1, 2, 2:nk-1) -      xc(:, 2:ni-1, 3, 2:nk-1) & ! j
        + 0.5*xc(:, 2:ni-1, 1, 1:nk-2) +  0.5*xc(:, 2:ni-1, 1, 3:nk  ) & ! k
    )

    ! j=nj
    x(:, 2:ni-1, nj, 2:nk-1) = sf1*xc(:, 2:ni-1, nj, 2:nk-1) + sfu3*( &
          0.5*xc(:, 1:ni-2, nj,   2:nk-1) +  0.5*xc(:, 3:ni,   nj,   2:nk-1) & ! i
        + 2.0*xc(:, 2:ni-1, nj-1, 2:nk-1) -      xc(:, 2:ni-1, nj-2, 2:nk-1) & ! j
        + 0.5*xc(:, 2:ni-1, nj,   1:nk-2) +  0.5*xc(:, 2:ni-1, nj,   3:nk  ) & ! k
    )

    ! k=1
    x(:, 2:ni-1, 2:nj-1, 1) = sf1*xc(:, 2:ni-1, 2:nj-1, 1) + sfu3*( &
          0.5*xc(:, 1:ni-2, 2:nj-1, 1) +  0.5*xc(:, 3:ni,   2:nj-1, 1) &  ! i
        + 0.5*xc(:, 2:ni-1, 1:nj-2, 1) +  0.5*xc(:, 2:ni-1, 3:nj,   1) &  ! j
        + 2.0*xc(:, 2:ni-1, 2:nj-1, 2) -      xc(:, 2:ni-1, 2:nj-1, 3) &  ! k
    )

    ! k=nk
    x(:, 2:ni-1, 2:nj-1, nk) = sf1*xc(:, 2:ni-1, 2:nj-1, nk) + sfu3*( &
          0.5*xc(:, 1:ni-2, 2:nj-1, nk  ) +  0.5*xc(:, 3:ni  , 2:nj-1, nk  ) & ! i
        + 0.5*xc(:, 2:ni-1, 1:nj-2, nk  ) +  0.5*xc(:, 2:ni-1, 3:nj,   nk  ) & ! j
        + 2.0*xc(:, 2:ni-1, 2:nj-1, nk-1) -      xc(:, 2:ni-1, 2:nj-1, nk-2) & ! k
    )

    ! twelve edges

    ! i=1,  j=1
    x(:, 1, 1, 2:nk-1) = sf1*xc(:, 1, 1, 2:nk-1) + sfu3*( &
          2.0*xc(:, 2, 1, 2:nk-1) -      xc(:, 3, 1, 2:nk-1) & ! i
        + 2.0*xc(:, 1, 2, 2:nk-1) -      xc(:, 1, 3, 2:nk-1) & ! j
        + 0.5*xc(:, 1, 1, 1:nk-2) +  0.5*xc(:, 1, 1, 3:nk  ) & ! k
    )

    ! i=1,  j=nj
    x(:, 1, nj, 2:nk-1) = sf1*xc(:, 1, nj, 2:nk-1) + sfu3*( &
          2.0*xc(:, 2, nj,   2:nk-1) -      xc(:, 3, nj,   2:nk-1) & ! i
        + 2.0*xc(:, 1, nj-1, 2:nk-1) -      xc(:, 1, nj-2, 2:nk-1) & ! j
        + 0.5*xc(:, 1, nj,   1:nk-2) +  0.5*xc(:, 1, nj,   3:nk  ) & ! k
    )

    ! i=ni,  j=1
    x(:, ni, 1, 2:nk-1) = sf1*xc(:, ni, 1, 2:nk-1) + sfu3*( &
          2.0*xc(:, ni-1, 1, 2:nk-1) -      xc(:, ni-2, 1, 2:nk-1) & ! i
        + 2.0*xc(:, ni,   2, 2:nk-1) -      xc(:, ni,   3, 2:nk-1) & ! j
        + 0.5*xc(:, ni,   1, 1:nk-2) +  0.5*xc(:, ni,   1, 3:nk  ) & ! k
    )

    ! i=ni,  j=nj
    x(:, ni, nj, 2:nk-1) = sf1*xc(:, ni, nj, 2:nk-1) + sfu3*( &
          2.0*xc(:, ni-1, nj,   2:nk-1) -      xc(:, ni-2, nj,   2:nk-1) & ! i
        + 2.0*xc(:, ni,   nj-1, 2:nk-1) -      xc(:, ni,   nj-2, 2:nk-1) & ! j
        + 0.5*xc(:, ni,   nj,   1:nk-2) +  0.5*xc(:, ni,   nj,   3:nk  ) & ! k
    )

    ! i=1, k=1
    x(:, 1, 2:nj-1, 1) = sf1*xc(:, 1, 2:nj-1, 1) + sfu3*( &
          2.0*xc(:, 2, 2:nj-1, 1) -      xc(:, 3, 2:nj-1, 1) &  ! i
        + 0.5*xc(:, 1, 1:nj-2, 1) +  0.5*xc(:, 1, 3:nj,   1) &  ! j
        + 2.0*xc(:, 1, 2:nj-1, 2) -      xc(:, 1, 2:nj-1, 3) &  ! k
    )

    ! i=1, k=nk
    x(:, 1, 2:nj-1, nk) = sf1*xc(:, 1, 2:nj-1, nk) + sfu3*( &
          2.0*xc(:, 2, 2:nj-1, nk  ) -     xc(:, 3, 2:nj-1, nk  ) &  ! i
        + 0.5*xc(:, 1, 1:nj-2, nk  ) + 0.5*xc(:, 1, 3:nj,   nk  ) &  ! j
        + 2.0*xc(:, 1, 2:nj-1, nk-1) -     xc(:, 1, 2:nj-1, nk-2) &  ! k
    )

    ! i=ni, k=1
    x(:, ni, 2:nj-1, 1) = sf1*xc(:, ni, 2:nj-1, 1) + sfu3*( &
          2.0*xc(:, ni-1, 2:nj-1, 1) -     xc(:, ni-2, 2:nj-1, 1) &  ! i
        + 0.5*xc(:, ni,   1:nj-2, 1) + 0.5*xc(:, ni,   3:nj,   1) &  ! j
        + 2.0*xc(:, ni,   2:nj-1, 2) -     xc(:, ni,   2:nj-1, 3) &  ! k
    )

    ! i=ni, k=nk
    x(:, ni, 2:nj-1, nk) = sf1*xc(:, ni, 2:nj-1, nk) + sfu3*( &
          2.0*xc(:, ni-1, 2:nj-1, nk  ) -     xc(:, ni-2, 2:nj-1, nk  ) &  ! i
        + 0.5*xc(:, ni,   1:nj-2, nk  ) + 0.5*xc(:, ni,   3:nj,   nk  ) &  ! j
        + 2.0*xc(:, ni,   2:nj-1, nk-1) -     xc(:, ni,   2:nj-1, nk-2) &  ! k
    )

    ! j=1, k=1
    x(:, 2:ni-1, 1, 1) = sf1*xc(:, 2:ni-1, 1, 1) + sfu3*( &
          0.5*xc(:, 1:ni-2, 1, 1) + 0.5*xc(:, 3:ni,   1, 1) &  ! i
        + 2.0*xc(:, 2:ni-1, 2, 1) -     xc(:, 2:ni-1, 3, 1) &  ! j
        + 2.0*xc(:, 2:ni-1, 1, 2) -     xc(:, 2:ni-1, 1, 3) &  ! k
    )

    ! j=1, k=nk
    x(:, 2:ni-1, 1, nk) = sf1*xc(:, 2:ni-1, 1, nk) + sfu3*( &
          0.5*xc(:, 1:ni-2, 1, nk  ) + 0.5*xc(:, 3:ni,   1, nk  ) &  ! i
        + 2.0*xc(:, 2:ni-1, 2, nk  ) -     xc(:, 2:ni-1, 3, nk  ) &  ! j
        + 2.0*xc(:, 2:ni-1, 1, nk-1) -     xc(:, 2:ni-1, 1, nk-2) &  ! k
    )

    ! j=nj, k=1
    x(:, 2:ni-1, nj, 1) = sf1*xc(:, 2:ni-1, nj, 1) + sfu3*( &
          0.5*xc(:, 1:ni-2, nj,   1) + 0.5*xc(:, 3:ni,   nj,   1) &  ! i
        + 2.0*xc(:, 2:ni-1, nj-1, 1) -     xc(:, 2:ni-1, nj-2, 1) &  ! j
        + 2.0*xc(:, 2:ni-1, nj,   2) -     xc(:, 2:ni-1, nj,   3) &  ! k
    )

    ! j=nj, k=nk
    x(:, 2:ni-1, nj, nk) = sf1*xc(:, 2:ni-1, nj, nk) + sfu3*( &
          0.5*xc(:, 1:ni-2, nj,   nk  ) + 0.5*xc(:, 3:ni,   nj,   nk  ) &  ! i
        + 2.0*xc(:, 2:ni-1, nj-1, nk  ) -     xc(:, 2:ni-1, nj-2, nk  ) &  ! j
        + 2.0*xc(:, 2:ni-1, nj,   nk-1) -     xc(:, 2:ni-1, nj,   nk-2) &  ! k
    )

    ! eight vertices

    ! i=1, j=1, k=1
    x(:, 1, 1, 1) = sf1*xc(:, 1, 1, 1) + sfu3*( &
          2.0*xc(:, 2, 1, 1) - xc(:, 3, 1, 1) &  ! i
        + 2.0*xc(:, 1, 2, 1) - xc(:, 1, 3, 1) &  ! j
        + 2.0*xc(:, 1, 1, 2) - xc(:, 1, 1, 3) &  ! k
    )

    ! i=1, j=1, k=nk
    x(:, 1, 1, nk) = sf1*xc(:, 1, 1, nk) + sfu3*( &
          2.0*xc(:, 2, 1, nk  ) - xc(:, 3, 1, nk  ) &  ! i
        + 2.0*xc(:, 1, 2, nk  ) - xc(:, 1, 3, nk  ) &  ! j
        + 2.0*xc(:, 1, 1, nk-1) - xc(:, 1, 1, nk-2) &  ! k
    )

    ! i=1, j=nj, k=1
    x(:, 1, nj, 1) = sf1*xc(:, 1, nj, 1) + sfu3*( &
          2.0*xc(:, 2, nj,   1) - xc(:, 3, nj,   1) &  ! i
        + 2.0*xc(:, 1, nj-1, 1) - xc(:, 1, nj-2, 1) &  ! j
        + 2.0*xc(:, 1, nj,   2) - xc(:, 1, nj,   3) &  ! k
    )

    ! i=1, j=nj, k=nk
    x(:, 1, nj, nk) = sf1*xc(:, 1, nj, nk) + sfu3*( &
          2.0*xc(:, 2, nj,     nk) - xc(:, 3, nj,   nk  ) &  ! i
        + 2.0*xc(:, 1, nj-1,   nk) - xc(:, 1, nj-2, nk  ) &  ! j
        + 2.0*xc(:, 1, nj,   nk-1) - xc(:, 1, nj,   nk-2) &  ! k
    )


    ! i=ni, j=1, k=1
    x(:, ni, 1, 1) = sf1*xc(:, ni, 1, 1) + sfu3*( &
          2.0*xc(:, ni-1, 1, 1) - xc(:, ni-2, 1, 1) &  ! i
        + 2.0*xc(:, ni,   2, 1) - xc(:, ni,   3, 1) &  ! j
        + 2.0*xc(:, ni,   1, 2) - xc(:, ni,   1, 3) &  ! k
    )

    ! i=ni, j=1, k=nk
    x(:, ni, 1, nk) = sf1*xc(:, ni, 1, nk) + sfu3*( &
          2.0*xc(:, ni-1, 1, nk  ) - xc(:, ni-2, 1, nk  ) &  ! i
        + 2.0*xc(:, ni, 2,   nk  ) - xc(:, ni,   3, nk  ) &  ! j
        + 2.0*xc(:, ni, 1,   nk-1) - xc(:, ni,   1, nk-2) &  ! k
    )

    ! i=ni, j=nj, k=1
    x(:, ni, nj, 1) = sf1*xc(:, ni, nj, 1) + sfu3*( &
          2.0*xc(:, ni-1,  nj,   1) - xc(:, ni-2,  nj,   1) &  ! i
        + 2.0*xc(:, ni, nj-1, 1) - xc(:, ni, nj-2, 1) &  ! j
        + 2.0*xc(:, ni, nj,   2) - xc(:, ni, nj,   3) &  ! k
    )

    ! i=ni, j=nj, k=nk
    x(:, ni, nj, nk) = sf1*xc(:, ni, nj, nk) + sfu3*( &
          2.0*xc(:, ni-1, nj,   nk  ) - xc(:, ni-2, nj,   nk  ) &  ! i
        + 2.0*xc(:, ni,   nj-1, nk  ) - xc(:, ni,   nj-2, nk  ) &  ! j
        + 2.0*xc(:, ni,   nj,   nk-1) - xc(:, ni,   nj,   nk-2) &  ! k
    )

end subroutine
