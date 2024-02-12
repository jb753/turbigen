subroutine smooth(x, sf, xs, np, ni, nj, nk)

    implicit none

    integer, intent (in)  :: np
    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real, intent (in)  :: x(np, ni, nj, nk)
    real, intent (out) :: xs(np, ni, nj, nk)

    real, intent (in)  :: sf
    real :: sf1
    real :: sfu3
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
