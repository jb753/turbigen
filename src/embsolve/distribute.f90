! Routines for distributing node/face/cell values around

subroutine node_to_face(xn, xi, xj, xk, ni, nj, nk, np)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real, intent (in)  :: xn(ni, nj, nk, np)

    ! Note: seems to go faster if the outputs are 'inout'
    real, intent (inout)  :: xi(ni, nj-1, nk-1, np)
    real, intent (inout)  :: xj(ni-1, nj, nk-1, np)
    real, intent (inout)  :: xk(ni-1, nj-1, nk, np)

    !$omp workshare
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
    !$omp end workshare

end subroutine


subroutine node_to_cell(xn, xc, ni, nj, nk, np)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real, intent (in)  :: xn(ni, nj, nk, np)

    ! Note: seems to go faster if the outputs are 'inout'
    real, intent (inout)  :: xc(ni-1, nj-1, nk-1, np)

    !$omp workshare
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
    !$omp end workshare


end subroutine

subroutine cell_to_node(xc, xn, ni, nj, nk, np)

    implicit none

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk
    integer, intent (in)  :: np

    real, intent (in)  :: xc(ni-1, nj-1, nk-1, np)

    ! Note: seems to go faster if the outputs are 'inout'
    real, intent (inout)  :: xn(ni, nj, nk, np)

    !$omp workshare
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
    !$omp end workshare

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
    real, intent (in)  :: xc(ni-1, nj-1, nk-1, np)

    ! Note: seems to go faster if the outputs are 'inout'
    real, intent (inout)  :: xi(ni, nj-1, nk-1, np)
    real, intent (inout)  :: xj(ni-1, nj, nk-1, np)
    real, intent (inout)  :: xk(ni-1, nj-1, nk, np)

    !$omp workshare
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
    !$omp end workshare

end subroutine
