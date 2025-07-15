subroutine smatvec(A, x, y, m, n, k)
    implicit none
    integer, intent(in) :: m, n, k
    real(4), intent(in) :: A(m, n, k)
    real(4), intent(in) :: x(n, k)
    real(4), intent(out) :: y(m, k)
    integer :: i

    do i = 1, k
        y(:, i) = matmul(A(:, :, i), x(:, i))
    end do

end subroutine smatvec


subroutine smatmat(A, B, C, m, n, p, k)
    implicit none
    integer, intent(in) :: m, n, p, k
    real(4), intent(in) :: A(m, n, k)
    real(4), intent(in) :: B(n, p, k)
    real(4), intent(out) :: C(m, p, k)
    integer :: i

    do i = 1, k
        C(:, :, i) = matmul(A(:, :, i), B(:, :, i))
    end do

end subroutine smatmat


subroutine dmatvec(A, x, y, m, n, k)
    implicit none
    integer, intent(in) :: m, n, k
    real(8), intent(in) :: A(m, n, k)
    real(8), intent(in) :: x(n, k)
    real(8), intent(out) :: y(m, k)
    integer :: i

    do i = 1, k
        y(:, i) = matmul(A(:, :, i), x(:, i))
    end do

end subroutine dmatvec


subroutine dmatmat(A, B, C, m, n, p, k)
    implicit none
    integer, intent(in) :: m, n, p, k
    real(8), intent(in) :: A(m, n, k)
    real(8), intent(in) :: B(n, p, k)
    real(8), intent(out) :: C(m, p, k)
    integer :: i

    do i = 1, k
        C(:, :, i) = matmul(A(:, :, i), B(:, :, i))
    end do

end subroutine dmatmat
