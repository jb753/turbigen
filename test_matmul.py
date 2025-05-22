import numpy as np
import time
import matvec  # Assumes compiled Fortran module

# Parameters
k, m, n = 65 * 65, 5, 5
dtype = np.float32
N = 10000

# Random inputs
A_f = np.asfortranarray(np.random.rand(m, n, k).astype(dtype))
B_f = np.asfortranarray(np.random.rand(m, n, k).astype(dtype))
x_f = np.asfortranarray(np.random.rand(n, k).astype(dtype))

# Also prepare NumPy-style (k, m, n) for matmul
A_np = np.transpose(A_f, (2, 0, 1))  # (k, m, n)
B_np = np.transpose(B_f, (2, 0, 1))  # (k, m, n)
x_np = x_f.T[..., None]  # (k, n)

print("matvec")

# Benchmark Fortran
start = time.perf_counter()
for _ in range(N):
    y_f = matvec.matvec(A_f, x_f)
end = time.perf_counter()
print(f"Fortran matvec time: {end - start:.6f} seconds")

# Benchmark NumPy
start = time.perf_counter()
for _ in range(N):
    y_np = np.matmul(A_np, x_np)
end = time.perf_counter()
print(f"NumPy matmul time: {end - start:.6f} seconds")

# Check correctness
assert y_f.dtype == dtype == y_np.dtype
max_diff = np.max(np.abs(y_f - y_np.T))
print(f"Max difference between Fortran and NumPy results: {max_diff:.3e}")

print("matmat")
N = 2000

# Benchmark Fortran
start = time.perf_counter()
for _ in range(N):
    C_f = matvec.matmat(A_f, B_f)
end = time.perf_counter()
print(f"Fortran matmat time: {end - start:.6f} seconds")

# Benchmark NumPy
start = time.perf_counter()
for _ in range(N):
    C_np = np.matmul(A_np, B_np)
end = time.perf_counter()
print(f"NumPy matmul time: {end - start:.6f} seconds")

# Transpose C_np to match Fortran output
C_np = np.transpose(C_np, (1, 2, 0))  # (k, n, m)

# Check correctness
assert C_f.dtype == dtype == C_np.dtype
max_diff = np.max(np.abs(C_f - C_np))
print(f"Max difference between Fortran and NumPy results: {max_diff:.3e}")
