import numpy as np
import time

N = 65 * 65
a = np.full((N,), 1.0)
b = np.full((N,), 2.0)
c = np.full((N,), 3.0)
d = np.full((N,), 4.0)

k = 5000

data = (
    (a, b, a, a, a),
    (c, d, c, a, a),
    (b, c, d, a, a),
    (c, d, c, a, a),
    (b, c, d, a, a),
)

start = time.perf_counter()
for _ in range(k):
    Af = np.stack(data)
end = time.perf_counter()
print(f"stack to Fortran shape: {end - start:.6f} seconds")

start = time.perf_counter()
for _ in range(k):
    Ac = np.stack([np.stack(row, axis=-1) for row in data], axis=-2)
end = time.perf_counter()
print(f"nested stack to C shape: {end - start:.6f} seconds")

start = time.perf_counter()
for _ in range(k):
    m = len(data)
    n = len(data[0])
    p = len(data[0][0])
    out = np.empty((m, n, p))
    for icol in range(n):
        for irow in range(m):
            out[irow, icol, ...] = data[irow][icol]
end = time.perf_counter()
assert np.allclose(Af[..., 0], out[..., 0])
print(f"preallocated F shape: {end - start:.6f} seconds")

start = time.perf_counter()
for _ in range(k):
    m = len(data)
    n = len(data[0])
    p = len(data[0][0])
    out = np.empty((p, m, n))
    for icol in range(n):
        for irow in range(m):
            out[..., irow, icol] = data[irow][icol]
end = time.perf_counter()
assert np.allclose(Ac[0, ...], out[0, ...])
print(f"preallocated C shape: {end - start:.6f} seconds")
