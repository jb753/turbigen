"""Functions to compute the discrete Laplacian."""
import numpy as np
from scipy.signal import convolve


# Coefficients for central difference second derivative order 2, 4, 6 accurate
D2_CENTRAL = np.array(
    [
        [0, 0, 1, -2, 1, 0, 0],
        [0, -1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12, 0],
        [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
    ]
)

# Coefficients for boundary difference second derivative order 1, 3, 5 accurate
D2_BOUNDARY = np.array(
    [
        [1, -2, 1, 0, 0, 0, 0, 0],
        [35 / 12, -26 / 3, 19 / 2, -14 / 3, 11 / 12, 0, 0, 0],
        [203 / 45, -87 / 5, 117 / 4, -254 / 9, 33 / 2, -27 / 5, 137 / 180, 0],
    ]
)


def make_stencil(ityp, jtyp, ktyp, order):
    """Function to generate stencils at different locations.

    Parameters
    ----------
    ityp, jtyp, ktyp: int
        0 for interior eg. (i-1, i, i+1)
        1 for start eg. (1, 2, 3)
        -1 for end eg. (ni-3, ni-2, ni-1)
    order: int
        2 for second-order or 4 for fourth-order accurate

    """

    # Get 1D coefficients
    iord = order // 2 - 1
    cent = D2_CENTRAL[iord]
    bound = D2_BOUNDARY[iord]

    # Preallocate stencil size
    iz = np.nonzero(cent)[0]
    N = len(iz)
    N2 = N // 2
    cent = cent[iz]
    bound = bound[:N]
    D = np.zeros((N, N, N))

    def get_origin(typ):
        if typ == 1:
            o = 0
        elif typ == 0:
            o = N2
        elif typ == -1:
            o = -1
        else:
            raise Exception("typ should be in (0,1,-1)")
        return o

    # Select the 'origin' of the stencil
    io = get_origin(ityp)
    jo = get_origin(jtyp)
    ko = get_origin(ktyp)

    # Set the coefficients for i/j/k in turn

    for i in range(N):
        ii = i - N2 + order // 2
        if ityp:
            D[i, jo, ko] += bound[ii]
        else:
            D[i, jo, ko] += cent[ii]

    for j in range(N):
        jj = j - N2 + order // 2
        if jtyp:
            D[io, j, ko] += bound[jj]
        else:
            D[io, j, ko] += cent[jj]

    for k in range(N):
        kk = k - N2 + order // 2
        if ktyp:
            D[io, jo, k] += bound[kk]
        else:
            D[io, jo, k] += cent[kk]

    return D

# Second-order accurate
KERN2 = {
    # Interior
    "ijk": make_stencil(0, 0, 0, 2),
    # 8 Faces
    "i0": make_stencil(1, 0, 0, 2),
    "ni": make_stencil(-1, 0, 0, 2),
    "j0": make_stencil(0, 1, 0, 2),
    "nj": make_stencil(0, -1, 0, 2),
    "k0": make_stencil(0, 0, 1, 2),
    "nk": make_stencil(0, 0, -1, 2),
    # 12 Edges
    # i=0
    "i0j0": make_stencil(1, 1, 0, 2),
    "i0nj": make_stencil(1, -1, 0, 2),
    "i0k0": make_stencil(1, 0, 1, 2),
    "i0nk": make_stencil(1, 0, -1, 2),
    # i=ni
    "nij0": make_stencil(-1, 1, 0, 2),
    "ninj": make_stencil(-1, -1, 0, 2),
    "nik0": make_stencil(-1, 0, 1, 2),
    "nink": make_stencil(-1, 0, -1, 2),
    # j
    "j0k0": make_stencil(0, 1, 1, 2),
    "j0nk": make_stencil(0, 1, -1, 2),
    "njk0": make_stencil(0, -1, 1, 2),
    "njnk": make_stencil(0, -1, -1, 2),
    # 8 Vertices
    "i0j0k0": make_stencil(1, 1, 1, 2),
    "i0njk0": make_stencil(1, -1, 1, 2),
    "i0j0nk": make_stencil(1, 1, -1, 2),
    "i0njnk": make_stencil(1, -1, -1, 2),
    "nij0k0": make_stencil(-1, 1, 1, 2),
    "ninjk0": make_stencil(-1, -1, 1, 2),
    "nij0nk": make_stencil(-1, 1, -1, 2),
    "ninjnk": make_stencil(-1, -1, -1, 2),
}


def laplacian2(x, sf=1.):
    # x has shape [ni,nj,nk]
    # return smoothed nodal values, second-order accuarte

    xs = np.full_like(x, np.nan)

    def conv(y, sten):
        return convolve(y, sf*KERN2[sten], mode="valid").squeeze()

    # Interior
    xs[1:-1, 1:-1, 1:-1] = conv(x, "ijk")

    # 6 Faces
    xs[0, 1:-1, 1:-1] = conv(x[:3, :, :], "i0")
    xs[-1, 1:-1, 1:-1] = conv(x[-3:, :, :], "ni")
    xs[1:-1, 0, 1:-1] = conv(x[:, :3, :], "j0")
    xs[1:-1, -1, 1:-1] = conv(x[:, -3:, :], "nj")
    xs[1:-1, 1:-1, 0] = conv(x[:, :, :3], "k0")
    xs[1:-1, 1:-1, -1] = conv(x[:, :, -3:], "nk")

    # 12 Edges
    # i0
    xs[0, 0, 1:-1] = conv(x[:3, :3, :], "i0j0")
    xs[0, -1, 1:-1] = conv(x[:3, -3:, :], "i0nj")
    xs[0, 1:-1, 0] = conv(x[:3, :, :3], "i0k0")
    xs[0, 1:-1, -1] = conv(x[:3, :, -3:], "i0nk")
    # ni
    xs[-1, 0, 1:-1] = conv(x[-3:, :3, :], "nij0")
    xs[-1, -1, 1:-1] = conv(x[-3:, -3:, :], "ninj")
    xs[-1, 1:-1, 0] = conv(x[-3:, :, :3], "nik0")
    xs[-1, 1:-1, -1] = conv(x[-3:, :, -3:], "nink")
    # j
    xs[1:-1, 0, 0] = conv(x[:, :3, :3], "j0k0")
    xs[1:-1, 0, -1] = conv(x[:, :3, -3:], "j0nk")
    xs[1:-1, -1, 0] = conv(x[:, -3:, :3], "njk0")
    xs[1:-1, -1, -1] = conv(x[:, -3:, -3:], "njnk")

    # 8 Corners
    # i=0
    xs[0, 0, 0] = conv(x[:3, :3, :3], "i0j0k0")
    xs[0, -1, 0] = conv(x[:3, -3:, :3], "i0njk0")
    xs[0, 0, -1] = conv(x[:3, :3, -3:], "i0j0nk")
    xs[0, -1, -1] = conv(x[:3, -3:, -3:], "i0njnk")
    # ni
    xs[-1, 0, 0] = conv(x[-3:, :3, :3], "nij0k0")
    xs[-1, -1, 0] = conv(x[-3:, -3:, :3], "ninjk0")
    xs[-1, 0, -1] = conv(x[-3:, :3, -3:], "nij0nk")
    xs[-1, -1, -1] = conv(x[-3:, -3:, -3:], "ninjnk")

    assert not np.isnan(xs).any()

    return xs

