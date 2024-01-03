"""Functions for smoothing meshes."""

import numpy as np
import turbigen.util
from enum import IntEnum
import scipy.interpolate


class BCond(IntEnum):
    FIX = 0
    EXTRAP_Y = 1
    ORTHOGONAL = 2
    DISTANCE = 3


def smooth_block(xy, bci, bck, rtol=np.inf, maxiter=10000):
    """Smooth blade-to-blade mesh by solving Poisson's equation.

    * The coordinates are modified in-place.
    * Assumes that j is the spanwise index.
    * Control near-wall grid spacing by setting a "smooth fraction"?
      But what about orthogonality control?

    Parameters
    ----------
    xy: (2, ni, nj, nk) array
        Blade-to-blade coordinates of an entire block (all three dimensions).
    bci: (2, nk) array
        Boundary condition types for i=0 and i=ni-1 surfaces.
    bck: (2, ni) array
        Boundary condition types for k=0 and k=nk-1 surfaces.
    rtol: float
        Relative tolerance to terminate smoothing loop, default to one iteration.

    """

    # Perform some checks on the input data
    assert xy.ndim == 4
    assert xy.shape[0] == 2
    assert xy.shape[1] > 4
    assert xy.shape[3] > 4

    Nwall = xy.shape[3] // 2 - 3
    rf_wall = np.linspace(0.0, 1.0, Nwall)
    # rf_wall = np.ones_like(rf_wall)*0.5
    rf_wall = np.stack((rf_wall, np.flip(rf_wall))).reshape(1, 1, 1, Nwall, 2)

    # Calculate normals for k=0 and k=nk-1 surfaces
    dxy_wall = np.diff(xy[:, :, :, (0, -1)], n=1, axis=1)
    # Cell centered normal vectors
    vec_wall_cell = np.stack((-dxy_wall[1], dxy_wall[0]))
    lvec_wall_cell = np.sqrt(np.sum(vec_wall_cell**2.0, axis=0, keepdims=True))
    vec_wall_cell /= lvec_wall_cell
    # Nodal normal vectors
    vec_wall = np.concatenate(
        (
            vec_wall_cell[:, (0,), ...],
            0.5 * (vec_wall_cell[:, 1:, ...] + vec_wall_cell[:, :-1, ...]),
            vec_wall_cell[:, (-1,), ...],
        ),
        axis=1,
    )

    # Store the starting wall distance on k=0 and k=nk-1
    dwall0 = np.sqrt(
        np.sum(np.diff(xy[..., : (Nwall + 1)], n=1, axis=-1) ** 2.0, axis=0)
    ).squeeze()
    dwallnk = np.sqrt(
        np.sum(np.diff(xy[..., -(Nwall + 1) :], n=1, axis=-1) ** 2.0, axis=0)
    ).squeeze()

    dwall = np.stack((dwall0, dwallnk), axis=-1)
    dwall = np.expand_dims(dwall, 0)
    dwall = np.cumsum(dwall, axis=-2)
    dwall[:, :, :, :, 0] = dwall[:, :, :, :, 0] - dwall[:, :, :, (0,), 0]
    dwall[:, :, :, :, 1] = dwall[:, :, :, :, 1] - dwall[:, :, :, (-1,), 1]

    vec_wall = np.expand_dims(vec_wall, 3)
    # vec_ortho = dwall * vec_wall

    xy_wall_ortho = vec_wall * dwall + xy[..., None, (0, -1)]

    # Store logical indices for where to extrapolate
    i0ex = bci[0] == BCond.EXTRAP_Y
    niex = bci[1] == BCond.EXTRAP_Y

    # Logical indices for where to set orthogonal
    # k0orth = bck[0] == BCond.ORTHOGONAL
    # nkorth = bck[1] == BCond.ORTHOGONAL

    # Logical indices for where to set distance
    k0dist = bck[0] == BCond.DISTANCE
    nkdist = bck[1] == BCond.DISTANCE

    # xy_new = xy.copy()
    # # Orthogonal on k boundaries
    # rf = np.linspace(1.
    # xy_new[:, k0orth, :, 1] = (
    #     rf * xy_wall_ortho[:, k0orth, :, 0] + (1.0 - rf) * xy[:, k0orth, :, 1]
    # )
    # xy_new[:, nkorth, :, -2] = (
    #     rf * xy_wall_ortho[:, nkorth, :, 1] + (1.0 - rf) * xy[:, k0orth, :, -2]
    # )
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(xy_new[0, :, 0, :], xy_new[1, :, 0, :], "k-",lw=0.5)
    # ax.plot(xy_new[0, :, 0, :].T, xy_new[1, :, 0, :].T, "k-",lw=0.5)
    # ax.axis("equal")
    # plt.show()
    # quit()

    # Set absolute tolerance on the larger of x and y ranges
    Lref = np.max([np.ptp(xyi) for xyi in xy])
    tol = Lref * rtol

    # Loop over smoothing iterations
    for n in range(maxiter):
        # Central differencing for first and second derivatives wrt i and k indices
        dxyi = xy[:, 2:, :, 1:-1] - xy[:, :-2, :, 1:-1]
        dxyk = xy[:, 1:-1, :, 2:] - xy[:, 1:-1, :, :-2]
        d2xyik = (
            xy[:, 2:, :, 2:]
            - xy[:, 2:, :, :-2]
            + xy[:, :-2, :, :-2]
            - xy[:, :-2, :, 2:]
        )

        # Mean coords
        xymi = 0.5 * (xy[:, 2:, :, 1:-1] + xy[:, :-2, :, 1:-1])
        xymk = 0.5 * (xy[:, 1:-1, :, 2:] + xy[:, 1:-1, :, :-2])

        # First derivative combinations
        gamma = dxyi[0, ...] ** 2.0 + dxyi[1, ...] ** 2.0
        alpha = dxyk[0, ...] ** 2.0 + dxyk[1, ...] ** 2.0
        beta = dxyi[0, ...] * dxyk[0, ...] + dxyi[1, ...] * dxyk[1, ...]

        # Evaluate new interior coordinates
        xy_new = (xymi * alpha - beta * d2xyik * 0.25 + xymk * gamma) / (alpha + gamma)

        # Store residual
        resid_xy = np.abs(xy_new - xy[:, 1:-1, :, 1:-1])

        # Assign to grid
        xy[:, 1:-1, :, 1:-1] = xy_new

        # Now apply boundary conditions

        # Extrapolate on i boundaries
        xy[1, 0, :, i0ex] = 2.0 * xy[1, 1, :, i0ex] - xy[1, 2, :, i0ex]
        xy[1, -1, :, niex] = 2.0 * xy[1, -2, :, niex] - xy[1, -3, :, niex]

        # Distance control on k boundaries

        # # Store the starting wall distance on k=0 and k=nk-1
        # dwall0_now = np.diff(xy[..., :2], n=1, axis=-1)
        # dwallnk_now = np.diff(xy[..., -2:], n=1, axis=-1)
        # vec_wall_now = np.stack((dwall0_now, dwallnk_now), axis=-1)
        # lvec_wall_now = np.sqrt(np.sum(vec_wall_now**2.0, axis=0, keepdims=True))
        # vec_wall_now /= lvec_wall_now
        # xy_wall_dist = vec_wall_now * dwall + xy[..., None, (0, -1)]

        # Orthogonal on k boundaries
        rf_orth = 0.1
        Nwall_orth = 3
        xy[:, k0dist, :, :Nwall_orth] = (
            rf_orth * xy_wall_ortho[:, k0dist, ...][..., :Nwall_orth, 0]
            + (1.0 - rf_orth) * xy[:, k0dist, :, :Nwall_orth]
        )
        xy[:, nkdist, :, -Nwall_orth:] = (
            rf_orth * xy_wall_ortho[:, nkdist, ...][..., -Nwall_orth:, 1]
            + (1.0 - rf_orth) * xy[:, nkdist, :, -Nwall_orth:]
        )

        if np.mod(n, 5) == 0 or n == maxiter - 1:
            # Set wall distance on k boundaries
            for i in range(bck.shape[1]):
                rfk = 0.5
                if k0dist[i]:
                    # Interpolator for this i-line
                    dist_now = turbigen.util.cum_arc_length(xy[:, i, :, :], axis=-1)

                    for j in range(dist_now.shape[0]):
                        interp_now = scipy.interpolate.interp1d(
                            dist_now[j], xy[:, i, j, :]
                        )
                        xy[:, i, j, :Nwall] = (
                            rfk * interp_now(dwall[0, i, j, :, 0])
                            + (1.0 - rfk) * xy[:, i, j, :Nwall]
                        )

                if nkdist[i]:
                    # Interpolator for this i-line
                    dist_now = turbigen.util.cum_arc_length(xy[:, i, :, :], axis=-1)
                    for j in range(dist_now.shape[0]):
                        dist_now[j] -= dist_now[j][-1]
                        interp_now = scipy.interpolate.interp1d(
                            dist_now[j], xy[:, i, j, :], axis=-1
                        )
                        xy[:, i, j, -Nwall:] = (
                            rfk * interp_now(dwall[0, i, j, :, 1])
                            + (1.0 - rfk) * xy[:, i, j, -Nwall:]
                        )

        # # Orthogonal on k boundaries
        # rf_now = np.min((n*.02, 1.)) * rf_wall
        # xy[:, k0dist, :, :(Nwall)] = (
        #     rf_now[..., 0] * xy_wall_dist[:, k0dist, ...][..., 0]
        #     + (1.0 - rf_now[..., 0]) * xy[:, k0dist, :, :(Nwall)]
        # )
        # xy[:, nkdist, :, -(Nwall):] = (
        #     rf_now[..., 1] * xy_wall_dist[:, nkdist, ...][..., 1]
        #     + (1.0 - rf_now[..., 1]) * xy[:, nkdist, :, -(Nwall):]
        # )

        # print(n, resid_xy.max() / Lref)
        if np.isnan(resid_xy).any():
            raise Exception("Smoothing diverged")

        # If maximum movement less than tolerance, break out of loop
        if (resid_xy.max() < tol).all():
            break

        # xy[:, k0dist, :, :(Nwall)] = (
        #     rf_now[..., 0] * xy_wall_dist[:, k0dist, ...][..., 0]
        #     + (1.0 - rf_now[..., 0]) * xy[:, k0dist, :, :(Nwall)]
        # )
        # xy[:, nkdist, :, -(Nwall):] = (
        #     rf_now[..., 1] * xy_wall_dist[:, nkdist, ...][..., 1]
        #     + (1.0 - rf_now[..., 1]) * xy[:, nkdist, :, -(Nwall):]
        # )

    # dist = turbigen.util.cum_arc_length(xy, axis=-1)
    # ddist = np.diff(dist, axis=-1)
    # ER = ddist[:,:,1:]/ddist[:,:,:-1]
    # ER[ER<1.] = 1./ER[ER<1.]

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10.,6.))
    # ax.plot(ER[100,0,:],'k-x')
    # plt.show()
    # quit()

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(figsize=(10.0, 6.0))
    # pitch = xy[1, 0, 0, :].ptp()
    # ax.plot(xy[0, :, 0, 0], xy[1, :, 0, 0], "r-", lw=1)
    # ax.plot(xy[0, :, 0, -1], xy[1, :, 0, -1], "r-", lw=1)
    # ax.plot(xy[0, :, 0, :], xy[1, :, 0, :], "k-", lw=0.5)
    # ax.plot(xy[0, :, 0, :].T, xy[1, :, 0, :].T, "k-", lw=0.5)
    # ax.plot(xy[0, :, 0, :], xy[1, :, 0, :] + pitch, "k-", lw=0.5)
    # ax.plot(xy[0, :, 0, :].T, xy[1, :, 0, :].T + pitch, "k-", lw=0.5)
    # ax.plot(xy[0, :, 0, 0], xy[1, :, 0, 0] + pitch, "r-", lw=1)
    # ax.plot(xy[0, :, 0, -1], xy[1, :, 0, -1] + pitch, "r-", lw=1)
    # ax.axis("equal")
    # plt.show()
    # quit()

    return xy
