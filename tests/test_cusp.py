"""Run a periodic grid with CFL=0 to check which nodes have periodicity applied."""

import pytest
import turbigen.grid
import turbigen.solvers.ember as ember
import numpy as np
import matplotlib.pyplot as plt
from turbigen import util

logger = util.make_logger()
logger.setLevel("INFO")


def make_sector():
    # Geometry
    L = 0.1
    rm = 10.0
    dr = 0.1

    r1 = rm - dr / 2.0
    r2 = rm + dr / 2.0

    nj = 13
    ni = 25
    nk = 9

    Nb = int(2.0 * np.pi * rm / dr)
    pitch = 2.0 * np.pi / Nb

    xv = np.linspace(0, L, ni)
    rv = np.linspace(r1, r2, nj)
    tv = np.linspace(-pitch / 2.0, pitch / 2.0, nk)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

    return xrt, Nb


@pytest.mark.slow
def test_periodic():
    xrt, Nb = make_sector()

    L = np.ptp(xrt[0])
    dtdx = np.tan(np.radians(30.0))
    xrt[2] += dtdx * xrt[0]

    patches = [
        turbigen.grid.InletPatch(i=0),
        turbigen.grid.OutletPatch(i=-1),
        turbigen.grid.CuspPatch(k=0),
        turbigen.grid.CuspPatch(k=-1),
    ]

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, Nb, patches)

    g = turbigen.grid.Grid([block])
    g.check_coordinates()
    g.match_patches()

    # Boundary conditions
    cp = 1005.0
    ga = 1.4
    mu = 1.8e-5
    Po1 = 1e5
    To1 = 300.0
    Alpha = 0.0
    Beta = 0.0
    rgas = cp * ga / (ga - 1.0)
    cv = rgas / ga
    rho0 = Po1 / (rgas * To1)
    u0 = cv * To1

    # Set an initial guess
    block.set_rho_u(rho0, u0)
    block.Vx = 50.0
    block.Vr = 0.0
    block.Vt = 0.0
    block.cp = cp
    block.gamma = ga
    block.Omega = 0.0
    block.mu = mu

    P1 = Po1 * 0.7
    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_P_T(Po1, To1)
    g.apply_inlet(So1, Alpha, Beta)
    g.calculate_wall_distance()
    g.apply_outlet(P1)

    ember.Ember(
        n_step=4000,
        n_step_avg=1000,
        i_loss=0,
    ).run(g)

    C = g[0][:, g[0].nj // 2, :]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.contourf(C.x, C.rt, C.P)

    fig, ax = plt.subplots()
    ax.contourf(C.x, C.rt, C.To)

    fig, ax = plt.subplots()
    ax.contourf(C.x, C.rt, C.s)

    fig, ax = plt.subplots()
    m = ax.contourf(C.x, C.rt, C.Alpha)
    plt.colorbar(m, ax=ax)

    plt.show()


@pytest.mark.slow
def test_cusp():
    xrt, Nb = make_sector()

    ni = xrt.shape[1]

    ite = int(ni * 0.6)
    icusp = ite - 4
    print(icusp, ite, ni)

    patches = [
        turbigen.grid.InletPatch(i=0),
        turbigen.grid.OutletPatch(i=-1),
        turbigen.grid.PeriodicPatch(k=0, i=(ite, -1)),
        turbigen.grid.PeriodicPatch(k=-1, i=(ite, -1)),
        turbigen.grid.CuspPatch(k=0, i=(icusp + 1, ite)),
        turbigen.grid.CuspPatch(k=-1, i=(icusp + 1, ite)),
    ]

    pitch = 2.0 * np.pi / float(Nb)
    rm = 0.5 * (xrt[1, 0, 0, 0] + xrt[1, -1, 0, 0])

    x = xrt[0, :, 0, 0]
    L = x[ite]
    Lcusp = x[ite] - x[icusp]

    xn = (x - x[ite]) / L
    tmax = L * 0.04
    thick = np.interp(xn, [-1.0, -Lcusp / L, 0.0, 1.0], [tmax, tmax, 0.0, 0.0])
    pitch_frac = xrt[2] / pitch / 2.0
    xrt[2, :, :, :] -= pitch_frac * thick[:, None, None]

    # Make a bit assymetric
    xn2 = (x - x[0]) / np.ptp(x)
    xrt[2, :, :, :] += tmax / rm * 4 * xn2[:, None, None] ** 2

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, Nb, patches)

    nj = xrt.shape[2]
    # C = block[:, nj, :]
    # plt.plot(C.x, C.rt, "k-", lw=0.5)
    # plt.plot(C.x.T, C.rt.T, "k-", lw=0.5)
    # plt.show()
    # quit()

    g = turbigen.grid.Grid(
        [
            block,
        ]
    )
    g.check_coordinates()
    g.match_patches()

    assert g[0].patches[-1].match is g[0].patches[-2], "Cusp patches do not match"
    assert g[0].patches[-2].match is g[0].patches[-1], "Cusp patches do not match"

    # Boundary conditions
    cp = 1005.0
    ga = 1.4
    mu = 1.8e-5
    Po1 = 1e5
    To1 = 300.0
    Alpha = 0.0
    Beta = 0.0
    rgas = cp * ga / (ga - 1.0)
    cv = rgas / ga
    rho0 = Po1 / (rgas * To1)
    u0 = cv * To1

    # Set an initial guess
    L = np.ptp(block.x)
    pitch_frac = block.t / block.pitch
    length_frac = block.x / L
    block.set_rho_u(rho0, u0)
    block.Vx = 50.0
    block.Vr = 0.0
    block.Vt = 0.0
    block.cp = cp
    block.gamma = ga
    block.Omega = 0.0
    block.mu = mu

    P1 = Po1 * 0.7
    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_P_T(Po1, To1)
    g.apply_inlet(So1, Alpha, Beta)
    g.calculate_wall_distance()
    g.apply_outlet(P1)

    ember.Ember(
        n_step=4000,
        n_step_avg=1000,
        i_loss=0,
    ).run(g)

    C = g[0][:, g[0].nj // 2, :]
    P = C.P[:, (0, -1)]

    # Should be unloaded from icusp+1 onwards
    Ptol = 1e-6 * P1
    assert (np.abs(np.diff(P[icusp + 1 :, :], axis=-1)) < Ptol).all(), (
        "Cusp patch not unloaded correctly"
    )


if __name__ == "__main__":
    test_periodic()
