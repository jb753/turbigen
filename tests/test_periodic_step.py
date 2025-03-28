"""Run a periodic grid with CFL=0 to check which nodes have periodicity applied."""

import turbigen.grid
import turbigen.solvers.emb as embsolve
import numpy as np


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


def test_periodic():
    xrt, Nb = make_sector()

    ni = xrt.shape[1]

    ile = ni // 3
    ite = ni // 3 * 2

    patches = [
        # turbigen.grid.InletPatch(i=0),
        # turbigen.grid.OutletPatch(i=-1),
        turbigen.grid.PeriodicPatch(k=0, i=(0, ile)),
        turbigen.grid.PeriodicPatch(k=-1, i=(0, ile)),
        turbigen.grid.PeriodicPatch(k=0, i=(ite, -1)),
        turbigen.grid.PeriodicPatch(k=-1, i=(ite, -1)),
    ]

    pitch = 2.0 * np.pi / float(Nb)
    pitch_frac = xrt[2] / pitch / 2.0
    xrt[2, (ile + 1) : ite, :, :] -= pitch_frac[(ile + 1) : ite, ...] * pitch * 0.2

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, Nb, patches)

    g = turbigen.grid.Grid(
        [
            block,
        ]
    )
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

    # Set an initial guess
    L = np.ptp(block.x)
    pitch_frac = block.t / block.pitch
    length_frac = block.x / L
    block.rho = pitch_frac + 2.0 + length_frac**2.0
    block.u = cp * To1
    block.Vx = 1.0
    block.Vr = 0.0
    block.Vt = 0.0
    block.cp = cp
    block.gamma = ga
    block.Omega = 0.0
    block.mu = mu

    P1 = Po1 * 0.8
    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_P_T(Po1, To1)
    g.apply_inlet(So1, Alpha, Beta)
    g.calculate_wall_distance()
    g.apply_outlet(P1)

    x = block.x[:, 0, (0,)]
    t = block.t[:, 0, (0, -1)]
    rho_pre = block.rho[:, :, (0, -1)]

    embsolve.Emb(
        n_step=1,
        n_step_avg=1,
        n_step_log=1,
        CFL=1e-9,
    ).run(g)

    rho_post = block.rho[:, :, (0, -1)]

    rho_target = (t / block.pitch + 2.0 + (x / L) ** 2)[:, None, :]
    rho_mean = np.mean(rho_target, axis=2, keepdims=True)

    assert np.allclose(rho_pre, rho_target)

    # Points on blade should not change
    # EXCLUDE the LE and TE from this
    assert np.allclose(
        rho_post[(ile + 1) : ite, :, :], rho_target[(ile + 1) : ite, :, :]
    )

    # Periodic point up and down should be averaged
    # INCLUDING the LE and TE
    assert np.allclose(rho_post[: (ile + 1), :, :], rho_mean[: (ile + 1), :, :])
    assert np.allclose(rho_post[ite:, :, :], rho_mean[ite:, :, :])


if __name__ == "__main__":
    test_periodic()
