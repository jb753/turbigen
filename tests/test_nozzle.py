"""Run a quasi-1D nozzle in the native solver."""

import turbigen.solvers.emb as embsolve
import turbigen.compflow_native as cf
import turbigen.grid
import turbigen.util
import numpy as np
from copy import copy
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
import pytest


def make_nozzle(
    xnAR,
    L_h=4.0,
    AR_pitch=1.0,
    skew=0.0,
    htr=0.99,
    dirn="r",
    xnRR=None,
    Alpha=0.0,
    tper=False,
    Ma1=0.3,
    rpm=0.0,
    Tu0=300.0,
    To1=300.0,
    dP=0.0,
):
    """Generate the grid."""

    # Geometry
    h = 0.1
    L = h * L_h
    rm = 0.5 * h * (1.0 + htr) / (1.0 - htr)
    rh = rm - 0.5 * h
    rt = rm + 0.5 * h

    # Boundary conditions
    ga = 1.4
    cp = 1005.0
    mu = 1.8e-5
    Beta = 0.0
    Po1 = 1e5
    To1 = To1

    # Rotating reference frame
    Omega = rpm / 60.0 * np.pi * 2.0
    U = Omega * rm

    # Set inlet Ma to get inlet static state
    V = cf.V_cpTo_from_Ma(Ma1, ga) * np.sqrt(cp * To1)
    P1 = Po1 / cf.Po_P_from_Ma(Ma1, ga)
    T1 = To1 / cf.To_T_from_Ma(Ma1, ga)

    # Relative flow angle
    Vt = V * np.sin(np.radians(Alpha))
    Vt_rel = Vt - U
    Alpha_rel = np.degrees(np.arctan2(Vt_rel, V))

    # Numbers of grid points
    nj = 17
    nk = 17
    ni = int(nj * L_h - 3)

    # Use pitchwise aspect ratio to find cell spacing, pitch and Nb
    pitch = h / (nj - 1) * (nk - 1) * AR_pitch
    Nb = int(2.0 * np.pi * rm / pitch)
    dt = 2.0 * np.pi / float(Nb)

    # Make the coordinates
    # tv = np.linspace(-dt / 2., dt / 2., nk)
    tv = np.linspace(0.0, dt, nk)
    xv = np.linspace(0.0, L, ni)
    rv = np.linspace(rh, rt, nj)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

    # Interpolate area at the x-coordinates
    fac_A = pchip_interpolate(*xnAR, xv / L)

    # Add on radius change
    if xnRR is not None:
        fac_R = pchip_interpolate(*xnRR, xv / L)
        xrt[1] *= np.expand_dims(fac_R, (1, 2))
        fac_A /= fac_R

    # Sentinel of None means skew along flow direction
    if skew is None:
        skew_max = 30.0
        skew = np.clip(-Alpha_rel, -skew_max, skew_max)

    # Apply skew
    xrt[2] += xrt[0] * np.tan(np.radians(skew)) / xrt[1]

    # Squeeze the nozzle
    if dirn == "r":
        xrt[1] = (xrt[1] - rm) * np.expand_dims(fac_A, (1, 2)) + rm
    elif dirn == "t":
        xrt[2] *= np.expand_dims(fac_A, (1, 2))

    # Split into blocks
    blocks = []
    nblock = 1
    istb = [ni // nblock * iblock for iblock in range(nblock)]
    ienb = [ni // nblock * (iblock + 1) + 1 for iblock in range(nblock)]
    ienb[-1] = ni

    for iblock in range(nblock):
        # Special case for only one block
        if nblock == 1:
            patches = [
                turbigen.grid.InletPatch(i=0),
                turbigen.grid.OutletPatch(i=-1),
            ]

        # First block has an inlet
        elif iblock == 0:
            patches = [
                turbigen.grid.InletPatch(i=0),
                turbigen.grid.PeriodicPatch(i=-1),
                # turbigen.grid.PeriodicPatch(k=0),
                # turbigen.grid.PeriodicPatch(k=-1),
            ]

        # Last block has outlet
        elif iblock == (nblock - 1):
            patches = [
                turbigen.grid.PeriodicPatch(i=0),
                turbigen.grid.OutletPatch(i=-1),
                # turbigen.grid.PeriodicPatch(k=0),
                # turbigen.grid.PeriodicPatch(k=-1),
            ]

        # Middle blocks are both periodic
        else:
            patches = [
                turbigen.grid.PeriodicPatch(i=0),
                turbigen.grid.PeriodicPatch(i=-1),
                # turbigen.grid.PeriodicPatch(k=0),
                # turbigen.grid.PeriodicPatch(k=-1),
            ]

        if tper:
            patches.extend(
                [
                    turbigen.grid.PeriodicPatch(k=0),
                    turbigen.grid.PeriodicPatch(k=-1),
                ]
            )

        block = turbigen.grid.PerfectBlock.from_coordinates(
            xrt[:, istb[iblock] : ienb[iblock], :, :], Nb, patches
        )
        block.label = f"b{iblock}"

        blocks.append(block)

    # Make the grid object
    g = turbigen.grid.Grid(blocks)
    g.check_coordinates()

    # Boundary conditions
    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_Tu0(Tu0)
    So1.set_P_T(Po1, To1)
    g.apply_inlet(So1, Alpha, Beta)
    g.calculate_wall_distance()
    g.apply_outlet(P1)

    # Fluid props
    for b in g:
        b.cp = cp
        b.gamma = ga
        b.mu = mu
        b.Omega = Omega
        b.set_Tu0(Tu0)

    # Evaulate 1D analytical
    Q1 = cf.mcpTo_APo_from_Ma(Ma1, ga)
    Ma = cf.Ma_from_mcpTo_APo(Q1 / fac_A, ga)
    P = Po1 / cf.Po_P_from_Ma(Ma, ga)
    T = To1 / cf.To_T_from_Ma(Ma, ga)
    V = np.sqrt(cp * To1) * cf.V_cpTo_from_Ma(Ma, ga)

    F = g[0].empty(shape=(ni,))
    F.Omega = Omega
    F.Vx = V
    F.Vr = 0.0
    F.Vt = 0.0
    F.set_P_T(P, T)
    F.x = xv
    F.r = rm
    F.t = 0.0
    F.set_Tu0(Tu0)

    Ve = np.expand_dims(V, (1, 2))
    Te = np.expand_dims(T, (1, 2))
    Pe = np.expand_dims(P, (1, 2))

    # Initial guess
    for ib, b in enumerate(g):
        b.Vx = Ve[istb[ib] : ienb[ib]]
        b.Vr = 0.0
        b.Vt = Ve[istb[ib] : ienb[ib]] * np.tan(np.radians(Alpha))
        b.set_P_T(Pe[istb[ib] : ienb[ib]] + dP, Te[istb[ib] : ienb[ib]])

    g.match_patches()

    return g, F


settings = {
    "n_step": 5000,
    "n_step_avg": 500,
    "n_step_log": 100,
    "i_loss": 0,
    "nstep_damp": 100,
}


def plot_nozzle(g, F):
    """Make debugging plots."""

    L = np.ptp(F.x)

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f"C{ib}"
        C = b[:, b.nj // 2, b.nk // 2]
        ax.plot(C.x / L, C.Ma, "-.", color=cs)
    ax.plot(F.x / L, F.Ma, "k-")
    ax.set_title("Ma")
    ax.set_ylim((0, 1))

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f"C{ib}"
        C = b[:, b.nj // 2, b.nk // 2]
        ax.plot(C.x / L, C.P / F.P[-1], color=cs)
    ax.plot(F.x / L, F.P / F.P[-1], "k-")
    ax.set_title("P/nom P exit")
    # ax.set_ylim(bottom=0.)

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f"C{ib}"
        C = b[:, b.nj // 2, b.nk // 2]
        ax.plot(C.x / L, C.rVt, color=cs)
    ax.plot(F.x / L, F.rVt, "k-")
    ax.set_title("rVt")

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f"C{ib}"
        C = b[:, b.nj // 2, b.nk // 2]
        ax.plot(C.x / L, C.Alpha, color=cs)
    ax.set_title("Alpha")

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f"C{ib}"
        C = b[:, b.nj // 2, b.nk // 2]
        ax.plot(C.x / L, C.Alpha_rel, color=cs)
    ax.set_title("Alpha_rel")

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f"C{ib}"
        C = b[:, b.nj // 2, b.nk // 2]
        ax.plot(C.x / L, C.Beta, color=cs)
    ax.set_title("Beta")

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f"C{ib}"
        C = b[:, b.nj // 2, b.nk // 2]
        ax.plot(C.x / L, (C.ho - F.ho[0]) / F[0].V ** 2, color=cs)
    ax.set_title("(Ho - ho_nom)/(V_nom**2)")

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f"C{ib}"
        C = b[:, b.nj // 2, b.nk // 2]
        ax.plot(C.x / L, (C.s - F.s[0]) / F.rgas, color=cs)
    ax.set_title("(s - s_nom)/Rgas")

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        C = b[:, :, b.nk // 2]
        ax.contourf(C.x / L, C.r / L, C.s)
    ax.set_title("ent")
    # ax.set_ylim((0,1))

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        C = b[:, :, b.nk // 2]
        ax.contourf(C.x / L, C.r / L, C.ho)
    ax.set_title("ho")
    # ax.set_ylim((0,1))

    fig, ax = plt.subplots()
    b = g[-1]
    C = b[-1, :, :]
    ax.contourf(C.y, C.z, C.P)
    print(C.Vx.min(), C.Vx.max())
    ax.set_title("Pexit")
    # ax.set_ylim((0,1))

    # plt.show()


def post_nozzle(g, F):
    """Extract errors."""

    Ma = np.concatenate([b.Ma[:-1, b.nj // 2, b.nk // 2] for b in g])
    err_Ma = Ma - F.Ma[:-1]

    print(
        f"Mach error: mean={err_Ma.mean():.2e}, min={err_Ma.min():.2e},"
        f" max={err_Ma.max():.2e}"
    )

    T2 = F.T[-1]
    ho1 = F.ho[0]
    s1 = F.s[0]
    V1 = F.Vx[0]
    s = np.concatenate([b.s[:-1, b.nj // 2, b.nk // 2] for b in g])
    Ys = (s - s1) * T2 / (0.5 * V1**2)

    print(
        f"Entropy conservation error Ys: mean={Ys.mean():.3e}, min={Ys.min():.3e},"
        f" max={Ys.max():.3e}"
    )

    ho = np.concatenate([b.ho[:-1, b.nj // 2, b.nk // 2] for b in g])
    Cho = (ho - ho1) / (0.5 * V1**2)

    print(
        f"Energy conservation error Cho: mean={Cho.mean():.3e}, min={Cho.min():.3e},"
        f" max={Cho.max():.3e}"
    )

    return err_Ma, Ys, Cho


@pytest.mark.parametrize("dirn", ("r", "t"))
def test_condi(dirn):
    """Run subsonic con-di nozzles."""

    xA = np.array([[0.0, 0.02, 0.3, 0.98, 1.0], [1.0, 1.0, 0.6, 1.0, 1.0]])
    g, F = make_nozzle(xA, dirn=dirn)

    embsolve.Emb(**settings).run(g)

    # plot_nozzle(g, F)
    # plt.show()

    err_Ma, Ys, Cho = post_nozzle(g, F)

    rtol_Ma = 5e-2
    assert (np.abs(err_Ma) < rtol_Ma).all()
    rtol_sh = 5e-3
    assert (np.abs(Ys) < rtol_sh).all()
    assert (np.abs(Cho) < rtol_sh).all()


@pytest.mark.parametrize("Tu0", (0.0, 150.0, 300.0))
def test_Tu0(Tu0):
    """Check that the internal energy datum makes no difference."""

    xA = np.array([[0.0, 0.01, 0.99, 1.0], [1.0, 1.0, 1.0, 1.0]])
    g, F = make_nozzle(xA, Tu0=Tu0)

    embsolve.Emb(**settings).run(g)

    # plot_nozzle(g,F)
    # plt.show()

    err_Ma, Ys, Cho = post_nozzle(g, F)

    rtol = 5e-4

    assert (np.abs(err_Ma) < rtol).all()
    assert (np.abs(Ys) < rtol).all()
    assert (np.abs(Cho) < rtol).all()


@pytest.mark.parametrize("Alpha", (-30.0, 0.0, 30.0))
def test_uniform(Alpha):
    """Run the most basic parallel annulus, grid aligned with flow."""

    xA = np.array([[0.0, 0.01, 0.99, 1.0], [1.0, 1.0, 1.0, 1.0]])

    g, F = make_nozzle(xA, Alpha=Alpha, skew=Alpha)

    embsolve.Emb(**settings).run(g)

    err_Ma, Ys, Cho = post_nozzle(g, F)

    rtol = 2.5e-4

    assert (np.abs(err_Ma) < rtol).all()
    assert (np.abs(Ys) < rtol).all()
    assert (np.abs(Cho) < rtol).all()


@pytest.mark.parametrize("Ma", (0.6, 0.9))
def test_Ma(Ma):
    """Run uniform flow at different Mach."""

    xA = np.array([[0.0, 0.01, 0.99, 1.0], [1.0, 1.0, 1.0, 1.0]])

    g, F = make_nozzle(xA, Ma1=Ma)

    embsolve.Emb(**settings).run(g)

    err_Ma, Ys, Cho = post_nozzle(g, F)

    rtol = 3.0e-5
    assert (np.abs(err_Ma) < rtol).all()
    assert (np.abs(Ys) < rtol).all()
    assert (np.abs(Cho) < rtol).all()


@pytest.mark.parametrize("Alpha", (-30.0, 0.0, 30.0))
def test_skew(Alpha):
    """Run an axial flow with skewed grid."""

    xA = np.array([[0.0, 0.01, 0.99, 1.0], [1.0, 1.0, 1.0, 1.0]])

    g, F = make_nozzle(xA, skew=Alpha, tper=True)

    embsolve.Emb(**settings).run(g)

    err_Ma, Ys, Cho = post_nozzle(g, F)

    rtol = 2e-4
    assert (np.abs(err_Ma) < rtol).all()
    assert (np.abs(Ys) < rtol).all()
    assert (np.abs(Cho) < rtol).all()


@pytest.mark.parametrize("Alpha", (-30.0, 0.0, 30.0))
def test_radius(Alpha):
    """Constant area with radius change."""

    xA = np.array([[0.0, 0.01, 0.99, 1.0], [1.0, 1.0, 1.0, 1.0]])
    xR = np.array([[0.0, 0.02, 0.98, 1.0], [1.0, 1.0, 0.9, 0.9]])

    g, F = make_nozzle(xA, xnRR=xR, htr=0.9, Alpha=Alpha, skew=Alpha, tper=True)

    embsolve.Emb(**settings).run(g)

    _, Ys, Cho = post_nozzle(g, F)

    if Alpha:
        rVt = np.concatenate([b.rVt[:-1, b.nj // 2, b.nk // 2] for b in g])
        err_rVt = rVt / rVt[0] - 1.0
        tol_rVt = 1e-2
        print(
            f"Angular momentum conservation error drVt mean={err_rVt.mean():.2e},"
            f" min={err_rVt.min():.2e}, max={err_rVt.max():.2e}"
        )
        assert (err_rVt < tol_rVt).all()

    tol_sh = 5e-4
    assert (np.abs(Ys) < tol_sh).all()
    assert (np.abs(Cho) < tol_sh).all()


def test_patch_A_avg():
    """Check patch area averaging weights."""

    # Make an arbitrary grid
    xA = np.array([[0.0, 0.02, 0.3, 0.98, 1.0], [1.0, 1.0, 0.6, 1.0, 1.0]])
    g, F = make_nozzle(xA)
    block = g[-1]
    patch = block.outlet_patches[0]

    # Calculate area average using weight
    ind = patch.get_flat_indices(order="F")
    w = patch.get_A_avg_weights(order="F")
    rsqavg = np.sum((block.r.ravel(order="F")[ind] ** 2.0) * w) / np.sum(w)

    # Manually check the area average
    C = patch.get_cut()
    dA = turbigen.util.vecnorm(C.dAi)
    rsqavg_check = np.sum((C.r_face[0] ** 2) * dA) / np.sum(dA)
    assert np.isclose(rsqavg, rsqavg_check)


@pytest.mark.parametrize("rpm", (-100.0, 100.0))
def test_rpm(rpm):
    xA = np.array([[0.0, 0.02, 0.3, 0.98, 1.0], [1.0, 1.0, 0.6, 1.0, 1.0]])
    g, F = make_nozzle(xA, rpm=rpm, tper=True, skew=None)

    sol = embsolve.Emb(**settings)
    sol.nstep_damp = -1
    sol.run(g)

    # plot_nozzle(g,F)
    # plt.show()

    err_Ma, Ys, Cho = post_nozzle(g, F)

    rtol_Ma = 5e-2
    assert (np.abs(err_Ma) < rtol_Ma).all()
    rtol_sh = 5e-3
    assert (np.abs(Ys) < rtol_sh).all()
    assert (np.abs(Cho) < rtol_sh).all()


@pytest.mark.parametrize("To1", (200.0, 300.0, 400.0))
def test_To1(To1):
    """Run the most basic parallel annulus, grid aligned with flow."""

    xA = np.array([[0.0, 0.01, 0.99, 1.0], [1.0, 1.0, 1.0, 1.0]])

    g, F = make_nozzle(xA, To1=To1)

    embsolve.Emb(**settings).run(g)

    err_Ma, Ys, Cho = post_nozzle(g, F)

    # plot_nozzle(g, F)
    # plt.show()

    rtol = 3e-4

    assert (np.abs(err_Ma) < rtol).all()
    assert (np.abs(Ys) < rtol).all()
    assert (np.abs(Cho) < rtol).all()


@pytest.mark.parametrize("dP", (-1000.0, 1000.0))
def test_offset(dP):
    """Set an initial guess with different static pressure."""

    xA = np.array([[0.0, 0.01, 0.99, 1.0], [1.0, 1.0, 1.0, 1.0]])

    g, F = make_nozzle(xA, dP=dP)

    embsolve.Emb(**settings).run(g)

    err_Ma, Ys, Cho = post_nozzle(g, F)

    # plot_nozzle(g, F)
    # plt.show()

    rtol = 2.5e-4

    assert (np.abs(err_Ma) < rtol).all()
    assert (np.abs(Ys) < rtol).all()
    assert (np.abs(Cho) < rtol).all()


if __name__ == "__main__":
    # test_condi("t")
    test_rpm(-100.0)
    pass
