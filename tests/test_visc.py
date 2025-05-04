"""Viscous test cases."""

import turbigen.solvers.emb as embsolve
import turbigen.compflow_native as cf
import turbigen.grid
import turbigen.clusterfunc
import turbigen.util
import numpy as np
import matplotlib.pyplot as plt

# With quasi-2D periodic grids, and no halo cells,
# a little bit of 2nd order smoothing is needed to
# prevent instability
settings = {
    "n_step": 16000,
    "n_step_avg": 1000,
    "n_step_log": 100,
    "nstep_damp": 500,
    "xllim_pitch": 0.0,
    "smooth4": 0.001,
    "smooth2_adapt": 0.5,
    "smooth2_const": 0.002,
}


def make_plate(mu, Tu0=300.0):
    """Generate the grid."""

    AR_merid = 1.0
    AR_pitch = 1.0
    htr = 0.9
    Alpha1 = 0.0
    Ma1 = 0.3
    skew = 0.0
    L_h = 4.0

    # Geometry
    h = 0.5
    L = L_h * h
    rm = 0.5 * h * (1.0 + htr) / (1.0 - htr)
    rh = rm - 0.5 * h
    rt = rm + 0.5 * h

    # Boundary conditions
    ga = 1.4
    cp = 1005.0
    Beta = 0.0
    Po1 = 1e5
    To1 = 300.0

    # Set inlet Ma to get inlet static state
    rgas = cp * (ga - 1.0) / ga
    V = cf.V_cpTo_from_Ma(Ma1, ga) * np.sqrt(cp * To1)
    P1 = Po1 / cf.Po_P_from_Ma(Ma1, ga)
    T1 = To1 / cf.To_T_from_Ma(Ma1, ga)
    rho1 = P1 / rgas / T1

    # Radial grid points
    ER = 1.05
    d1 = 0.005 * 0.2
    dmax = 0.1 * 0.2
    # rv = turbigen.clusterfunc.double.free(d1, d2, dmax, ER, rh, rt)
    rv = turbigen.clusterfunc.single.free(d1, dmax, ER, rh, rt)
    dmax1 = np.diff(rv).max()
    nj = len(rv)

    # Circumferential grid points
    # Use pitchwise aspect ratio to find cell spacing, pitch and Nb
    nk = 9
    pitch = dmax1 * (nk - 1) * AR_pitch
    Nb = int(2.0 * np.pi * rm / pitch)
    dt = 2.0 * np.pi / float(Nb)
    tv = np.linspace(0.0, dt, nk)

    # Axial grid points
    di = dmax * AR_merid
    clule = 0.1
    di1 = di * clule
    ERi = 1.2
    xup = np.flip(turbigen.clusterfunc.single.free(di1, di, ERi, 0.0, -h))
    ile = len(xup)
    xdn = turbigen.clusterfunc.single.free(di1, di, ERi, 0.0, L)
    xv = np.concatenate((xup, xdn[1:]))
    # ni = int(np.round((len(xv)-1)/8)*8 +1)
    ni = int(np.round((len(xv) - 1) / 8) * 8 + 1)
    xv = xv[:ni]

    # ni = int((L+h)/di)
    # xv = np.linspace(-h, L, ni)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

    # Calculate Blasius displacement thickness
    xv0 = xv[xv > 0.0]
    delstar = 1.72 * np.sqrt(mu * xv0 / V / rho1)

    # # # Stretch vertically to account for displacement thickness
    # xn = xv/xv[-1]
    # stretch = np.expand_dims(np.interp(xv, xv0, delstar/h+1.), (1,2))
    # xrt[1] = (xrt[1] - rh)*stretch + rh

    # fig, ax = plt.subplots()
    # ax.plot(xrt[0,:,:,0], xrt[1,:,:,0],'k-',lw=0.5)
    # ax.plot(xrt[0,:,:,0].T, xrt[1,:,:,0].T,'k-',lw=0.5)
    # ax.axis('equal')
    # plt.show()
    # quit()

    # Split into blocks
    blocks = []
    nblock = 1
    istb = [ni // nblock * iblock for iblock in range(nblock)]
    ienb = [ni // nblock * (iblock + 1) + 1 for iblock in range(nblock)]
    ienb[-1] = ni

    if ile > ienb[0]:
        raise Exception("Blocks too small")

    for iblock in range(nblock):
        # Special case for only one block
        if nblock == 1:
            patches = [
                turbigen.grid.InletPatch(i=0),
                turbigen.grid.OutletPatch(i=-1),
                turbigen.grid.InviscidPatch(i=(0, ile), j=0),
            ]

        # First block has an inlet
        elif iblock == 0:
            patches = [
                turbigen.grid.InletPatch(i=0),
                turbigen.grid.PeriodicPatch(i=-1),
                turbigen.grid.InviscidPatch(i=(0, ile), j=0),
            ]

        # Last block has outlet
        elif iblock == (nblock - 1):
            patches = [
                turbigen.grid.PeriodicPatch(i=0),
                turbigen.grid.OutletPatch(i=-1),
            ]

        # Middle blocks are both periodic
        else:
            patches = [
                turbigen.grid.PeriodicPatch(i=0),
                turbigen.grid.PeriodicPatch(i=-1),
            ]

        patches.extend(
            [
                turbigen.grid.PeriodicPatch(k=0),
                turbigen.grid.PeriodicPatch(k=-1),
                turbigen.grid.InviscidPatch(j=-1),
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
    So1.set_P_T(Po1, To1)
    So1.set_Tu0(Tu0)
    g.apply_inlet(So1, Alpha1, Beta)
    g.calculate_wall_distance()
    g.apply_outlet(P1)

    # Initial guess
    for b in g:
        b.Vx = V
        b.Vr = 0.0
        b.Vt = 0.0
        b.cp = cp
        b.gamma = ga
        b.mu = mu
        b.Omega = 0.0
        b.set_P_T(P1, T1)
        b.set_Tu0(Tu0)

    g.match_patches()

    return g


def make_pipe():
    """Generate the grid."""

    L_h = 8.0
    AR_merid = 4.0
    AR_pitch = 0.5
    htr = 0.95

    # Geometry
    h = 0.1
    L = h * L_h
    rm = 0.5 * h * (1.0 + htr) / (1.0 - htr)
    rh = rm - 0.5 * h
    rt = rm + 0.5 * h

    # Boundary conditions
    Alpha1 = 0.0
    Ma1 = 0.2
    ga = 1.4
    cp = 1005.0
    mu = 5e-2
    Beta = 0.0
    Po1 = 1e5
    To1 = 320.0

    # Set inlet Ma to get inlet static state
    rgas = cp * (ga - 1.0) / ga
    V = cf.V_cpTo_from_Ma(Ma1, ga) * np.sqrt(cp * To1)
    P1 = Po1 / cf.Po_P_from_Ma(Ma1, ga)
    T1 = To1 / cf.To_T_from_Ma(Ma1, ga)

    dw = 0.002
    dmax = 0.04
    ER = 1.1
    cluv = turbigen.clusterfunc.symmetric.free(dw, dmax, ER)
    ddmax = np.diff(cluv).max() * h

    # Numbers of grid points
    nj = len(cluv)
    nk = 9
    ni = int(L / ddmax / AR_merid)
    print(ni, nj, nk)

    rv = rh + cluv * h

    # Use pitchwise aspect ratio to find cell spacing, pitch and Nb
    pitch = dmax * h * (nk - 1) * AR_pitch
    Nb = int(2.0 * np.pi * rm / pitch)
    dt = 2.0 * np.pi / float(Nb)

    # Make the coordinates
    # tv = dt * cluv
    tv = np.linspace(0.0, dt, nk)
    xv = np.linspace(0.0, L, ni)
    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

    # # Open up to make dpdx 0
    # fac_A = np.linspace(1.,1.5,ni)
    # xrt[1] = (xrt[1] - rm)*np.expand_dims(fac_A, (1,2)) + rm

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
            ]

        # Last block has outlet
        elif iblock == (nblock - 1):
            patches = [
                turbigen.grid.PeriodicPatch(i=0),
                turbigen.grid.OutletPatch(i=-1),
            ]

        # Middle blocks are both periodic
        else:
            patches = [
                turbigen.grid.PeriodicPatch(i=0),
                turbigen.grid.PeriodicPatch(i=-1),
            ]

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

        print(f"{block}")
        print(f"xmin = {block.x.min()}")
        print(f"xmax = {block.x.max()}")
        for p in patches:
            print(p)
        print("")

        blocks.append(block)

    # Make the grid object
    g = turbigen.grid.Grid(blocks)
    g.check_coordinates()

    # Boundary conditions
    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_P_T(Po1, To1)
    g.apply_inlet(So1, Alpha1, Beta)
    g.calculate_wall_distance()
    g.apply_outlet(P1)

    # fig, ax = plt.subplots()
    # lev = np.linspace(0,h/2,11)
    # b = g[-1]
    # C = b[0,:,:]
    # ax.contourf(C.z, C.y, C.w, lev)
    # ax.axis('equal')
    # plt.show()

    # fig, ax = plt.subplots()
    # lev = np.linspace(0,h/2,11)
    # b = g[-1]
    # C = b[0,:,:]
    # ax.contourf(C.z, C.y, C.w, lev)
    # ax.axis('equal')
    # plt.show()

    # fig, ax = plt.subplots()
    # lev = np.linspace(0,h/2,11)
    # for b in g:
    #     C = b[:,:,-1]
    #     ax.contourf(C.x, C.r, C.w, lev)
    # ax.axis('equal')
    # plt.show()
    # quit()

    # Initial guess
    for b in g:
        b.Vx = V
        b.Vr = 0.0
        b.Vt = V * np.tan(np.radians(Alpha1))
        b.cp = cp
        b.gamma = ga
        b.mu = mu
        b.Omega = 0.0
        b.set_P_T(P1, T1)

    # # Evaulate 1D analytical
    # Q1 = cf.mcpTo_APo_from_Ma(Ma1,ga)
    # Ma = cf.Ma_from_mcpTo_APo(Q1/fac_A, ga)
    # P = Po1/cf.Po_P_from_Ma(Ma, ga)
    # T = To1/cf.To_T_from_Ma(Ma, ga)
    # V = np.sqrt(cp*To1)*cf.V_cpTo_from_Ma(Ma, ga)

    F = g[0].empty(shape=(ni,))
    F.Vx = V
    F.Vr = 0.0
    F.Vt = 0.0
    F.set_P_T(P1, T1)
    F.x = xv
    F.r = rm
    F.t = 0.0

    g.match_patches()

    # fig, ax = plt.subplots()
    # b = g[0]
    # C = b[0, :, :]
    # ax.plot(C.z, C.y, 'k-')
    # ax.plot(C.z.T, C.y.T, 'k-')
    # ax.axis('equal')
    # plt.show()

    return g, F


def test_plate_turb():
    """Run boundary layer with yplus ~ 30."""

    # g = make_plate(mu=0.5e-4, Tu0=0.)
    # set_ts3 = {'ilos': 1, 'xllim': 0., 'xllim_free': 0., 'workdir': 'runs/plate_turb/', 'nstep': 10000, 'nstep_avg': 1000, 'dampin': 1e9, 'sfin': 0., 'facsecin': 0.0005, 'fmgrid': 0.}
    # import turbigen.solvers.ts3
    # turbigen.solvers.ts3.run(g, set_ts3, None)

    g = make_plate(mu=0.5e-4)

    solver = embsolve.Emb(**settings)
    solver.xllim_pitch = 1e5
    solver.run(g)

    # Extract skin friction
    b = g[0]
    Cj2 = b[1:, 2, 0]
    Cj1 = b[1:, 1, 0]
    Cj0 = b[1:, 0, 0]
    Cjm = b[1:, b.nj // 2, 0]
    Vinf = Cjm.Vx
    rhoinf = Cjm.rho
    dVdy = (Cj2.Vx - Cj1.Vx) / (Cj2.r - Cj1.r)
    mu = Cj0.mu
    tauw = dVdy * mu

    cf = tauw / (0.5 * rhoinf * Vinf * Vinf)
    x = Cjm.x

    # fig, ax = plt.subplots()
    # ax.plot(b.Vx[-10, :, 0], b.r[-10, :, 0], "-x")

    # xcf_ts3 = np.savetxt('tests/xcf_yp5_turb.csv', np.stack((x,cf)))

    # Setup figure
    # fig, ax = plt.subplots()
    # ax.set_ylim((0.,0.006))

    # # Plot skin friction
    # xcf_ts3 = np.loadtxt("tests/xcf_yp5_turb.csv")
    # ax.plot(x, cf, "-", label="embsolve")
    # ax.plot(*xcf_ts3, "-", label="TS3")

    # # Plot correlation
    # x0 = 0.0
    xx = x[x > 0.0]
    # # ax.plot(xx, cf_corr,'k--', label='Blasius')

    # ax.set_ylabel("Skin Friction Coefficient, $C_f$")
    # ax.set_xlabel("Streamwise Distance, $x/L$")
    # ax.legend()
    # plt.tight_layout(pad=0.1)
    # # plt.savefig('tests/blasius_cf.pdf')

    # # Get error
    # err = (cf[x > 0.0] / xcf_ts3[1][x > 0.0] - 1.0)[xx > 0.25]
    # print(np.abs(err).mean() )
    # # assert np.abs(err).mean() < 0.05

    # print("TS cf rel error")
    # print("mean", np.abs(err).mean())
    # print("max", err.max())
    # print("min", err.min())

    # Momentum flux
    rho = b.rho.mean(axis=2)
    Vx = b.Vx.mean(axis=2)
    r = b.r.mean(axis=2)
    P = b.P.mean(axis=2)
    x = b.x[:, 0, 0]
    dr = np.diff(r, axis=-1)
    rm = 0.5 * (r[:, 1:] + r[:, :-1])
    rho = 0.5 * (rho[:, 1:] + rho[:, :-1])
    Vx = 0.5 * (Vx[:, 1:] + Vx[:, :-1])
    P = 0.5 * (P[:, 1:] + P[:, :-1])
    mom = np.sum((rho * Vx * Vx + P) * 2 * np.pi * rm * dr, axis=-1)

    # Force on plate = mom in - mom out
    force = mom[0] - mom  # [N]
    force_width = force / (2 * np.pi * b.r.min())

    # Drag coefficient
    dyn_head = 0.5 * rhoinf.mean() * (Vinf.mean() ** 2)
    Cd = force_width[x > 0.0] / dyn_head / x[x > 0.0]

    # Cd = (mom-mom[0])[1:][x>0.]/(xx*0.5*rhoinf.mean()*Vinf.mean()**2)
    # np.savetxt('tests/xcd_yp5_turb.csv', np.stack((xx, Cd)))
    Cdts3 = np.loadtxt("tests/xcd_yp5_turb.csv")

    fig, ax = plt.subplots()

    err = (Cd / Cdts3[1] - 1.0)[xx > 0.25]
    assert np.abs(err).mean() < 0.05

    print("TS rel drag error")
    print("mean", np.abs(err).mean())
    print("max", err.max())
    print("min", err.min())

    # Cdb /= Cdb[-1]/Cd[-1]
    xxn = xx / xx[-1]
    xxts3 = Cdts3[0] / xx[-1]
    ax.plot(xxn, Cd, label="embsolve")
    ax.plot(xxts3, Cdts3[1], label="TS3")
    # ax.plot(xxn, Cdb, 'k--', label='Blasius')
    # ax.set_ylim([0.,0.020])
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Drag Coefficient, $C_D$")
    ax.set_xlabel("Streamwise Distance, $x/L$")
    ax.legend()
    plt.tight_layout(pad=0.1)
    # plt.show()
    plt.savefig("tests/turb_cd.pdf")
    plt.close()


def test_plate_lam():
    """Run boundary layer with yplus ~ 5."""

    # g = make_plate(mu=8e-4, Tu0=0.)
    # set_ts3 = {'ilos': 1, 'xllim': 0., 'xllim_free': 0., 'workdir': 'runs/plate_yp5/', 'nstep': 100000, 'nstep_avg': 1000, 'dampin': 1e9, 'sfin': 0., 'facsecin': 0.0005, 'fmgrid': 0.}
    # import turbigen.solvers.ts3
    # turbigen.solvers.ts3.run(g, set_ts3, None)

    g = make_plate(mu=8e-4)

    embsolve.Emb(**settings).run(g)

    # Extract skin friction
    b = g[0]
    Cj2 = b[1:, 2, 0]
    Cj1 = b[1:, 1, 0]
    Cj0 = b[1:, 0, 0]
    Cjm = b[1:, b.nj // 2, 0]
    Vinf = Cjm.Vx
    rhoinf = Cjm.rho
    dVdy = (Cj2.Vx - Cj1.Vx) / (Cj2.r - Cj1.r)
    mu = Cj0.mu
    tauw = dVdy * mu

    cf = tauw / (0.5 * rhoinf * Vinf * Vinf)
    x = Cjm.x
    Lref = x[-1] * 0.8

    # xcf_ts3 = np.savetxt('tests/xcf_yp5_ts3.csv', np.stack((x,cf)))

    # Setup figure
    fig, ax = plt.subplots()
    ax.set_ylim((0.0, 6))

    # Plot skin friction
    xcf_ts3 = np.loadtxt("tests/xcf_yp5_ts3.csv")
    xcf_ts3[1] /= 1e-3
    xcf_ts3[0] /= Lref
    ax.plot(x / Lref, cf / 1e-3, "-", label="EMB")
    ax.plot(*xcf_ts3, "-", label="TS3")

    # Plot correlation
    x0 = 0.0
    xx = x[x > 0.0]
    Rex = rhoinf[x > 0.0] * Vinf[x > 0.0] * (xx - x0) / mu
    cf_corr = 0.644 / np.sqrt(Rex)
    ax.plot(xx / Lref, cf_corr * 1e3, "k--", label="Blasius")
    ax.set_xlim([0.0, 1.0])

    ax.set_ylabel(r"Skin Friction Coefficient, $C_f\,/10^{-3}$")
    ax.set_xlabel("Streamwise Distance, $x/L$")
    ax.set_yticks(np.arange(0, 8, 2))
    ax.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig("plots/blasius_cf.pdf")
    plt.close()

    # Get error
    err = (cf[x > 0.0] - cf_corr)[xx > 0.25]
    assert np.abs(err).mean() < 1e-4

    print("Blasius cf error")
    print("mean", np.abs(err).mean())
    print("max", err.max())
    print("min", err.min())


def test_poiseuille():
    g, F = make_pipe()

    embsolve.Emb(**settings).run(g)

    print("Processing last block...")
    b = g[-1]
    iplot = int(b.ni * 0.9)
    print(f"iplot={iplot}")
    C = b[:, b.nj // 2, b.nk // 2]
    C2 = b[-1, :, :]
    h = np.ptp(C2.r)
    print(f"span={h}")
    dPdx = np.gradient(C.P, C.x)
    print(f"dPdx={dPdx.min()}, {dPdx.mean()}, {dPdx.max()}")
    mu = F.mu
    print(f"mu={mu}")
    K = dPdx[iplot] / 2.0 / mu * h * h
    print(f"K={K}")

    fig, ax = plt.subplots(layout="constrained")
    b = g[-1]
    C = b[-1, :, b.nk // 2]
    r = C.r
    h = np.ptp(r)
    rn = (r - r.min()) / h
    denom = -dPdx[iplot] * h * h / 2.0 / mu
    Vn = C.Vx / denom
    Vn_analytical = (1.0 - rn) * rn
    ax.plot(Vn, rn, "-", label="EMB")
    ax.plot(Vn_analytical, rn, "--", label="Poiseuille")
    ax.set_xlabel(
        r"Velocity, $\left. V \middle/ \left( \frac{-h^2}{2\mu}\frac{\mathrm{d} p}{\mathrm{d} x}\right)\right.$"
    )
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 0.3])
    ax.set_ylabel(r"Channel Height, $y/h$")
    ax.set_xticks(np.arange(0, 0.4, 0.1))
    # ax.set_yticks([0, 0.5, 1.0])
    ax.legend()
    plt.savefig("plots/poiseuille.pdf")
    plt.close()
    # plt.show()

    Cm, A, _ = C2.mix_out()
    mdot = Cm.rho * Cm.Vm * A
    print(f"mdot={mdot}")
    rho = Cm.rho
    print(f"rho={rho}")
    w = 2.0 * np.pi * 0.5 * (C2.r.min() + C2.r.max())
    print(f"width={w}")
    mdot_analytical = -rho * w * h * K / 6.0

    err = mdot_analytical / mdot - 1.0
    print(
        f"mdot acutal={mdot:.2f}, theory={mdot_analytical:.2f}, error={err * 100:.2f}%"
    )
    assert np.abs(err) < 0.01


if __name__ == "__main__":
    # test_plate_turb()
    # test_plate_lam()
    test_poiseuille()
