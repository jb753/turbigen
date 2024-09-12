"""Run a periodic grid with CFL=0 to check which nodes have periodicity applied."""

import turbigen.grid
import turbigen.solvers.embsolve
import numpy as np
import turbigen.compflow_native as cf

import matplotlib.pyplot as plt


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
    tv = np.linspace(-pitch/2., pitch/2., nk)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

    return xrt, Nb


def test_periodic():

    xrt, Nb = make_sector()

    ni = xrt.shape[1]

    ile = ni//3
    ite = ni//3*2

    patches = [
        # turbigen.grid.InletPatch(i=0),
        # turbigen.grid.OutletPatch(i=-1),
        turbigen.grid.PeriodicPatch(k=0, i=(0, ile)),
        turbigen.grid.PeriodicPatch(k=-1, i=(0, ile)),
        turbigen.grid.PeriodicPatch(k=0, i=(ite, -1)),
        turbigen.grid.PeriodicPatch(k=-1, i=(ite, -1)),
    ]

    pitch = 2.*np.pi/float(Nb)
    pitch_frac = xrt[2] / pitch / 2.
    xrt[2,(ile+1):ite, :, :] -= pitch_frac[(ile+1):ite, ...] * pitch * 0.2

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, Nb, patches)

    g = turbigen.grid.Grid(
        [
            block,
        ]
    )
    g.check_coordinates()
    g.match_patches()

    C = block[:,0, :]

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(C.x, C.rt, 'k-')
    # ax.plot(C.x.T, C.rt.T, 'k-')
    # plt.show()

    # Boundary conditions
    cp = 1005.
    ga = 1.4
    mu = 1.8e-5
    Po1 = 1e5
    To1 = 300.
    Alpha = 0.
    Beta = 0.

    # Set an initial guess
    L = np.ptp(block.x)
    pitch_frac = block.t/block.pitch
    length_frac = block.x/L
    block.rho = pitch_frac + 2. + length_frac **2.
    block.u = cp * To1
    block.Vx = 1.
    block.Vr = 0.
    block.Vt = 0.
    block.cp = cp
    block.gamma = ga
    block.Omega = 0.0
    block.mu = mu

    P1 = Po1*0.8
    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_P_T(Po1, To1)
    g.apply_inlet(So1, Alpha, Beta)
    g.calculate_wall_distance()
    g.apply_outlet(P1)

    x = block.x[:,0,(0,)]
    t = block.t[:,0,(0,-1)]
    rho_pre = block.rho[:,:,(0,-1)]

    settings = {
        "n_step": 1,
        "n_step_avg": 1,
        "n_step_log": 1,
        "CFL": 1e-9,
    }
    turbigen.solvers.embsolve.run(g, settings)

    rho_post = block.rho[:,:,(0,-1)]

    rho_target = (t/block.pitch + 2. + (x/L)**2)[:,None,:]
    rho_mean = np.mean(rho_target,axis=2, keepdims=True)

    assert np.allclose(rho_pre, rho_target)

    # Points on blade should not change
    # EXCLUDE the LE and TE from this
    assert np.allclose(rho_post[(ile+1):ite,:,:], rho_target[(ile+1):ite, :, :])

    # Periodic point up and down should be averaged
    # INCLUDING the LE and TE
    assert np.allclose(rho_post[:(ile+1),:,:], rho_mean[:(ile+1), :, :])
    assert np.allclose(rho_post[ite:,:,:], rho_mean[ite:, :, :])

    xs = x.squeeze()
    fig, ax = plt.subplots()
    jplot = block.nj//2
    ax.plot(xs,rho_pre[:,jplot,:])
    ax.plot(xs,rho_post[:,jplot,:])
    ax.plot(xs[ile],rho_post[ile,jplot,0],'kx')
    ax.plot(xs[ite],rho_post[ite,jplot,0],'kx')
    # ax.plot(Ck0_step.x,n Ck0_step.rho, 'C2', '-')
    # ax.plot(Cnk_step.x, Cnk_step.rho, 'C3', '-')
    # ax.plot(Cnk_step.x.squeeze(), Cnk_step.rho[ile], 'kx')
    # ax.plot(Cnk_step.x[ite], Cnk_step.rho[ite], 'rx')
    plt.show()


if __name__ == "__main__":

    test_periodic()
