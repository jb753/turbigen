"""Viscous test cases."""
import turbigen.solvers.native
import turbigen.compflow_native as cf
import turbigen.grid
import turbigen.clusterfunc
import turbigen.util
import numpy as np
from timeit import default_timer as timer
import sys
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
import pytest

settings = {
    'n_step': 10000,
    # 'n_step': 1000,
    'n_step_avg': 1000,
    'n_step_log': 100,
    'plot_conv': True,
    # 'nstep_damp': -1,
    'xllim_pitch': 100.0,
    # 'i_loss': 0,
    # "damping_factor" : 25.,
    # "nstep_damp" : -1,
    # "smoothing_factor" : 0.005
}

# Check our MPI rank
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Jump to solver slave process if not first rank
    if rank > 0:
        turbigen.solvers.native.run_slave()
        sys.exit(0)
except ImportError:
    pass

def make_plate(mu, Tu0=300.):
    """Generate the grid."""

    AR_merid=1.
    AR_pitch=1.
    htr=0.9
    Alpha1=0.
    Ma1=0.3
    skew=0.
    L_h = 4.

    # Geometry
    h = 0.2
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
    rgas = cp * (ga-1.)/ga
    V = cf.V_cpTo_from_Ma(Ma1,ga)*np.sqrt(cp*To1)
    P1 = Po1/cf.Po_P_from_Ma(Ma1,ga)
    T1 = To1/cf.To_T_from_Ma(Ma1,ga)

    # Radial grid points
    ER = 1.05
    # d1 = 0.01*h
    # dmax = 0.1*h
    d1 = 0.005*h
    dmax = 0.1*h
    # rv = turbigen.clusterfunc.double.free(d1, d2, dmax, ER, rh, rt)
    rv = turbigen.clusterfunc.single.free(d1, dmax, ER, rh, rt)
    dmax1 = np.diff(rv).max()
    nj = len(rv)

    # Circumferential grid points
    # Use pitchwise aspect ratio to find cell spacing, pitch and Nb
    nk = 5
    pitch = dmax1*(nk-1)*AR_pitch
    Nb = int(2.0 * np.pi * rm / pitch)
    dt = 2.0 * np.pi / float(Nb)
    tv = np.linspace(0., dt, nk)

    # Axial grid points
    ni = int(L/dmax/AR_merid)
    xv = np.linspace(0., L, ni)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing='ij'))

    # Stretch vertically
    xn = xv/xv[-1]
    stretch = np.expand_dims(np.linspace(1., 1.06, ni), (1,2))
    xrt[1] = (xrt[1] - rh)*stretch + rh

    # Split into blocks
    blocks = []
    nblock = 1
    istb = [ni//nblock*iblock for iblock in range(nblock)]
    ienb = [ni//nblock*(iblock+1)+1 for iblock in range(nblock)]
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
        elif iblock==(nblock-1):
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

        patches.extend([
            turbigen.grid.PeriodicPatch(k=0),
            turbigen.grid.PeriodicPatch(k=-1),
            turbigen.grid.InviscidPatch(j=-1),
            ]
        )

        block = turbigen.grid.PerfectBlock.from_coordinates(
                xrt[:,istb[iblock]:ienb[iblock],:,:], Nb, patches
        )
        block.label=f'b{iblock}'

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
        b.Vr = 0.
        b.Vt = 0.
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

    L_h = 8.
    AR_merid=4.
    AR_pitch=0.5
    htr = 0.95

    # Geometry
    h = 0.1
    L = h * L_h
    rm = 0.5 * h * (1.0 + htr) / (1.0 - htr)
    rh = rm - 0.5 * h
    rt = rm + 0.5 * h

    # Boundary conditions
    Alpha1 = 0.
    Ma1 = 0.2
    ga = 1.4
    cp = 1005.0
    mu = 5e-2
    Beta = 0.0
    Po1 = 1e5
    To1 = 300.0

    # Set inlet Ma to get inlet static state
    rgas = cp * (ga-1.)/ga
    V = cf.V_cpTo_from_Ma(Ma1,ga)*np.sqrt(cp*To1)
    P1 = Po1/cf.Po_P_from_Ma(Ma1,ga)
    T1 = To1/cf.To_T_from_Ma(Ma1,ga)
    rho1 = P1/rgas/T1

    dw = 0.002
    dmax = 0.04
    ER = 1.1
    cluv = turbigen.clusterfunc.symmetric.free(dw, dmax, ER)
    ddmax = np.diff(cluv).max()*h

    # Numbers of grid points
    nj = len(cluv)
    nk = 5
    ni = int(L/ddmax/AR_merid)
    print(ni, nj, nk)

    rv = rh + cluv*h

    # Use pitchwise aspect ratio to find cell spacing, pitch and Nb
    pitch = dmax*h*(nk-1)*AR_pitch
    Nb = int(2.0 * np.pi * rm / pitch)
    dt = 2.0 * np.pi / float(Nb)

    # Make the coordinates
    # tv = dt * cluv
    tv = np.linspace(0., dt, nk)
    xv = np.linspace(0., L, ni)
    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing='ij'))


    # # Open up to make dpdx 0
    # fac_A = np.linspace(1.,1.5,ni)
    # xrt[1] = (xrt[1] - rm)*np.expand_dims(fac_A, (1,2)) + rm


    # Split into blocks
    blocks = []
    nblock = 4
    istb = [ni//nblock*iblock for iblock in range(nblock)]
    ienb = [ni//nblock*(iblock+1)+1 for iblock in range(nblock)]
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
        elif iblock==(nblock-1):
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

        patches.extend([
            turbigen.grid.PeriodicPatch(k=0),
            turbigen.grid.PeriodicPatch(k=-1),
            ]
        )

        block = turbigen.grid.PerfectBlock.from_coordinates(
                xrt[:,istb[iblock]:ienb[iblock],:,:], Nb, patches
        )
        block.label=f'b{iblock}'

        print(f'{block}')
        print(f'xmin = {block.x.min()}')
        print(f'xmax = {block.x.max()}')
        for p in patches:
            print(p)
        print('')

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
        b.Vr = 0.
        b.Vt = V*np.tan(np.radians(Alpha1))
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
    F.Vr = 0.
    F.Vt = 0.
    F.set_P_T(P1,T1)
    F.x = xv
    F.r = rm
    F.t = 0.

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

    # g = make_plate(mu=1.8e-4)
    # conf_ts3 = {'type': 'ts3', 'workdir': 'runs/visc', 'ilos': 1, 'xllim_ref': 'span', 'nstep': 20000, 'nstep_avg': 1000}
    # g.run(conf_ts3, None)

    settings = {
        'n_step': 10000,
        # 'n_step': 1000,
        'n_step_avg': 1000,
        'n_step_log': 100,
        'plot_conv': True,
        # 'nstep_damp': -1,
        'xllim_pitch': 100.0,
        # 'i_loss': 0,
        # "damping_factor" : 25.,
        # "nstep_damp" : -1,
        # "smoothing_factor" : 0.005
    }

    # g = make_plate()
    g = make_plate(mu=1.8e-4)

    np.set_printoptions(precision=2)
    turbigen.solvers.native.run(g, settings)

    fig, ax = plt.subplots()
    C = g[-1][-2,:,0]
    ax.plot(C.Vx, C.r, '-x')
    plt.show()

    cf = []
    x = []
    for b in g:
        Cj2 = b[1:,2,0]
        Cj1 = b[1:,1,0]
        Cj0 = b[1:,0,0]
        Cjm = b[1:,b.nj//2,0]
        Vinf = Cjm.Vx
        rhoinf = Cjm.rho
        dVdy = (Cj2.Vx-Cj1.Vx)/(Cj2.r-Cj1.r)
        mu = Cj0.mu
        tauw = dVdy * mu
        cf.append(tauw/(0.5*rhoinf*Vinf*Vinf))
        x.append(Cjm.x)
    x = np.concatenate(x)
    cf = np.concatenate(cf)
    fig, ax = plt.subplots()
    b = g[0]
    C = b[:,b.nj//2, :]
    ax.plot(x, cf, '-x')
    # ax.set_ylim([0., 0.08])

    # np.savetxt('tests/xcf_ts3.csv', np.stack((x, cf)))
    # xcf_ts3 = np.loadtxt('tests/xcf_ts3.csv')
    # ax.plot(*xcf_ts3, '-o')
    # np.savetxt('tests/xcf_ts3.csv', np.stack((x, cf)))
    xcf_ts3 = np.loadtxt('tests/xcf_turb_ts3.csv')
    ax.plot(*xcf_ts3, '-o')
    ax.set_ylim([0., 0.0002])



    # x0 = 0.02
    # xx = x[x>x0]-x0
    # cflam = 0.664*(rhoinf[0]*Vinf[0]/mu *xx)**-0.5
    # ax.plot(xx, cflam, 'k--')

    plt.show()

    # plt.savefig('beans.pdf')

def test_plate_lam_yp5():
    """Run boundary layer with yplus ~ 5."""

    g = make_plate(Tu0=0., mu=8e-4)
    conf_ts3 = {'type': 'ts3', 'workdir': 'runs/plate_yp5', 'ilos': 1, 'xllim': 0., 'nstep': 20000, 'nstep_avg': 1000, 'adaptive_smoothing': 0, 'facsecin': 0.01, 'sfin': 0.002}
    g.run(conf_ts3, None)

    # g = make_plate(mu=8e-4)

    # np.set_printoptions(precision=2)
    # settings = {
    #     'n_step': 10000,
    #     'n_step_avg': 1000,
    #     'n_step_log': 100,
    #     'plot_conv': True,
    #     'xllim_pitch': 0.0,
    # }
    # turbigen.solvers.native.run(g, settings)

    cf = []
    x = []
    for b in g:
        Cj2 = b[1:,2,0]
        Cj1 = b[1:,1,0]
        Cj0 = b[1:,0,0]
        Cjm = b[1:,b.nj//2,0]
        Vinf = Cjm.Vx
        rhoinf = Cjm.rho
        dVdy = (Cj2.Vx-Cj1.Vx)/(Cj2.r-Cj1.r)
        mu = Cj0.mu
        tauw = dVdy * mu
        cf.append(tauw/(0.5*rhoinf*Vinf*Vinf))
        x.append(Cjm.x)
    x = np.concatenate(x)
    cf = np.concatenate(cf)

    np.savetxt('tests/xcf_yp5_ts3.csv', np.stack((x, cf)))

    fig, ax = plt.subplots()
    b = g[0]
    C = b[:,b.nj//2, :]
    ax.plot(x, cf, '-x')

    fig, ax = plt.subplots()
    b = g[0]
    C = b[-2,:, 0]
    ax.plot(C.Vx, C.r, '-x')

    plt.show()

def test_plate_lam():
    """Run boundary layer with yplus < 1."""

    # g = make_plate(Tu0=0., mu=1.8e-2)
    # conf_ts3 = {'type': 'ts3', 'workdir': 'runs/visc', 'ilos': 1, 'xllim': 0., 'nstep': 20000, 'nstep_avg': 1000}
    # # g.run(conf_ts3, None)

    g = make_plate(mu=1.8e-2)

    np.set_printoptions(precision=2)
    settings = {
        'n_step': 10000,
        # 'n_step': 1000,
        'n_step_avg': 1000,
        'n_step_log': 100,
        'plot_conv': True,
        # 'nstep_damp': -1,
        'xllim_pitch': 0.0,
        # 'i_loss': 0,
        # "damping_factor" : 25.,
        # "nstep_damp" : -1,
        # "smoothing_factor" : 0.005
    }

    turbigen.solvers.native.run(g, settings)

    cf = []
    x = []
    for b in g:
        Cj2 = b[1:,2,0]
        Cj1 = b[1:,1,0]
        Cj0 = b[1:,0,0]
        Cjm = b[1:,b.nj//2,0]
        Vinf = Cjm.Vx
        rhoinf = Cjm.rho
        dVdy = (Cj2.Vx-Cj1.Vx)/(Cj2.r-Cj1.r)
        mu = Cj0.mu
        tauw = dVdy * mu
        cf.append(tauw/(0.5*rhoinf*Vinf*Vinf))
        x.append(Cjm.x)
    x = np.concatenate(x)
    cf = np.concatenate(cf)
    fig, ax = plt.subplots()
    b = g[0]
    C = b[:,b.nj//2, :]
    ax.plot(x, cf, '-x')
    # ax.set_ylim([0., 0.08])

    # np.savetxt('tests/xcf_ts3.csv', np.stack((x, cf)))
    xcf_ts3 = np.loadtxt('tests/xcf_turb_ts3.csv')
    ax.plot(*xcf_ts3, '-o')

    plt.show()


def test_poiseuille():

    g, F = make_pipe()


    # fig, ax = plt.subplots()
    # b = g[0]
    # C = b[:, :, b.nk//2]
    # ax.plot(C.x, C.r, 'k-',lw=0.2)
    # ax.plot(C.x.T, C.r.T, 'k-',lw=0.2)
    # ax.axis('equal')
    # plt.show()


    np.set_printoptions(precision=2)
    turbigen.solvers.native.run(g, settings)

    b = g[0]
    C = b[:, b.nj//2, b.nk//2]
    P = C.P
    Po1 = C.Po[0]
    P1 = C.P[0]

    fig, ax = plt.subplots()
    for b in g:
        C = b[:, b.nj//2, b.nk//2]
        dPdx = np.gradient(C.P,C.x)
        mu = F.mu
        Cp = (C.P-P1)/(Po1-P1)
        ax.plot(C.x, Cp, '-x')

    fig, ax = plt.subplots()
    for b in g:
        C = b[:, b.nj//2, b.nk//2]
        ax.plot(C.x, C.Vx, '-x')

    iplot = int(b.ni*0.9)

    b = g[-1]
    C = b[iplot, :, b.nk//2]
    h = C.r.ptp()
    rnorm = (C.r-C.r.min())/C.r.ptp()
    K = dPdx[iplot]/2./mu*h*h
    soln = -K * rnorm*(1.-rnorm)
    err = (C.Vx-soln)/soln.max()

    fig, ax = plt.subplots()
    ax.plot(C.Vx, rnorm, '-x')
    ax.plot(soln, rnorm, '-x')
    ax.set_title('r')
    plt.show()

    fig, ax = plt.subplots()
    b = g[0]
    C = b[-1, :, :]
    ax.plot(C.z, C.y, '-')
    ax.plot(C.z.T, C.y.T, '-')
    ax.axis('equal')

    Cm, A, _ = C.mix_out()
    mdot = Cm.rho * Cm.Vm * A
    rho = Cm.rho
    w = 2.*np.pi*0.5*(C.r.min()+C.r.max())
    mdot_analytical =  -rho * w*h * K / 6.

    print(f'Analytical solution error: {err.min()}, {err.max()}, {err.mean()}')
    print(f'mdot acutal={mdot:.2f}, theory={mdot_analytical:.2f}, error={(mdot_analytical/mdot-1.)*100:.2f}%')
    assert np.abs(err).mean()<0.05

def not_test_blasius():

    ni = 51
    nj = 37

    xr = np.loadtxt('tests/blasius_grid.dat').reshape(2,nj, ni).transpose((0, 2, 1))
    print(xr.shape)

    Minf = 0.1
    Pinf_imp = 6.0
    Tinf_imp = 700.
    mu_imp = 6.5044372E-04

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(xr[0], xr[1], 'k.-')
    ax.plot(xr[0].T, xr[1].T, 'k.-')
    ax.axis('equal')
    plt.show()
    quit()


if __name__=='__main__':

    # test_plate_turb()
    test_plate_lam_yp5()
    # test_poiseuille()
