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
    'n_step': 40000,
    # 'n_step': 1000,
    'n_step_avg': 1000,
    'n_step_log': 100,
    'plot_conv': True,
    # 'nstep_damp': -1,
    'xllim_pitch': 0.,
    # 'i_loss': 0,
    # "damping_factor" : 25.,
    # "nstep_damp" : -1,
    # "CFL" : 0.4,
    # "i_scheme" : 0,
    # "i_exit" : 0,
    # "smoothing_factor" : 0.001,
    # "smoothing_2nd_proportion" : 0.5
}

# Check our MPI rank
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Jump to solver slave process if not first rank
if rank > 0:
    turbigen.solvers.native.run_slave()
    sys.exit(0)

def make_pipe():
    """Generate the grid."""

    L_h = 6.
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
    Ma1 = 0.3
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

    # Calculate dwall for target yplus
    yplus = 1.
    Re = rho1 * V * h / mu
    Cf = (2.0 * np.log10(Re) - 0.65) ** -2.3
    tauw = Cf * 0.5 * (rho1 * V**2)
    Vtau = np.sqrt(tauw / rho1)
    Lvisc = mu / rho1 / Vtau
    dw = yplus * Lvisc/h
    # print(dw)
    dw = 0.001
    dmax = 0.04
    ER = 1.05
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
    nblock = 2
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
    plt.show()

    print(f'Analytical solution error: {err.min()}, {err.max()}, {err.mean()}')
    assert np.abs(err).mean()<0.05


if __name__=='__main__':

    test_poiseuille()
