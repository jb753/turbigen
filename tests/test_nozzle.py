"""Run a quasi-1D nozzle in the native solver."""
import turbigen.solvers.native
import turbigen.compflow_native as cf
import turbigen.grid
import numpy as np
from timeit import default_timer as timer
import sys
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
import pytest

# Check our MPI rank
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Jump to solver slave process if not first rank
if rank > 0:
    turbigen.solvers.native.run_slave()
    sys.exit(0)

def make_nozzle(xnAR, L_h = 4., AR_merid=2., AR_pitch=1., skew=0., htr=0.99, dirn='r', xnRR=None, Alpha=0.):

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
    To1 = 300.0

    # Set inlet Ma to get inlet static state
    Ma1 = 0.3
    rgas = cp * (ga-1.)/ga
    V = cf.V_cpTo_from_Ma(Ma1,ga)*np.sqrt(cp*To1)
    P1 = Po1/cf.Po_P_from_Ma(Ma1,ga)
    T1 = To1/cf.To_T_from_Ma(Ma1,ga)

    # Numbers of grid points
    nj = 17
    nk = 17
    ni = int(nj*L_h)

    # Use pitchwise aspect ratio to find cell spacing, pitch and Nb
    pitch = h/(nj-1)*(nk-1)*AR_pitch
    Nb = int(2.0 * np.pi * rm / pitch)
    dt = 2.0 * np.pi / float(Nb)


    # Make the coordinates
    tv = np.linspace(-dt / 2., dt / 2., nk)
    xv = np.linspace(0., L, ni)
    rv = np.linspace(rh, rt, nj)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing='ij'))


    # Interpolate area at the x-coordinates
    fac_A = pchip_interpolate(*xnAR, xv/L)

    # Add on radius change
    if not xnRR is None:
        fac_R = pchip_interpolate(*xnRR, xv/L)
        xrt[1] *= np.expand_dims(fac_R,(1,2))
        fac_A /= fac_R


    # Apply skew
    xrt[2] += xrt[0] * np.tan(np.radians(skew))/xrt[1]

    # Squeeze the nozzle
    if dirn=='r':
        xrt[1] = (xrt[1] - rm)*np.expand_dims(fac_A, (1,2)) + rm
    elif dirn=='t':
        xrt[2] *= np.expand_dims(fac_A, (1,2))

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
                # turbigen.grid.PeriodicPatch(k=0),
                # turbigen.grid.PeriodicPatch(k=-1),
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

        block = turbigen.grid.PerfectBlock.from_coordinates(
                xrt[:,istb[iblock]:ienb[iblock],:,:], Nb, patches
        )
        block.label=f'b{iblock}'

        blocks.append(block)

    # Make the grid object
    g = turbigen.grid.Grid(blocks)
    g.match_patches()
    g.check_coordinates()

    # Boundary conditions
    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_P_T(Po1, To1)
    g.apply_inlet(So1, Alpha, Beta)
    g.calculate_wall_distance()
    g.apply_outlet(P1)

    # Initial guess
    for b in g:
        b.Vx = V
        b.Vr = 0.
        b.Vt = V*np.tan(np.radians(Alpha))
        b.cp = cp
        b.gamma = ga
        b.mu = mu
        b.Omega = 0.0
        b.set_P_T(P1, T1)

    g.apply_periodic()

    # Evaulate 1D analytical
    Q1 = cf.mcpTo_APo_from_Ma(Ma1,ga)
    Ma = cf.Ma_from_mcpTo_APo(Q1/fac_A, ga)
    P = Po1/cf.Po_P_from_Ma(Ma, ga)
    T = To1/cf.To_T_from_Ma(Ma, ga)
    V = np.sqrt(cp*To1)*cf.V_cpTo_from_Ma(Ma, ga)

    F = g[0].empty(shape=(ni,))
    F.Vx = V
    F.Vr = 0.
    F.Vt = 0.
    F.set_P_T(P,T)
    F.x = xv
    F.r = rm
    F.t = 0.

    return g, F

settings = {
    'n_step': 10000,
    'n_step_avg': 1000,
    'n_step_log': 1000,
    'i_loss': 0,
}

def plot_nozzle(g, F):

    L = F.x.ptp()

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f'C{ib}'
        C = b[:, :, b.nk//2]
        ax.plot(C.x[:,b.nj//2]/L, C.r[:,0]/L, color=cs)
        ax.plot(C.x[:,b.nj//2]/L, C.r[:,-1]/L, color=cs)
    ax.axis('equal')

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f'C{ib}'
        C = b[:, b.nj//2, b.nk//2]
        ax.plot(C.x/L, C.Ma, color=cs)
    ax.plot(F.x/L, F.Ma, 'k-')
    ax.set_ylim(bottom=0.)

    plt.show()

def post_nozzle(g, F):

    Ma = np.concatenate([b.Ma[:-1,b.nj//2, b.nk//2] for b in g])
    err_Ma = Ma-F.Ma[:-1]

    print(f'Mach error: mean={err_Ma.mean():.3e}, min={err_Ma.min():.3e}, max={err_Ma.max():.3e}')

    T2 = F.T[-1]
    ho1 = F.ho[0]
    s1 = F.s[0]
    V1 = F.Vx[0]
    s = np.concatenate([b.s[:-1,b.nj//2, b.nk//2] for b in g])
    Ys = (s-s1)*T2/(0.5*V1**2)

    print(f'Entropy conservation error Ys: mean={Ys.mean():.3e}, min={Ys.min():.3e}, max={Ys.max():.3e}')

    ho = np.concatenate([b.ho[:-1,b.nj//2, b.nk//2] for b in g])
    Cho = (ho-ho1)/(0.5*V1**2)

    print(f'Energy conservation error Cho: mean={Cho.mean():.3e}, min={Cho.min():.3e}, max={Cho.max():.3e}')

    return err_Ma, Ys, Cho

@pytest.mark.parametrize("dirn", ('r','t'))
def test_nozzle(dirn, plot=False):

    xA = np.array(
        [
            [0.,0.02, 0.3, 0.98, 1.],
            [1.,1., 0.6, 1., 1.]
        ]
    )

    g, F = make_nozzle(xA, dirn=dirn)


    np.set_printoptions(precision=2)

    turbigen.solvers.native.run(g, settings)

    err_Ma, Ys, Cho = post_nozzle(g, F)

    tol_Ma = 0.02  # Quite loose because flow not really 1D
    assert (np.abs(err_Ma)<tol_Ma).all()
    tol_s = 0.001
    assert (np.abs(Ys)<tol_s).all()
    tol_ho = 0.002
    assert (np.abs(Cho)<tol_ho).all()

    if plot:
        plot_nozzle(g, F)

def test_radius():

    xA = np.array(
        [
            [0.,0.02, 0.3, 0.98, 1.],
            [1.,1., 0.6, 1., 1.]
        ]
    )
    xR = np.array(
        [
            [0.,0.02, 0.98, 1.],
            [1.,1., 0.9, 0.9]
        ]
    )

    g, F = make_nozzle(xA, dirn='r',xnRR=xR, htr=0.9, Alpha=0.,skew=0.)

    np.set_printoptions(precision=2)

    turbigen.solvers.native.run(g, settings)

    _, Ys, Cho = post_nozzle(g, F)

    # tol_s = 0.001
    # assert (np.abs(Ys)<tol_s).all()
    tol_ho = 0.01
    # assert (np.abs(Cho)<tol_ho).all()

    fig, ax = plt.subplots()
    ax.plot(Ys)

    fig, ax = plt.subplots()
    ax.plot(Cho)

    # if plot:
    plot_nozzle(g, F)

test_radius()
# test_nozzle(dirn='r', plot=True)
