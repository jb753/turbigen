"""Run a quasi-1D nozzle in the native solver."""
import turbigen.solvers.native
import turbigen.compflow_native as cf
import turbigen.grid
import turbigen.util
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

def make_nozzle(xnAR, L_h = 4., AR_merid=2., AR_pitch=1., skew=0., htr=0.99, dirn='r', xnRR=None, Alpha=0., tper=False):
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
    # tv = np.linspace(-dt / 2., dt / 2., nk)
    tv = np.linspace(0., dt, nk)
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

        if tper:
            patches.extend([
                turbigen.grid.PeriodicPatch(k=0),
                turbigen.grid.PeriodicPatch(k=-1),
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

    g.match_patches()
    g.apply_periodic()

    return g, F

settings = {
    'n_step': 20000,
    'n_step_avg': 1000,
    'n_step_log': 100,
    'i_loss': 0,
    'i_exit': 1,
    'i_inlet': 1,
}

def plot_nozzle(g, F):
    """Make debugging plots."""

    L = F.x.ptp()

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f'C{ib}'
        C = b[:, :, b.nk//2]
        ax.plot(C.x[:,b.nj//2]/L, C.r[:,0]/L, color=cs)
        ax.plot(C.x[:,b.nj//2]/L, C.r[:,-1]/L, color=cs)
        ax.plot(C.x/L, C.r/L, 'k-', lw=0.1)
        ax.plot(C.x.T/L, C.r.T/L, 'k-', lw=0.1)
    ax.axis('equal')
    ax.set_ylabel('r')
    ax.set_xlabel('x')

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f'C{ib}'
        C = b[:, b.nj//2, :]
        ax.plot(C.x[:,0]/L, C.r[:,0]/L, color=cs)
        ax.plot(C.x[:,-1]/L, C.r[:,-1]/L, color=cs)
        ax.plot(C.x/L, C.rt/L, 'k-', lw=0.1)
        ax.plot(C.x.T/L, C.rt.T/L, 'k-', lw=0.1)
    ax.axis('equal')
    ax.set_ylabel('rt')
    ax.set_xlabel('x')

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f'C{ib}'
        C = b[:, b.nj//2, b.nk//2]
        ax.plot(C.x/L, C.Ma, '-.',color=cs)
    ax.plot(F.x/L, F.Ma, 'k-')
    ax.set_title('Ma')
    ax.set_ylim((0,1))

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f'C{ib}'
        C = b[:, b.nj//2, b.nk//2]
        ax.plot(C.x/L, C.P/F.Po, color=cs)
    ax.plot(F.x/L, F.P/F.Po, 'k-')
    ax.set_title('P')
    # ax.set_ylim(bottom=0.)

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f'C{ib}'
        C = b[:, b.nj//2, b.nk//2]
        ax.plot(C.x/L, C.rVt, color=cs)
    ax.plot(F.x/L, F.rVt, 'k-')
    ax.set_title('rVt')

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        cs = f'C{ib}'
        C = b[:, b.nj//2, b.nk//2]
        ax.plot(C.x/L, C.Alpha, color=cs)
    ax.set_title('Alpha')

    fig, ax = plt.subplots()
    for ib, b in enumerate(g):
        C = b[:, :, b.nk//2]
        ax.contourf(C.x/L, C.r/L, C.s)
    ax.set_title('ent')
    # ax.set_ylim((0,1))

    plt.show()

def post_nozzle(g, F):
    """Extract errors."""

    Ma = np.concatenate([b.Ma[:-1,b.nj//2, b.nk//2] for b in g])
    err_Ma = Ma-F.Ma[:-1]

    print(f'Mach error: mean={err_Ma.mean():.2e}, min={err_Ma.min():.2e}, max={err_Ma.max():.2e}')

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
def test_condi(dirn, plot=False):
    """Run subsonic con-di nozzles."""

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

    rtol_Ma = 5e-2
    assert (np.abs(err_Ma)<rtol_Ma).all()
    rtol_sh = 5e-3
    assert (np.abs(Ys)<rtol_sh).all()
    assert (np.abs(Cho)<rtol_sh).all()

@pytest.mark.parametrize("Alpha", (-30.,0.,30.))
def test_uniform(Alpha):
    """Run the most basic parallel annulus, grid aligned with flow."""

    xA = np.array(
        [
            [0., 0.01,0.99, 1.],
            [1.,1., 1., 1.]
        ]
    )

    g, F = make_nozzle(xA, Alpha=Alpha, skew=Alpha)

    np.set_printoptions(precision=2)

    turbigen.solvers.native.run(g, settings)

    err_Ma, Ys, Cho = post_nozzle(g, F)

    rtol = 2e-4
    assert (np.abs(err_Ma)<rtol).all()
    assert (np.abs(Ys)<rtol).all()
    assert (np.abs(Cho)<rtol).all()

@pytest.mark.parametrize("Alpha", (-30.,0.,30.))
def test_skew(Alpha):
    """Run an axial flow with skewed grid."""

    xA = np.array(
        [
            [0., 0.01,0.99, 1.],
            [1.,1., 1., 1.]
        ]
    )

    g, F = make_nozzle(xA, skew=Alpha, tper=True)

    np.set_printoptions(precision=2)

    turbigen.solvers.native.run(g, settings)

    err_Ma, Ys, Cho = post_nozzle(g, F)

    rtol = 1e-4
    assert (np.abs(err_Ma)<rtol).all()
    assert (np.abs(Ys)<rtol).all()
    assert (np.abs(Cho)<rtol).all()

@pytest.mark.parametrize("Alpha", (-30.,0.,30.))
def test_radius(Alpha):
    """Constant area with radius change."""

    xA = np.array(
        [
            [0., 0.01,0.99, 1.],
            [1.,1., 1., 1.]
        ]
    )
    xR = np.array(
        [
            [0.,0.02, 0.98, 1.],
            [1.,1., 0.9, .9]
        ]
    )

    g, F = make_nozzle(xA, xnRR=xR, htr=0.9, Alpha=Alpha, skew=Alpha, tper=True)

    np.set_printoptions(precision=2)

    turbigen.solvers.native.run(g, settings)

    _, Ys, Cho = post_nozzle(g, F)

    if Alpha:
        rVt = np.concatenate([b.rVt[:-1,b.nj//2, b.nk//2] for b in g])
        err_rVt = rVt/rVt[0]-1.
        tol_rVt = 1e-2
        print(f'Angular momentum conservation error drVt mean={err_rVt.mean():.2e}, min={err_rVt.min():.2e}, max={err_rVt.max():.2e}')
        assert (err_rVt<tol_rVt).all()

    tol_sh = 1e-3
    assert (np.abs(Ys)<tol_sh).all()
    assert (np.abs(Cho)<tol_sh).all()

def test_patch_A_avg():
    """Check patch area averaging weights."""

    # Make an arbitrary grid
    xA = np.array(
        [
            [0.,0.02, 0.3, 0.98, 1.],
            [1.,1., 0.6, 1., 1.]
        ]
    )
    g, F = make_nozzle(xA)
    block = g[0]
    patch = block.outlet_patches[0]

    # Calculate area average using weight
    ind = patch.get_flat_indices(order='F')
    w = patch.get_A_avg_weights(order='F')
    rsqavg = np.sum((block.r.ravel(order='F')[ind]**2.)*w)

    # Manually check the area average
    C = patch.get_cut()
    dA = turbigen.util.vecnorm(C.dAi)
    rsqavg_check = np.sum((C.r_face[0]**2)*dA)/np.sum(dA)
    assert np.isclose(rsqavg, rsqavg_check)

def not_test_exit(Alpha):

    xA = np.array(
        [
            [0., 0.01,0.99, 1.],
            [1.,1., 1., 1.]
        ]
    )

    g, F = make_nozzle(xA, Alpha=Alpha, skew=Alpha, htr=0.7)

    np.set_printoptions(precision=2)

    settings = {
        'n_step': 8000,
        'n_step_avg': 100,
        'n_step_log': 100,
        'i_loss': 0,
    }

    turbigen.solvers.native.run(g, settings)

    fig, ax = plt.subplots()
    b = g[-1]
    Cout = b[-1,:,b.nk//2]
    ax.plot(Cout.x,Cout.Ma)
    plt.show()


    fig, ax = plt.subplots()
    b = g[-1]
    Cout = b[-1,:,b.nk//2]
    ax.plot(Cout.P/F.P[-1]-1., Cout.r)
    ax.set_title('P/Pout')
    plt.show()

if __name__=='__main__':


    # print('testing exit, aligned grid')
    # test_patch_A_avg()
    # test_exit(0.)

    # print('testing uniform, aligned grid')
    test_uniform(30.)

    # print('testing uniform, skewed grid')
    # test_skew(30.)

    # print('testing radius change, aligned grid')
    # test_radius(30.)

    # print('testing con-di nozzles')
    # test_condi('t')
