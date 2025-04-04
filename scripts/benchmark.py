import turbigen.solvers.emb as embsolve
import turbigen.solvers.ts3
import turbigen.compflow_native as cf
import turbigen.grid
import turbigen.clusterfunc
import turbigen.util
import numpy as np
from timeit import default_timer as timer
import sys
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt

try:
    # Check our MPI rank
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Jump to solver slave process if not first rank
    if rank > 0:
        embsolve.run_slave()
        sys.exit(0)
except ImportError as e:
    print(e)
    pass


def make_nozzle(
    nblock,
    xnAR,
    L_h=4.0,
    AR_merid=2.0,
    AR_pitch=1.0,
    skew=0.0,
    htr=0.99,
    dirn="r",
    xnRR=None,
    Alpha=0.0,
    Ma1=0.3,
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
    To1 = 300.0

    # Set inlet Ma to get inlet static state
    rgas = cp * (ga - 1.0) / ga
    V = cf.V_cpTo_from_Ma(Ma1, ga) * np.sqrt(cp * To1)
    P1 = Po1 / cf.Po_P_from_Ma(Ma1, ga)
    T1 = To1 / cf.To_T_from_Ma(Ma1, ga)

    # ~10^6 grid points
    ni = 169
    nj = 81
    nk = 73

    # ni = 81
    # nj = 33
    # nk = 25

    # Use pitchwise aspect ratio to find cell spacing, pitch and Nb
    pitch = h / (nj - 1) * (nk - 1) * AR_pitch
    Nb = int(2.0 * np.pi * rm / pitch)
    dt = 2.0 * np.pi / float(Nb)

    # Make the coordinates
    tv = np.linspace(0.0, dt, nk)
    xv = np.linspace(0.0, L, ni)
    rv = np.linspace(rh, rt, nj)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

    # Interpolate area at the x-coordinates
    fac_A = pchip_interpolate(*xnAR, xv / L)

    # Add on radius change
    if not xnRR is None:
        fac_R = pchip_interpolate(*xnRR, xv / L)
        xrt[1] *= np.expand_dims(fac_R, (1, 2))
        fac_A /= fac_R

    # Apply skew
    xrt[2] += xrt[0] * np.tan(np.radians(skew)) / xrt[1]

    # Squeeze the nozzle
    if dirn == "r":
        xrt[1] = (xrt[1] - rm) * np.expand_dims(fac_A, (1, 2)) + rm
    elif dirn == "t":
        xrt[2] *= np.expand_dims(fac_A, (1, 2))

    # Split into blocks
    blocks = []
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
                turbigen.grid.InviscidPatch(k=0),
                turbigen.grid.InviscidPatch(k=-1),
                turbigen.grid.InviscidPatch(j=0),
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
    g.apply_inlet(So1, Alpha, Beta)
    # g.calculate_wall_distance()
    g.apply_outlet(P1)

    # Fluid props
    for b in g:
        b.w = 0.0
        b.cp = cp
        b.gamma = ga
        b.mu = mu
        b.Omega = 0.0

    # Evaulate 1D analytical
    Q1 = cf.mcpTo_APo_from_Ma(Ma1, ga)
    Ma = cf.Ma_from_mcpTo_APo(Q1 / fac_A, ga)
    P = Po1 / cf.Po_P_from_Ma(Ma, ga)
    T = To1 / cf.To_T_from_Ma(Ma, ga)
    V = np.sqrt(cp * To1) * cf.V_cpTo_from_Ma(Ma, ga)

    F = g[0].empty(shape=(ni,))
    F.Vx = V
    F.Vr = 0.0
    F.Vt = 0.0
    F.set_P_T(P, T)
    F.x = xv
    F.r = rm
    F.t = 0.0

    Ve = np.expand_dims(V, (1, 2))
    Te = np.expand_dims(T, (1, 2))
    Pe = np.expand_dims(P, (1, 2))

    # Initial guess
    for ib, b in enumerate(g):
        b.Vx = Ve[istb[ib] : ienb[ib]]
        b.Vr = 0.0
        b.Vt = Ve[istb[ib] : ienb[ib]] * np.tan(np.radians(Alpha))
        b.set_P_T(Pe[istb[ib] : ienb[ib]], Te[istb[ib] : ienb[ib]])

    g.match_patches()

    return g, F


xA = np.array([[0.0, 0.02, 0.3, 0.98, 1.0], [1.0, 1.0, 0.6, 1.0, 1.0]])

# def make_cylinder(nblock):

#     # Geometry
#     L = 0.4
#     rm = 1.0
#     dr = 0.1

#     r1 = rm - dr / 2.0
#     r2 = rm + dr / 2.0

#     ni = 169
#     nj = 81
#     nk = 73
#     ntot = ni*nj*nk

#     Nb = np.round(2*np.pi * rm / dr).astype(int)
#     pitch = 2.0 * np.pi / Nb

#     xv = np.linspace(0, L, ni)
#     rv = np.linspace(r1, r2, nj)
#     tv = np.linspace(0.0, pitch, nk)

#     xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

#     # Split into blocks
#     blocks = []
#     istb = [ni // nblock * iblock for iblock in range(nblock)]
#     ienb = [ni // nblock * (iblock + 1) + 1 for iblock in range(nblock)]
#     ienb[-1] = ni

#     for iblock in range(nblock):

#         # Special case for only one block
#         if nblock == 1:
#             patches = [
#                 turbigen.grid.InletPatch(i=0),
#                 turbigen.grid.OutletPatch(i=-1),
#             ]

#         # First block has an inlet
#         elif iblock == 0:
#             patches = [
#                 turbigen.grid.InletPatch(i=0),
#                 turbigen.grid.PeriodicPatch(i=-1),
#             ]

#         # Last block has outlet
#         elif iblock == (nblock - 1):
#             patches = [
#                 turbigen.grid.PeriodicPatch(i=0),
#                 turbigen.grid.OutletPatch(i=-1),
#             ]

#         # Middle blocks are both periodic
#         else:
#             patches = [
#                 turbigen.grid.PeriodicPatch(i=0),
#                 turbigen.grid.PeriodicPatch(i=-1),
#             ]

#         patches.extend(
#             [
#                 turbigen.grid.PeriodicPatch(k=0),
#                 turbigen.grid.PeriodicPatch(k=-1),
#             ]
#         )

#         block = turbigen.grid.PerfectBlock.from_coordinates(
#             xrt[:, istb[iblock] : ienb[iblock], :, :], Nb, patches
#         )
#         block.label = f"b{iblock}"

#         blocks.append(block)


#     g = turbigen.grid.Grid(blocks)
#     g.check_coordinates()

#     # Boundary conditions
#     cp = 1005.
#     ga = 1.4
#     mu = 1.84e-5
#     Po1 = 1e5
#     To1 = 300.
#     Ma1 = 0.4
#     V1 = cf.V_cpTo_from_Ma(Ma1, ga) * np.sqrt(cp * To1)
#     P1 = Po1 / cf.Po_P_from_Ma(Ma1, ga)
#     T1 = To1 / cf.To_T_from_Ma(Ma1, ga)
#     So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
#     So1.set_P_T(Po1, To1)
#     So1.set_Tu0(To1)
#     g.apply_inlet(So1, 0., 0.)
#     g.calculate_wall_distance()
#     g.apply_outlet(P1)
#     g.match_patches()

#     # Initial guess
#     for b in g:
#         b.Vx = V1
#         b.Vr = 0.0
#         b.Vt = 0.0
#         b.cp = cp
#         b.gamma = ga
#         b.mu = mu
#         b.Omega = 0.0
#         b.set_P_T(P1, T1)
#         b.set_Tu0(To1)

#     return g


def run_embsolve(g):
    solver = embsolve.Emb(
        n_step=200,
        n_step_log=50,
        n_step_ramp=0,
        n_step_avg=100,
        nstep_damp=0,
        print_conv=False,
    )
    solver.run(g)
    conv = solver.convergence

    return conv.tpnps, conv.err_mdot[-1]


def run_ts3(g):
    conf = turbigen.solvers.ts3.Config(
        nstep=20000, nstep_avg=100, workdir="ts3", ilos=1
    )
    tstart = timer()
    turbigen.solvers.ts3.run(g, conf, None)
    tend = timer()
    return tend - tstart


if __name__ == "__main__":
    # g, _ = make_nozzle(1, xA)
    # ts3_time = run_ts3(g)
    # quit()
    # Turbostream cost
    # £ factor * tpnps * cfl factor
    cost_ts3 = 55.0 * 4e-9 * 0.7 / 0.4
    time_ts3 = 4e-9 * 0.7 / 0.4

    g, _ = make_nozzle(size, xA)
    tpnps, err = run_embsolve(g)
    cost = tpnps * size
    with open("plots/bench.dat", "a") as f:
        f.write(
            f"{size}, {err}, {tpnps}, {cost}, {cost / cost_ts3}, {tpnps / time_ts3}\n"
        )
