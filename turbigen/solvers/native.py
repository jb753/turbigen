import numpy as np
import turbigen.util
import turbigen.fluid
import turbigen.flowfield
import turbigen.grid
from turbigen.solvers.base import BaseSolver
from turbigen.compiled import (
    smooth,
    node_to_cell,
    node_to_face,
    residual,
    step,
    damp,
    calculate_secondary,
    viscous_force
)
from timeit import default_timer as timer
from mpi4py import MPI
import logging

logger = turbigen.util.make_logger()

logger.setLevel(level=logging.INFO)

typ = np.float32



class NativeConfig(BaseSolver):
    """Settings with default values for the native solver."""

    _name = "Native"

    smoothing_factor = 0.01
    """Artificial dissipation to suppress central-differencing instability and
    reduce overshoots at sharp discontinuities. Increased values are more
    robust, but less accurate."""

    smoothing_2nd_proportion = 0.2

    CFL = 0.7
    """Courant--Friedrichs--Lewy number, time step normalised by local wave
    speed and cell size. Reduced values are more stable but slower to
    converge."""

    n_step = 5000
    """Number of time steps to run for."""

    n_step_dt = 10
    """Number of time steps between updates of the local time step."""

    n_step_log = 100
    """Number of time steps between log prints."""

    n_step_avg = 1
    """Number of time steps to average over."""

    conv_lim = 1e-9

    damping_factor = 25.0
    """Negative feedback to damp down high residuals. Lower values are more stable."""

    i_scheme = 1

class SolverBlock:
    """Hold just the data we need for a CFD solution."""

    def __init__(self, block):
        """Initialise from a standard Block object."""


        # Primaries
        self.conserved = np.asfortranarray(np.moveaxis(block.conserved, 0, -1)).astype(typ)

        self.mu = block.mu

        self.ho = np.asfortranarray(block.ho).astype(typ)
        self.P = np.asfortranarray(block.P).astype(typ)

        self.halfVsq = np.asfortranarray(0.5 * block.V**2).astype(typ)
        self.u = np.asfortranarray(block.u).astype(typ)

        self.x = np.asfortranarray(block.x).astype(typ)
        self.r = np.asfortranarray(block.r).astype(typ)
        self.t = np.asfortranarray(block.t).astype(typ)

        # Geometry
        self.r = np.asfortranarray(block.r).astype(typ)
        self.dAi = np.asfortranarray(np.moveaxis(block.dAi, 0, -1)).astype(typ)
        self.dAj = np.asfortranarray(np.moveaxis(block.dAj, 0, -1)).astype(typ)
        self.dAk = np.asfortranarray(np.moveaxis(block.dAk, 0, -1)).astype(typ)
        self.vol = np.asfortranarray(block.vol).astype(typ)
        self.dlmin = np.asfortranarray(block.dlmin).astype(typ)
        self.Omega = block.Omega.mean().astype(typ)

        self.dU1 = self.conserved.copy(order="F").astype(typ) * np.nan
        self.dU2 = self.conserved.copy(order="F").astype(typ) * np.nan
        self._flag_scree = False

        self.conserved_avg = self.conserved.copy(order="F").astype(np.double) * 0.

        ni, nj, nk = block.shape
        self.f = np.zeros((ni-1, nj-1, nk-1, 5), order="F", dtype=typ)

        # Get wall indicators
        # These are three arrays of shape
        #   i faces: (ni, nj-1, nk-1)
        #   j faces: (ni-1, nj, nk-1)
        #   k faces: (ni-1, nj-1, nk)
        # equal to one if the face is a wall, zero otherwise
        self.wall_indicators = [np.asfortranarray(w) for w in get_wall(block)]
        self.wall_nodes = block.get_wall(trim=0)

        # ni, nj, nk = self.wall_nodes.shape
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(self.wall_nodes[:,nj//2, 0])
        # plt.show()
        # quit()

        # # Convert wall indicators to wall indices
        # # Which are indices into the flattend face arrays
        # self.walls = [np.where((w > 0.99).flat)[0] for w in self.wall_indicators]

        self.inlets = [patch.get_inlet_data() for patch in block.inlet_patches]
        self.outlets = [patch.get_outlet_data() for patch in block.outlet_patches]

        if isinstance(block, turbigen.grid.PerfectBlock):
            self.state = turbigen.fluid.PerfectState(shape=block.shape, order="F")
            self.state._metadata = block._metadata
            self.state.set_rho_u(block.rho, block.u)
            self.state_inlets = [
                turbigen.fluid.PerfectState(shape=inlet[0].shape, order="F")
                for inlet in self.inlets
            ]
        else:
            raise NotImplementedError()

        # Preallocate stored inlet density
        for inlet, state_inlet in zip(self.inlets, self.state_inlets):
            state_inlet._metadata = block._metadata
            rho_inlet = block.rho.ravel(order="F")[inlet[0]]
            u_inlet = block.u.ravel(order="F")[inlet[0]]
            state_inlet.set_rho_u(rho_inlet, u_inlet)

    def set_inlets(self, rfin):
        """Set conserved variables on inlets by relaxing density changes."""

        # Change inlet patches
        for patch, state in zip(self.inlets, self.state_inlets):

            # Expand patch data
            ind, Po, To, Alpha, Beta, rhoo, ho, r = patch

            # Relax changes in density
            rho_now = (
                rfin * self.conserved[..., 0].ravel(order="F")[ind] + (1.-rfin) * state.rho
            )

            # Check for flow reversal
            rho_now[rho_now > rhoo] = rhoo * 0.999

            # Isentropic expansion from stagnation state
            state.set_rho_s(rho_now, state.s)

            # Pull out vars we need
            h, u, P = state.h, state.u, state.P

            # Get the velocity
            dhin = ho - h
            Vinsq = 2.0 * dhin
            Vin = np.sqrt(Vinsq)

            # Resolve velocity components
            tanAlpha = turbigen.util.tand(Alpha)
            tanBeta = turbigen.util.tand(Beta)
            Vxin = Vin / np.sqrt((1.0 + tanAlpha**2) * (1.0 + tanBeta**2))
            Vrin = Vxin * tanBeta
            Vmin = np.sqrt(Vxin**2 + Vrin**2)
            Vtin = Vmin * tanAlpha

            # Reset conserved vars on inlet
            # Do not reset the density - seems to compromise stability
            self.conserved[..., 1].ravel(order="F")[ind] = rho_now * Vxin  # rhoVx
            self.conserved[..., 2].ravel(order="F")[ind] = rho_now * Vrin  # rhoVr
            self.conserved[..., 3].ravel(order="F")[ind] = rho_now * r * Vtin  # rhorVt
            self.conserved[..., 4].ravel(order="F")[ind] = rho_now * (
                u + 0.5 * Vinsq
            ) 

            # Reset pressure and hstag on inlet
            self.ho.ravel(order="F")[ind] = h + 0.5 * Vin**2
            self.P.ravel(order="F")[ind] = P

    def set_outlets(self):
        """Set static pressure on outlets."""
        for outlet in self.outlets:
            self.P.ravel(order="F")[outlet[0]] = outlet[1]

    def set_walls(self):
        """Zero the momentums on a wall."""
        self.conserved[...,1][self.wall_nodes] = 0.
        self.conserved[...,2][self.wall_nodes] = 0.
        self.conserved[...,3][self.wall_nodes] = 0.
        self.conserved[...,4][self.wall_nodes] = self.conserved[...,0][self.wall_nodes]*self.u[self.wall_nodes]

    def set_timestep(self, CFL):
        Vx = self.conserved[..., 1] / self.conserved[..., 0]
        Vr = self.conserved[..., 2] / self.conserved[..., 0]
        Vt = self.conserved[..., 3] / self.conserved[..., 0] / self.r
        V = np.sqrt(Vx**2 + Vr**2 + Vt**2)

        a = self.state.a

        ni, nj, nk = self.r.shape

        Va_node = np.asfortranarray(np.stack((V, a), axis=-1)).astype(typ)
        Va_cell = np.empty((ni - 1, nj - 1, nk - 1, 2), order="F", dtype=typ)
        node_to_cell(Va_node, Va_cell)
        Vref = Va_cell[..., 0]
        aref = Va_cell[..., 1]
        self.dt = CFL * self.dlmin / (aref + Vref)

    # @profile

    def residual(self):
        residual(
            self.conserved,
            self.P,
            self.ho,
            self.r,
            self.f,
            self.Omega,
            *self.wall_indicators,
            self.dt,
            self.dAi,
            self.dAj,
            self.dAk,
            self.vol,
            self.dU1,
        )

    def step(self, istep, istep_avg, nstep_avg, ischeme):
        step(
            self.conserved,
            self.conserved_avg,
            self.dU1,
            self.dU2,
            istep,
            istep_avg,
            nstep_avg,
            ischeme,
        )


    def set_secondary(self):

        calculate_secondary(self.r, self.conserved, self.halfVsq, self.u)
        self.state.set_rho_u(self.conserved[..., 0], self.u)
        self.ho[:] = self.state.h + self.halfVsq
        self.P[:] = self.state.P

    def smooth(self, sf2, sf4):
        smooth(self.conserved, sf2, sf4)

    def damp(self, fdamp):
        damp(self.dU1, fdamp)

    def calculate_viscous(self):
        viscous_force(
            self.conserved, self.f, self.mu, *self.wall_indicators, self.vol, self.dAi, self.dAj, self.dAk, self.r
        )




def get_periodics(g, procids):

    periodics = []
    seen = []
    pid = 0

    for patch in g.periodic_patches:

        if patch in seen:
            continue
        else:
            seen.append(patch)
            seen.append(patch.match)

        bid, ind, nxbid, nxind = patch.get_periodic_data()
        periodics.append((pid, bid, procids[bid], ind, nxbid, procids[nxbid], nxind))
        pid += 1

    return periodics


def set_periodic(b1, b2, ind1, ind2):
    conserved1 = b1.conserved.ravel(order="F")
    conserved2 = b2.conserved.ravel(order="F")
    avg = 0.5 * (conserved1[ind1] + conserved2[ind2])
    conserved1[ind1] = avg
    conserved2[ind2] = avg


def send_slave(block_split, procids, periodics, settings):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    for iproc in range(1, size):
        comm.send(block_split[iproc], dest=iproc)

    comm.Barrier()

    for iproc in range(1, size):
        comm.send(periodics, dest=iproc)

    comm.Barrier()

    for iproc in range(1, size):
        comm.send(settings, dest=iproc)

    comm.Barrier()

def exchange_periodics(blocks, bid_local, periodics, variable='conserved'):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Update periodic boundaries
    for patch in periodics:
        pid, bid, procid, ind, nxbid, nxprocid, nxind = patch
        count = len(ind)

        b1 = blocks[bid_local[bid]]

        if variable=='conserved':
            v1 = b1.conserved.ravel(order='F')
        elif variable=='residual':
            v1 = b1.dU1.ravel(order='F')

        # Just set the periodic if on same rank
        if nxprocid == rank:

            b2 = blocks[bid_local[nxbid]]
            if variable=='conserved':
                v2 = b2.conserved.ravel(order='F')
            elif variable=='residual':
                v2 = b2.dU1.ravel(order='F')

            avg = 0.5 * (v1[ind] + v2[nxind])
            v1[ind] = avg
            v2[nxind] = avg

        # Otherwise, communication is needed
        else:

            # Preallocate a buffer to recieve data
            nxv = np.empty((count,), dtype=typ)

            # If our rank is lower than next rank, send first
            if rank < nxprocid:
                comm.Send([v1[ind], count, MPI.REAL4], dest=nxprocid, tag=pid)
                comm.Recv([nxv, count, MPI.REAL4], source=nxprocid, tag=pid)
            # If our rank is higher than next rank, recieve first
            else:
                comm.Recv([nxv, count, MPI.REAL4], source=nxprocid, tag=pid)
                comm.Send([v1[ind], count, MPI.REAL4], dest=nxprocid, tag=pid)
            v1[ind] = 0.5 * (v1[ind] + nxv)


def run_slave(blocks=None, periodics_all=None, nodes=None, conf=None):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if blocks is None:
        blocks = comm.recv()
        comm.Barrier()
        periodics_all = comm.recv()
        comm.Barrier()
        conf = comm.recv()
        comm.Barrier()
        master_flag = False
    else:
        master_flag = True
        dUlog = []


    # Calculate smoothing and inlet relaxation scaled by CFL
    sf = conf.CFL * conf.smoothing_factor
    sf2 = sf * conf.smoothing_2nd_proportion
    sf4 = sf * (1.0 - conf.smoothing_2nd_proportion)
    rfin = 0.5#0.2/conf.CFL

    # Only keep relevent periodics
    # And rearrange the periodics so that foreign procid is always nx
    periodics = []
    for patch in periodics_all:
        pid, bid, procid, ind, nxbid, nxprocid, nxind = patch
        if procid == rank:
            periodics.append(patch)
        elif nxprocid == rank:
            periodics.append((pid, nxbid, nxprocid, nxind, bid, procid, ind))

    bids = [b.bid for b in blocks]

    # Lookup of local bid from global bid
    bid_local = {bid: ibid for ibid, bid in enumerate(bids)}

    nblock = len(blocks)

    tstart = timer()

    dUe_ref = None

    # Start the main time stepping loop
    for istep in range(conf.n_step):

        if not np.mod(istep, conf.n_step_dt):
            exchange_periodics(blocks, bid_local, periodics, variable='conserved')

        # Calculate residual for all blocks
        for iblock in range(nblock):

            sb = blocks[iblock]

            if not np.mod(istep, conf.n_step_dt):
                sb.set_timestep(conf.CFL)

            sb.set_inlets(rfin)

            sb.set_outlets()

            # sb.set_walls()

            sb.residual()

        # Send residuals to other blocks
        exchange_periodics(blocks, bid_local, periodics, variable='residual')

        # Now integrate forward
        istep_avg =  conf.n_step-conf.n_step_avg
        for iblock in range(nblock):
            sb = blocks[iblock]

            if conf.damping_factor:
                sb.damp(conf.damping_factor)

            sb.step(istep, istep_avg, conf.n_step_avg, conf.i_scheme)

            sb.smooth(sf2, sf4)

            sb.set_secondary()

            if not np.mod(istep, 5) and istep > 100:
                sb.calculate_viscous()

        if not np.mod(istep, conf.n_step_log) and istep > 0:

            # Send residuals to master proc
            dUnow = np.stack([np.abs(b.dU1.mean(axis=(0,1,2))) for b in blocks])

            if rank:
                comm.send(dUnow, dest=0)
                terminate = np.empty((1,),dtype=int)
                comm.Recv([terminate, 1, MPI.INT], source=0, tag=rank)
                if terminate:
                    comm.send(blocks, dest=0)
                    return
            else:
                dUall = [dUnow,]
                for iproc in range(1, size):
                    dUall.append(comm.recv(source=iproc))
                dUall = np.concatenate(dUall)

                ten = timer()
                tpnps = (ten - tstart) / nodes / conf.n_step_log
                tstart = ten

                logger.info(f"{istep}: tpnps={tpnps:.3e}")
                for ib, dU in enumerate(dUall):
                    logger.info(f"  block {ib}: {dU}")

                dUlognow = np.stack(dUall).mean(axis=0)
                dUlog.append(dUlognow)

                if not dUe_ref:
                    dUe_ref = dUlognow[-1]*conf.conv_lim

                if dUlognow[-1] < dUe_ref:
                    terminate = np.array((1,), dtype=int)
                else:
                    terminate = np.array((0,), dtype=int)
                for iproc in range(1, size):
                    comm.Send([terminate, 1, MPI.INT], dest=iproc, tag=iproc)

                if terminate:
                    return blocks, dUlog


    if master_flag:
        return blocks, dUlog
    else:
        comm.send(blocks, dest=0)


def run(grid, settings={}, machine=None):

    conf = NativeConfig(**settings)

    logger.info("Intialising native solver...")

    nodes = np.sum([b.size for b in grid])

    blocks = [SolverBlock(b) for b in grid]
    for ib, b in enumerate(blocks):
        b.bid = ib

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    logger.info(f"Patitioning onto {size} processors...")
    procids = grid.partition(size)
    periodics = get_periodics(grid, procids)

    # Split into lists for each procid
    block_split = []
    for iproc in range(size):
        block_split.append([])
        for ib, b in enumerate(blocks):
            if iproc == procids[ib]:
                block_split[-1].append(b)

    logger.info("Sending data to processors...")
    send_slave(block_split, procids, periodics, conf)

    logger.info("Starting the main time-stepping loop...")
    block_split[0], dUlog = run_slave(block_split[0], periodics, nodes, conf)

    for iproc in range(1, size):
        block_split[iproc] = comm.recv(source=iproc)

    blocks_out = []
    for bsi in block_split:
        blocks_out.extend(bsi)

    for b, sb in zip(grid, blocks_out):
        # sb.set_secondary()
        cons_avg = np.moveaxis(sb.conserved_avg, -1, 0)
        b.set_conserved(cons_avg)


    mdot_in = 0.
    for patch in grid.inlet_patches:
        Cm, A, _ = patch.get_cut().mix_out()
        mdot_in += Cm.rho * Cm.Vm * A

    mdot_out = 0.
    for patch in grid.outlet_patches:
        Cm, A, _ = patch.get_cut().mix_out()
        mdot_out += Cm.rho * Cm.Vm * A

    logger.info(f'Mass flow error: {(mdot_in/mdot_out-1.)*100.:.1f}%')


    dUlog = np.array(dUlog)
    dUlog /= dUlog[0]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.semilogy(dUlog)
    ax.legend((r'$\rho$',r'$\rho V_x$',r'$\rho V_r$',r'$\rho r V_\theta$',r'$\rho e$'))

    ip, jp, kp = np.unravel_index(np.argmax(blocks_out[0].dU1[...,-1]),blocks_out[0].dU1[...,-1].shape)
    jp = grid[0].shape[1]//2

    fig, ax = plt.subplots()
    for b in grid:
        ni, nj, nk = b.shape
        c = b[:,nj//2,0].squeeze()
        ax.plot(c.x, c.P, '-')
        c = b[:,nj//2,nk//2].squeeze()
        ax.plot(c.x, c.P, '-')
        # c = b[:,nj//2,1].squeeze()
        # c = b[:,nj//2,1].squeeze()
        # ax.plot(c.x, c.P, '-')
        # c = b[:,nj//2,-2].squeeze()
        # ax.plot(c.x, c.P, '-')
        c = b[:,nj//2,-1].squeeze()
        ax.plot(c.x, c.P, '-')
        ax.set_title('P')


    # wallk = blocks[0].wall_indicators[2]
    # print(wallk.shape)
    # fig, ax = plt.subplots()
    # ax.plot(wallk[:,nj//2, 0],'o-')
    # ax.plot(wallk[:,nj//2, 2],'+-')
    # ax.plot(wallk[:,nj//2, -1],'^-')
    # ax.plot(wallk[:,nj//2, -3],'x-')
    # plt.show()
    # quit()

    fig, ax = plt.subplots()

    for b, sb in zip(grid, blocks):


        # get face-centered x,rt
        ni, nj, nk = b.shape
        xrtf = [
            np.empty((ni, nj - 1, nk - 1, 3), order="F", dtype=typ),
            np.empty((ni - 1, nj, nk - 1, 3), order="F", dtype=typ),
            np.empty((ni - 1, nj - 1, nk, 3), order="F", dtype=typ),
        ]
        xrtn = np.asfortranarray(np.stack((b.x,b.r, b.rt),axis=-1)).astype(typ)
        node_to_face(xrtn, *xrtf)
        print(xrtn.shape)
        ii = 2
        print(xrtf[ii].shape)
        print(sb.wall_indicators[ii].shape)
        xface = xrtf[ii][...,0][sb.wall_indicators[ii]==1]
        rface = xrtf[ii][...,1][sb.wall_indicators[ii]==1]
        rtface = xrtf[ii][...,2][sb.wall_indicators[ii]==1]

        c = b[:,jp,:].squeeze()
        rmean = c.r.mean()
        ikeep = np.abs(rface/rmean-1.)<0.007
        if ikeep.any():
            xface = xface[ikeep]
            rface = rface[ikeep]
            rtface = rtface[ikeep]
            ax.plot(xface, rtface, 'bo')

        ni, nj, nk = b.shape
        ax.plot(c.x, c.rt, 'k-',lw=0.2)
        ax.plot(c.x.T, c.rt.T, 'k-',lw=0.2)
        # ax.plot(
        ax.axis('equal')
        ax.grid('P')

    plt.show()

def get_geom(b):
    # Areas and volumes
    dAi = np.asfortranarray(np.moveaxis(b.dAi, 0, -1))
    dAj = np.asfortranarray(np.moveaxis(b.dAj, 0, -1))
    dAk = np.asfortranarray(np.moveaxis(b.dAk, 0, -1))
    vol = np.asfortranarray(b.vol)

    # Shortest side length
    dli = turbigen.util.vecnorm(b.dli)
    dlj = turbigen.util.vecnorm(b.dlj)
    dlk = turbigen.util.vecnorm(b.dlk)
    dli = 0.25 * (
        dli[:, :-1, :-1] + dli[:, :-1, 1:] + dli[:, 1:, :-1] + dli[:, :-1, :-1]
    )
    dlj = 0.25 * (
        dlj[:-1, :, :-1] + dlj[:-1, :, 1:] + dlj[1:, :, :-1] + dlj[:-1, :, :-1]
    )
    dlk = 0.25 * (
        dlk[:-1, :-1, :] + dlk[:-1, 1:, :] + dlk[1:, :-1, :] + dlk[:-1, :-1, :]
    )
    dlmin = np.asfortranarray(np.minimum(dli, dlj, dlk))

    # Wall locations
    wall = get_wall(b)

    # Reference frame angular velocity
    Omega = b.Omega.mean()

    return dAi, dAj, dAk, vol, dlmin, wall, Omega


# def get_timestep(b, dlmin, CFL):
#     ni, nj, nk = b.shape
#     Va_node = np.asfortranarray(np.stack((b.V, b.a), axis=-1))
#     Va_cell = np.empty((ni - 1, nj - 1, nk - 1, 2), order="F")
#     node_to_cell(Va_node, Va_cell)
#     Vref = Va_cell[..., 0]
#     aref = Va_cell[..., 1]
#     dt = CFL * dlmin / (aref + Vref)
#     return np.asfortranarray(dt)


def get_wall(b, trim=1):
    # Find logical indices that zero the fluxes on wall faces
    thresh = 0.99  # To allow for floating point error

    # Preallocate face indices
    ni, nj, nk = b.shape
    wf = [
        np.empty((ni, nj - 1, nk - 1, 1), order="F", dtype=typ),
        np.empty((ni - 1, nj, nk - 1, 1), order="F", dtype=typ),
        np.empty((ni - 1, nj - 1, nk, 1), order="F", dtype=typ),
    ]
    wn = np.asfortranarray(np.expand_dims(b.get_wall(trim), -1).astype(typ))

    # Calculate nodal values of wall indicator
    node_to_face(wn, *wf)

    wfl = [(wfn > thresh)[..., 0].astype(np.int8) for wfn in wf]

    return wfl
