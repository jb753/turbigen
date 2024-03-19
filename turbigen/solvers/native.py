import numpy as np
import turbigen.util
import turbigen.fluid
import turbigen.flowfield
import turbigen.grid
from turbigen.compiled import (
    smooth,
    node_to_cell,
    node_to_face,
    step,
    calculate_secondary
)
from timeit import default_timer as timer
from mpi4py import MPI
import logging

logger = turbigen.util.make_logger()

logger.setLevel(level=logging.DEBUG)


# class NativeConfig(BaseSolver):
#     """Settings with default values for the TS4 solver."""

#     _name = "Native"


def flatwhere(x):
    return np.where(x.flat)[0]


nstep_dt = 100
nstep_log = 50
nstep = 5000
nrk = 4
dampin = 10.0
rfin = 0.2
rfin1 = 1.0 - rfin

sfin = 0.001
fac_2nd = 0.2
CFL = 0.4
sf = CFL * sfin
sf2 = sf * fac_2nd
sf4 = sf * (1.0 - fac_2nd)


class SolverBlock:
    """Hold just the data we need for a CFD solution."""

    def __init__(self, block):
        """Initialise from a standard Block object."""

        # Primaries
        self.conserved = np.asfortranarray(np.moveaxis(block.conserved, 0, -1)).astype(np.single)

        self.ho = np.asfortranarray(block.ho).astype(np.single)
        self.P = np.asfortranarray(block.P).astype(np.single)

        self.halfVsq = np.asfortranarray(0.5 * block.V**2).astype(np.single)
        self.u = np.asfortranarray(block.u).astype(np.single)

        self.x = np.asfortranarray(block.x).astype(np.single)
        self.r = np.asfortranarray(block.r).astype(np.single)
        self.t = np.asfortranarray(block.t).astype(np.single)

        # Geometry
        self.r = np.asfortranarray(block.r).astype(np.single)
        self.dAi = np.asfortranarray(np.moveaxis(block.dAi, 0, -1)).astype(np.single)
        self.dAj = np.asfortranarray(np.moveaxis(block.dAj, 0, -1)).astype(np.single)
        self.dAk = np.asfortranarray(np.moveaxis(block.dAk, 0, -1)).astype(np.single)
        self.vol = np.asfortranarray(block.vol).astype(np.single)
        self.dlmin = np.asfortranarray(block.dlmin).astype(np.single)
        self.Omega = block.Omega.mean().astype(np.single)

        self.dU1 = self.conserved.copy(order="F").astype(np.single) * np.nan
        self.dU2 = self.conserved.copy(order="F").astype(np.single) * np.nan
        self._flag_scree = False

        # Get wall indicators
        # These are three arrays of shape
        #   i faces: (ni, nj-1, nk-1)
        #   j faces: (ni-1, nj, nk-1)
        #   k faces: (ni-1, nj-1, nk)
        # equal to one if the face is a wall, zero otherwise
        self.wall_indicators = [np.asfortranarray(w) for w in get_wall(block)]

        # Convert wall indicators to wall indices
        # Which are indices into the flattend face arrays
        self.walls = [flatwhere(w > 0.99) for w in self.wall_indicators]

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

    def set_inlets(self):
        """Set conserved variables on inlets by relaxing density changes."""

        # Change inlet patches
        for patch, state in zip(self.inlets, self.state_inlets):

            # Expand patch data
            ind, Po, To, Alpha, Beta, rhoo, ho, r = patch

            # Relax changes in density
            rho_now = (
                rfin * self.conserved[..., 0].ravel(order="F")[ind] + rfin1 * state.rho
            )

            # Check for flow reversal
            rho_now[rho_now > rhoo] = rhoo * 0.99999

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
            self.conserved[..., 0].ravel(order="F")[ind] = rho_now  # rho
            self.conserved[..., 1].ravel(order="F")[ind] = rho_now * Vxin  # rhoVx
            self.conserved[..., 2].ravel(order="F")[ind] = rho_now * Vrin  # rhoVr
            self.conserved[..., 3].ravel(order="F")[ind] = rho_now * r * Vtin  # rhorVt
            self.conserved[..., 4].ravel(order="F")[ind] = rho_now * (
                u + 0.5 * Vinsq
            )  # rhoe

            # Reset pressure and hstag on inlet
            self.ho.ravel(order="F")[ind] = h + 0.5 * Vin**2
            self.P.ravel(order="F")[ind] = P

    def set_outlets(self):
        """Set static pressure on outlets."""
        for outlet in self.outlets:
            self.P.ravel(order="F")[outlet[0]] = outlet[1]

    def set_timestep(self):
        Vx = self.conserved[..., 1] / self.conserved[..., 0]
        Vr = self.conserved[..., 2] / self.conserved[..., 0]
        Vt = self.conserved[..., 3] / self.conserved[..., 0] / self.r
        V = np.sqrt(Vx**2 + Vr**2 + Vt**2)

        a = self.state.a

        ni, nj, nk = self.r.shape

        Va_node = np.asfortranarray(np.stack((V, a), axis=-1)).astype(np.single)
        Va_cell = np.empty((ni - 1, nj - 1, nk - 1, 2), order="F", dtype=np.single)
        node_to_cell(Va_node, Va_cell)
        Vref = Va_cell[..., 0]
        aref = Va_cell[..., 1]
        self.dt = CFL * self.dlmin / (aref + Vref)

    # @profile
    def step(self, start_flag):

        step(
            self.conserved,
            self.P,
            self.ho,
            self.r,
            self.Omega,
            *self.wall_indicators,
            self.dt,
            self.dAi,
            self.dAj,
            self.dAk,
            self.vol,
            self.dU1,
            self.dU2,
            start_flag,
        )

        self.smooth()

        calculate_secondary(self.r, self.conserved, self.halfVsq, self.u)

        self.state.set_rho_u(self.conserved[..., 0], self.u)

        self.ho[:] = self.state.h + self.halfVsq
        self.P[:] = self.state.P

    def smooth(self):
        smooth(self.conserved, sf2, sf4)


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


def send_slave(block_split, procids, periodics):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    for iproc in range(1, size):
        comm.send(block_split[iproc], dest=iproc)

    for iproc in range(1, size):
        comm.send(periodics, dest=iproc)


def run_slave(blocks=None, periodics_all=None, nodes=None):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if blocks is None:
        blocks = comm.recv()
        periodics_all = comm.recv()
        master_flag = False
    else:
        master_flag = True

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

    # Start the main time stepping loop
    for istep in range(nstep):

        # Update periodic boundaries
        for patch in periodics:
            pid, bid, procid, ind, nxbid, nxprocid, nxind = patch
            count = len(ind)

            # Just set the periodic if on same rank
            if nxprocid == rank:
                set_periodic(
                    blocks[bid_local[bid]], blocks[bid_local[nxbid]], ind, nxind
                )
            # Otherwise, communication is needed
            else:
                conserved = blocks[bid_local[bid]].conserved.ravel(order="F")
                nxconserved = np.empty((count,), dtype=np.single)
                # If our rank is lower than next rank, send first
                if rank < nxprocid:
                    comm.Send(
                        [conserved[ind], count, MPI.REAL4], dest=nxprocid, tag=pid
                    )
                    comm.Recv([nxconserved, count, MPI.REAL4], source=nxprocid, tag=pid)
                # If our rank is higher than next rank, recieve first
                else:
                    comm.Recv([nxconserved, count, MPI.REAL4], source=nxprocid, tag=pid)
                    comm.Send(
                        [conserved[ind], count, MPI.REAL4], dest=nxprocid, tag=pid
                    )
                conserved[ind] = 0.5 * (conserved[ind] + nxconserved)

        # Loop over blocks
        for iblock in range(nblock):

            sb = blocks[iblock]

            # Update time stegs
            if not np.mod(istep, nstep_dt):
                sb.set_timestep()

            sb.set_inlets()

            sb.set_outlets()

            start_flag = 1 if istep == 0 else 0
            sb.step(start_flag)

            # sb.smooth()

        if not np.mod(istep, nstep_log) and istep > 0 and master_flag:
            log_line = f"{istep}: {np.abs(blocks[0].dU1).mean(axis=(0,1,2))}"
            logger.info(log_line)
            ten = timer()
            tpnps = (ten - tstart) / nodes / nstep_log
            logger.info(f"tpnps={tpnps:.3e}")
            tstart = ten

    if master_flag:
        return blocks
    else:
        comm.send(blocks, dest=0)


def run(grid, settings={}, machine=None):

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
    send_slave(block_split, procids, periodics)

    logger.info("Starting the main time-stepping loop...")
    block_split[0] = run_slave(block_split[0], periodics, nodes)

    for iproc in range(1, size):
        block_split[iproc] = comm.recv(source=iproc)

    blocks_out = []
    for bsi in block_split:
        blocks_out.extend(bsi)

        # mdot_in = 0.0
        # for patch in grid.inlet_patches:
        #     Cm, Ann, _ = patch.get_cut().mix_out()
        #     mdot_in += Cm.rho * Cm.Vm * Ann

        # mdot_out = 0.0
        # for patch in grid.outlet_patches:
        #     Cm, Ann, _ = patch.get_cut().mix_out()
        #     mdot_out += Cm.rho * Cm.Vm * Ann
        #     print(
        #         f"mass flows {mdot_in:.3e}, {mdot_out:.3e}, "
        #         f"err={(mdot_in/mdot_out-1.)*100:.1f}%"
        #     )
        #     mdot_all[istep // nstep_log] = (mdot_in, mdot_out)

    for b, sb in zip(grid, blocks_out):
        b.set_conserved(np.moveaxis(sb.conserved, -1, 0))


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


def get_timestep(b, dlmin):
    ni, nj, nk = b.shape
    Va_node = np.asfortranarray(np.stack((b.V, b.a), axis=-1))
    Va_cell = np.empty((ni - 1, nj - 1, nk - 1, 2), order="F")
    node_to_cell(Va_node, Va_cell)
    Vref = Va_cell[..., 0]
    aref = Va_cell[..., 1]
    dt = CFL * dlmin / (aref + Vref)
    return np.asfortranarray(dt)


def get_wall(b):
    # Find logical indices that zero the fluxes on wall faces
    thresh = 0.99  # To allow for floating point error

    # Preallocate face indices
    ni, nj, nk = b.shape
    wf = [
        np.empty((ni, nj - 1, nk - 1, 1), order="F", dtype=np.single),
        np.empty((ni - 1, nj, nk - 1, 1), order="F", dtype=np.single),
        np.empty((ni - 1, nj - 1, nk, 1), order="F", dtype=np.single),
    ]
    wn = np.asfortranarray(np.expand_dims(b.get_wall(), -1).astype(np.single))

    # Calculate nodal values of wall indicator
    node_to_face(wn, *wf)

    wfl = [(wfn > thresh)[..., 0].astype(np.int8) for wfn in wf]

    return wfl
