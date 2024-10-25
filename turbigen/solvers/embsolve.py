import numpy as np
from copy import copy

import turbigen.util

import logging
from dataclasses import dataclass

# from turbigen.embsolve import embsolve
from pathlib import Path
from timeit import default_timer as timer

import turbigen.flowfield
import turbigen.fluid
import turbigen.grid
from turbigen.solvers.base import BaseSolver
from turbigen.solvers.embsolvec import embsolve

util = turbigen.util
logger = turbigen.util.make_logger()

logger.setLevel(level=logging.INFO)

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    mpi_single = MPI.REAL4
    mpi_double = MPI.REAL8
except ImportError:
    size = 1
    rank = 0
    comm = None
    mpi_single = None
    mpi_double = None


@dataclass
class Config(BaseSolver):
    """Settings with default values for the native solver."""

    _name = "Native"

    workdir: Path = None

    smooth4: float = 0.01
    """Fourth-order smoothing factor."""

    smooth2_adapt: float = 1.0
    """Second-order smoothing factor, adaptive on pressure."""

    smooth2_const: float = 0.0
    """Second-order smoothing factor, constant throughout the flow."""

    CFL: float = 0.65
    """Courant--Friedrichs--Lewy number, time step normalised by local wave
    speed and cell size. Reduced values are more stable but slower to
    converge."""

    n_step: int = 5000
    """Number of time steps to run for."""

    n_step_dt: int = 10
    """Number of time steps between updates of the local time step."""

    n_step_log: int = 100
    """Number of time steps between log prints."""

    n_step_avg: int = 1
    """Number of time steps to average over."""

    n_step_ramp: int = 250
    """Number of time steps to ramp smoothing and damping."""

    n_loss: int = 5
    """Number of time steps between viscous force updates."""

    nstep_damp: int = 500
    """Number of steps to apply damping."""

    damping_factor: float = 25.0
    """Negative feedback to damp down high residuals. Lower values are more stable."""

    Pr_turb: float = 1.0
    """Turbulent Prandtl number."""

    xllim_pitch: float = 0.03

    precision: int = 1

    i_scheme: int = 1

    i_loss: int = 1

    i_exit: int = 1
    i_inlet: int = 1
    K_exit: float = 0.9
    K_inlet: float = 0.7

    plot_conv: bool = False
    print_conv: bool = True

    tauw_lam_mult: float = 1.0
    tauw_turb_mult: float = 1.0

    fmgrid: float = 0.2
    multigrid: tuple = (2, 2, 2)


def get_dw(block, typ):
    # Cell height in each of i,j,k dirns
    dli = turbigen.util.vecnorm(block.dli)
    dlj = turbigen.util.vecnorm(block.dlj)
    dlk = turbigen.util.vecnorm(block.dlk)

    ni, nj, nk = block.shape

    dwi = np.asfortranarray(np.zeros((ni, nj - 1, nk - 1), dtype=typ))
    dwi[0, :, :] = util.node_to_face2(dli[0, :, :])
    dwi[-1, :, :] = util.node_to_face2(dli[-1, :, :])

    dwj = np.asfortranarray(np.zeros((ni - 1, nj, nk - 1), dtype=typ))
    dwj[:, 0, :] = util.node_to_face2(dlj[:, 0, :])
    dwj[:, -1, :] = util.node_to_face2(dlj[:, -1, :])

    dwk = np.asfortranarray(np.zeros((ni - 1, nj - 1, nk), dtype=typ))
    dwk[:, :, 0] = util.node_to_face2(dlk[:, :, 0])
    dwk[:, :, -1] = util.node_to_face2(dlk[:, :, -1])

    return dwi, dwj, dwk


def to_fort_type(x, typ):
    if x.ndim > 3:
        x2 = np.moveaxis(x, 0, -1)
    else:
        x2 = x
    return np.asfortranarray(x2).astype(typ)


class SolverBlock:
    """Hold just the data we need for a CFD solution."""

    def __init__(self, block, conf):
        """Initialise from a standard Block object."""

        # Select precision
        if conf.precision == 1:
            typ = np.float32
            mpi_typ = mpi_single
        else:
            typ = np.float64
            mpi_typ = mpi_double

        def to_fort(x):
            return to_fort_type(x, typ)

        # Primaries
        self.cons = to_fort(block.conserved)

        self.conf = conf
        self.Nb = block.Nb

        self.mu = typ(block.mu)
        self.cp = typ(block.cp)
        self.Pr_turb = typ(conf.Pr_turb)

        self.ho = to_fort(block.ho)
        self.P = to_fort(block.P)
        self.Pref = to_fort(self.P.mean())

        self.halfVsq = to_fort(0.5 * block.V**2)
        self.u = to_fort(block.u)
        self.T = to_fort(block.T)

        self.dw = get_dw(block, typ)
        self.pitch = block.pitch

        # Geometry
        self.x = to_fort(block.x)
        self.r = to_fort(block.r)
        self.t = to_fort(block.t)
        self.rf = [to_fort(r) for r in block.r_face]
        self.rc = to_fort(block.r_cell)

        self.dAi = to_fort(block.dAi_new)
        self.dAj = to_fort(block.dAj_new)
        self.dAk = to_fort(block.dAk_new)

        self.Vxrt = to_fort(block.Vxrt)
        self.Omega = block.Omega.mean().astype(typ).item()
        xllim = (
            block.pitch * 0.5 * (block.r.max() + block.r.min()) * self.conf.xllim_pitch
        )

        self.cons_avg = self.cons.copy(order="F").astype(np.double) * 0.0

        Omega = block.Omega.mean()
        self.U = to_fort(Omega * block.r)
        self.Uf = [to_fort(Omega * r) for r in block.r_face]

        # Residual storage
        ni, nj, nk = block.shape
        self.fb = np.zeros((ni - 1, nj - 1, nk - 1, 5), order="F", dtype=typ)
        self.dUc = np.zeros((ni - 1, nj - 1, nk - 1, 5, 2), order="F", dtype=typ)
        self.dUn = np.zeros((ni, nj, nk, 5), order="F", dtype=typ)

        xlength = np.asfortranarray(np.clip(block.w, 0.0, xllim)).astype(typ)
        xlength = (0.41 * xlength) ** 2.0
        self.xlength = to_fort(np.zeros((ni - 1, nj - 1, nk - 1)))
        embsolve.node_to_cell(xlength, self.xlength)

        # Get indices for multigrid
        self.ijk_multigrid = (
            get_multigrid_indices((ni - 1, nj - 1, nk - 1), conf.multigrid) + 1
        )
        self.vol = get_multigrid_volumes(
            to_fort(block.vol_new), self.ijk_multigrid, typ
        )
        self.dlmin = get_multigrid_lengths(block, conf.multigrid, typ)
        self.dt_vol = self.dlmin * 0.0

        # Get wall indices
        # These are ijk (3, n) for each of ifaces, jfaces, kfaces, nodes
        # Note 1-indexed for Fortran
        *self.ijk_wall_face, self.ijk_wall_node = [
            np.asfortranarray(np.argwhere(wall).T + 1).astype(np.int16)
            for wall in block.get_wall()
        ]

        # Get indices for wall functions
        *self.ijk_wall_face_slip, _ = [
            np.asfortranarray(np.argwhere(wall).T + 1).astype(np.int16)
            for wall in block.get_wall(ignore_slip=True)
        ]

        # Get indices to first node off the wall
        iwall1, jwall1, kwall1 = [ijk + 0 for ijk in self.ijk_wall_face_slip]
        iwall1[0, iwall1[0, :] == ni] -= 1
        jwall1[1, jwall1[1, :] == nj] -= 1
        kwall1[2, kwall1[2, :] == nk] -= 1

        # Get wall cell size
        self.dw_face = [
            embsolve.get_by_ijk(to_fort(dl), ijk)
            for dl, ijk in zip(block.get_dwall(), [iwall1, jwall1, kwall1])
        ]

        # Get wall area magnitudes
        dAijk = [
            np.sqrt((self.dAi**2).sum(axis=-1)),
            np.sqrt((self.dAj**2).sum(axis=-1)),
            np.sqrt((self.dAk**2).sum(axis=-1)),
        ]

        self.dA_face = [
            embsolve.get_by_ijk(dA, ijk)
            for dA, ijk in zip(dAijk, self.ijk_wall_face_slip)
        ]

        # Put dummy values in zero-length ijk
        for n in range(3):
            ijk = self.ijk_wall_face_slip[n]
            if ijk.shape[-1] == 0:
                ijkdum = np.asfortranarray(-np.ones((3, 1))).astype(np.int16)
                self.ijk_wall_face_slip[n] = ijkdum
                self.dw_face[n] = to_fort(np.ones((1,)))
                self.dA_face[n] = to_fort(np.ones((1,)))

        # Get nodal smoothing scaling factors

        # Wall length scales at nodes
        dli = turbigen.util.vecnorm(block.dli)
        dlj = turbigen.util.vecnorm(block.dlj)
        dlk = turbigen.util.vecnorm(block.dlk)

        # Distribute length scales to cells
        dli = np.stack(
            (
                dli[:, :-1, :-1],
                dli[:, 1:, :-1],
                dli[:, :-1, 1:],
                dli[:, 1:, 1:],
            )
        ).mean(axis=0)
        dlj = np.stack(
            (
                dlj[:-1, :, :-1],
                dlj[1:, :, :-1],
                dlj[:-1, :, 1:],
                dlj[1:, :, 1:],
            )
        ).mean(axis=0)
        dlk = np.stack(
            (
                dlk[:-1, :-1, :],
                dlk[1:, :-1, :],
                dlk[:-1, 1:, :],
                dlk[1:, 1:, :],
            )
        ).mean(axis=0)

        # Smoothing scale factors in each volume
        Lref = block.vol_new ** (1 / 3)
        L = to_fort(
            np.stack(
                (
                    dli / Lref,
                    dlj / Lref,
                    dlk / Lref,
                ),
                axis=0,
            )
        )
        Ls = L.sum(axis=-1, keepdims=True)
        L = L / Ls * 3.0

        # Now distribute to nodes
        self.L = to_fort(np.ones((3, ni, nj, nk)))
        embsolve.cell_to_node(L, self.L, ni, nj, nk, 3)
        # Disable scaling
        # self.L = to_fort(np.ones((3, ni, nj, nk))/3.)

        self.tau = [
            to_fort(np.zeros((6, ni, nj - 1, nk - 1))),
            to_fort(np.zeros((6, ni - 1, nj, nk - 1))),
            to_fort(np.zeros((6, ni - 1, nj - 1, nk))),
        ]

        self.bconds = []
        self.bconds += [InletBoundary(patch) for patch in block.inlet_patches]
        self.bconds += [OutletBoundary(patch) for patch in block.outlet_patches]

        if isinstance(block, turbigen.grid.PerfectBlock):
            self.state = turbigen.fluid.PerfectState(
                shape=block.shape, order="F", typ=typ
            )
            self.state.gamma = typ(block.gamma)
            self.state.cp = typ(block.cp)
            self.state.mu = typ(block.mu)
            self.state.set_rho_u(block.rho, block.u)
            self.state.set_Tu0(block.Tu0)

        else:
            raise NotImplementedError()

        del to_fort

    def set_timestep(self, CFL, relax=0.0):
        embsolve.set_timesteps(
            self.dt_vol,
            self.vol,
            self.state.a,
            self.Vxrt,
            self.U,
            self.dlmin,
            self.ijk_multigrid,
            CFL,
        )

    def residual(self, fmgrid, damp, ischeme):
        embsolve.residual(
            self.cons,
            self.Vxrt,
            self.P,  # Only pressure differences matter
            self.Pref,
            self.ho,
            self.fb,
            self.U,
            *self.Uf,
            self.r,
            *self.rf,
            self.dAi,
            self.dAj,
            self.dAk,
            self.vol,
            self.dt_vol,
            *self.ijk_wall_face,
            self.ijk_multigrid,
            fmgrid,
            damp,
            self.dUc,
            self.dUn,
            ischeme,
        )

    def step(self, istep, ischeme):
        embsolve.step(
            self.cons,
            self.dUc,
            self.dUn,
            ischeme,
        )

    def set_secondary(self):
        embsolve.secondary(self.r, self.cons, self.Vxrt, self.halfVsq, self.u)
        self.state.set_rho_u(self.cons[..., 0], self.u)
        self.ho[:] = self.state.h + self.halfVsq
        self.P[:] = self.state.P
        self.T[:] = self.state.T

    def smooth(self, sf2, sf4, sf2min):
        embsolve.smooth(self.cons, self.P, self.L, sf4, sf2, sf2min)

    def damp(self, fdamp):
        embsolve.damp(self.dU1, fdamp)

    def set_viscous_stress(self):
        embsolve.shear_stress(
            self.cons,
            self.Vxrt,
            self.T,
            self.mu,
            self.cp,
            self.Pr_turb,
            self.xlength,
            self.vol[..., 0],
            self.dAi,
            self.dAj,
            self.dAk,
            self.Omega,
            self.r,
            self.rc,
            *self.rf,
            *self.ijk_wall_face_slip,
            *self.dw_face,
            *self.dA_face,
            self.fb,
        )


class Periodic:
    """Encapsulate information needed for periodic boundary."""

    def __init__(self, patch, pid, procids, typ):
        match = patch.match
        perm, flip = match.get_match_perm_flip()

        self.pid = pid
        self.bid = patch.block.grid.index(patch.block)
        self.nxbid = match.block.grid.index(match.block)

        self.ijk = ijk = np.asfortranarray(patch.get_indices().reshape(3, -1)).astype(
            np.int16
        )
        self.nxijk = nxijk = np.asfortranarray(
            match.get_indices(perm, flip).reshape(3, -1)
        ).astype(np.int16)

        # Check the coords match
        b1 = patch.block
        b2 = patch.match.block

        Npts = ijk.shape[-1]
        for n in range(Npts):
            ijknow = tuple(ijk[:, n])
            nxijknow = tuple(nxijk[:, n])

            assert np.isclose(
                b1.x[ijknow],
                b2.x[nxijknow],
            )
            assert np.isclose(
                b1.r[ijknow],
                b2.r[nxijknow],
            )

            t1 = np.mod(b1.t[ijknow], b1.pitch) + 1.0
            t2 = np.mod(b2.t[nxijknow], b2.pitch) + 1.0
            assert np.allclose(t1, t2)

        # Check we have the correct number of points
        npt = patch.get_cut().to_unstructured().shape[0]
        assert ijk.shape[1] == npt
        assert nxijk.shape[1] == npt
        self.N = npt * 5

        # Check the indices are in correct range
        assert ijk.min() >= 0
        assert ijk[0].max() < b1.ni
        assert ijk[1].max() < b1.nj
        assert ijk[2].max() < b1.nk

        assert nxijk.min() >= 0
        assert nxijk[0].max() < b2.ni
        assert nxijk[1].max() < b2.nj
        assert nxijk[2].max() < b2.nk

        # Add one for 1-based Fortran indices
        self.ijk += 1
        self.nxijk += 1

        # Store required data
        self.procid = procids[self.bid]
        self.nxprocid = procids[self.nxbid]

        self.buffer = np.empty((self.N), order="F").astype(typ)
        self.nxbuffer = np.empty((self.N), order="F").astype(typ)

    def reversed(self):
        p = copy(self)
        p.bid, p.nxbid = p.nxbid, p.bid
        p.procid, p.nxprocid = p.nxprocid, p.procid
        p.ijk, p.nxijk = p.nxijk, p.ijk
        return p

    def setup_communication(self, comm, mpi_typ):
        if self.procid == self.nxprocid:
            return
        self.Send = comm.Send_init(
            buf=[self.buffer, self.N, mpi_typ],
            dest=self.nxprocid,
            tag=self.pid,
        )
        self.Recv = comm.Recv_init(
            buf=[self.nxbuffer, self.N, mpi_typ],
            source=self.nxprocid,
            tag=self.pid,
        )


def get_mixers(grid, procids, typ):
    mixers = []
    seen = []
    pid = 0
    for patch in grid.mixing_patches:
        if patch in seen:
            continue
        else:
            seen.append(patch)
            seen.append(patch.match)
        mixers.append(MixingBoundary(patch, pid, procids, typ))
        pid += 1
        mixers.append(MixingBoundary(patch.match, pid, procids, typ))
        pid += 1
        mixers[-2].nxpid = mixers[-1].pid
        mixers[-1].nxpid = mixers[-2].pid
        if mixers[0].procid == mixers[1].procid:
            mixers[0].nxbuffer = mixers[1].buffer
            mixers[1].nxbuffer = mixers[0].buffer
    return mixers


def get_periodics(g, procids, typ):
    periodics = []
    seen = []
    pid = 0

    for patch in g.periodic_patches:
        if patch in seen:
            continue
        else:
            seen.append(patch)
            seen.append(patch.match)

        periodics.append(Periodic(patch, pid, procids, typ))
        pid += 1

    return periodics


def send_slave(block_split, procids, periodics, mixers):
    for iproc in range(1, size):
        comm.send(block_split[iproc], dest=iproc)

    comm.Barrier()

    for iproc in range(1, size):
        comm.send(periodics, dest=iproc)

    comm.Barrier()

    for iproc in range(1, size):
        comm.send(mixers, dest=iproc)

    comm.Barrier()


def exchange_mixing(blocks, bid_local, mixers, log):
    # Prepare to recieve into away buffers
    for mixer in mixers:
        if not mixer.nxprocid == rank:
            mixer.Recv.Start()

    # Populate the home buffer with pitchwise-averaged
    # fluxes and conserved vars and send away
    for mixer in mixers:
        b1 = blocks[bid_local[mixer.bid]]
        mixer.pull(b1)
        mixer.fill_buffer()
        if not mixer.nxprocid == rank:
            mixer.Send.Start()

    # We now use populated buffers to get flux differences and
    # side-averaged mean flow
    for mixer in mixers:
        # Wait for communication before unpacking the buffers form each side
        if not mixer.nxprocid == rank:
            mixer.Recv.Wait()

        b1 = blocks[bid_local[mixer.bid]]
        mixer.apply(b1, log)


def exchange_periodic(blocks, bid_local, periodics):
    # Update periodic boundaries

    # Prepare to recieve into away buffers
    for patch in periodics:
        if not patch.nxprocid == rank:
            patch.Recv.Start()

    # Loop to populate home buffer and send away buffer
    for patch in periodics:
        # Load flow field into our buffer
        b1 = blocks[bid_local[patch.bid]].cons
        patch.buffer[:] = embsolve.get_by_ijk(b1, patch.ijk)

        # Can directly set away buffer if same rank
        if patch.nxprocid == rank:
            b2 = blocks[bid_local[patch.nxbid]].cons
            patch.nxbuffer[:] = embsolve.get_by_ijk(b2, patch.nxijk)

        # Otherwise, communication is needed
        else:
            patch.Send.Start()

    # Once the communication completes, take average of home
    # and away buffers and assign back to grid
    for patch in periodics:
        # Wait for communication if needed
        if not patch.nxprocid == rank:
            patch.Recv.Wait()

        # Take average and assign to home block
        bavg = 0.5 * (patch.buffer + patch.nxbuffer)
        b1 = blocks[bid_local[patch.bid]].cons
        embsolve.set_by_ijk(b1, bavg, patch.ijk)

        # If we are on same proc, then we have to set other side as well
        if patch.nxprocid == rank:
            b2 = blocks[bid_local[patch.nxbid]].cons
            embsolve.set_by_ijk(b2, bavg, patch.nxijk)


def run_slave(blocks=None, periodics_all=None, mixers_all=None, nodes=None, conf=None):
    if blocks is None:
        blocks = comm.recv()
        comm.Barrier()
        periodics_all = comm.recv()
        comm.Barrier()
        mixers_all = comm.recv()
        comm.Barrier()
        master_flag = False
    else:
        master_flag = True
        dUlog = []
        Yslog = []
        merrlog = []

    # Calculate smoothing and inlet relaxation scaled by CFL
    CFL_ref = 0.7
    conf = blocks[0].conf
    sf2 = conf.smooth2_adapt * conf.CFL / CFL_ref
    sf4 = conf.smooth4 * conf.CFL / CFL_ref
    sf2min = conf.smooth2_const * conf.CFL / CFL_ref
    K_inlet = conf.K_inlet * conf.CFL / CFL_ref
    K_exit = conf.K_exit  # * conf.CFL / CFL_ref
    rfin = 0.2

    if blocks[0].conf.precision == 1:
        typ = np.float32
        mpi_typ = mpi_single
    else:
        typ = np.float64
        mpi_typ = mpi_double

    # Only keep relevent periodics
    # And rearrange the periodics so that foreign procid is always nx
    periodics = []
    for patch in periodics_all:
        if patch.procid == rank:
            periodics.append(patch)
        elif patch.nxprocid == rank:
            periodics.append(patch.reversed())
    mixers = []
    for patch in mixers_all:
        if patch.procid == rank:
            mixers.append(patch)

    # Setup MPI communication
    for patch in periodics + mixers:
        patch.setup_communication(comm, mpi_typ)

    bids = [b.bid for b in blocks]

    # Lookup of local bid from global bid
    bid_local = {bid: ibid for ibid, bid in enumerate(bids)}

    nblock = len(blocks)

    dUnow = np.empty((conf.n_step_log, nblock, 5))

    # Now integrate forward
    istep_avg = conf.n_step - conf.n_step_avg

    # Initialise a conservative time step
    for iblock in range(nblock):
        blocks[iblock].set_timestep(conf.CFL * 0.5)

    try:
        tstart = timer()
        tfirst = tstart + 0.0

        for iblock in range(nblock):
            blocks[iblock].set_secondary()

        # Start the main time stepping loop
        for istep in range(conf.n_step):
            # Ramping factors
            damping_ramp = np.interp(istep, [0, conf.n_step_ramp], [0.5, 1.0])
            smoothing_ramp = np.interp(istep, [0, conf.n_step_ramp], [2.0, 1.0])
            cfl_ramp = np.interp(istep, [0, conf.n_step_ramp], [0.5, 1.0])
            fmgrid_ramp = np.interp(istep, [0, conf.n_step_ramp], [0.0, 1.0])

            # Exchange conserved variables across periodic patches
            exchange_periodic(blocks, bid_local, periodics)

            # Update boundary conditions and calculate residual for all blocks
            for iblock in range(nblock):
                sb = blocks[iblock]

                # Update pressure, ho, velocities
                sb.set_secondary()

                # Accumulate time average
                if istep >= istep_avg:
                    sb.cons_avg += sb.cons / float(conf.n_step_avg)

                # Update time steps using current local Mach
                if not np.mod(istep, conf.n_step_dt):
                    sb.set_timestep(conf.CFL * cfl_ramp)

                # If this is a viscous calculation
                # Update the viscous forces every nloss time steps
                if not np.mod(istep, conf.n_loss) and conf.i_loss > 0:
                    sb.set_viscous_stress()

                # Damping factor for this time step
                if conf.damping_factor and (
                    istep < conf.nstep_damp or conf.nstep_damp < 0
                ):
                    damp = conf.damping_factor * damping_ramp
                else:
                    damp = 1e6

                # Sum fluxes for each cell and distribute to the nodes
                i_scheme = -1 if not istep else conf.i_scheme
                sb.residual(
                    conf.fmgrid * fmgrid_ramp,
                    damp,
                    i_scheme,
                )

                # Apply boundary conditions
                for bc in sb.bconds:
                    bc.apply(sb)

            # Exchange fluxes across mixing patches
            exchange_mixing(
                blocks, bid_local, mixers, not np.mod(istep, conf.n_step_log)
            )

            for iblock in range(nblock):
                sb = blocks[iblock]

                sb.cons += sb.dUn

                sb.smooth(
                    sf2 * smoothing_ramp, sf4 * smoothing_ramp, sf2min * smoothing_ramp
                )

            # Record residuals
            iilog = np.mod(istep - 1, conf.n_step_log)
            dUnow[iilog] = np.stack(
                [np.abs(b.dUc[..., 0].mean(axis=(0, 1, 2))) for b in blocks]
            )

            # Intermittently print convergence
            if (not np.mod(istep, conf.n_step_log)) and (istep > 0):
                # Send residuals to master proc
                if rank:
                    comm.send(dUnow, dest=0)

                else:
                    dUall = [
                        dUnow,
                    ]
                    for iproc in range(1, size):
                        dUall.append(comm.recv(source=iproc))

                    dUall = np.concatenate(dUall, axis=1)

                    ten = timer()
                    tpnps = (ten - tstart) / nodes / conf.n_step_log
                    tstart = ten

                    if conf.print_conv:
                        logger.info(f"{istep}: tpnps={tpnps:.3e}")
                        for ib, dU in enumerate(dUall.mean(axis=0)):
                            logger.info(
                                f"  block {ib}: "
                                f"{dU[0]:.2e} {dU[1]:.2e} {dU[2]:.2e} "
                                f"{dU[3]:.2e} {dU[4]:.2e}"
                            )

                    dUlognow = np.stack(dUall).mean(axis=1)
                    dUlog.append(dUlognow)

        tlast = timer()

    except KeyboardInterrupt:
        tlast = timer()
        for iblock in range(nblock):
            sb = blocks[iblock]
            sb.cons_avg = sb.cons

    if master_flag:
        tpnps = (tlast - tfirst) / nodes / conf.n_step
        logger.info(f"Elapsed time {tlast-tfirst:.2f}s")
        logger.info(f"Average tpnps={tpnps:.3e}")
        return blocks, dUlog, merrlog, Yslog, tpnps
    else:
        comm.send(blocks, dest=0)


def run(grid, conf, machine=None):
    if isinstance(conf, dict):
        conf = Config(**conf)

    if conf.skip:
        logger.info("Skipping, doing nothing.")
        return

    logger.info("Initialising native solver...")
    t1 = timer()

    nodes = np.sum([b.size for b in grid])

    # Select precision
    if conf.precision == 1:
        typ = np.float32
    else:
        typ = np.float64

    blocks = [SolverBlock(b, conf) for b in grid]
    for ib, b in enumerate(blocks):
        b.bid = ib

    logger.info(f"Patitioning onto {size} processors...")
    # procids is a list of length nblocks, of which processor is alocated to each block
    procids = grid.partition(size)
    periodics = get_periodics(grid, procids, typ)
    mixers = get_mixers(grid, procids, typ)

    # Split into lists for each procid
    block_split = []
    for iproc in range(size):
        block_split.append([])
        for ib, b in enumerate(blocks):
            if iproc == procids[ib]:
                block_split[-1].append(b)

    t2 = timer()
    logger.info(f"Elapsed time {t2-t1:.2f}s")

    if comm:
        logger.info("Sending data to processors...")
        tst = timer()
        send_slave(block_split, procids, periodics, mixers)
        ten = timer()
        logger.info(f"Elapsed time {ten-tst:.2f}s")

    logger.info("Starting the main time-stepping loop...")
    block_split[0], dUlog, merrlog, Yslog, tpnps = run_slave(
        block_split[0], periodics, mixers, nodes
    )

    logger.info("Recieving data from processors...")
    tst = timer()
    for iproc in range(1, size):
        block_split[iproc] = comm.recv(source=iproc)
    ten = timer()
    logger.info(f"Elapsed time {ten-tst:.2f}s")

    blocks_out = []
    for bsi in block_split:
        blocks_out.extend(bsi)

    isort = np.argsort([b.bid for b in blocks_out])
    blocks_out = [blocks_out[i] for i in isort]

    for b, sb in zip(grid, blocks_out):
        cons_avg = np.moveaxis(sb.cons_avg, -1, 0)
        b.set_conserved(cons_avg)

    mdot_in = 0.0
    for patch in grid.inlet_patches:
        Cm, A, _ = patch.get_cut().mix_out()
        mdot_in += Cm.rho * Cm.Vm * A

    mdot_out = 0.0
    for patch in grid.outlet_patches:
        Cm, A, _ = patch.get_cut().mix_out()
        mdot_out += Cm.rho * Cm.Vm * A

    if not mdot_out == 0.0:
        merr = mdot_in / mdot_out - 1.0
        logger.info(f"Mass flow error: {merr*100.:.1f}%")
    else:
        merr = -1.0

    if conf.plot_conv:
        dUlog = np.concatenate(dUlog, axis=0)
        r_ref = np.mean(blocks_out[0].r)
        dUlog[:, 3] /= r_ref
        ii = tuple(range(conf.n_step_log))
        drho_ref = dUlog[ii, 0].max()
        drhoVx_ref = dUlog[ii, 1].max()
        drhoVr_ref = dUlog[ii, 2].max()
        drhoVt_ref = dUlog[ii, 3].max()
        drhoV_ref = np.max((drhoVx_ref, drhoVr_ref, drhoVt_ref))
        drhoe_ref = dUlog[ii, 4].max()

        dUlog[:, 0] /= drho_ref
        dUlog[:, 1:4] /= drhoV_ref
        dUlog[:, 4] /= drhoe_ref

        dUlog = turbigen.util.moving_average(dUlog, conf.n_step_log)

        import matplotlib.pyplot as plt

        omin = dUlog[conf.n_step_log :, :].min()

        fig, ax = plt.subplots()
        ax.semilogy(dUlog)
        ax.set_ylim(bottom=omin)
        plt.tight_layout()
        plt.savefig("conv.pdf")
        plt.close()

        # fig, ax = plt.subplots()
        # ax.plot(Yslog)
        # ax.set_ylabel("Entropy Loss Coefficient, $Y_s$")
        # plt.tight_layout()

        # fig, ax = plt.subplots()
        # ax.plot(merrlog)
        # ax.set_ylabel(r"Mass Conservation Error $\varepsilon \dot{m}/\%$")
        # plt.tight_layout()
        # plt.show()
    return tpnps, merr


def get_multigrid_indices(shape, nb):
    """For a block of a given shape and a set of multigrid levels,
    evaluate the indices of every fine mesh point into each of the
    coarse grid levels.

    Returns
    -------
    ijkmg: array (nlevels, ni, nj, nk, 3)"""

    # Preallocate output array
    ni, nj, nk = shape
    nlev = len(nb)
    ijkmg = np.asfortranarray(np.full((3,) + shape + (nlev,), -1, dtype=np.int16))
    nbf = np.asfortranarray(nb, dtype=np.int16)
    embsolve.multigrid_indices(ijkmg, nbf)
    assert (ijkmg >= 0).all()
    return ijkmg


def get_multigrid_volumes(vol, ijkmg, typ):
    nlev = ijkmg.shape[-1]
    volmg = np.asfortranarray(np.zeros(vol.shape + (nlev + 1,), dtype=typ))
    embsolve.multigrid_volumes(volmg, vol, ijkmg)

    assert np.ptp(np.sum(volmg, axis=(0, 1, 2))) / np.sum(vol) < 1e-3

    return volmg


def arange_including_end(ni, di):
    ii = np.arange(0, ni, di)
    if not (ii[-1] == (ni - 1)):
        ii = np.append(ii, ni - 1)
    assert ii[-1] == (ni - 1)
    assert np.allclose(np.diff(ii[:-1]), di)
    return ii


def get_multigrid_lengths(block, nb, typ):
    # Preallocate output array
    ni, nj, nk = block.shape
    ijkmg = get_multigrid_indices((ni - 1, nj - 1, nk - 1), nb)
    nlev = len(nb)

    dlmg = np.asfortranarray(
        np.zeros(
            (
                ni - 1,
                nj - 1,
                nk - 1,
                nlev + 1,
            ),
            dtype=typ,
        )
    )

    nimg = np.max(ijkmg[0, ...], axis=(0, 1, 2)) + 1
    njmg = np.max(ijkmg[1, ...], axis=(0, 1, 2)) + 1
    nkmg = np.max(ijkmg[2, ...], axis=(0, 1, 2)) + 1

    # Finest grid level is trivial
    dlmg[..., 0] = block.dlmin_new

    # Loop over multigrid levels
    for ilev in range(nlev):
        # Number of cells along each side of this
        # multigrid level is product of all previous
        nbi = np.prod(nb[: ilev + 1])

        # Assemble a list of ijk with correct step size
        iimg = arange_including_end(ni, nbi)
        jjmg = arange_including_end(nj, nbi)
        kkmg = arange_including_end(nk, nbi)
        data_lev = np.full((block.nprop, len(iimg), len(jjmg), len(kkmg)), np.nan)

        # Loop over coarse cells
        for i, img in enumerate(iimg):
            for j, jmg in enumerate(jjmg):
                for k, kmg in enumerate(kkmg):
                    data_lev[:, i, j, k] = block._data[:, img, jmg, kmg]
        blk_lev = block.empty()
        blk_lev._data = data_lev
        assert blk_lev.dlmin.shape == (nimg[ilev], njmg[ilev], nkmg[ilev])
        dlmg[: nimg[ilev], : njmg[ilev], : nkmg[ilev], ilev + 1] = blk_lev.dlmin_new

    return dlmg


class Boundary:
    """Store flow field on a boundary condition."""

    def __init__(self, patch):
        """Set up the boundary condition using a patch object."""

        # Store slicing data for this patch so we can exchange
        # information with the block on which the patch resides
        self.slice = patch.get_slice()

        # Cut out flowfield from the block
        C = patch.block[self.slice]

        # Preallocate a working fluid object for all nodes on patch
        self.state = C.copy()
        self.shape = self.state.shape
        self.size = self.state.size

        # Preallocate nodal conserved variable changes
        # We apply boundary conditions by intercepting them
        self.dUn = np.empty(self.shape + (5,))

        # Store face area
        if patch.cdir == 0:
            self.dA = C.dAi.squeeze()
        elif patch.cdir == 1:
            self.dA = C.dAk.squeeze()
        elif patch.cdir == 2:
            self.dA = C.dAj.squeeze()
        self.dA = util.vecnorm(self.dA)

        # Determine a permutation order such that
        # first axis is spanwise, second axis is pitchwise
        ax_theta = np.argmax([np.ptp(C.t, axis=n).mean() for n in range(3)]).item()
        ax_stream = np.argmax(np.array(C.shape) == 1).item()
        ax_span = np.setdiff1d([0, 1, 2], [ax_theta, ax_stream]).item()
        self.order = (ax_stream, ax_span, ax_theta)
        if not self.order == (0, 1, 2):
            raise Exception("Boundary conditions must be on const-i faces.")

        # Get normal vectors pointing into the domain
        C0 = C.copy().transpose(self.order)
        C1 = patch.get_cut(offset=1).transpose(self.order)
        dxr = C1.xr - C0.xr
        self.normal = dxr / turbigen.util.vecnorm(dxr)

        # Angular pitch and cell widths for integration
        self.pitch = C0.pitch + 0.0
        self.dt = np.diff(C0.t.squeeze(), axis=1)

        # Check that theta gridlines are at constant x and r
        Lref = np.maximum(np.ptp(C0.x), np.ptp(C0.r))
        rtol = 1e-3
        assert (np.ptp(C0.squeeze().x, axis=1) / Lref < rtol).all()
        assert (np.ptp(C0.squeeze().r, axis=1) / Lref < rtol).all()

    # def clip_velocities(self, Ma_min=0.01):
    #     """Limit the minimum absolute throughflow velocity to avoid singular transformation matrices."""
    #     V_min = self.state.a.mean() * Ma_min
    #     ind_clip = np.abs(self.state.Vx) < V_min
    #     self.state.Vx[ind_clip] = V_min * np.sign(self.state.Vx[ind_clip])

    def pull(self, block):
        """Update stored state using solution from parent block."""

        # Extract the variables we need at all nodes on patch
        rho = block.cons[self.slice][..., 0]
        u = block.u[self.slice]
        self.state.set_rho_u(rho, u)

        # Note that the solver blocks use Fortran axis order
        # So that e.g. Vx = Vxrt[...,0] is contiguous
        # This is opposite to the C axis order used within state objects
        # So we have to move the component axis to first posn
        self.state.Vxrt = np.moveaxis(block.Vxrt[self.slice], -1, 0)

        # Nodal residuals
        # We keep always in Fortran order, no moveaxis here
        self.dUn[:] = block.dUn[self.slice]

    def push(self, block):
        """Send modified residuals back to the parent block."""
        block.dUn[self.slice] = self.dUn

    def apply(self, block):
        """Apply a characteristic boundary condition by altering nodal changes."""

        # Get flow field from block
        self.pull(block)

        # Take outwards-running chics from interior
        dchic_outwards = self.outward_chics()

        # Set inwards-running chics using prescribed boundary conditions
        dchic_inwards = self.inward_chics() * 0.5

        # Transform to conserved variable changes
        dcons = self.state.chic_to_conserved @ (dchic_outwards + dchic_inwards)

        # Send the nodal changes back to the block
        self.dUn[:] = dcons[..., 0]
        self.push(block)

    def slice_inward(self):
        """Index the chics propagating into domain."""
        raise NotImplementedError()

    def inward_chics(self):
        """Index the chics propagating into domain."""
        raise NotImplementedError()

    def outward_chics(self):
        """Get chics propagating out of domain from nodal changes."""
        # Transform conserved changes to chics
        dchic = self.state.conserved_to_chic @ self.dUn[..., None]
        # Zero out the inwards chics
        dchic[..., self.slice_inward(), 0] = 0.0
        return dchic


class OutletBoundary(Boundary):
    def __init__(self, patch):
        # Set up the common features of all boundaries
        super().__init__(patch)

        # Store the target static pressure
        self.P_target = patch.Pout

    def slice_inward(self):
        """Index the upstream-running chics (inwards thro outlet)."""
        return slice(0, 1, None)

    def inward_chics(self):
        """Use static pressure target to set upstream-running wave."""

        dP = self.P_target - self.state.P
        dVx = -dP / self.state.rho / self.state.a  # from c2=0
        dc1 = dP - self.state.rho * self.state.a * dVx  # definition of c1
        dc = np.zeros(self.shape + (5, 1))
        dc[..., 0, 0] = dc1
        return dc


class InletBoundary(Boundary):
    def __init__(self, patch):
        # Set up the common features of all boundaries
        super().__init__(patch)

        # Store the target ho, s, flow angles
        bcond_target = np.array(
            [
                patch.state.h,
                patch.state.s,
                turbigen.util.tand(patch.Alpha),
                turbigen.util.tand(patch.Beta),
            ]
        )
        self.bcond_target = np.tile(bcond_target, self.shape + (1,))[..., None]

    def slice_inward(self):
        """Index the downstream-running chics (inwards thro inlet)."""
        return slice(1, None, None)

    def inward_chics(self):
        """Use target inlet conditions to set downstream-propagating waves."""

        # Evaluate bcond error to get desired bcond changes
        bcond_now = np.stack(
            (self.state.ho, self.state.s, self.state.tanAlpha, self.state.tanBeta),
            axis=-1,
        )[..., None]
        dbcond = self.bcond_target - bcond_now

        # Convert to chics
        dchic = self.state.inlet_to_chic @ dbcond

        # Prepend a zero for upstream-running wave
        dc1 = np.zeros(self.shape + (1, 1))
        dchic = np.concatenate((dc1, dchic), axis=3)

        return dchic


class MixingBoundary(Boundary):
    def __init__(self, patch, pid, procids, typ):
        """Define the mixing boundary with patch and communication info."""
        # Set up the common features of all boundaries
        super().__init__(patch)

        # Store ids for communication
        match = patch.match
        self.pid = pid
        self.bid = patch.block.grid.index(patch.block)
        self.nxbid = match.block.grid.index(match.block)
        self.procid = procids[self.bid]
        self.nxprocid = procids[self.nxbid]

        # Determine a common radial grid vector
        C = patch.get_cut()
        self.spf = C.spf[0, :, 0]

        # Buffers for communication
        self.ncomm = len(self.spf) * 5 * 2
        self.buffer = np.full((self.ncomm,), np.nan).astype(typ)
        self.nxbuffer = np.full((self.ncomm,), np.nan).astype(typ)

        # Pitch-avg normal grid vector
        self.normal_avg = self.normal[:, 0, :, :].mean(axis=-1)

        # Common pitch-averaged state
        self.state_avg = self.state.empty(shape=(len(self.spf),))
        self.state_avg.xrt = C.xrt[:, 0, :, 0]

        # Preallocate pitch-avg flux changes
        self.dflux_avg = np.zeros((1, len(self.spf), 1, 5, 1))

        # Set direction
        self.is_inlet = np.ones_like(self.spf, dtype=bool)

    @property
    def is_outlet(self):
        return np.logical_not(self.is_inlet)

    def setup_communication(self, comm, mpi_typ):
        self.Send = comm.Send_init(
            buf=[self.buffer, self.ncomm, mpi_typ],
            dest=self.nxprocid,
            tag=self.pid,
        )
        self.Recv = comm.Recv_init(
            buf=[self.nxbuffer, self.ncomm, mpi_typ],
            source=self.nxprocid,
            tag=self.nxpid,
        )

    def pitchwise_average(self, y):
        """Area-average a variable in the circumferential direction."""
        return 0.5 * np.sum((y[..., 1:] + y[..., :-1]) * self.dt, axis=-1) / self.pitch

    def fill_buffer(self):
        """Prepare pitch-avg fluxes and conserved vars to send."""
        flux_avg = self.pitchwise_average(self.state.fluxes)
        cons_avg = self.pitchwise_average(self.state.conserved)
        self.buffer[:] = np.stack((flux_avg, cons_avg)).reshape(-1)

    def unpack_buffers(self):
        """Average home and away buffers to get common fluxes and conserved."""

        buffer = self.buffer.reshape(2, 5, -1)
        nxbuffer = self.nxbuffer.reshape(2, 5, -1)

        dflux = (buffer[0] - nxbuffer[0]) / 2.0
        cons_avg = 0.5 * (buffer[1] + nxbuffer[1])

        # Store the results
        self.state_avg.set_conserved(cons_avg)
        self.dflux_avg[:] = -np.expand_dims(dflux.T, (0, 2, -1))

    def set_direction(self):
        """Use current avg velocity and normals to get flow direction."""
        Vxr_avg = self.state_avg.Vxr
        Vxr_avg /= turbigen.util.vecnorm(Vxr_avg)
        self.is_inlet[:] = (
            np.sign(np.einsum("i...,i...", Vxr_avg, self.normal_avg)) > 0.0
        )

    def outward_chics(self):
        """Get chics propagating out of domain using local flow dirn."""
        # Transform conserved changes to chics
        dchic = self.state.conserved_to_chic @ self.dUn[..., None]
        # Where the pitch-avg flow is into the domain
        # zero the downstream-running chics
        dchic[:, self.is_inlet, :, 1:, 0] = 0.0
        # Where the pitch-avg flow is out of the domain
        # zero the upstream-running chic
        dchic[:, self.is_outlet, :, 0, 0] = 0.0
        return dchic

    def inward_chics(self):
        """Set inward chics to drive flux error to zero at uniform ho and s."""

        # First calculate chic changes due to flux error
        flux_to_chic = np.expand_dims(self.state_avg.flux_to_chic, (0, 2))
        dchic = np.tile(flux_to_chic @ self.dflux_avg, (1, 1, self.shape[2], 1, 1))

        # Relax
        dchic *= 0.5

        # Discard the outwards-running chics
        # Where the pitch-avg flow is into the domain like an inlet
        # zero the upstream-running chic
        dchic[:, self.is_inlet, :, 0, 0] = 0.0
        # Where the pitch-avg flow is out of the domain like an outlet
        # zero the downstream-running chic
        dchic[:, self.is_outlet, :, 1:, 0] = 0.0

        # Second, we need to enforce uniform ho and s on inlet
        if self.is_inlet.any():
            # Calculate perturbations to drive towards uniformity
            ho_avg = self.pitchwise_average(self.state.ho)[..., None]
            s_avg = self.pitchwise_average(self.state.s)[..., None]
            dho = ho_avg - self.state.ho
            ds = s_avg - self.state.s

            # Assemble a change in inlet bcond vector
            # Do not alter flow angles
            # Cannot control static P because set by upstream-running chic
            Z = np.zeros_like(dho)
            dinlet_local = np.stack((dho, ds, Z, Z), axis=-1)[..., None]

            # Convert to downstream-propagating chics
            dchic_local = self.state.inlet_to_chic @ dinlet_local

            # Prepend a zero for upstream-running wave
            dc1 = np.zeros(self.shape + (1, 1))
            dchic_local = np.concatenate((dc1, dchic_local), axis=3)

            # We only apply local changes where flow is into domain
            dchic[:, self.is_inlet, ...] += dchic_local[:, self.is_inlet, ...]

        return dchic

    def apply(self, block, log):
        self.unpack_buffers()

        if log:
            dflux_avg = self.dflux_avg.mean(axis=(0, 1, 2, 4))
            if dflux_avg.mean() > 0.0:
                flux_avg = self.state_avg.fluxes.mean(axis=-1)
                flux_ref = flux_avg.copy()
                flux_ref[2] = flux_ref[1]
                flux_ref[3] = self.state_avg.r.mean() * flux_ref[1]
                print(dflux_avg / flux_ref)

        self.set_direction()

        # Take outwards-running chics from interior
        dchic_outwards = self.outward_chics()

        # Set inwards-running chics using prescribed boundary conditions
        dchic_inwards = self.inward_chics()

        # Transform to conserved variable changes
        dcons = self.state.chic_to_conserved @ (dchic_outwards + dchic_inwards)

        # Send the nodal changes back to the block
        self.dUn[:] = dcons[..., 0]
        self.push(block)
