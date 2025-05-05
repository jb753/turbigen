import numpy as np
from copy import copy

import turbigen.util
import sys

from dataclasses import dataclass

from timeit import default_timer as timer

import turbigen.flowfield
import turbigen.fluid
import turbigen.grid
import turbigen.solvers.base

from embsolvec import embsolve

util = turbigen.util
logger = turbigen.util.make_logger()


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


def get_memory_usage():
    with open("/proc/self/status") as f:
        mem = f.read().split("VmRSS:")[1].split("\n")[0][:-3].strip()

    return float(mem) / 1000.0


@dataclass
class Emb(turbigen.solvers.base.BaseSolver):
    """
    ember
    -----

    :program:`ember` is the native flow solver built into :program:`turbigen`.

    """

    smooth4: float = 0.01
    """Fourth-order smoothing factor."""

    smooth2_adapt: float = 1.0
    """Second-order smoothing factor, adaptive on pressure."""

    smooth2_const: float = 0.0
    """Second-order smoothing factor, constant throughout the flow."""

    smooth_ratio_min: float = 0.1
    """Largest reduction in smoothing on a non-isotropic grid. Unity disables directional scaling, lower values clip the local smoothing factor to be sf_local >= sf * smooth_ratio_min."""

    CFL: float = 0.65
    """Courant--Friedrichs--Lewy number, time step normalised by local wave speed and cell size. Reduced values are more stable but slower to converge."""

    n_step: int = 5000
    """Number of time steps to run for."""

    n_step_mix: int = 5
    """Number of time steps between mixing plane updates."""

    n_step_dt: int = 10
    """Number of time steps between updates of the local time step."""

    n_step_log: int = 500
    """Number of time steps between log prints."""

    n_step_avg: int = 1
    """Number of time steps to average over."""

    n_step_ramp: int = 250
    """Number of time steps to ramp smoothing and damping."""

    n_loss: int = 5
    """Number of time steps between viscous force updates."""

    nstep_damp: int = -1
    """Number of steps to apply damping."""

    damping_factor: float = 25.0
    """Negative feedback to damp down high residuals. Lower values are more stable."""

    Pr_turb: float = 1.0
    """Turbulent Prandtl number."""

    xllim_pitch: float = 0.03
    """Maximum mixing length as a fraction of the pitch."""

    precision: int = 1
    """Precision of the solver. 1: single, 2: double."""

    i_scheme: int = 1
    """Which time-stepping scheme to use. 0: scree, 1: super."""

    i_loss: int = 1
    """Viscous loss model. 0: inviscid, 1: viscous."""

    K_exit: float = 0.5
    """Relaxation factor for outlet forcing."""

    K_inlet: float = 0.5
    """Relaxation factor for inlet forcing."""

    K_mix: float = 0.1
    """Relaxation factor for mixing plane forcing."""

    sf_mix: float = 0.01
    """Smoothing factor for uniform enthalpy and entropy downstream of mixing plane."""

    print_conv: bool = True
    """Print convergence history in the log."""

    fmgrid: float = 0.2
    """Factor scaling the multigrid residual."""

    multigrid: tuple = (2, 2, 2)
    """Number of cells forming each multigrid level. (2, 2, 2) gives coarse cells of side length 2, 4, and 8 fine cells."""

    area_avg_Pout: bool = True
    """Force area-averaged outlet pressure to target, otherwise use uniform outlet pressure."""

    def robust(self):
        """Create a copy of the config with more robust settings."""
        return self.replace(
            damping_factor=3.0,
            smooth2_adapt=1.0,
            smooth4=0.02,
            fmgrid=0.0,
            CFL=0.4,
            i_scheme=0,
        )

    def restart(self):
        """Create a copy of the config to smoothly restart."""
        return self.replace(
            n_step_ramp=0,
        )

    def run(self, grid, machine=None, workdir=None):
        logger.debug(
            f"Entering embsolve run, memory usage on rank {rank}: {get_memory_usage():.0f}MB"
        )

        del machine, workdir
        conf = self

        logger.info("Initialising ember...")
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
        mixers = get_mixers(grid, procids, typ, conf.K_mix, conf.sf_mix)

        # Split into lists for each procid
        block_split = []
        for iproc in range(size):
            block_split.append([])
            for ib, b in enumerate(blocks):
                if iproc == procids[ib]:
                    block_split[-1].append(b)

        t2 = timer()
        logger.debug(f"Elapsed time {t2 - t1:.2f}s")

        if comm:
            logger.debug("Sending data to processors...")
            tst = timer()
            send_slave(block_split, procids, periodics, mixers)
            ten = timer()
            logger.debug(f"Elapsed time {ten - tst:.2f}s")

        logger.info("Starting the main time-stepping loop...")
        block_split[0], mixers_out, tpnps, dUlog = run_slave(
            block_split[0], periodics, mixers, nodes
        )

        logger.debug("Recieving data from processors...")
        tst = timer()
        for iproc in range(1, size):
            block_split[iproc] = comm.recv(source=iproc)
        ten = timer()
        logger.debug(f"Elapsed time {ten - tst:.2f}s")

        blocks_out = []
        for bsi in block_split:
            blocks_out.extend(bsi)

        isort = np.argsort([b.bid for b in blocks_out])
        blocks_out = [blocks_out[i] for i in isort]

        # Assemble a convergence history
        mhos = np.full(
            (
                2,
                3,
                conf.n_step,
            ),
            np.nan,
        )
        nnow = conf.n_step
        for b in blocks_out:
            for bc in b.bconds:
                mhos_now = np.array(bc.convergence_log).T
                nnow = mhos_now.shape[-1]
                if isinstance(bc, InletBoundary):
                    mhos[0, :, :nnow] = mhos_now
                elif isinstance(bc, OutletBoundary):
                    mhos[1, :, :nnow] = mhos_now

        istep = np.arange(conf.n_step)
        istep_avg = conf.n_step - conf.n_step_avg
        resid = np.concatenate(dUlog, axis=0)[:, 0]
        state_conv = blocks_out[0].state.empty(shape=(2, nnow))
        state_conv.set_h_s(mhos[:, 1, :], mhos[:, 2, :])
        conv = turbigen.solvers.base.ConvergenceHistory(
            istep, istep_avg, resid, mhos[:, 0], state_conv
        )
        conv.tpnps = tpnps

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
            logger.info(
                f"mdot_in/out={mdot_in:.3g}/{mdot_out:.3g}, err={merr * 100.0:.1f}%"
            )
        else:
            merr = -1.0

        self.convergence = conv  # , tpnps, merr


class SolverBlock:
    """Hold just the data we need for a CFD solution."""

    def __init__(self, block, conf):
        """Initialise from a standard Block object."""

        # Config settings
        self.conf = conf
        self.shape = tuple(block.shape)
        self.shape_cell = tuple([i - 1 for i in block.shape])

        # Block-level scalars
        Omega = block.Omega.mean()
        self.Omega = self.cast_scalar(Omega)

        # Primaries
        self.cons = self.cast_array(block.conserved)

        # Geometry
        self.r = self.cast_array(block.r)
        self.dAi = self.cast_array(block.dAi)
        self.dAj = self.cast_array(block.dAj)
        self.dAk = self.cast_array(block.dAk)

        # Multigrid setup
        # Note all stored indices are 1-indexed for Fortran
        self.ijk_multigrid = self.get_multigrid_indices()
        self.vol = self.get_multigrid_volumes(block.vol)
        self.dlmin = self.get_multigrid_lengths(block)

        # Mixing length
        self.xlength = self.get_xlength(block)

        # Get wall indices
        # These are ijk (3, n) for each of ifaces, jfaces, kfaces, nodes
        # Including slip walls for enforcing no-flow condition
        *self.ijk_wall_face, self.ijk_wall_node = self.get_wall_indices(
            block, ignore_slip=False
        )
        # Excluding slip walls for wall functions
        *self.ijk_wall_face_slip, _ = self.get_wall_indices(block, ignore_slip=True)

        # Get wall cell size
        self.dw_face = self.get_wall_cell_size(block)

        # Store boundary conditions
        self.bconds = [
            InletBoundary(patch, conf.K_inlet) for patch in block.inlet_patches
        ] + [
            OutletBoundary(patch, conf.K_exit, conf.area_avg_Pout)
            for patch in block.outlet_patches
        ]

        # Initialise the state object for this block
        # With the correct data type
        if isinstance(block, turbigen.grid.PerfectBlock):
            state = self.state = turbigen.fluid.PerfectState(
                shape=block.shape, order="F", typ=self.get_data_type()
            )
            state.gamma = self.cast_scalar(block.gamma)
            state.cp = self.cast_scalar(block.cp)
            state.mu = self.cast_scalar(block.mu)
            state.set_rho_u(block.rho, block.u)
            state.set_Tu0(block.Tu0)
        else:
            raise NotImplementedError()

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

        # Get length scales along each side of the cell
        L = self.cast_array(
            np.stack(
                (
                    dli,
                    dlj,
                    dlk,
                ),
                axis=0,
            )
        )
        # Normalise to sum to 3.0
        # In an isotropic grid all elements should be 1.0
        L /= L.sum(axis=-1, keepdims=True) / 3.0

        # Clip to a maximum reduction in smoothing and re-normalise
        L = np.clip(L, conf.smooth_ratio_min, None)
        L /= L.sum(axis=-1, keepdims=True) / 3.0

        # Now distribute cell length scales to nodes
        ni, nj, nk = self.shape
        self.L = self.cast_array(np.ones((3, ni, nj, nk)))
        embsolve.cell_to_node(L, self.L, ni, nj, nk, 3)

        self.rf = [self.cast_array(r) for r in block.r_face]
        self.rc = self.cast_array(block.r_cell)

        #
        # Don't need block after this
        #

    def setup_temporary(self):
        # Can initialise these from the state object after data transfer
        self.mu = self.cast_scalar(self.state.mu)
        self.cp = self.cast_scalar(self.state.cp)
        self.Pr_turb = self.cast_scalar(self.conf.Pr_turb)
        self.Pref = self.cast_scalar(self.state.P.mean())

        # Intialise from geometry after data transfer
        self.dA_wall_face = self.get_wall_area_magnitude()
        self.set_dummy_ijk()

        # Preallocate
        self.u = self.preallocate()
        self.Vxrt = self.preallocate(self.shape + (3,))
        self.cons_avg = self.preallocate(self.shape + (5,))
        self.fb = self.preallocate(self.shape_cell + (5,))
        self.dUc = self.preallocate(self.shape_cell + (5, 2))
        self.dUn = self.preallocate(self.shape + (5,))
        self.dt_vol = self.dlmin * 0.0

    def set_dummy_ijk(self):
        """Where the lists of wall faces have zero length, set sentinel values."""
        for n in range(3):
            ijk = self.ijk_wall_face_slip[n]
            if ijk.shape[-1] == 0:
                ijkdum = np.asfortranarray(-np.ones((3, 1))).astype(np.int16)
                self.ijk_wall_face_slip[n] = ijkdum
                self.dw_face[n] = self.cast_array(np.ones((1,)))
                self.dA_face[n] = self.cast_array(np.ones((1,)))

    def get_data_type(self):
        """Return configured single or double numeric data type."""
        if self.conf.precision == 1:
            typ = np.float32
        else:
            typ = np.float64
        return typ

    def get_mpi_data_type(self):
        """Return configured single or double numeric data type for MPI."""
        if self.conf.precision == 1:
            typ = mpi_single
        else:
            typ = mpi_double
        return typ

    def cast_scalar(self, x):
        """Convert a scalar value to block data type."""
        return self.get_data_type()(util.asscalar(x))

    def cast_array(self, x):
        """Convert an array to Fortran type of correct precision, moving axis for 4D."""
        if x.ndim > 3:
            x2 = np.moveaxis(x, 0, -1)
        else:
            x2 = x
        return np.asfortranarray(x2).astype(self.get_data_type())

    def preallocate(self, shape=None):
        if shape is None:
            shape = self.shape
        return np.asfortranarray(np.zeros(shape, dtype=self.get_data_type()))

    def get_multigrid_indices(self):
        """For a block of a given shape and a set of multigrid levels,
        evaluate the indices of every fine mesh point into each of the
        coarse grid levels.

        Returns
        -------
        ijkmg: array (nlevels, ni, nj, nk, 3)"""

        ni, nj, nk = self.shape
        shape = (ni - 1, nj - 1, nk - 1)
        nb = self.conf.multigrid
        # Preallocate output array
        ni, nj, nk = shape
        nlev = len(nb)
        ijkmg = np.asfortranarray(np.full((3,) + shape + (nlev,), -1, dtype=np.int16))
        nbf = np.asfortranarray(nb, dtype=np.int16)
        embsolve.multigrid_indices(ijkmg, nbf)
        assert (ijkmg >= 0).all()
        return ijkmg + 1

    def get_multigrid_volumes(self, vol):
        nlev = self.ijk_multigrid.shape[-1]
        vol = self.cast_array(vol)
        volmg = self.preallocate(vol.shape + (nlev + 1,))
        embsolve.multigrid_volumes(volmg, vol, self.ijk_multigrid)
        assert np.ptp(np.sum(volmg, axis=(0, 1, 2))) / np.sum(vol) < 1e-3
        return volmg

    def get_multigrid_lengths(self, block):
        nb = self.conf.multigrid

        # Preallocate output array
        ni, nj, nk = block.shape
        ijkmg = self.ijk_multigrid - 1
        nlev = len(nb)

        dlmg = self.preallocate(
            (
                ni - 1,
                nj - 1,
                nk - 1,
                nlev + 1,
            )
        )

        nimg = np.max(ijkmg[0, ...], axis=(0, 1, 2)) + 1
        njmg = np.max(ijkmg[1, ...], axis=(0, 1, 2)) + 1
        nkmg = np.max(ijkmg[2, ...], axis=(0, 1, 2)) + 1

        # Finest grid level is trivial
        dlmg[..., 0] = block.dlmin

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
            dlmg[: nimg[ilev], : njmg[ilev], : nkmg[ilev], ilev + 1] = blk_lev.dlmin

        return dlmg

    def get_xlength(self, block):
        # Mixing length limit
        xllim = (
            block.pitch * 0.5 * (block.r.max() + block.r.min()) * self.conf.xllim_pitch
        )
        # Nodal xlength
        xlength = self.cast_array(np.clip(block.w, 0.0, xllim))
        # Times von Karman and squared
        xlength = (0.41 * xlength) ** 2.0
        # Distribute to cellss
        xlength_cell = self.preallocate(self.shape_cell)
        embsolve.node_to_cell(xlength, xlength_cell)
        return xlength_cell

    def get_wall_indices(self, block, ignore_slip):
        return [
            np.asfortranarray(np.argwhere(wall).T + 1).astype(np.int16)
            for wall in block.get_wall(ignore_slip)
        ]

    def get_wall_cell_size(self, block):
        ni, nj, nk = self.shape
        iwall1, jwall1, kwall1 = [ijk + 0 for ijk in self.ijk_wall_face_slip]
        iwall1[0, iwall1[0, :] == ni] -= 1
        jwall1[1, jwall1[1, :] == nj] -= 1
        kwall1[2, kwall1[2, :] == nk] -= 1
        return [
            embsolve.get_by_ijk(self.cast_array(dl), ijk)
            for dl, ijk in zip(block.get_dwall(), [iwall1, jwall1, kwall1])
        ]

    def get_wall_area_magnitude(self):
        dAijk = [
            np.sqrt((self.dAi**2).sum(axis=-1)),
            np.sqrt((self.dAj**2).sum(axis=-1)),
            np.sqrt((self.dAk**2).sum(axis=-1)),
        ]
        self.dA_face = [
            embsolve.get_by_ijk(dA, ijk)
            for dA, ijk in zip(dAijk, self.ijk_wall_face_slip)
        ]

    def set_timestep(self, CFL, relax=0.0):
        embsolve.set_timesteps(
            self.dt_vol,
            self.vol,
            self.state.a,
            self.Vxrt,
            self.Omega * self.r,
            self.dlmin,
            self.ijk_multigrid,
            CFL,
        )

    def residual(self, fmgrid, damp, ischeme):
        embsolve.residual(
            self.cons,
            self.Vxrt,
            self.state.P,
            self.Pref,
            self.state.h,
            self.fb,
            self.Omega,
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
        """Calculate velocity components and update thermodynamic state."""
        embsolve.secondary(self.r, self.cons, self.Vxrt, self.u)
        self.state.set_rho_u(self.cons[..., 0], self.u)

    def smooth(self, sf2, sf4, sf2min):
        embsolve.smooth(self.cons, self.state.P, self.L, sf4, sf2, sf2min)

    def damp(self, fdamp):
        embsolve.damp(self.dU1, fdamp)

    def set_viscous_stress(self):
        # Assemble args in a dictionary and send as keywords
        # so we don't need to be careful with order
        # note keys must be all lower case
        kwargs = {
            "cons": self.cons,
            "v": self.Vxrt,
            "t": self.state.T,
            "mu": self.mu,
            "cp": self.cp,
            "pr_turb": self.Pr_turb,
            "xlength": self.xlength,
            "vol": self.vol[..., 0],  # Only fine grid volumes
            "dai": self.dAi,
            "daj": self.dAj,
            "dak": self.dAk,
            "omega": self.Omega,
            "r": self.r,
            "rc": self.rc,
            "ri": self.rf[0],
            "rj": self.rf[1],
            "rk": self.rf[2],
            "ijk_iwall": self.ijk_wall_face_slip[0],
            "ijk_jwall": self.ijk_wall_face_slip[1],
            "ijk_kwall": self.ijk_wall_face_slip[2],
            "dw_iwall": self.dw_face[0],
            "dw_jwall": self.dw_face[1],
            "dw_kwall": self.dw_face[2],
            "da_iwall": self.dA_face[0],
            "da_jwall": self.dA_face[1],
            "da_kwall": self.dA_face[2],
            "fvisc": self.fb,
        }
        embsolve.shear_stress(**kwargs)


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

        self.buffer = np.zeros((self.N), order="F").astype(typ)
        self.nxbuffer = np.zeros((self.N), order="F").astype(typ)

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


def get_mixers(grid, procids, typ, K_mix, sf_mix):
    mixers = []
    seen = []
    pid = 0
    for patch in grid.mixing_patches:
        if patch in seen:
            continue
        else:
            seen.append(patch)
            seen.append(patch.match)
        mixers.append(MixingBoundary(patch, pid, procids, typ, K_mix, sf_mix))
        pid += 1
        mixers.append(MixingBoundary(patch.match, pid, procids, typ, K_mix, sf_mix))
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


def exchange_mixing(mixers):
    # Prepare to recieve into away buffers
    for mixer in mixers:
        if not mixer.nxprocid == rank:
            mixer.Recv.Start()

    # Populate the home buffer with pitchwise-averaged
    # fluxes and conserved vars and send away
    for mixer in mixers:
        mixer.fill_buffer()
        if not mixer.nxprocid == rank:
            mixer.Send.Start()

    # We now use populated buffers to get flux differences and
    # side-averaged mean flow
    for mixer in mixers:
        # Wait for communication before unpacking the buffers form each side
        if not mixer.nxprocid == rank:
            mixer.Recv.Wait()

        mixer.unpack_buffers()
        mixer.set_direction()

    # Whether Send actually blocks is implementation-dependent
    # so we have to explicitly wait for it to finish
    for mixer in mixers:
        if not mixer.nxprocid == rank:
            mixer.Send.Wait()


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
            # Send to away buffer
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

    # Whether Send actually blocks is implementation-dependent
    # so we have to explicitly wait for it to finish
    for patch in periodics:
        if not patch.nxprocid == rank:
            patch.Send.Wait()


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

    # Calculate smoothing and inlet relaxation scaled by CFL
    CFL_ref = 0.7
    conf = blocks[0].conf
    sf2 = conf.smooth2_adapt * conf.CFL / CFL_ref
    sf4 = conf.smooth4 * conf.CFL / CFL_ref
    sf2min = conf.smooth2_const * conf.CFL / CFL_ref

    if blocks[0].conf.precision == 1:
        mpi_typ = mpi_single
    else:
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

    dUnow = np.zeros((conf.n_step_log, nblock, 5))

    # Now integrate forward
    istep_avg = conf.n_step - conf.n_step_avg

    logger.debug(f"Memory usage on rank {rank}: {get_memory_usage():.0f}MB")

    # Allocate working vars
    for iblock in range(nblock):
        blocks[iblock].setup_temporary()

    logger.debug(
        f"After allocation Memory usage on rank {rank}: {get_memory_usage():.0f}MB"
    )

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

                # Check for NaNs
                if not np.mod(istep, 10):
                    if np.any(np.isnan(sb.cons)):
                        logger.iter(f"NaN at step {istep} in block {iblock}")
                        sys.exit(3)

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

            # Apply mixers
            for mixer in mixers:
                b1 = blocks[bid_local[mixer.bid]]
                mixer.pull(b1)
                mixer.apply(b1)

            # Exchange fluxes across mixing patches
            if not np.mod(istep, conf.n_step_mix):
                exchange_mixing(mixers)

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
            if (
                (not np.mod(istep, conf.n_step_log))
                and (istep > 0)
                or (istep == conf.n_step - 1)
            ):
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
                    remaining = (ten - tfirst) / (istep + 1) * (conf.n_step - istep - 1)
                    tstart = ten
                    # Print remaining time as minutes and seconds
                    remaining_minutes = int(remaining // 60)
                    remaining_seconds = int(remaining % 60)

                    if conf.print_conv:
                        logger.info(
                            f"{istep}: tpnps={tpnps:.3e}, remaining={remaining_minutes}m{remaining_seconds}s"
                        )
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
        logger.info(f"Elapsed time {tlast - tfirst:.2f}s")
        logger.info(f"Average tpnps={tpnps:.3e}")
        return blocks, mixers, tpnps, dUlog
    else:
        comm.send(blocks, dest=0)
        comm.send(mixers, dest=0)


def arange_including_end(ni, di):
    ii = np.arange(0, ni, di)
    if not (ii[-1] == (ni - 1)):
        ii = np.append(ii, ni - 1)
    assert ii[-1] == (ni - 1)
    assert np.allclose(np.diff(ii[:-1]), di)
    return ii


class Boundary:
    """Store flow field on a boundary condition."""

    def __init__(self, patch, K):
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
        self.dUn = np.zeros(self.shape + (5,))

        # Determine a permutation order such that
        # first axis is spanwise, second axis is pitchwise
        ax_theta = np.argmax([np.ptp(C.t, axis=n).mean() for n in range(3)]).item()
        ax_stream = np.argmax(np.array(C.shape) == 1).item()
        ax_span = np.setdiff1d([0, 1, 2], [ax_theta, ax_stream]).item()
        self.order = (ax_stream, ax_span, ax_theta)
        if not self.order == (0, 1, 2):
            raise Exception("Boundary conditions must be on const-i faces.")

        self.Nb = C.Nb

        # Store weights for area integral
        # Preallocate zeros
        self.wA = np.zeros((3,) + self.shape)
        # Distribute face areas to nodes
        self.wA[:, :, :-1, :-1] += C.dAi
        self.wA[:, :, 1:, :-1] += C.dAi
        self.wA[:, :, :-1, 1:] += C.dAi
        self.wA[:, :, 1:, 1:] += C.dAi
        self.wA /= 4.0
        self.wAabs = turbigen.util.vecnorm(self.wA)
        self.A = turbigen.util.vecnorm(C.dAi.squeeze()).sum()
        assert np.isclose(self.wAabs.sum(), self.A)

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

        # Store relaxation factor
        self.K = K

        # Initialise lists for convergence recording
        self.convergence_log = []

    def record_flows(self):
        """Append mass flow and mass-averaged ho and s to convergence log."""
        flux_mass = self.state.flux_mass * self.Nb
        s = self.state.s
        ho = self.state.ho
        mdot = np.sum(self.wA * flux_mass).astype(float)
        hodot = np.sum(self.wA * flux_mass * ho).astype(float) / mdot
        sdot = np.sum(self.wA * flux_mass * s).astype(float) / mdot
        self.convergence_log.append((mdot, hodot, sdot))

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

        # Record in convergence history
        self.record_flows()

        # Take outwards-running chics from interior
        dchic_outwards = self.outward_chics()

        # Set inwards-running chics using prescribed boundary conditions
        dchic_inwards = self.inward_chics() * self.K

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
    def __init__(self, patch, K, area_avg):
        # Set up the common features of all boundaries
        super().__init__(patch, K)

        # Store the target static pressure
        self.P_target = patch.Pout
        self.area_avg = area_avg

    def slice_inward(self):
        """Index the upstream-running chics (inwards thro outlet)."""
        return slice(0, 1, None)

    def inward_chics(self):
        """Use static pressure target to set upstream-running wave."""

        if self.area_avg:
            # Force to an area-averaged static pressure
            P_Aavg = np.sum(self.wAabs * self.state.P) / self.A
            dP = self.P_target - P_Aavg
        else:
            # Force to uniform at target pressure
            dP = self.P_target - self.state.P

        # Calculate the chic wave
        dVx = -dP / self.state.rho / self.state.a  # from c2=0
        dc1 = dP - self.state.rho * self.state.a * dVx  # definition of c1
        dc = np.zeros(self.shape + (5, 1))
        dc[..., 0, 0] = dc1
        return dc


class InletBoundary(Boundary):
    def __init__(self, patch, K):
        # Set up the common features of all boundaries
        super().__init__(patch, K)

        # Store the target ho, s, flow angles
        if np.isscalar(patch.Alpha):
            bcond_target = np.array(
                [
                    patch.state.h,
                    patch.state.s,
                    turbigen.util.tand(patch.Alpha),
                    turbigen.util.tand(patch.Beta),
                ]
            )
            self.bcond_target = np.tile(bcond_target, self.shape + (1,))[..., None]
        else:
            bcond_target = np.stack(
                [
                    patch.state.h,
                    patch.state.s,
                    turbigen.util.tand(patch.Alpha),
                    turbigen.util.tand(patch.Beta),
                ],
                axis=-1,
            )
            self.bcond_target = bcond_target[..., None]

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
    def __init__(self, patch, pid, procids, typ, K_mix, sf_mix):
        """Define the mixing boundary with patch and communication info."""
        # Set up the common features of all boundaries
        super().__init__(patch, K_mix)

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
        cons_avg = self.pitchwise_average(self.state.conserved)
        self.state_avg.set_conserved(cons_avg.squeeze())

        # Preallocate pitch-avg flux changes
        self.dflux_avg = np.zeros((1, len(self.spf), 1, 5, 1))

        # Set direction
        self.is_inlet = np.ones_like(self.spf, dtype=bool)

        # Store smoothing factors
        self.sf_mix = sf_mix

    @property
    def is_outlet(self):
        return np.logical_not(self.is_inlet)

    def smooth_pitchwise(self):
        """Force ho, s, and Beta to be uniform in circumferential direction.

        Directly change conserved variables, not chics.

        """

        # Get the average values
        ho_avg = self.pitchwise_average(self.state.ho)[..., None]
        s_avg = self.pitchwise_average(self.state.s)[..., None]
        tanBe_avg = self.pitchwise_average(self.state.tanBeta)[..., None]

        # Get the differences
        dho = ho_avg - self.state.ho
        ds = s_avg - self.state.s
        dtanBe = tanBe_avg - self.state.tanBeta

        # Assemble a change in bcond vector [ho, s, tanAl, tanBe, P]
        # Do not need to change Alpha because periodic
        # Cannot control static P because set by upstream-running chic
        # So they are both zero
        Z = np.zeros_like(dho)
        dinlet_local = np.stack((dho, ds, Z, dtanBe, Z), axis=-1)[..., None]

        # Conversion matrices
        prim_to_inlet = self.state.primitive_to_bcond
        prim_to_cons = self.state.primitive_to_conserved
        inlet_to_cons = prim_to_cons @ np.linalg.inv(prim_to_inlet)

        # Calculate conserved changes
        dcons = (inlet_to_cons @ dinlet_local)[..., 0]
        dcons_relaxed = self.sf_mix * dcons[:, self.is_inlet, ...]

        # Set the changes in the buffer
        self.dUn[:, self.is_inlet, ...] += dcons_relaxed

    def setup_communication(self, comm, mpi_typ):
        if self.procid == self.nxprocid:
            return
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

        # Store the averaged state and flux error
        self.state_avg.set_conserved(cons_avg)
        self.dflux_avg[:] = -np.expand_dims(dflux.T, (0, 2, -1))

        # Limit the minimum absolute throughflow velocity to avoid singular transformation matrices."""
        Ma_min = 0.01
        V_min = self.state_avg.a.mean() * Ma_min
        ind_clip = np.abs(self.state_avg.Vx) < V_min
        self.state_avg.Vx[ind_clip] = V_min * np.sign(self.state_avg.Vx[ind_clip])

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
        conserved_to_chic = np.expand_dims(self.state_avg.conserved_to_chic, (0, 2))
        dchic = conserved_to_chic @ self.dUn[..., None]
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
        dchic *= self.K

        # Discard the outwards-running chics
        # Where the pitch-avg flow is into the domain like an inlet
        # zero the upstream-running chic
        dchic[:, self.is_inlet, :, 0, 0] = 0.0
        # Where the pitch-avg flow is out of the domain like an outlet
        # zero the downstream-running chic
        dchic[:, self.is_outlet, :, 1:, 0] = 0.0

        return dchic

    def apply(self, block):
        # Take outwards-running chics from interior
        dchic_outwards = self.outward_chics()

        # Set inwards-running chics using prescribed boundary conditions
        dchic_inwards = self.inward_chics()

        # Transform to conserved variable changes
        chic_to_conserved = np.expand_dims(self.state_avg.chic_to_conserved, (0, 2))
        dcons = chic_to_conserved @ (dchic_outwards + dchic_inwards)

        # Store the nodal changes
        self.dUn[:] = dcons[..., 0]

        # Pitchwise smooth ho, s, and Beta to uniformity
        # By changing dUn in place
        if self.is_inlet.any():
            self.smooth_pitchwise()

        # Send the nodal changes back to the block
        self.push(block)
