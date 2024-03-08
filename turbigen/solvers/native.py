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
    calculate_secondary,
)
from timeit import default_timer as timer
import logging

logger = turbigen.util.make_logger()

logger.setLevel(level=logging.INFO)


# class NativeConfig(BaseSolver):
#     """Settings with default values for the TS4 solver."""

#     _name = "Native"


def flatwhere(x):
    return np.where(x.flat)[0]


nstep_dt = 100
nstep_log = 100
nstep = 4000
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
        self.conserved = np.asfortranarray(np.moveaxis(block.conserved, 0, -1))

        self.ho = np.asfortranarray(block.ho)
        self.P = np.asfortranarray(block.P)

        self.Vxrt = np.asfortranarray(np.moveaxis(block.Vxrt, 0, -1))
        self.u = np.asfortranarray(block.u)

        # Geometry
        self.r = np.asfortranarray(block.r)
        self.dAi = np.asfortranarray(np.moveaxis(block.dAi, 0, -1))
        self.dAj = np.asfortranarray(np.moveaxis(block.dAj, 0, -1))
        self.dAk = np.asfortranarray(np.moveaxis(block.dAk, 0, -1))
        self.vol = np.asfortranarray(block.vol)
        self.dlmin = np.asfortranarray(block.dlmin)
        self.Omega = block.Omega.mean()

        self.dU1 = self.conserved.copy(order="F") * np.nan
        self.dU2 = self.conserved.copy(order="F") * np.nan
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
            rho_inlet = block.rho.flat[inlet[0]]
            u_inlet = block.u.flat[inlet[0]]
            state_inlet.set_rho_u(rho_inlet, u_inlet)

    def set_inlets(self):
        """Set conserved variables on inlets by relaxing density changes."""

        # Change inlet patches
        for patch, state in zip(self.inlets, self.state_inlets):

            # Expand patch data
            ind, Po, To, Alpha, Beta, rhoo, ho, r = patch

            # Relax changes in density
            rho_now = rfin * self.conserved[..., 0].flat[ind] + rfin1 * state.rho

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
            self.conserved[..., 0].flat[ind] = rho_now  # rho
            self.conserved[..., 1].flat[ind] = rho_now * Vxin  # rhoVx
            self.conserved[..., 2].flat[ind] = rho_now * Vrin  # rhoVr
            self.conserved[..., 3].flat[ind] = rho_now * r * Vtin  # rhorVt
            self.conserved[..., 4].flat[ind] = rho_now * (u + 0.5 * Vinsq)  # rhoe

            # Reset pressure and hstag on inlet
            self.ho.flat[ind] = h + 0.5 * Vin**2
            self.P.flat[ind] = P

    def set_outlets(self):
        """Set static pressure on outlets."""
        for outlet in self.outlets:
            self.P.flat[outlet[0]] = outlet[1]

    def set_timestep(self):
        Vx = self.conserved[..., 1] / self.conserved[..., 0]
        Vr = self.conserved[..., 2] / self.conserved[..., 0]
        Vt = self.conserved[..., 3] / self.conserved[..., 0] / self.r
        V = np.sqrt(Vx**2 + Vr**2 + Vt**2)

        a = self.state.a

        ni, nj, nk = self.r.shape

        Va_node = np.asfortranarray(np.stack((V, a), axis=-1))
        Va_cell = np.empty((ni - 1, nj - 1, nk - 1, 2), order="F")
        node_to_cell(Va_node, Va_cell)
        Vref = Va_cell[..., 0]
        aref = Va_cell[..., 1]
        self.dt = CFL * self.dlmin / (aref + Vref)

    def step(self):

        Phor = np.asfortranarray(np.stack((self.P, self.ho, self.r), axis=-1))
        step(
            self.conserved,
            Phor,
            self.Omega,
            *self.wall_indicators,
            self.dt,
            self.dAi,
            self.dAj,
            self.dAk,
            self.vol,
            self.dU1,
        )

        if self._flag_scree:
            self.conserved += 2.0 * self.dU1 - self.dU2
        else:
            self.conserved += self.dU1
            self._flag_scree = True

        self.dU2[:] = self.dU1

        calculate_secondary(Phor[..., 2], self.conserved, self.Vxrt, self.u)

        self.state.set_rho_u(self.conserved[..., 0], self.u)

        Vsq = np.sum(self.Vxrt**2, axis=-1)
        self.ho[:] = self.state.h + 0.5 * Vsq
        self.P[:] = self.state.P

    def smooth(self):
        smooth(self.conserved, sf2, sf4)


# def set_periodic(g)
def get_periodics(g):

    periodics = []
    seen = []

    for patch in g.periodic_patches:

        if patch in seen:
            continue
        else:
            seen.append(patch)
            seen.append(patch.match)

        periodics.append(patch.get_periodic_data())

    return periodics
    # [patch.get_periodic_data() for patch in block.periodic_patches]


# @profile
def run(grid, settings={}, machine=None):

    nblock = len(grid)
    nodes = np.sum([b.size for b in grid])

    sg = [SolverBlock(b) for b in grid]
    periodics = get_periodics(grid)
    print(len(periodics))
    print(periodics[0])
    quit()

    # sb.set_outlets()
    # sb.set_inlets()
    # sb.set_timestep()
    # sb.step()
    # quit()

    logger.info("Starting native solver...")

    logger.info("Starting the main time-stepping loop...")

    tstart = timer()

    # Start the main time stepping loop
    for istep in range(nstep):

        # Update periodic boundaries
        # grid.apply_periodic()

        # Loop over blocks
        for iblock in range(nblock):

            sb = sg[iblock]

            # Update time stegs
            if not np.mod(istep, nstep_dt):
                sb.set_timestep()
                print(sb.dt.min(), sb.dt.max())

            sb.set_inlets()

            sb.set_outlets()

            sb.step()

            sb.smooth()

            if not np.mod(istep, nstep_log):

                log_line = f"{istep}: {np.abs(sb.dU1).mean(axis=(0,1,2))}"
                logger.info(log_line)
                ten = timer()
                tpnps = (ten - tstart) / nodes / nstep_log
                logger.info(f"tpnps={tpnps:.3e}")
                tstart = ten

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

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # ax.plot(mdot_all / mdot_all[-1].mean())
    # plt.show()

    for b, sb in zip(grid, sg):
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
        np.empty((ni, nj - 1, nk - 1, 1), order="F"),
        np.empty((ni - 1, nj, nk - 1, 1), order="F"),
        np.empty((ni - 1, nj - 1, nk, 1), order="F"),
    ]
    wn = np.asfortranarray(np.expand_dims(b.get_wall(), -1).astype(float))

    # Calculate nodal values of wall indicator
    node_to_face(wn, *wf)

    wfl = [(wfn > thresh)[..., 0].astype(np.int8) for wfn in wf]

    return wfl


# @profile
def apply_bconds(b):
    """Return properties needed to time march after in/out boundaries applied"""

    P = b.P
    ho = b.ho

    if b._conserved_store is None:
        conserved = b.conserved
        rho, rhoVx, rhoVr, rhorVt, rhoe = conserved
    else:
        conserved = b._conserved_store
        # rho = conserved[..., 0]
        rhoVx = conserved[..., 1]
        rhoVr = conserved[..., 2]
        rhorVt = conserved[..., 3]
        rhoe = conserved[..., 4]

    # Adjust static pressure for exit boundary conditions
    for patch in b.outlet_patches:
        P[patch.get_slice()] = patch.Pout  # * 0.1 + 0.9 * (patch.get_cut().P)

    # Change inlet patches
    for patch in b.inlet_patches:

        ipatch = patch.get_slice()
        inlet = b[ipatch]
        rfin = 0.2  # *patch.rfin

        # Relax changes in density if we have a stored state
        if patch.rho_store is None:
            rho_now = inlet.rho.copy()
        else:
            rho_now = rfin * inlet.rho + (1.0 - rfin) * patch.rho_store
            # Check for flow reversal
            rho_now[rho_now > patch.state.rho] = patch.state.rho * 0.9999
            # Isentropic expansion from stagnation state
            inlet.set_rho_s(rho_now, patch.state.s)

        patch.rho_store = rho_now

        # Get the velocity
        dhin = patch.state.h - inlet.h
        Vin = np.sqrt(2.0 * dhin)

        tanAlpha = turbigen.util.tand(patch.Alpha)
        tanBeta = turbigen.util.tand(patch.Beta)

        # Resolve velocity components
        Vxin = Vin / np.sqrt((1.0 + tanAlpha**2) * (1.0 + tanBeta**2))
        Vrin = Vxin * tanBeta
        Vmin = np.sqrt(Vxin**2 + Vrin**2)
        Vtin = Vmin * tanAlpha

        # Reset conserved vars on inlet
        # rho[ipatch] = inlet.rho
        rhoVx[ipatch] = inlet.rho * Vxin
        rhoVr[ipatch] = inlet.rho * Vrin
        rhorVt[ipatch] = inlet.rho * inlet.r * Vtin
        rhoe[ipatch] = inlet.rho * (inlet.u + 0.5 * Vin**2)

        # Reset pressure and hstag on inlet
        ho[ipatch] = inlet.h + 0.5 * Vin**2
        P[ipatch] = inlet.P

    Pref = 1e5
    P -= Pref

    # All nodal variables are ready
    if b._conserved_store is None:
        conserved = np.asfortranarray(np.moveaxis(conserved, 0, -1))

    return conserved, np.asfortranarray(np.stack((P, ho, b.r), axis=-1))
