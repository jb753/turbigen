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
    viscous_force,
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

    nstep_damp = 500
    """Number of steps to apply damping."""

    i_scheme = 1

    i_loss = 1

    i_exit = 1
    i_inlet = 0
    K_exit = 0.9

    plot_conv = False


def get_dw(block):
    # Cell height in each of i,j,k dirns
    dli = turbigen.util.vecnorm(block.dli)
    dlj = turbigen.util.vecnorm(block.dlj)
    dlk = turbigen.util.vecnorm(block.dlk)

    def node_to_face2(x):
        return np.stack(
            (
                x[:-1, :-1],
                x[1:, 1:],
                x[:-1, 1:],
                x[1:, :-1],
            )
        ).mean(axis=0)

    ni, nj, nk = block.shape

    dwi = np.asfortranarray(np.zeros((ni, nj - 1, nk - 1), dtype=typ))
    dwi[0, :, :] = node_to_face2(dli[0, :, :])
    dwi[-1, :, :] = node_to_face2(dli[-1, :, :])

    dwj = np.asfortranarray(np.zeros((ni - 1, nj, nk - 1), dtype=typ))
    dwj[:, 0, :] = node_to_face2(dlj[:, 0, :])
    dwj[:, -1, :] = node_to_face2(dlj[:, -1, :])

    dwk = np.asfortranarray(np.zeros((ni - 1, nj - 1, nk), dtype=typ))
    dwk[:, :, 0] = node_to_face2(dlk[:, :, 0])
    dwk[:, :, -1] = node_to_face2(dlk[:, :, -1])

    return dwi, dwj, dwk


class SolverBlock:
    """Hold just the data we need for a CFD solution."""

    def __init__(self, block):
        """Initialise from a standard Block object."""

        # Primaries
        self.conserved = np.asfortranarray(np.moveaxis(block.conserved, 0, -1)).astype(
            typ
        )

        self.mu = block.mu

        self.ho = np.asfortranarray(block.ho).astype(typ)
        self.P = np.asfortranarray(block.P).astype(typ)

        self.halfVsq = np.asfortranarray(0.5 * block.V**2).astype(typ)
        self.u = np.asfortranarray(block.u).astype(typ)

        self.xrt = np.asfortranarray(np.moveaxis(block.xrt, 0, -1)).astype(typ)

        self.dw = get_dw(block)
        self.pitch = block.pitch

        # Geometry
        self.r = np.asfortranarray(block.r).astype(typ)
        self.dAi = np.asfortranarray(np.moveaxis(block.dAi_new, 0, -1)).astype(typ)
        self.dAj = np.asfortranarray(np.moveaxis(block.dAj_new, 0, -1)).astype(typ)
        self.dAk = np.asfortranarray(np.moveaxis(block.dAk_new, 0, -1)).astype(typ)
        self.vol = np.asfortranarray(block.vol_new).astype(typ)
        self.dlmin = np.asfortranarray(block.dlmin).astype(typ)
        self.Omega = block.Omega.mean().astype(typ)
        xllim = block.pitch * 0.5 * (block.r.max() + block.r.min()) * 0.03
        xlength = np.asfortranarray(np.clip(block.w, 0.0, xllim)).astype(typ)
        xlength = (0.41 * xlength) ** 2.0
        self.xlength = np.asfortranarray(block.vol).astype(typ) * np.nan
        node_to_cell(xlength, self.xlength)
        self.mu_turb = np.asfortranarray(block.vol).astype(typ) * 0.0

        self.dU1 = self.conserved.copy(order="F").astype(typ) * np.nan
        self.dU2 = self.conserved.copy(order="F").astype(typ) * np.nan
        self._flag_scree = False

        self.conserved_avg = self.conserved.copy(order="F").astype(np.double) * 0.0

        ni, nj, nk = block.shape
        self.f = np.zeros((ni - 1, nj - 1, nk - 1, 5), order="F", dtype=typ)

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

        self.inlets = [get_inlet_data(patch) for patch in block.inlet_patches]
        self.outlets = [get_outlet_data(patch) for patch in block.outlet_patches]

        if isinstance(block, turbigen.grid.PerfectBlock):

            self.state = turbigen.fluid.PerfectState(shape=block.shape, order="F")
            self.state._metadata = block._metadata
            self.state.set_rho_u(block.rho, block.u)

            self.state_inlets = [
                turbigen.fluid.PerfectState(shape=inlet[0].shape, order="F")
                for inlet in self.inlets
            ]

            self.state_outlets = [
                turbigen.fluid.PerfectState(shape=outlet[0].shape, order="F")
                for outlet in self.outlets
            ]

        else:
            raise NotImplementedError()

        # Preallocate stored inlet density
        for inlet, state_inlet in zip(self.inlets, self.state_inlets):
            state_inlet._metadata = block._metadata
            rho_inlet = block.rho.ravel(order="F")[inlet[0]]
            u_inlet = block.u.ravel(order="F")[inlet[0]]
            state_inlet.set_rho_u(rho_inlet, u_inlet)

        # Initialise  outlet states
        for outlet, state_outlet in zip(self.outlets, self.state_outlets):
            state_outlet._metadata = block._metadata
            rho_out = block.rho.ravel(order="F")[outlet[0]]
            u_out = block.u.ravel(order="F")[outlet[0]]
            state_outlet.set_rho_u(rho_out, u_out)

    def set_inlets(self, rfin, i_inlet):
        """Set conserved variables on inlets by relaxing density changes."""

        # Change inlet patches
        for patch, state in zip(self.inlets, self.state_inlets):

            # Expand patch data
            ind, Po, To, Alpha, Beta, rhoo, hoin, sin, r, normal = patch

            tanAlpha = turbigen.util.tand(Alpha)
            tanBeta = turbigen.util.tand(Beta)
            cosBeta = turbigen.util.cosd(Beta)
            sinBeta = turbigen.util.sind(Beta)

            if i_inlet == 0:

                # Relax changes in density
                rho_now = (
                    rfin * self.conserved[..., 0].ravel(order="F")[ind]
                    + (1.0 - rfin) * state.rho
                )

                # Check for flow reversal
                rho_now[rho_now > rhoo] = rhoo * 0.9999

                # Isentropic expansion from stagnation state
                state.set_rho_s(rho_now, state.s)

                # Pull out vars we need
                h, u, P = state.h, state.u, state.P

                # Get the velocity
                dhin = hoin - h
                Vinsq = 2.0 * dhin
                Vin = np.sqrt(Vinsq)

                # Resolve velocity components
                Vxin = Vin / np.sqrt((1.0 + tanAlpha**2) * (1.0 + tanBeta**2))
                Vrin = Vxin * tanBeta
                Vmin = np.sqrt(Vxin**2 + Vrin**2)
                Vtin = Vmin * tanAlpha

                # Reset conserved vars on inlet
                # Not sure about reseting the inlet density -
                # seems to compromise stability
                # Needs a lower rfin if we reset the inlet density
                # self.conserved[..., 0].ravel(order="F")[ind] = rho_now  # rho
                self.conserved[..., 1].ravel(order="F")[ind] = rho_now * Vxin  # rhoVx
                self.conserved[..., 2].ravel(order="F")[ind] = rho_now * Vrin  # rhoVr
                self.conserved[..., 3].ravel(order="F")[ind] = (
                    rho_now * r * Vtin
                )  # rhorVt
                self.conserved[..., 4].ravel(order="F")[ind] = rho_now * (
                    u + 0.5 * Vinsq
                )

                # Reset pressure and hstag on inlet
                self.ho.ravel(order="F")[ind] = h + 0.5 * Vin**2
                self.P.ravel(order="F")[ind] = P

            else:

                # Extract properties from soln
                rho = self.conserved[..., 0].ravel(order="F")[ind]
                rhoVx = self.conserved[..., 1].ravel(order="F")[ind]
                rhoVr = self.conserved[..., 2].ravel(order="F")[ind]
                rhorVt = self.conserved[..., 3].ravel(order="F")[ind]
                P = self.P.ravel(order="F")[ind]
                ho = self.ho.ravel(order="F")[ind]

                # Calculate velocities
                Vx = rhoVx / rho
                Vr = rhoVr / rho
                Vt = rhorVt / rho / r
                Vm = np.sqrt(Vx**2 + Vr**2)

                # Update the inlet state object using soln P and rho
                # so we can read off other thermodynamic properties
                state.set_P_rho(P, rho)
                a = state.a
                s = state.s

                # Calculate the inlet residuals

                # # Giles (1988) Eqn. (5.11)
                # R = np.stack(
                #     (
                #         P * (s - sin) / cv * 2.0,
                #         rho * a * (Vt - tanAlpha * Vm),
                #         rho * (ho - hoin),
                #     )
                # )

                drho = np.empty_like(rho)
                dVm = np.empty_like(rho)
                dVt = np.empty_like(rho)
                dP = np.empty_like(rho)

                for i in range(len(ind)):

                    # Dimensional residuals
                    eps = (
                        -np.stack(
                            (
                                s[i] - sin,
                                Vt[i] - tanAlpha * Vm[i],
                                ho[i] - hoin,
                            )
                        )
                        * 0.2
                    )

                    depsdF = np.stack(
                        [
                            [state.dsdrho_P[i], 0.0, 0.0, state.dsdP_rho[i]],
                            [0.0, -tanAlpha, 1.0, 0.0],
                            [state.dhdrho_P[i], Vm[i], Vt[i], state.dhdP_rho[i]],
                        ]
                    )

                    dFdc = np.stack(
                        [
                            [-1.0 / a[i] ** 2, 0.0, 0.5 / a[i] ** 2],
                            [0.0, 0.0, 0.5 / rho[i] / a[i]],
                            [0.0, 1.0 / rho[i] / a[i], 0.0],
                            [0.0, 0.0, 0.5],
                        ]
                    )

                    B = np.linalg.inv(depsdF @ dFdc)

                    c = B @ eps

                    # Calculate primitive changes
                    ai = a[i]
                    rhoi = rho[i]
                    drho[i] = (-c[0] + 0.5 * c[2]) / ai / ai
                    dVm[i] = 0.5 * c[2] / rhoi / ai
                    dVt[i] = c[1] / rhoi / ai
                    dP[i] = 0.5 * c[2]

                # print(dVt.mean())
                # quit()
                # # # Calculate chics Eqn. (5.15)
                # # Mm = Vm / a
                # # Mt = Vt / a
                # # gfac = 1.0 / (ga - 1.0)
                # # Mtot = 1.0 + Mm + Mt * tanAlpha
                # # dc1 = -R[0]
                # # dc2 = (
                # #     -R[0] * gfac * tanAlpha + R[1] * (1.0 + Mm) + R[2] * tanAlpha
                # # ) / -Mtot
                # # dc3 = (-R[0] * gfac - R[1] * Mt + R[2]) / -Mtot

                # # Calculate primitive changes
                # drho = (-dc1 + dc3) / a / a
                # dVm = dc3 / rho / a
                # dVt = dc2 / rho / a
                # dP = dc3

                # Force onto the target beta
                Vm_new = Vm + dVm
                Vx_new = Vm_new * cosBeta
                Vr_new = Vm_new * sinBeta
                Vt_new = Vt + dVt

                # Now evaluate new flow field
                # dVx = dVm * normal[0]
                # dVr = dVm * normal[1]
                rho_new = rho + drho
                P_new = P + dP

                state.set_P_rho(P_new, rho_new)

                halfVsq_new = 0.5 * (Vx_new**2 + Vr_new**2 + Vt_new**2)

                # Reset conserved vars on inlet
                self.conserved[..., 0].ravel(order="F")[ind] = rho_new
                self.conserved[..., 1].ravel(order="F")[ind] = rho_new * Vx_new  # rhoVx
                self.conserved[..., 2].ravel(order="F")[ind] = rho_new * Vr_new  # rhoVr
                self.conserved[..., 3].ravel(order="F")[ind] = rho_new * r * Vt_new
                self.conserved[..., 4].ravel(order="F")[ind] = rho_new * (
                    state.u + halfVsq_new
                )

                # # Perturb the conserved vars
                # rho[:] = rho_new
                # rhoVx[:] = rho_new * Vx_new
                # rhoVr[:] = rho_new * Vr_new
                # rhorVt[:] = rho_new * r * Vt_new
                # rhoe[:] = rho_new * (state.u + halfVsq_new)

                # Update secondary vars
                self.u.ravel(order="F")[ind] = state.u
                self.P.ravel(order="F")[ind] = state.P
                self.ho.ravel(order="F")[ind] = state.h + halfVsq_new

    def set_outlets(self, i_exit, K_exit):
        """Set static pressure on outlets."""

        for patch, state in zip(self.outlets, self.state_outlets):

            # Extract patch data
            ind, P_exit, wA, normal, r = patch

            if i_exit == 0:

                # Stagnation enthalpy from interior and imposed exit pressure
                # set the outlet state
                ho_exit = self.ho.ravel(order="F")[ind]
                halfVsq_exit = self.halfVsq.ravel(order="F")[ind]
                h_exit = ho_exit - halfVsq_exit
                state.set_P_h(P_exit, h_exit)

                # Update conserved vars
                # rho and u change, V stay the same
                fac_rho = state.rho / self.conserved[..., 0].ravel(order="F")[ind]
                for i in range(4):
                    self.conserved[..., i].ravel(order="F")[ind] *= fac_rho
                self.conserved[..., 4].ravel(order="F")[ind] = state.rho * (
                    state.u + halfVsq_exit
                )

                # Update secondary vars
                self.u.ravel(order="F")[ind] = state.u
                self.P.ravel(order="F")[ind] = P_exit

            elif i_exit == 1:

                # Characteristic boundary condition

                # Extract the conserved vars from solution
                rho = self.conserved[..., 0].ravel(order="F")[ind]
                rhoVx = self.conserved[..., 1].ravel(order="F")[ind]
                rhoVr = self.conserved[..., 2].ravel(order="F")[ind]
                rhorVt = self.conserved[..., 3].ravel(order="F")[ind]
                rhoe = self.conserved[..., 4].ravel(order="F")[ind]

                # Update the state for this outlet from solution
                u = self.u.ravel(order="F")[ind]
                state.set_rho_u(rho, u)
                a = state.a
                P = state.P

                # Apply a uniform correction
                # Pav = np.sum(state.P*wA)
                # dP = K_exit*(P_exit - Pav)

                # Apply a local correction
                dP = (P_exit - P) * K_exit

                # dP = (P_exit - P) + (P_exit - Pav)

                # # Force towards P_exit preserving a fraction of the variation
                # # wrt area average
                # Pav = np.sum(state.P*wA)
                # dPav = P - Pav
                # dP = K_exit*((P_exit+0.5*dPav) - P)

                # Eqn. (80) from Giles (1992) gives perturbations to
                # primative properties in terms of chics
                # We want to send a wave upstream to drive the area-averaged
                # pressure towards the target without altering the other chics
                # So set c1 = c2 = c3 = 0
                # and solve for perturbations due to c4 alone
                dVm = -dP / rho / a
                drho = dP / a / a
                dVx = dVm * normal[0]
                dVr = dVm * normal[1]

                # We need to know u at the perturbed P and rho
                # So set the outlet state to new values
                rho_new = rho + drho
                P_new = state.P + dP
                state.set_P_rho(P_new, rho_new)

                # New velocities
                Vx_new = rhoVx / rho + dVx
                Vr_new = rhoVr / rho + dVr
                Vt_new = rhorVt / rho / r
                halfVsq_new = 0.5 * (Vx_new**2 + Vr_new**2 + Vt_new**2)

                # Perturb the conserved vars
                rho[:] = rho_new
                rhoVx[:] = rho_new * Vx_new
                rhoVr[:] = rho_new * Vr_new
                rhorVt[:] = rho_new * r * Vt_new
                rhoe[:] = rho_new * (state.u + halfVsq_new)

                # Update secondary vars
                self.u.ravel(order="F")[ind] = state.u
                self.P.ravel(order="F")[ind] = state.P
                self.ho.ravel(order="F")[ind] = state.h + halfVsq_new

    def set_walls(self):
        """Zero the momentums on a wall."""
        self.conserved[..., 1][self.wall_nodes] = 0.0
        self.conserved[..., 2][self.wall_nodes] = 0.0
        self.conserved[..., 3][self.wall_nodes] = 0.0
        self.conserved[..., 4][self.wall_nodes] = (
            self.conserved[..., 0][self.wall_nodes] * self.u[self.wall_nodes]
        )

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
            self.conserved,
            self.f,
            self.mu,
            self.mu_turb,
            self.xlength,
            *self.wall_indicators,
            self.vol,
            self.dAi,
            self.dAj,
            self.dAk,
            self.r,
        )


def get_periodic_data(patch):
    ind = patch.get_flat_indices("F")
    match = patch.match
    perm, flip = match.get_match_perm_flip()
    nxind = match.get_flat_indices("F", perm, flip)
    bid = patch.block.grid.index(patch.block)
    nxbid = match.block.grid.index(match.block)
    return bid, ind, nxbid, nxind


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

        bid, ind, nxbid, nxind = get_periodic_data(patch)
        periodics.append((pid, bid, procids[bid], ind, nxbid, procids[nxbid], nxind))
        pid += 1

    return periodics


def get_inlet_data(patch):

    _, di, dj, dk = np.shape(patch.get_indices())
    C = patch.get_cut()
    if di == 1:
        dA = C.dAi
    else:
        raise NotImplementedError("This assumes the patch is on a const. i face")
    normal = (dA / turbigen.util.vecnorm(dA)).mean(axis=(1, 2, 3))

    return (
        patch.get_flat_indices(order="F"),
        patch.state.P + 0.0,
        patch.state.T + 0.0,
        patch.Alpha + 0.0,
        patch.Beta + 0.0,
        patch.state.rho,
        patch.state.h,
        patch.state.s,
        patch.get_cut().r.reshape(-1),
        normal,
    )


def get_outlet_data(patch):

    # Calculate a normal vector
    # Assume the patch is flat
    _, di, dj, dk = np.shape(patch.get_indices())

    C = patch.get_cut()
    if di == 1:
        dA = C.dAi
    else:
        raise NotImplementedError("This assumes the patch is on a const. i face")
    normal = (dA / turbigen.util.vecnorm(dA)).mean(axis=(1, 2, 3))

    wA = patch.get_A_avg_weights(order="F")

    # Nodal radii
    r = C.r.reshape(-1)

    return patch.get_flat_indices(order="F"), (patch.Pout + 0.0), wA, normal, r


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


def exchange_periodics(blocks, bid_local, periodics, variable="conserved"):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Update periodic boundaries
    for patch in periodics:
        pid, bid, procid, ind, nxbid, nxprocid, nxind = patch

        b1 = blocks[bid_local[bid]]

        if variable == "conserved":
            v1 = b1.conserved
        elif variable == "residual":
            v1 = b1.dU1
        elif variable == "coords":
            v1 = b1.xrt
        nv = v1.shape[-1]

        # Just set the periodic if on same rank
        if nxprocid == rank:

            b2 = blocks[bid_local[nxbid]]
            if variable == "conserved":
                v2 = b2.conserved
            elif variable == "residual":
                v2 = b2.dU1
            elif variable == "coords":
                v2 = b2.xrt

            for i in range(nv):
                v1i = v1[..., i].ravel(order="F")
                v2i = v2[..., i].ravel(order="F")

                if variable == "coords":
                    v1ii = v1i[ind].copy()
                    v2ii = v2i[nxind].copy()
                    # Take mod wrt pitch
                    if i == 2:
                        v1ii = np.mod(v1ii, b1.pitch) + 1.0
                        v2ii = np.mod(v2ii, b2.pitch) + 1.0
                    assert np.allclose(v1ii, v2ii)
                    continue

                else:

                    avg = 0.5 * (v1i[ind] + v2i[nxind])
                    v1i[ind] = avg
                    v2i[nxind] = avg

        # Otherwise, communication is needed
        else:

            # Preallocate a buffer to recieve data
            di = len(ind)
            count = di * nv
            nxv = np.empty((count,), dtype=typ)

            # Assemble data to send
            vs = np.empty((count,), dtype=typ)
            for i in range(nv):
                ist = i * di
                ien = (i + 1) * di
                vs[ist:ien] = v1[..., i].ravel(order="F")[ind]

            # If our rank is lower than next rank, send first
            if rank < nxprocid:
                comm.Send([vs, count, MPI.REAL4], dest=nxprocid, tag=pid)
                comm.Recv([nxv, count, MPI.REAL4], source=nxprocid, tag=pid)
            # If our rank is higher than next rank, recieve first
            else:
                comm.Recv([nxv, count, MPI.REAL4], source=nxprocid, tag=pid)
                comm.Send([vs, count, MPI.REAL4], dest=nxprocid, tag=pid)

            # Take average over both sides
            vavg = 0.5 * (vs + nxv)

            # Assign back to the grid
            for i in range(nv):
                ist = i * di
                ien = (i + 1) * di
                v1[..., i].ravel(order="F")[ind] = vavg[ist:ien]


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
    rfin = 0.1

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

    # Check periodics
    exchange_periodics(blocks, bid_local, periodics, variable="coords")

    # Start the main time stepping loop
    for istep in range(conf.n_step):

        # if not np.mod(istep, conf.n_step_dt):
        exchange_periodics(blocks, bid_local, periodics, variable="conserved")

        # Calculate residual for all blocks
        for iblock in range(nblock):

            sb = blocks[iblock]

            if not np.mod(istep, conf.n_step_dt):
                sb.set_timestep(conf.CFL)

            sb.set_inlets(rfin, conf.i_inlet)

            sb.set_outlets(conf.i_exit, conf.K_exit)

            sb.residual()

        # Send residuals to other blocks
        exchange_periodics(blocks, bid_local, periodics, variable="residual")

        # Now integrate forward
        istep_avg = conf.n_step - conf.n_step_avg
        for iblock in range(nblock):
            sb = blocks[iblock]

            if conf.damping_factor and istep > conf.nstep_damp:
                sb.damp(conf.damping_factor)

            sb.step(istep, istep_avg, conf.n_step_avg, conf.i_scheme)

            sb.smooth(sf2, sf4)

            sb.set_secondary()

            if not np.mod(istep, 2) and istep > 100 and conf.i_loss > 0:
                sb.calculate_viscous()

        if conf.n_step_log > 0 and not np.mod(istep, conf.n_step_log) and istep > 0:

            # Send residuals to master proc
            dUnow = np.stack([np.abs(b.dU1.mean(axis=(0, 1, 2))) for b in blocks])

            if rank:
                comm.send(dUnow, dest=0)
                terminate = np.empty((1,), dtype=int)
                comm.Recv([terminate, 1, MPI.INT], source=0, tag=rank)
                if terminate:
                    comm.send(blocks, dest=0)
                    return
            else:
                dUall = [
                    dUnow,
                ]
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
                    dUe_ref = dUlognow[-1] * conf.conv_lim

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

    mdot_in = 0.0
    for patch in grid.inlet_patches:
        Cm, A, _ = patch.get_cut().mix_out()
        mdot_in += Cm.rho * Cm.Vm * A

    mdot_out = 0.0
    for patch in grid.outlet_patches:
        Cm, A, _ = patch.get_cut().mix_out()
        mdot_out += Cm.rho * Cm.Vm * A

    logger.info(f"Mass flow error: {(mdot_in/mdot_out-1.)*100.:.1f}%")

    dUlog = np.array(dUlog)

    if conf.plot_conv:
        drho_ref = dUlog[1, 0]
        drhoVx_ref = dUlog[1, 1]
        drhoVr_ref = dUlog[1, 2]
        drhoVt_ref = dUlog[1, 3] / np.mean(blocks_out[0].r)
        drhoV_ref = np.max((drhoVx_ref, drhoVr_ref, drhoVt_ref))
        drhoe_ref = dUlog[1, 4]

        dUlog[:, 0] /= drho_ref
        dUlog[:, 1:4] /= drhoV_ref
        dUlog[:, 4] /= drhoe_ref

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.semilogy(dUlog)
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
