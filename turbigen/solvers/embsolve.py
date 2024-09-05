import numpy as np
import turbigen.util
import turbigen.fluid
import turbigen.flowfield
import turbigen.grid
from dataclasses import dataclass
from turbigen.solvers.base import BaseSolver

# from turbigen.embsolve import embsolve

from pathlib import Path

from turbigen.solvers.embsolvec import *

from timeit import default_timer as timer

import logging

logger = turbigen.util.make_logger()

logger.setLevel(level=logging.INFO)

typ = np.float32

try:
    from mpi4py import MPI
except ImportError:
    pass


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

    xllim_pitch: float = 0.03

    i_scheme: int = 1

    i_loss: int = 1

    i_exit: int = 1
    i_inlet: int = 1
    K_exit: float = 0.9
    K_inlet: float = 0.7

    plot_conv: bool = False

    tauw_lam_mult: float = 1.0
    tauw_turb_mult: float = 1.0

    rf_periodic: float = 1.0

    fmgrid: float = 0.2
    multigrid: tuple = (2, 2, 2)


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


def to_fort(x):
    if x.ndim > 3:
        x2 = np.moveaxis(x, 0, -1)
    else:
        x2 = x
    return np.asfortranarray(x2).astype(typ)


class SolverBlock:
    """Hold just the data we need for a CFD solution."""

    def __init__(self, block, conf):
        """Initialise from a standard Block object."""

        # Primaries
        self.cons = to_fort(block.conserved)

        self.conf = conf
        self.Nb = block.Nb

        self.mu = block.mu

        self.ho = to_fort(block.ho)
        self.P = to_fort(block.P)
        self.Pref = to_fort(self.P.mean())

        self.halfVsq = to_fort(0.5 * block.V**2)
        self.u = to_fort(block.u)
        self.T = to_fort(block.T)

        self.dw = get_dw(block)
        self.pitch = block.pitch

        # Geometry
        self.x = to_fort(block.x)
        self.r = to_fort(block.r)
        self.t = to_fort(block.t)
        self.rf = [to_fort(r) for r in block.r_face]

        self.rc = to_fort(np.zeros_like(block.vol))
        embsolve.node_to_cell(self.r, self.rc)

        self.dAi = to_fort(block.dAi_new)
        self.dAj = to_fort(block.dAj_new)
        self.dAk = to_fort(block.dAk_new)
        self.vol = get_multigrid_volumes(block.vol_new, conf.multigrid)
        self.dlmin = get_multigrid_lengths(block, conf.multigrid)
        self.dt = self.dlmin * 0.0

        self.Vxrt = to_fort(block.Vxrt)
        self.Omega = block.Omega.mean().astype(typ)
        xllim = (
            block.pitch * 0.5 * (block.r.max() + block.r.min()) * self.conf.xllim_pitch
        )

        self.cons_avg = self.cons.copy(order="F").astype(np.double) * 0.0

        ni, nj, nk = block.shape
        self.fb = np.zeros((ni - 1, nj - 1, nk - 1, 5), order="F", dtype=typ)
        self.dU1 = np.zeros((ni - 1, nj - 1, nk - 1, 5), order="F", dtype=typ)
        self.dU2 = np.zeros((ni - 1, nj - 1, nk - 1, 5), order="F", dtype=typ)

        xlength = np.asfortranarray(np.clip(block.w, 0.0, xllim)).astype(typ)
        xlength = (0.41 * xlength) ** 2.0
        self.xlength = to_fort(np.zeros((ni - 1, nj - 1, nk - 1)))
        embsolve.node_to_cell(xlength, self.xlength)

        # Get indices for multigrid
        self.ijk_multigrid = (
            get_multigrid_indices((ni - 1, nj - 1, nk - 1), conf.multigrid) + 1
        )

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

        # self.dA_face = [
        #     to_fort(embsolve.get_by_ijk(dA, ijk).reshape(-1,3).T)
        #     for dA, ijk in zip(dAijk, self.ijk_wall_face_slip)
        # ]

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
        # self.L = to_fort(np.ones((3, ni, nj, nk))/3.)

        # print('at ni//2, nj//2, k=0')
        # print(self.L[ni//2, nj//2, 0, :])
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(self.L[ni//2,nj//2,:,1])
        # plt.show()

        self.tau = [
            to_fort(np.zeros((6, ni, nj - 1, nk - 1))),
            to_fort(np.zeros((6, ni - 1, nj, nk - 1))),
            to_fort(np.zeros((6, ni - 1, nj - 1, nk))),
        ]

        self.inlets = [get_inlet_data(patch) for patch in block.inlet_patches]
        self.outlets = [get_outlet_data(patch) for patch in block.outlet_patches]

        if isinstance(block, turbigen.grid.PerfectBlock):
            self.state = turbigen.fluid.PerfectState(
                shape=block.shape, order="F", typ=typ
            )
            self.state.gamma = typ(block.gamma)
            self.state.cp = typ(block.cp)
            self.state.mu = typ(block.mu)
            self.state.set_rho_u(block.rho, block.u)

            self.state_inlets = [
                turbigen.fluid.PerfectState(shape=inlet[0].shape, order="F", typ=typ)
                for inlet in self.inlets
            ]

            self.state_outlets = [
                turbigen.fluid.PerfectState(shape=outlet[0].shape, order="F", typ=typ)
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

    def set_inlets(self, rfin, i_inlet, K_inlet):
        """Set cons variables on inlets by relaxing density changes."""

        # Change inlet patches
        for patch, state in zip(self.inlets, self.state_inlets):
            # Expand patch data
            ind, Po, To, Alpha, Beta, rhoo, hoin, sin, r, _ = patch

            tanAl = turbigen.util.tand(Alpha)
            tanBeta = turbigen.util.tand(Beta)
            cosBeta = turbigen.util.cosd(Beta)
            sinBeta = turbigen.util.sind(Beta)

            if i_inlet == 0:
                # Relax changes in density
                rho_now = (
                    rfin * self.cons[..., 0].ravel(order="F")[ind]
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
                Vxin = Vin / np.sqrt((1.0 + tanAl**2) * (1.0 + tanBeta**2))
                Vrin = Vxin * tanBeta
                Vmin = np.sqrt(Vxin**2 + Vrin**2)
                Vtin = Vmin * tanAl

                # Reset cons vars on inlet
                # Not sure about reseting the inlet density -
                # seems to compromise stability
                # Needs a lower rfin if we reset the inlet density
                # self.cons[..., 0].ravel(order="F")[ind] = rho_now  # rho
                self.cons[..., 1].ravel(order="F")[ind] = rho_now * Vxin  # rhoVx
                self.cons[..., 2].ravel(order="F")[ind] = rho_now * Vrin  # rhoVr
                self.cons[..., 3].ravel(order="F")[ind] = rho_now * r * Vtin  # rhorVt
                self.cons[..., 4].ravel(order="F")[ind] = rho_now * (u + 0.5 * Vinsq)

                # Reset pressure and hstag on inlet
                self.ho.ravel(order="F")[ind] = h + 0.5 * Vin**2
                self.P.ravel(order="F")[ind] = P

            else:
                # Extract properties from soln
                rho = self.cons[..., 0].ravel(order="F")[ind]
                rhoVx = self.cons[..., 1].ravel(order="F")[ind]
                rhoVr = self.cons[..., 2].ravel(order="F")[ind]
                rhorVt = self.cons[..., 3].ravel(order="F")[ind]
                P = self.P.ravel(order="F")[ind]
                ho = self.ho.ravel(order="F")[ind]

                # Calculate velocities
                Vx = rhoVx / rho
                Vr = rhoVr / rho
                Vt = rhorVt / rho / r
                Vm = np.sqrt(Vx**2 + Vr**2)
                V = np.sqrt(Vx**2 + Vr**2 + Vt**2)

                # Update the inlet state object using soln P and rho
                # so we can read off other thermodynamic properties
                state.set_P_rho(P, rho)
                a = state.a
                s = state.s
                dsdrho = state.dsdrho_P
                dhdrho = state.dhdrho_P
                dsdP = state.dsdP_rho
                dhdP = state.dhdP_rho
                rhoa = rho * a
                asq = a * a

                # Scaling of changes based on patch Mach
                # Reduce K_inlet at high Mach
                Ma_ref = np.mean(V / a)
                scale_Ma = np.interp(Ma_ref, [0.3, 1.0], [1.0, 0.5])

                # Calculate the inlet residuals

                # Dimensional residuals
                eps = (
                    np.stack(
                        (
                            s - sin,
                            Vt - tanAl * Vm,
                            ho - hoin,
                        ),
                        axis=-1,
                    )
                    * -K_inlet
                    * scale_Ma
                )[..., None]

                # Jacobian
                fac = (
                    -Vm * dsdrho / rhoa
                    - Vt * dsdrho * tanAl / rhoa
                    - dhdP * dsdrho
                    + dhdrho * dsdP
                )
                dcdeps = (
                    np.stack(
                        [
                            [
                                Vm * a / rho
                                + Vt * a * tanAl / rho
                                + asq * dhdP
                                + dhdrho,
                                Vt * (asq * dsdP + dsdrho),
                                -asq * dsdP - dsdrho,
                            ],
                            [
                                dhdrho * tanAl,
                                -Vm * dsdrho
                                - rhoa * dhdP * dsdrho
                                + rhoa * dhdrho * dsdP,
                                -dsdrho * tanAl,
                            ],
                            [dhdrho, Vt * dsdrho, -dsdrho],
                        ]
                    )
                    / fac
                )
                dcdeps[2] *= 2
                dcdeps = np.moveaxis(dcdeps, 2, 0)

                c = dcdeps @ eps
                c = c.squeeze().T

                # Calculate primitive changes
                drho = (-c[0] + 0.5 * c[2]) / asq
                dVm = 0.5 * c[2] / rhoa
                dVt = c[1] / rhoa
                dP = 0.5 * c[2]

                # Now evaluate new flow field
                # Force onto the target beta
                rho_new = rho + drho
                P_new = P + dP
                Vm_new = Vm + dVm
                Vx_new = Vm_new * cosBeta
                Vr_new = Vm_new * sinBeta
                Vt_new = Vt + dVt

                state.set_P_rho(P_new, rho_new)

                halfVsq_new = 0.5 * (Vx_new**2 + Vr_new**2 + Vt_new**2)

                # Reset cons vars on inlet
                self.cons[..., 0].ravel(order="F")[ind] = rho_new
                self.cons[..., 1].ravel(order="F")[ind] = rho_new * Vx_new  # rhoVx
                self.cons[..., 2].ravel(order="F")[ind] = rho_new * Vr_new  # rhoVr
                self.cons[..., 3].ravel(order="F")[ind] = rho_new * r * Vt_new
                self.cons[..., 4].ravel(order="F")[ind] = rho_new * (
                    state.u + halfVsq_new
                )

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

                # Update cons vars
                # rho and u change, V stay the same
                fac_rho = state.rho / self.cons[..., 0].ravel(order="F")[ind]
                for i in range(4):
                    self.cons[..., i].ravel(order="F")[ind] *= fac_rho
                self.cons[..., 4].ravel(order="F")[ind] = state.rho * (
                    state.u + halfVsq_exit
                )

                # Update secondary vars
                self.u.ravel(order="F")[ind] = state.u
                self.P.ravel(order="F")[ind] = P_exit

            elif i_exit == 1:
                # Characteristic boundary condition

                # Extract the cons vars from solution
                rho = self.cons[..., 0].ravel(order="F")[ind]
                rhoVx = self.cons[..., 1].ravel(order="F")[ind]
                rhoVr = self.cons[..., 2].ravel(order="F")[ind]
                rhorVt = self.cons[..., 3].ravel(order="F")[ind]
                rhoe = self.cons[..., 4].ravel(order="F")[ind]

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

                # Perturb the cons vars
                rho[:] = rho_new
                rhoVx[:] = rho_new * Vx_new
                rhoVr[:] = rho_new * Vr_new
                rhorVt[:] = rho_new * r * Vt_new
                rhoe[:] = rho_new * (state.u + halfVsq_new)

                # Update secondary vars
                self.u.ravel(order="F")[ind] = state.u
                self.P.ravel(order="F")[ind] = state.P
                self.ho.ravel(order="F")[ind] = state.h + halfVsq_new

    def set_timestep(self, CFL, relax=0.0):
        embsolve.set_timesteps(
            self.dt,
            self.vol,
            self.state.a,
            self.Vxrt,
            self.dlmin,
            self.ijk_multigrid,
            CFL,
            relax,
        )

    def residual(self, fmgrid, damp, ischeme, sf2, sf4, sf2min):
        embsolve.residual(
            self.cons,
            self.Vxrt,
            self.P,  # Only pressure differences matter
            self.Pref,
            self.ho,
            self.fb,
            self.Omega,
            self.r,
            *self.rf,
            self.dAi,
            self.dAj,
            self.dAk,
            self.vol,
            self.dt,
            *self.ijk_wall_face,
            self.ijk_multigrid,
            fmgrid,
            damp,
            sf2,
            sf4,
            sf2min,
            self.L,
            self.dU1,
            self.dU2,
            ischeme,
        )

    def step(self, istep, ischeme):
        embsolve.step(
            self.cons,
            self.dU1,
            self.dU2,
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

    # def multigrid(self, fmgrid):
    #     embsolve.multigrid(self.dU1, self.ijk_multigrid, fmgrid)

    @profile
    def set_viscous_stress(self):
        embsolve.shear_stress(
            self.cons,
            self.Vxrt,
            self.mu,
            self.xlength,
            self.vol[..., 0],
            self.dAi,
            self.dAj,
            self.dAk,
            self.r,
            self.rc,
            *self.rf,
            *self.ijk_wall_face_slip,
            *self.dw_face,
            *self.dA_face,
            self.fb,
        )

    def set_viscous_force(self):
        embsolve.viscous_force(
            self.fb,
            self.cons,
            *self.tau,
            self.dAi,
            self.dAj,
            self.dAk,
            self.r,
            *self.rf,
            *self.ijk_wall_face_slip,
            *self.dw_face,
            *self.dA_face,
            self.mu,
            self.conf.tauw_lam_mult,
            self.conf.tauw_turb_mult,
        )


def trim_i(ijk):
    if ijk[0, 0, 0, 0] < ijk[0, -1, 0, 0]:
        # i is in ascending order
        ijk = ijk[:, :-1, :, :]
    else:
        # i is in descending order
        ijk = ijk[:, 1:, :, :]
    return ijk


def trim_j(ijk):
    if ijk[1, 0, 0, 0] < ijk[1, 0, -1, 0]:
        # j is in ascending order
        ijk = ijk[:, :, :-1, :]
    else:
        # j is in descending order
        ijk = ijk[:, :, 1:, :]
    return ijk


def trim_k(ijk):
    if ijk[2, 0, 0, 0] < ijk[2, 0, 0, -1]:
        # k is in ascending order
        ijk = ijk[:, :, :, :-1]
    else:
        # k is in descending order
        ijk = ijk[:, :, :, 1:]
    return ijk


def face_indices(ijk):
    # Given (3, di, dj, dk) array of nodal indices, get face indices

    # Choose direction
    _, di, dj, dk = ijk.shape

    if di == 1:
        # Constant-i face
        # Trim off the highest valued j and k
        ijk = trim_j(ijk)
        ijk = trim_k(ijk)

    elif dj == 1:
        # Constant-j face
        # Trim off the highest valued i and k
        ijk = trim_i(ijk)
        ijk = trim_k(ijk)

    elif dk == 1:
        # Constant-k face
        # Trim off the highest valued i and j
        ijk = trim_i(ijk)
        ijk = trim_j(ijk)

    return ijk


def get_periodic_data(patch):
    match = patch.match
    perm, flip = match.get_match_perm_flip()

    ijk = patch.get_indices()
    nxijk = match.get_indices(perm, flip)

    ijkf = face_indices(ijk).reshape(3, -1)
    nxijkf = face_indices(nxijk).reshape(3, -1)

    bid = patch.block.grid.index(patch.block)
    nxbid = match.block.grid.index(match.block)

    d = ijk.shape.index(1)
    nxd = nxijk.shape.index(1)

    ijk = ijk.reshape(3, -1)
    nxijk = nxijk.reshape(3, -1)

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

    # Check the indices are in correct range
    assert ijk.min() >= 0
    assert ijk[0].max() < b1.ni
    assert ijk[1].max() < b1.nj
    assert ijk[2].max() < b1.nk

    assert nxijk.min() >= 0
    assert nxijk[0].max() < b2.ni
    assert nxijk[1].max() < b2.nj
    assert nxijk[2].max() < b2.nk

    # For Fortran
    ijk = np.asfortranarray(ijk + 1).astype(np.int16)
    nxijk = np.asfortranarray(nxijk + 1).astype(np.int16)
    ijkf = np.asfortranarray(ijkf + 1).astype(np.int16)
    nxijkf = np.asfortranarray(nxijkf + 1).astype(np.int16)

    return bid, ijk, ijkf, d, nxbid, nxijk, nxijkf, nxd


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

        bid, ind, indf, d, nxbid, nxind, nxindf, nxd = get_periodic_data(patch)
        periodics.append(
            (
                pid,
                bid,
                procids[bid],
                ind,
                indf,
                d,
                nxbid,
                procids[nxbid],
                nxind,
                nxindf,
                nxd,
            )
        )
        pid += 1

    return periodics


def get_inlet_data(patch):
    _, di, dj, dk = np.shape(patch.get_indices())

    wA = patch.get_A_avg_weights(order="F")

    return (
        patch.get_flat_indices(order="F"),
        patch.state.P + 0.0,
        patch.state.T + 0.0,
        patch.Alpha + 0.0,
        patch.Beta + 0.0,
        patch.state.rho,
        patch.state.h,
        patch.state.s,
        to_fort(patch.get_cut().r.reshape(-1)),
        wA,
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


def send_slave(block_split, procids, periodics):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    for iproc in range(1, size):
        comm.send(block_split[iproc], dest=iproc)

    comm.Barrier()

    for iproc in range(1, size):
        comm.send(periodics, dest=iproc)

    comm.Barrier()


def exchange_cons(blocks, bid_local, periodics, rf):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Update periodic boundaries
    for patch in periodics:
        pid, bid, procid, ijk, _, _, nxbid, nxprocid, nxijk, _, _ = patch

        b1 = blocks[bid_local[bid]]

        v1 = b1.cons

        # Just set the periodic if on same rank
        if nxprocid == rank:
            b2 = blocks[bid_local[nxbid]]
            v2 = b2.cons

            # # print(ijk.shape)
            # # quit()
            # for ipt in range(ijk.shape[0]):
            #     i1, j1, k1 = ijk[:,ipt]-1
            #     i2, j2, k2 = nxijk[:,ipt]-1
            #     assert np.isclose(b1.r[i1, j1, k1], b2.r[i2, j2, k2])
            #     assert np.isclose(b1.x[i1, j1, k1], b2.x[i2, j2, k2])

            embsolve.average_by_ijk(v1, v2, ijk, nxijk, rf)

        # Otherwise, communication is needed
        else:
            # Assemble data to send
            vs = embsolve.get_by_ijk(v1, ijk)
            count = len(vs)

            # Preallocate a buffer to recieve
            nxv = np.empty_like(vs, dtype=typ)

            # If our rank is lower than next rank, send first
            if rank < nxprocid:
                comm.Send([vs, count, MPI.REAL4], dest=nxprocid, tag=pid)
                comm.Recv([nxv, count, MPI.REAL4], source=nxprocid, tag=pid)
            # Otherwise, recieve first
            else:
                comm.Recv([nxv, count, MPI.REAL4], source=nxprocid, tag=pid)
                comm.Send([vs, count, MPI.REAL4], dest=nxprocid, tag=pid)

            # Take average over both sides
            vavg = 0.5 * (vs + nxv)
            vnew = vavg * rf + vs * (1.0 - rf)
            embsolve.set_by_ijk(v1, vnew, ijk)


def exchange_tau(blocks, bid_local, periodics):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Update periodic boundaries
    for patch in periodics:
        pid, bid, procid, ijk, ijkf, d, nxbid, nxprocid, nxijk, nxijkf, nxd = patch

        # Extract home block
        b1 = blocks[bid_local[bid]]

        # Chose ijk face direction for home patch
        tau1 = b1.tau[d - 1]

        # Just set the periodic if on same rank
        if nxprocid == rank:
            # Extract away block
            b2 = blocks[bid_local[nxbid]]

            # Chose ijk face direction for away patch
            tau2 = b2.tau[nxd - 1]

            embsolve.average_by_ijk(tau1, tau2, ijkf, nxijkf, 1.0)

        # Otherwise, communication is needed
        else:
            # Assemble data to send
            vs = embsolve.get_by_ijk(tau1, ijkf)
            count = len(vs)

            # Preallocate a buffer to recieve
            nxv = np.empty_like(vs, dtype=typ)

            # If our rank is lower than next rank, send first
            if rank < nxprocid:
                comm.Send([vs, count, MPI.REAL4], dest=nxprocid, tag=pid)
                comm.Recv([nxv, count, MPI.REAL4], source=nxprocid, tag=pid)
            # Otherwise, recieve first
            else:
                comm.Recv([nxv, count, MPI.REAL4], source=nxprocid, tag=pid)
                comm.Send([vs, count, MPI.REAL4], dest=nxprocid, tag=pid)

            # Take average over both sides
            vavg = 0.5 * (vs + nxv)
            embsolve.set_by_ijk(tau1, vavg, ijkf)


# @profile
def run_slave(blocks=None, periodics_all=None, nodes=None, conf=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if blocks is None:
        blocks = comm.recv()
        comm.Barrier()
        periodics_all = comm.recv()
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
    rfin = 0.1

    # Only keep relevent periodics
    # And rearrange the periodics so that foreign procid is always nx
    periodics = []
    for patch in periodics_all:
        pid, bid, procid, ind, indf, d, nxbid, nxprocid, nxind, nxindf, nxd = patch
        if procid == rank:
            periodics.append(patch)
        elif nxprocid == rank:
            periodics.append(
                (pid, nxbid, nxprocid, nxind, nxindf, nxd, bid, procid, ind, indf, d)
            )

    bids = [b.bid for b in blocks]

    # Lookup of local bid from global bid
    bid_local = {bid: ibid for ibid, bid in enumerate(bids)}

    nblock = len(blocks)

    tstart = timer()

    dUnow = np.empty((conf.n_step_log, nblock, 5))

    # Now integrate forward
    istep_avg = conf.n_step - conf.n_step_avg

    # Initialise a conservative time step
    for iblock in range(nblock):
        blocks[iblock].set_timestep(conf.CFL * 0.5)

    try:
        # Start the main time stepping loop
        for istep in range(conf.n_step):

            # Ramping factors
            damping_ramp = np.interp(istep, [0, conf.n_step_ramp], [0.5, 1.0])
            smoothing_ramp = np.interp(istep, [0, conf.n_step_ramp], [2.0, 1.0])
            cfl_ramp = typ(np.interp(istep, [0, conf.n_step_ramp], [0.5, 1.0]))
            fmgrid_ramp = typ(np.interp(istep, [0, conf.n_step_ramp], [0.0, 1.0]))

            # Exchange conserved variables across periodic patches
            exchange_cons(blocks, bid_local, periodics, conf.rf_periodic)

            # Update boundary conditions and calculate residual for all blocks
            for iblock in range(nblock):
                sb = blocks[iblock]

                sb.set_secondary()

                # Accumulate average
                if istep >= istep_avg:
                    sb.cons_avg += sb.cons / float(conf.n_step_avg)

                # Update time steps using local Mach
                # With a relaxation factor so they do not change too rapidly
                if not np.mod(istep, conf.n_step_dt):
                    sb.set_timestep(conf.CFL * cfl_ramp, relax=0.25)

                # Apply boundary conditions
                sb.set_inlets(rfin, conf.i_inlet, K_inlet)
                sb.set_outlets(conf.i_exit, K_exit)

                # If this is a viscous calculation
                # Update the viscous forces every nloss time steps
                # Not right at the start of the calculation for stability
                if not np.mod(istep, conf.n_loss) and istep > 100 and conf.i_loss > 0:
                    # Evaluate the components of viscous stress tensor
                    sb.set_viscous_stress()

                    # Average the stresses on periodic faces
                    # exchange_tau(blocks, bid_local, periodics)

                # Using the updated flow field, sum fluxes for each cell
                # and store the residual
                # Apply damping to cell residuals if requested
                if conf.damping_factor and (
                    istep < conf.nstep_damp or conf.nstep_damp < 0
                ):
                    damp = conf.damping_factor * damping_ramp
                else:
                    damp = 0.0

                i_scheme = -1 if not istep else conf.i_scheme
                sb.residual(
                    conf.fmgrid * fmgrid_ramp,
                    damp,
                    i_scheme,
                    sf2 * smoothing_ramp,
                    sf4 * smoothing_ramp,
                    sf2min * smoothing_ramp,
                )

            # Record residuals
            iilog = np.mod(istep - 1, conf.n_step_log)
            dUnow[iilog] = np.stack(
                [np.abs(b.dU1.mean(axis=(0, 1, 2))) for b in blocks]
            )

            # Intermittently print convergence
            if (not np.mod(istep, conf.n_step_log)) and (istep > 0):
                # Inlet mass-avg props
                mdot1 = 0.0
                s1 = 0.0
                ho1 = 0.0
                h1 = 0.0
                A1 = 0.0
                for sb in blocks:
                    for patch in sb.inlets:
                        ind, _, _, _, _, _, _, _, _, wA = patch
                        rhoVxrt = np.stack(
                            (
                                sb.cons[:, :, :, 1].ravel(order="F")[ind],
                                sb.cons[:, :, :, 2].ravel(order="F")[ind],
                                sb.cons[:, :, :, 3].ravel(order="F")[ind],
                            )
                        )
                        s = sb.state.s.ravel(order="F")[ind]
                        h = sb.state.h.ravel(order="F")[ind]
                        ho = sb.ho.ravel(order="F")[ind]
                        mdotnow = (wA * rhoVxrt).sum()
                        mdot1 += mdotnow * sb.Nb
                        s1 += (wA * rhoVxrt * s).sum()
                        ho1 += (wA * rhoVxrt * ho).sum()
                        h1 += (wA * h).sum()
                        A1 += wA.sum()

                # Outlet mass-avg props
                mdot2 = 0.0
                s2 = 0.0
                ho2 = 0.0
                h2 = 0.0
                A2 = 0.0
                T2 = 0.0
                for sb in blocks:
                    for patch, state in zip(sb.outlets, sb.state_outlets):
                        ind, _, wA, _, _ = patch
                        rhoVxrt = np.stack(
                            (
                                sb.cons[:, :, :, 1].ravel(order="F")[ind],
                                sb.cons[:, :, :, 2].ravel(order="F")[ind],
                                sb.cons[:, :, :, 3].ravel(order="F")[ind],
                            )
                        )
                        s = state.s
                        ho = sb.ho.ravel(order="F")[ind]
                        T = state.T
                        h = state.h
                        mdotnow = (wA * rhoVxrt).sum()
                        mdot2 += mdotnow * sb.Nb
                        s2 += (wA * rhoVxrt * s).sum()
                        ho2 += (wA * rhoVxrt * ho).sum()
                        T2 += (wA * T).sum()
                        h2 += (wA * h).sum()
                        A2 += wA.sum()

                mhs = np.array([mdot1, ho1, s1, mdot2, ho2, s2, h1, h2, A1, A2, T2])

                # Send residuals to master proc
                if rank:
                    comm.send(dUnow, dest=0)
                    comm.send(mhs, dest=0)

                else:
                    dUall = [
                        dUnow,
                    ]
                    for iproc in range(1, size):
                        dUall.append(comm.recv(source=iproc))
                        mhs += comm.recv(source=iproc)

                    dUall = np.concatenate(dUall, axis=1)

                    with np.errstate(divide="ignore", invalid="ignore"):

                        # Mass weight all inlet/exits
                        mhs[1:3] /= mhs[0]
                        mhs[4:6] /= mhs[3]
                        merr = (mhs[3] / mhs[0] - 1.0) * 100.0
                        merrlog.append(merr)

                        # Area weight static enthalpy
                        mhs[6] /= mhs[8]
                        mhs[7] /= mhs[9]
                        mhs[10] /= mhs[9]
                        Ys = mhs[10] * (mhs[5] - mhs[2]) / (mhs[4] - mhs[7])
                    Yslog.append(Ys)

                    ten = timer()
                    tpnps = (ten - tstart) / nodes / conf.n_step_log
                    tstart = ten

                    logger.info(f"{istep}: tpnps={tpnps:.3e}")
                    logger.info(
                        "  Mass in, out, err = "
                        f"{mhs[0]:.1e} / {mhs[3]:.1e} / {merr:.1f}%"
                    )
                    logger.info(f"   Ys = {Ys:.3e}")
                    for ib, dU in enumerate(dUall.mean(axis=0)):
                        logger.info(
                            f"  block {ib}: "
                            f"{dU[0]:.2e} {dU[1]:.2e} {dU[2]:.2e} "
                            f"{dU[3]:.2e} {dU[4]:.2e}"
                        )

                    dUlognow = np.stack(dUall).mean(axis=1)
                    dUlog.append(dUlognow)

    except KeyboardInterrupt:
        for iblock in range(nblock):
            sb = blocks[iblock]
            sb.cons_avg = sb.cons

    if master_flag:
        return blocks, dUlog, merrlog, Yslog
    else:
        comm.send(blocks, dest=0)


def run(grid, conf, machine=None):

    if isinstance(conf, dict):
        conf = Config(**conf)

    if conf.skip:
        logger.info("Skipping, doing nothing.")
        return

    logger.info("Intialising native solver...")

    nodes = np.sum([b.size for b in grid])

    blocks = [SolverBlock(b, conf) for b in grid]
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
    block_split[0], dUlog, merrlog, Yslog = run_slave(block_split[0], periodics, nodes)

    for iproc in range(1, size):
        block_split[iproc] = comm.recv(source=iproc)

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

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.contourf(grid[0].x[:,:,1], grid[0].r[:,:,1],blocks_out[0].dU1[:,:,1,0])
    # ax.plot(grid[0].x[:,:,1], grid[0].r[:,:,1],'k-',lw=0.2)
    # ax.plot(grid[0].x[:,:,1].T, grid[0].r[:,:,1].T,'k-',lw=0.2)
    # ax.axis('equal')
    # fig, ax = plt.subplots()
    # b = grid[0]
    # ax.contourf(b.y[b.ni//2,:,:], b.z[b.ni//2,:,:],blocks_out[0].dU1[b.ni//2,:,:,0])
    # ax.plot(b.y[b.ni//2,:,:], b.z[b.ni//2,:,:],'k-',lw=0.2)
    # ax.plot(b.y[b.ni//2,:,:].T, b.z[b.ni//2,:,:].T,'k-',lw=0.2)
    # ax.axis('equal')
    # plt.show()

    # fig, ax = plt.subplots()
    # # ax.contourf(grid[0].x[:,:,1], grid[0].r[:,:,1],blocks_out[0].nu[:,:,1,0])
    # # ax.axis('equal')
    # x = grid[0].x[:, 0, 1]
    # nu = blocks_out[0].nu[:, 0, 1, 0]
    # k2 = 1.0
    # eps4 = 0.005
    # sf2 = k2 * nu
    # sf4 = np.maximum(0.0, eps4 - sf2)
    # ax.plot(x, sf2)
    # ax.plot(x, sf4)
    # plt.show()

    if not mdot_out == 0.0:
        logger.info(f"Mass flow error: {(mdot_in/mdot_out-1.)*100.:.1f}%")

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

        fig, ax = plt.subplots()
        ax.plot(Yslog)
        ax.set_ylabel("Entropy Loss Coefficient, $Y_s$")
        plt.tight_layout()

        fig, ax = plt.subplots()
        ax.plot(merrlog)
        ax.set_ylabel(r"Mass Conservation Error $\varepsilon \dot{m}/\%$")
        plt.tight_layout()
        plt.show()


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

    # Loop over multigrid levels
    for ilev in range(nlev):

        # Number of cells along each side of this
        # multigrid level is product of all previous
        nbi = np.prod(nb[: ilev + 1])

        # Loop over all points in the block
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):

                    # Integer divide to find the multigrid block index
                    ijkmg[:, i, j, k, ilev] = (
                        i // nbi,
                        j // nbi,
                        k // nbi,
                    )

    assert (ijkmg >= 0).all()
    return ijkmg


def get_multigrid_volumes(vol, nb):

    # Preallocate output array
    ijkmg = get_multigrid_indices(vol.shape, nb)
    nlev = len(nb)
    volmg = np.asfortranarray(np.zeros(vol.shape + (nlev + 1,), dtype=typ))
    ni, nj, nk = vol.shape

    # Finest grid level is trivial
    volmg[..., 0] = vol

    # Loop over multigrid levels
    for ilev in range(nlev):
        # Loop over all points in the block
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):

                    # Get the coarse block index for this fine point
                    ib, jb, kb = ijkmg[:, i, j, k, ilev]

                    # Accumulate fine volume onto the coarse volume
                    volmg[ib, jb, kb, ilev + 1] += vol[i, j, k]

    assert np.ptp(np.sum(volmg, axis=(0, 1, 2))) / np.sum(vol) < 1e-3

    return volmg


def arange_including_end(ni, di):
    ii = np.arange(0, ni, di)
    if not (ii[-1] == (ni - 1)):
        ii = np.append(ii, ni - 1)
    assert ii[-1] == (ni - 1)
    assert np.allclose(np.diff(ii[:-1]), di)
    return ii


def get_multigrid_lengths(block, nb):

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


def get_multigrid_timestep(dlmin, vol, Vxrt, a, CFL, relax, ijkmg, dt_old):

    # Get a+V cell-centered
    ni, nj, nk, _ = Vxrt.shape
    V = np.sqrt(np.sum(Vxrt**2, axis=-1))
    Vref_node = np.asfortranarray(V + a).astype(typ)
    Vref_cell = np.empty((ni - 1, nj - 1, nk - 1), order="F", dtype=typ)
    embsolve.node_to_cell(Vref_node, Vref_cell)

    # Preallocate
    Vrefmg = np.ones(dlmin.shape, order="F", dtype=typ) * np.nan
    nlev = ijkmg.shape[-1]

    # Trivial first level
    Vrefmg[..., 0] = Vref_cell

    # Loop over multigrid levels
    for ilev in range(nlev):
        # Loop over all points in the block
        for i in range(ni - 1):
            for j in range(nj - 1):
                for k in range(nk - 1):

                    # Get the coarse block index for this fine point
                    ib, jb, kb = ijkmg[:, i, j, k, ilev]

                    # Accumulate fine Vref onto coarse
                    vol_fac = vol[i, j, k, 0] / vol[ib, jb, kb, ilev]
                    Vrefmg[ib, jb, kb, ilev] += Vref_cell[i, j, k] * vol_fac

    # Now calculate time steps
    dt_new = CFL * dlmin / Vrefmg

    # Relax
    if relax:
        return relax * dt_new + (1.0 - relax) * dt_old
    else:
        return dt_new
