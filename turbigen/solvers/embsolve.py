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
        # Disable scaling
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

        self.bconds = [
            Boundary(patch) for patch in block.inlet_patches + block.outlet_patches
        ]

        self.mixers = []
        seen = []
        for patch in block.mixing_patches:
            if patch not in seen:
                mixer = [Boundary(patch), Boundary(patch.match)]
                self.bconds += mixer
                self.mixers.append(mixer)

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

        # # Preallocate stored inlet density
        # for inlet, state_inlet in zip(self.inlets, self.state_inlets):
        #     state_inlet._metadata = block._metadata
        #     rho_inlet = block.rho.ravel(order="F")[inlet[0]]
        #     u_inlet = block.u.ravel(order="F")[inlet[0]]
        #     state_inlet.set_rho_u(rho_inlet, u_inlet)

        # # Initialise  outlet states
        # for outlet, state_outlet in zip(self.outlets, self.state_outlets):
        #     state_outlet._metadata = block._metadata
        #     rho_out = block.rho.ravel(order="F")[outlet[0]]
        #     u_out = block.u.ravel(order="F")[outlet[0]]
        #     state_outlet.set_rho_u(rho_out, u_out)

        del to_fort

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

    # def multigrid(self, fmgrid):
    #     embsolve.multigrid(self.dU1, self.ijk_multigrid, fmgrid)

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

    return np.asfortranarray(ijk).reshape(3, -1).astype(np.int16)


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


def get_mixers(grid, procids):
    mixers = []
    seen = []
    for patch in grid.mixing_patches:
        if patch in seen:
            continue
        else:
            seen.append(patch)
            seen.append(patch.match)
        bid = patch.block.grid.index(patch.block)
        nxbid = patch.match.block.grid.index(patch.match.block)
        procid = procids[bid]
        nxprocid = procids[nxbid]
        mix_now = (
            MixingPlane(patch, bid, procid),
            MixingPlane(patch.match, nxbid, nxprocid),
        )
        # We also need a state object to hold the side-averaged flow conditions
        state = patch.block.empty((mix_now[0].nspan,))
        mixers.append(mix_now + (state,))
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


def get_inlet_data(patch, typ):
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
        to_fort_type(patch.get_cut().r.reshape(-1), typ),
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


def exchange_mixing(blocks, bid_local, mixers, typ, mpi_typ, plot=False):
    # Update periodic boundaries
    for mix1, mix2 in mixers:
        blk1 = blocks[bid_local[mix1.bid]]

        # Get pitchwise-avgeraged conditions on this side
        flux1, prim1 = mix1.get_averages(blk1)

        # Other side same rank
        if mix2.procid == rank:
            blk2 = blocks[bid_local[mix2.bid]]
            flux2, prim2 = mix2.get_averages(blk2)

        # Otherwise, communication is needed
        else:
            raise NotImplementedError()
        print("beans")
        quit()
        # Form the side-averaged flow conditions
        rho, u, Vs, Vt, Vn, _, _ = prim = 0.5 * (prim1 + prim2)

        # Update the thermodynamic state
        state.set_rho_u(rho, u)

        # Limit the minimum mach number
        Ma_min = 0.1
        Vn_min = Ma_min * state.a.mean()
        Vn[np.abs(Vn) < Vn_min] = Vn_min

        # Update the interface velocities
        state.Vxrt = [Vs, Vt, Vn]

        # Read off properties we will need
        ho = state.ho
        dhdrho = state.dhdrho_P
        dhdP = state.dhdP_rho
        dudP = state.dudP_rho
        dudrho = state.dudrho_P
        a = state.a
        e = state.e

        # Convert flux changes into primative changes
        # Matrix A from Holmes (2008) eqn. (A1)
        rhoVn = rho * Vn
        Z = np.zeros_like(rho)
        one = np.ones_like(rho)
        A = np.moveaxis(
            np.stack(
                (
                    (Vn, Z, Z, rho, Z),
                    (Vn * Vs, rhoVn, Z, rho * Vs, Z),
                    (Vn * Vt, Z, rhoVn, rho * Vt, Z),
                    (Vn**2, Z, Z, 2.0 * rhoVn, one),
                    (
                        Vn * ho + rhoVn * dhdrho,
                        rhoVn * Vs,
                        rhoVn * Vt,
                        rho * ho + rhoVn * Vn,
                        rhoVn * dhdP,
                    ),
                )
            ),
            -1,
            0,
        )
        Ainv = np.linalg.inv(A)

        # Convert primative changes to characteristic changes
        # Matrix B from Holmes (2008) eqn. (A3-4)
        asqi = 1.0 / a**2
        asq = a**2
        rhoa = rho * a
        rhoai = 1.0 / rho / a
        B = np.moveaxis(
            np.stack(
                (
                    (-asq, Z, Z, Z, one),
                    (Z, rhoa, Z, Z, Z),
                    (Z, Z, rhoa, Z, Z),
                    (Z, Z, Z, rhoa, one),
                    (Z, Z, Z, -rhoa, one),
                )
            ),
            -1,
            0,
        )
        Binv = np.linalg.inv(B)

        # Convert primative to conserved perturbations
        # Matrix C from Holmes (2008) eqn. (A5-6)
        C = np.moveaxis(
            np.stack(
                (
                    (one, Z, Z, Z, Z),
                    (Vs, rho, Z, Z, Z),
                    (Vt, Z, rho, Z, Z),
                    (Vn, Z, Z, rho, Z),
                    (e + rho * dudrho, rho * Vs, rho * Vt, rho * Vn, rho * dudP),
                )
            ),
            -1,
            0,
        )
        Cinv = np.linalg.inv(C)

        # Select which characteristics to keep
        Dup = np.diag([0, 0, 0, 1, 0])[None, ...]
        Ddn = np.diag([1, 1, 1, 0, 1])[None, ...]

        # Resolve to rtx
        cospsi = mix1.cospsi.squeeze()
        sinpsi = mix1.sinpsi.squeeze()
        T = np.moveaxis(
            np.stack(
                (
                    (one, Z, Z, Z, Z),
                    (Z, cospsi, Z, -sinpsi, Z),
                    (Z, Z, one, Z, Z),
                    (Z, sinpsi, Z, cospsi, Z),
                    (Z, Z, Z, Z, one),
                )
            ),
            -1,
            0,
        ).transpose(0, 2, 1)

        # Assemble the overall transformations
        TCBinv = T @ C @ Binv
        BAinv = B @ Ainv
        Qup = TCBinv @ Dup @ BAinv
        Qdn = TCBinv @ Ddn @ BAinv

        # Flux differences with relaxation
        K_mix = 0.1
        DF = ((flux1 - flux2).T)[..., None]
        DF *= -K_mix
        err = flux1
        jplot = mix1.nspan // 2
        err = flux1[:, jplot] - flux2[:, jplot]
        err[0] /= flux1[0, jplot]
        flux_ref = np.max(np.abs(flux1[1:4, jplot]))
        err[1:4] /= flux_ref
        err[4] /= flux1[4, jplot]
        if plot:
            logger.info(f"Mixing plane err: {err}")

        # Say we have only a mass-flow error

        # Now calculate changes in conserved variables on each side
        dU_up = (Qup @ DF).squeeze()
        dU_dn = (Qdn @ DF).squeeze()

        # Holmes uses conseved vars [rho, rhoVr, rhoVt, rhoVx, rhoE]
        # We use conseved vars [rho, rhoVx, rhoVr, rhorVt, rhoE]
        # And left-handed coordinate system
        dU_up = dU_up[:, (0, 3, 1, 2, 4)]
        dU_dn = dU_dn[:, (0, 3, 1, 2, 4)]
        dU_up[:, 3] *= mix1.r
        dU_dn[:, 3] *= mix1.r
        # dU_up, dU_dn = dU_dn, dU_up

        # Use the sign of Vs to select which of 1 and 2 are up/downstream
        ind1 = Vs > 0.0

        # Preallocate nodal changes for each side
        dU1 = np.full_like(dU_up, np.nan)
        dU2 = np.full_like(dU_up, np.nan)

        # Use the downstream-propagating chics where flow is into the domain
        dU1[ind1, :] = dU_dn[ind1, :]
        dU1[~ind1, :] = dU_up[~ind1, :]
        dU2[ind1, :] = dU_up[ind1, :]
        dU2[~ind1, :] = dU_dn[~ind1, :]
        dU2 *= -1.0

        assert not np.isnan(dU1).any()
        assert not np.isnan(dU2).any()

        # print('dU1 upstream side', dU1[0,:])
        # print('dU2 downstream side', dU2[0,:])

        # Now apply to each side
        mix1.perturb_conserved(blk1, dU1)
        mix2.perturb_conserved(blk2, dU2)


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


def run_slave(blocks=None, periodics_all=None, mixers=None, nodes=None, conf=None):
    if blocks is None:
        blocks = comm.recv()
        comm.Barrier()
        periodics_all = comm.recv()
        comm.Barrier()
        mixers = comm.recv()
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
        # pid, bid, procid, ind, indf, d, nxbid, nxprocid, nxind, nxindf, nxd = patch
        if patch.procid == rank:
            periodics.append(patch)
        elif patch.nxprocid == rank:
            periodics.append(patch.reversed())

    # Setup MPI communication on periodics
    for patch in periodics:
        patch.setup_communication(comm, mpi_typ)

    # Rearrange mixers so that foreign is always second
    for mix1, mix2, _ in mixers:
        # Swap around if needed
        if mix2.procid == rank and not mix1.procid == rank:
            mix2, mix1 = mix1, mix2

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

            # Exchange fluxes across mixing patches
            # exchange_mixing(blocks, bid_local, mixers)

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

                # sb.set_inlets(rfin, conf.i_inlet, K_inlet)
                # sb.set_outlets(conf.i_exit, K_exit)

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

    mixers = get_mixers(grid, procids)

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
    """Store FlowField on a block boundary, force to target."""

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

        # Preallocate a pitchwise-averaged state
        self.nspan = self.shape[ax_span]
        self.state_avg = self.state.empty((self.nspan,))

        # Get normal vectors pointing into the domain
        C0 = C.copy().transpose(self.order).squeeze()
        C1 = patch.get_cut(offset=1).transpose(self.order).squeeze()
        dxr = C1.xr - C0.xr
        self.normal = np.mean(dxr / turbigen.util.vecnorm(dxr), axis=-1)

        # Angular pitch and cell widths for integration
        self.pitch = C0.pitch + 0.0
        self.dt = np.diff(C0.t, axis=1)

        # Check that theta gridlines are at constant x and r
        Lref = np.maximum(np.ptp(C0.x), np.ptp(C0.r))
        rtol = 1e-3
        assert (np.ptp(C0.x, axis=1) / Lref < rtol).all()
        assert (np.ptp(C0.r, axis=1) / Lref < rtol).all()

        # Preallocate boundary targets
        self.inlet_target = np.full(self.shape + (4, 1), np.nan)
        self.P_target = np.full(self.shape, np.nan)

        # Initialise indicator for inlet or outlet
        # and set the target boundary condition vars
        # Nodewise to allow for reversed flow across mixing plane
        if isinstance(patch, turbigen.grid.InletPatch):
            self.is_inlet = np.ones(self.shape, dtype=bool)
            tanAl = turbigen.util.tand(patch.Alpha)
            tanBe = turbigen.util.tand(patch.Beta)
            self.inlet_target[..., :, 0] = [patch.state.h, patch.state.s, tanAl, tanBe]

        elif isinstance(patch, turbigen.grid.OutletPatch):
            self.is_inlet = np.zeros(self.shape, dtype=bool)
            self.P_target[:] = np.full(self.shape, patch.Pout)

        elif isinstance(patch, turbigen.grid.turbigen.grid.MixingPatch):
            # As an initial guess, we need to decide if the mixing plane
            # is on the upsteram or downstream side
            # Do this by dotting meridional velocity vector with grid normal
            Vxr = self.pitchwise_average(C.Vxr).squeeze()
            Vxr /= turbigen.util.vecnorm(Vxr)
            dirn = np.einsum("i...,i...", Vxr, self.normal)
            dirn_avg = np.mean(np.sign(dirn))

            # Velocity vector is pointing into interior => an inlet
            if dirn_avg > 0.0:
                self.is_inlet = np.ones(self.shape, dtype=bool)
                self.inlet_target[..., :, 0] = np.moveaxis(self.state.bcond[:4], 0, -1)
            # Velocity vector is pointing away from interior => an outlet
            else:
                self.is_inlet = np.zeros(self.shape, dtype=bool)
                self.P_target[:] = self.state.P

    @property
    def is_outlet(self):
        return np.logical_not(self.is_inlet)

    def clip_velocities(self):
        """Limit the minimum absolute throughflow velocity to avoid singular transformation matrices."""
        Ma_min = 0.05
        V_min = self.state.a.mean() * Ma_min
        ind_clip = np.abs(self.state.Vxrt) < V_min
        self.state.Vxrt[ind_clip] = V_min * np.sign(self.state.Vxrt[ind_clip])

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

        # Velocities
        self.state.Vxrt = np.moveaxis(block.Vxrt[self.slice], -1, 0)

        # Nodal residuals
        self.dUn[:] = block.dUn[self.slice]

    def push(self, block):
        """Send modified residuals back to the parent block."""
        # Noting that we have to swap from C to Fortran ordering
        block.dUn[self.slice] = self.dUn

    def apply(self, block):
        self.pull(block)
        assert self.is_inlet.all() or self.is_outlet.all()
        dchic_ext = self.exterior_chics()
        dchic_int = self.interior_chics()
        if self.is_outlet.all():
            dchic_int *= 1.0
        dchic = dchic_ext + dchic_int
        dcons = self.state.chic_to_conserved @ dchic
        self.dUn[:] = dcons[..., 0]
        self.push(block)

    def interior_chics(self):
        """Get chics propagating out of domain from interior nodal changes."""

        # Form a diagonal selector matrices for upstream or downstream chics
        Ddn = np.diag([0, 1, 1, 1, 1])
        Dup = np.diag([1, 0, 0, 0, 0])

        # Interior chics are upstream-running at inlet, downstream-running at outlet
        D = np.empty(self.shape + (5, 5))
        D[self.is_inlet] = Dup
        D[self.is_outlet] = Ddn

        # Matrix transform conserved changes to chics, with selector
        dc = D @ self.state.conserved_to_chic @ self.dUn[..., None]

        return dc

    def exterior_chics(self):
        # Preallocate
        dc = np.zeros(self.shape + (5, 1))

        #
        # On outlet, use static pressure to set upstream-running wave
        #
        dP = self.P_target - self.state.P
        rho = self.state.rho
        a = self.state.a
        dVx = -dP / rho / a  # from c2=0
        c1 = dP - rho * a * dVx
        dc[self.is_outlet, 0, 0] = c1[self.is_outlet]

        #
        # On inlet, use ho, s, Al, Be to set downstream-running chics
        #

        # Convert downstream-running chics to prim changes
        # Omit first column corresponding to upstream-running chic
        chic_to_prim = self.state.chic_to_primitive[..., :, 1:]

        # Convert primitive to inlet changes
        # Omit last row corresponding to static pressure
        prim_to_inlet = self.state.primitive_to_bcond[..., :-1, :]

        # Complete transformation matrix
        chic_to_inlet = prim_to_inlet @ chic_to_prim
        inlet_to_chic = np.linalg.inv(chic_to_inlet)

        # Evaluate bcond error
        inlet_now = np.stack(
            (self.state.ho, self.state.s, self.state.tanAlpha, self.state.tanBeta),
            axis=-1,
        )[..., None]
        dinlet = self.inlet_target - inlet_now

        # Convert to chics
        dc_inlet = (inlet_to_chic @ dinlet)[self.is_inlet]

        # Insert into the preallocated chic array
        dc[self.is_inlet, 1:, :] = dc_inlet

        # Under-relax
        dc *= 0.5

        return dc

    def pitchwise_average(self, y):
        # Perform trapezoidal integration at every spanwise location
        return 0.5 * np.sum((y[..., 1:] + y[..., :-1]) * self.dt, axis=-1) / self.pitch

    def get_averages(self):
        return (
            self.pitchwise_average(self.state.fluxes),
            self.pitchwise_average(self.state.prim),
        )

    def integrate(self):
        """Get mass flow and mass-averaged boundary conditions."""
        rhoVx = self.state.rhoVx.squeeze()
        Po = self.state.Po.squeeze()
        To = self.state.To.squeeze()
        P = self.state.P.squeeze()
        rhoVx_face = util.node_to_face2(rhoVx)
        To_face = util.node_to_face2(To)
        Po_face = util.node_to_face2(Po)
        P_face = util.node_to_face2(P)
        A = self.dA.sum()
        mdot = (rhoVx_face * self.dA).sum()
        Po_avg = (rhoVx_face * Po_face * self.dA).sum() / mdot
        To_avg = (rhoVx_face * To_face * self.dA).sum() / mdot
        P_avg = (P_face * self.dA).sum() / A
        return mdot, Po_avg, To_avg, P_avg


class MixingPlane:
    def __init__(self, patch, bid, procid):
        self.bid = bid
        self.procid = procid

        # Determine indexing into the conserved variables
        # array for this patch
        self.slice = patch.get_slice()

        # Set up the coordinates
        # Determine a permutation order such that
        # first axis is spanwise, second axis is pitchwise
        C = patch.get_cut()
        xrt = C.xrt
        ax_theta = np.argmax([np.ptp(C.t, axis=n).mean() for n in range(3)]).item()
        ax_stream = np.argmax(np.array(C.shape) == 1).item()
        ax_span = np.setdiff1d([0, 1, 2], [ax_theta, ax_stream]).item()
        self.order = (ax_stream, ax_span, ax_theta)

        # Preallocate a state to hold a pitchwise-averaged state
        self.nspan = C.shape[ax_span]

        # Get normal vectors pointing into the domain
        C1 = patch.get_cut(offset=1)
        C1.transpose(self.order)
        C1 = C1.squeeze()
        dxr = C1.xr - C0.xr
        self.normal = np.mean(dxr / turbigen.util.vecnorm(dxr), axis=-1)

        # Check that theta gridlines are at constant x and r
        Lref = np.maximum(np.ptp(Ct.x), np.ptp(Ct.r))
        rtol = 1e-3
        assert (np.ptp(Ct.x, axis=1) / Lref < rtol).all()
        assert (np.ptp(Ct.r, axis=1) / Lref < rtol).all()

        # Work out nodal interface angle
        self.psi = turbigen.util.angle_curve_node(Ct[:, 0].xr) - 90.0
        self.cospsi = turbigen.util.cosd(self.psi)[:, None]
        self.sinpsi = turbigen.util.sind(self.psi)[:, None]

        self.pitch = Ct.pitch + 0.0
        self.dt = np.diff(Ct.t, axis=1)

        # Store initial guess of pitchwise-uniform boundary condition vars
        self.ho = Ct.ho.mean(axis=-1)
        self.s = Ct.s.mean(axis=-1)
        self.Alpha = Ct.Alpha.mean(axis=-1)
        self.Beta = Ct.Beta.mean(axis=-1)
        self.P = Ct.P.mean(axis=-1)

    def get_primative(self, block):
        """Extract the primative variables on mixing patch.
        Always comes out indexed [variable, spanwise, pitchwise]"""
        rho = block.cons[self.slice][..., 0].transpose(self.order).squeeze()
        u = block.u[self.slice].transpose(self.order).squeeze()
        P = block.P[self.slice].transpose(self.order).squeeze()
        ho = block.ho[self.slice].transpose(self.order).squeeze()
        Vxrt = block.Vxrt[self.slice].transpose((3,) + self.order).squeeze()
        return np.stack((rho, u, *Vxrt, P, ho))

    def pitchwise_average(self, y):
        # Along the final axis
        return 0.5 * np.sum((y[..., 1:] + y[..., :-1]) * self.dt, axis=-1) / self.pitch

    def perturb_conserved(self, block, dU):
        """Extract the primative variables on mixing patch.
        Always comes out indexed [variable, spanwise, pitchwise]"""
        cons = block.cons[self.slice].transpose(self.order + (3,))
        cons += dU[None, :, None, :]

    def get_averages(self, block):
        """Extract fluxes to be conserved across the mixing plane."""

        # Get primative flow variables
        primative = self.get_primative(block)
        rho, u, Vx, Vr, Vt, P, ho = primative

        # Resolve velocity spanwise and normal to interface
        Vn = Vx * self.cospsi + Vr * self.sinpsi
        Vs = -Vx * self.sinpsi + Vr * self.cospsi

        # Overwrite resolved velocities into primative
        # Warning using the Holmes velocity componend ordering
        Vx[:] = Vs
        Vr[:] = Vt.copy()
        Vt[:] = Vn

        # Form the nodal fluxes of conserved quantities
        rhoVn = rho * Vn
        fluxes = np.stack(
            (
                rhoVn,
                rhoVn * Vs,
                rhoVn * Vt,
                rhoVn * Vn + P,
                rhoVn * ho,
            )
        )

        # Average fluxes across the pitch
        # Cannot use numpy trapz because the x vector
        # is different for every spanwise location
        fluxes_avg = self.pitchwise_average(fluxes)

        # Pitchwise-average the primative variables
        primative_avg = self.pitchwise_average(primative)

        # We now have all the information that
        # needs to be exchanged with other side of mixing plane
        return fluxes_avg, primative_avg
