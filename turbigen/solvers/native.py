import numpy as np
import turbigen.util
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

nstep_dt = 100
nstep_log = 100
nstep = 8000
nrk = 4
dampin = 10.0

sfin = 0.01
fac_2nd = 0.2
CFL = 0.4
sf = CFL * sfin
sf2 = sf * fac_2nd
sf4 = sf * (1.0 - fac_2nd)


def run(grid, settings={}, machine=None):

    nblock = len(grid)
    nodes = np.sum([b.size for b in grid])

    logger.info("Starting native solver...")

    # Calculate areas etc just once
    logger.info("Calculating geometry...")

    geom = [get_geom(b) for b in grid]

    logger.info("Starting the main time-stepping loop...")

    tstart = timer()

    # # Pre-allocate residuals
    # dU_last = [np.full(b.conserved.shape) for b in grid]
    # dU = [np.full(b.conserved.shape) for b in grid]

    # Pre-allocate secondaries
    Vxrt = [np.empty(b.shape + (3,), order="F") for b in grid]
    u = [np.empty(b.shape, order="F") for b in grid]
    dU = [np.empty(b.shape + (5,), order="F") for b in grid]
    dU_last = [np.empty(b.shape + (5,), order="F") for b in grid]
    mdot_all = np.empty((nstep // nstep_log, 2))

    # Start the main time stepping loop
    for istep in range(nstep):

        # Update periodic boundaries
        grid.apply_periodic()

        # Update time stegs
        if not np.mod(istep, nstep_dt):
            dt = [
                get_timestep(grid[iblock], geom[iblock][4]) for iblock in range(nblock)
            ]

        # Loop over blocks
        for iblock in range(nblock):

            b = grid[iblock]

            dAi, dAj, dAk, vol, dlmin, wall, Omega = geom[iblock]

            conserved, Phor = apply_bconds(b)

            step(
                conserved,
                Phor,
                Omega,
                *wall,
                dt[iblock],
                dAi,
                dAj,
                dAk,
                vol,
                dU[iblock],
            )

            if istep == 0:

                conserved += dU[iblock]

            else:

                conserved += 2.0 * dU[iblock] - dU_last[iblock]

            dU_last[iblock][:] = dU[iblock]

            smooth(conserved, sf2, sf4)

            calculate_secondary(Phor[..., 2], conserved, Vxrt[iblock], u[iblock])

            b.Vxrt = np.moveaxis(Vxrt[iblock], -1, 0)

            try:
                b.set_rho_u(conserved[..., 0], u[iblock])
            except Exception as e:

                print(e)

                ijkmax = np.argmax(grid[0].V)
                imax, jmax, kmax = np.unravel_index(ijkmax, grid[0].shape)

                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                bb = grid[0][:, jmax, :].squeeze()
                cm = ax.contourf(bb.x, bb.rt, bb.Ma)
                ax.plot(grid[0].x.flat[ijkmax], grid[0].rt.flat[ijkmax], "k*")
                plt.colorbar(cm)
                plt.show()
                # quit()

            if not np.mod(istep, nstep_log):

                log_line = f"{istep}: {np.abs(dU[0]).mean(axis=(0,1,2))}"
                logger.info(log_line)
                ten = timer()
                tpnps = (ten - tstart) / nodes / nstep_log
                logger.info(f"tpnps={tpnps}")
                tstart = ten

                mdot_in = 0.0
                for patch in grid.inlet_patches:
                    Cm, Ann, _ = patch.get_cut().mix_out()
                    mdot_in += Cm.rho * Cm.Vm * Ann

                mdot_out = 0.0
                for patch in grid.outlet_patches:
                    Cm, Ann, _ = patch.get_cut().mix_out()
                    mdot_out += Cm.rho * Cm.Vm * Ann
                    print(
                        f"mass flows {mdot_in}, {mdot_out}, err={mdot_in/mdot_out-1.}"
                    )
                    mdot_all[istep // nstep_log] = (mdot_in, mdot_out)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(mdot_all / mdot_all[-1].mean())
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


def apply_bconds(b):
    """Return properties needed to time march after in/out boundaries applied"""

    P = b.P
    ho = b.ho
    conserved = b.conserved

    rho, rhoVx, rhoVr, rhorVt, rhoe = conserved

    # Adjust static pressure for exit boundary conditions
    for patch in b.outlet_patches:
        P[patch.get_slice()] = patch.Pout  # * 0.1 + 0.9 * (patch.get_cut().P)

    # Change inlet patches
    for patch in b.inlet_patches:

        ipatch = patch.get_slice()
        inlet = b[ipatch]
        rfin = 0.1  # *patch.rfin

        if patch.store:
            # Relax changes in density if we have a stored state
            rho_now = rfin * inlet.rho + (1.0 - rfin) * patch.store.rho
            # Check for flow reversal
            rho_now[rho_now > patch.state.rho] = patch.state.rho * 0.9999
            # Isentropic expansion from stagnation state
            inlet.set_rho_s(rho_now, patch.state.s)

        patch.store = inlet

        # Get the velocity
        dhin = patch.state.h - inlet.h
        if (dhin <= 0.0).any():
            assert False
        Vin = np.sqrt(2.0 * dhin)

        tanAlpha = turbigen.util.tand(patch.Alpha)
        tanBeta = turbigen.util.tand(patch.Beta)

        # Resolve velocity components
        Vxin = Vin * np.sqrt((1.0 + tanAlpha**2) * (1.0 + tanBeta**2))
        Vrin = Vxin * tanBeta
        Vmin = np.sqrt(Vxin**2 + Vrin**2)
        Vtin = Vmin * tanAlpha

        # Reset conserved vars on inlet
        rho[ipatch] = inlet.rho
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
    conserved = np.asfortranarray(np.moveaxis(conserved, 0, -1))
    return conserved, np.asfortranarray(np.stack((P, ho, b.r), axis=-1))
