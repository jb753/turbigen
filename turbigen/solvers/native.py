import numpy as np
import turbigen.util
from turbigen.compiled import (
    smooth,
    node_to_cell,
    node_to_face,
    step,
)
from timeit import default_timer as timer
import logging

logger = turbigen.util.make_logger()

logger.setLevel(level=logging.INFO)


# class NativeConfig(BaseSolver):
#     """Settings with default values for the TS4 solver."""

#     _name = "Native"

nstep_dt = 100
nstep_log = 50
nstep = 1000
nrk = 4

sfin = 0.05
fac_2nd = 0.2
CFL = 3.5
sf = CFL * sfin
sf2 = sf*fac_2nd
sf4 = sf*(1.-fac_2nd)


def run(grid, settings={}, machine=None):

    nblock = len(grid)
    nodes = np.sum([b.size for b in grid])

    logger.info("Starting native solver...")

    # Calculate areas etc just once
    logger.info("Calculating geometry...")

    geom = [get_geom(b) for b in grid]

    logger.info("Starting the main time-stepping loop...")

    tstart = timer()

    # Start the main time stepping loop
    for istep in range(nstep):

        # Update periodic boundaries
        # grid.apply_periodic()

        # Update time stegs
        if not np.mod(istep, nstep_dt):
            dt = [
                get_timestep(grid[iblock], geom[iblock][4]) for iblock in range(nblock)
            ]

        # Loop over blocks
        for iblock in range(nblock):

            b = grid[iblock]
            dAi, dAj, dAk, vol, dlmin, wall, Omega = geom[iblock]

            dU = np.zeros(b.shape)
            print(dU.shape)
            quit()

            conserved_start = np.asfortranarray(b.conserved.copy())

            for irk in range(nrk):
                frk = 1.0 / (irk + nrk)

                conserved, Phor = apply_bconds(b)

                dU = step(
                    conserved, Phor, Omega, *wall, dt[iblock] * frk, dAi, dAj, dAk, vol
                )

                if irk == nrk - 1:
                    b.set_conserved(smooth(conserved_start + dU, sf))
                else:
                    b.set_conserved(conserved_start + dU)

            if not np.mod(istep, nstep_log):
                log_line = f"{istep}: {np.abs(dU).mean(axis=(-1,-2,-3))}"
                logger.info(log_line)
                ten = timer()
                tpnps = (ten - tstart) / nodes / nstep_log
                logger.info(f"tpnps={tpnps}")
                tstart = ten


def get_geom(b):
    # Areas and volumes
    dAi = np.asfortranarray(b.dAi)
    dAj = np.asfortranarray(b.dAj)
    dAk = np.asfortranarray(b.dAk)
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
    wall = [np.asfortranarray(w) for w in get_wall(b)]

    # Reference frame angular velocity
    Omega = b.Omega.mean()

    return dAi, dAj, dAk, vol, dlmin, wall, Omega


def get_timestep(b, dlmin):
    Vref, aref = node_to_cell(np.asfortranarray(np.stack((b.V, b.a))))
    dt = CFL * dlmin / (aref + Vref)
    return np.asfortranarray(dt)


def get_wall(b):
    # Find logical indices that zero the fluxes on wall faces
    thresh = 0.99  # To allow for floating point error
    return [
        (w[0] > thresh).astype(np.int8)
        for w in node_to_face(
            np.asfortranarray(np.expand_dims(b.get_wall(), 0).astype(float))
        )
    ]


# @profile
def apply_bconds(b):
    """Return properties needed to time march after in/out boundaries applied"""

    P = b.P.copy()
    ho = b.ho.copy()
    conserved = np.asfortranarray(b.conserved.copy())

    rho, rhoVx, rhoVr, rhorVt, rhoe = conserved

    # Adjust static pressure for exit boundary conditions
    for patch in b.outlet_patches:
        P[patch.get_slice()] = patch.Pout * 0.1 + 0.9 * (patch.get_cut().P)

    # Change inlet patches
    for patch in b.inlet_patches:

        ipatch = patch.get_slice()
        inlet = b[ipatch]
        rfin = patch.rfin

        if patch.store:
            # Relax changes in density if we have a stored state
            rho_now = rfin * inlet.rho + (1.0 - rfin) * patch.store.rho
            # Check for flow reversal
            rho_now[rho_now > patch.state.rho] = patch.state.rho * 0.9999
            # Isentropic expansion from stagnation state
            inlet.set_rho_s(rho_now, patch.state.s)
            assert np.allclose(inlet.rho, rho_now)

        patch.store = inlet

        # Get the velocity
        dhin = patch.state.h - inlet.h
        if (dhin <= 0.0).any():
            assert False
        Vin = np.sqrt(2 * dhin)

        tanAlpha = turbigen.util.tand(patch.Alpha)
        tanBeta = turbigen.util.tand(patch.Beta)

        # Resolve velocity components
        Vxin = Vin * np.sqrt((1.0 + tanAlpha**2) * (1.0 + tanBeta**2))
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
    return conserved, np.asfortranarray(np.stack((P, ho, b.r)))
