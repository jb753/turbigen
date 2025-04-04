"""Velocity-cubed estimate of surface dissipation."""

import numpy as np

import turbigen.grid
import turbigen.util


def post(
    grid,
    machine,
    meanline,
    __,
    ___,
    Cd=0.002,  # machine arg not used  # postdir arg not used
):
    r"""calc_surf_dissipation(Cd=0.002)

    Estimate boundary layer loss using a dissipation coefficient.

    After Denton (1993), entropy generation due to surface dissipation at walls is

    .. math::
        \dot{S}_\mathrm{surf} = \int_\mathrm{surf} C_\mathrm{d} \frac{\rho V_s^3}{T} \dee{A}

    where :math:`C_\mathrm{d}` is a dissipation coefficient, and :math:`V_s` is the local isentropic velocity.
    This routine evaluates the above equation using the CFD-predicted flow field and a constant dissipation coefficient,
    then saves the results in the mean-line object for subsequent analysis.

    The calculated :math:`\dot{S}_\mathrm{surf}` is a 2D array, broken down by row on the first axis, and on the
    second axis 0 corresponds to the hub and casing surfaces, and 1 the blade surfaces.

    Parameters
    ----------
    Cd : float
        Dissipation coefficient, default to the classic Denton value of 0.002.

    """

    # Get inlet entropy
    Cin = grid.inlet_patches[0].get_cut().mix_out()[0]
    sin = Cin.s

    # Get the final station cut plane
    xr_cut = machine.ann.get_offset_planes(offset=0)[-1]

    # dist = turbigen.util.signed_distance(xr_cut, block.xr)

    def _integrate_cut(C, rel, w=None):
        C = C.squeeze()
        # Pull out the face-centered properties we need
        dA = np.linalg.norm(C.surface_area, axis=-1, ord=2)
        rho = turbigen.util.node_to_face(C.rho)
        T = turbigen.util.node_to_face(C.T)
        x = turbigen.util.node_to_face(C.x)
        r = turbigen.util.node_to_face(C.r)
        xr = np.stack((x, r))

        # Isentropic to local static pressure
        Cs = C.copy().set_P_s(C.P, sin)
        hs = turbigen.util.node_to_face(Cs.h)

        # Subtract isentropic static enthalpy from real relative stagnation
        # enthalpy to get relative isenstopic exit dynamic head
        if rel:
            ho_rel = turbigen.util.node_to_face(C.ho_rel)
            Vs = np.sqrt(2.0 * np.maximum(ho_rel - hs, 0.0))
        else:
            ho = turbigen.util.node_to_face(C.ho)
            Vs = np.sqrt(2.0 * np.maximum(ho - hs, 0.0))

        # Apply weight for cuts that are not all wall
        if w is None:
            wf = np.ones_like(dA)
        else:
            wf = turbigen.util.node_to_face(w.squeeze())

        # Exclude cells downstream of last cut plane
        dist = turbigen.util.signed_distance(xr_cut, xr)
        wf[dist > 0.0] = 0.0

        # Multiply by the wall indicator to zero out non-walls
        # Perform the integration and accumulate total
        # return Cd * np.sum(wf * rho * Vs**3.0 / T * dA) * C.Nb, np.sum(wf * dA) * C.Nb

        # Multiply by the wall indicator to zero out non-walls
        # Perform the integration and accumulate total
        A = np.sum(wf * dA) * C.Nb
        Sdot = Cd * np.sum(wf * rho * Vs**3 / T * dA) * C.Nb
        V3 = np.sum(Vs**3 * dA) * C.Nb
        return Sdot, V3, A

    Sdot = np.zeros((grid.nrow, 2))
    Asurf = np.zeros((grid.nrow, 2))
    V3 = np.zeros((grid.nrow, 2))
    for irow, row_block in enumerate(grid.row_blocks):
        for block in row_block:
            # Preallocate wall indicator
            is_wall = np.ones(block.shape)
            is_wall[1:-1, 1:-1, 1:-1] = 0.0
            is_rot = np.zeros(block.shape)

            # Loop over patches
            for patch in block.patches:
                # Unset wall indicator if patch is not wall
                if isinstance(patch, turbigen.grid.RotatingPatch):
                    is_rot[patch.get_slice()] = 1.0
                if type(patch) in turbigen.grid.NOT_WALL_PATCHES:
                    is_wall[patch.get_slice()] = 0.0

            # Hub and casing
            for ind in (0, -1):
                if is_rot[:, ind, :].all():
                    Sdot_now, V3_now, A_now = _integrate_cut(block[:, ind, :], True)
                else:
                    Sdot_now, V3_now, A_now = _integrate_cut(block[:, ind, :], False)
                Sdot[irow, 0] += Sdot_now
                V3[irow, 0] += V3_now
                Asurf[irow, 0] += A_now

            # Blade surfaces
            for ind in (0, -1):
                if is_rot[:, :, ind].all():
                    Sdot_now, V3_now, A_now = _integrate_cut(
                        block[:, :, ind], True, is_wall[:, :, ind]
                    )
                else:
                    Sdot_now, V3_now, A_now = _integrate_cut(
                        block[:, :, ind], False, is_wall[:, :, ind]
                    )
                Sdot[irow, 1] += Sdot_now
                V3[irow, 1] += V3_now
                Asurf[irow, 1] += A_now

    meanline.Sdot_surf = Sdot
    meanline.A_surf = Asurf
    meanline.Vcubed = V3
