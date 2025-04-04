"""Write coordinates to an ibl file.

This is heavily based on bl_write_ibl by James Taylor."""

import numpy as np
import os
import turbigen.util


logger = turbigen.util.make_logger()


def _format_point(xyz):
    return " ".join(f"{xyzi:.12}" for xyzi in xyz) + "\n"


def _write_surf(f, xyz):
    """Write coordinates for a 2D surface to file."""

    # Check shape
    assert xyz.shape[0] == 3
    assert xyz.ndim == 3

    # Loop over sections first, then points on the section
    for j in range(xyz.shape[2]):

        # Header
        f.write("begin section\n")
        f.write("begin curve\n")

        for i in range(xyz.shape[1]):
            f.write(_format_point(xyz[:, i, j]))


def _write_curve(f, xyz):
    """Write coordinates for a 1D line to file."""

    # Add zero theta if only xr
    if xyz.shape[0] == 2:
        xyz = np.append(xyz, np.zeros((1, xyz.shape[1])), axis=0)

    # Check shape
    assert xyz.shape[0] == 3
    assert xyz.ndim == 2

    # Header
    f.write("begin section\n")
    f.write("begin curve\n")

    # Coordinates
    for i in range(xyz.shape[1]):
        f.write(_format_point(xyz[:, i]))


def _write_blade(f, bld, nspan, nchord):

    # Span fractions to write at, linearly spaced
    spf = np.linspace(-0.01, 1.01, nspan)

    # Get pressure and suction sides
    xrt_ul = np.stack([bld.evaluate_section(spfi, nchord) for spfi in spf]).transpose(
        1, 2, 3, 0
    )

    # Join pressure and suction sides into an o-grid
    # Repeat the LE at both ends but drop one of the TE points
    xrt = np.concatenate((xrt_ul[0, :, :-1, :], np.flip(xrt_ul[1], axis=1)), axis=1)

    # Check that the curve is closed
    assert np.allclose(xrt[:, 0, :], xrt[:, -1, :])

    # Convert to Cartesian
    y = xrt[1] * np.cos(xrt[2])
    z = xrt[1] * np.sin(xrt[2])
    xyz = np.stack((xrt[0], y, z))

    # Fit a plane through central section to get a normal vector
    nj = xyz.shape[2]
    vec_norm = turbigen.util.fit_plane(xyz[:, :, nj // 2])
    basis1, basis2 = turbigen.util.basis_from_normal(vec_norm)

    # Project the points onto the plane basis
    projected = turbigen.util.project_onto_plane(xyz, basis1, basis2)

    # Verify that the signed area is same for all sections
    area = np.array(
        [turbigen.util.shoelace_formula(projected[..., j]) for j in range(nj)]
    )
    assert not np.diff(np.sign(area)).any()

    # Write 2D surface to file
    _write_surf(f, xyz)

    # Write LE and TE lines
    _write_curve(f, xyz[:, 0, :])
    _write_curve(f, xyz[:, -1, :])
    _write_curve(f, xyz[:, nchord - 1, :])


def post(grid, machine, meanline, _, postdir, nspan=22, nchord=100):
    """write_ibl()

    Write machine surface geometry to an ibl file.

    Save coordinates around blade sections and annulus lines to Creo format."""

    ibl_name = os.path.join(postdir, "machine.ibl")
    logger.info(f"Creo export: opening {ibl_name}")
    with open(ibl_name, "w") as f:

        # Header
        f.write("Closed Index arclength\n")

        # Loop over rows
        for irow in range(machine.Nrow):

            # Main blade if present
            if bld := machine.bld[irow]:
                logger.info(f"Writing row {irow} main blade")
                _write_blade(f, bld, nspan, nchord)

            # Splitter blade if present
            if machine.split:
                if bld := machine.split[irow]:
                    logger.info(f"Writing row {irow} splitter")
                    _write_blade(f, bld, nspan, nchord)

        # Hub and casing lines
        hub, cas = machine.ann.get_coords().transpose(0, 2, 1)
        logger.info("Writing annulus lines")
        _write_curve(f, hub)
        _write_curve(f, cas)

    logger.info("Finished ibl export.")
