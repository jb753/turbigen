"""Write coordinates to an stl file."""

import numpy as np
import os
import turbigen.util
import stl


logger = turbigen.util.make_logger()


def get_stl_data(section, close_tip):
    xrt_ps = section[0, ...]
    xrt_ss = section[1, ...]
    xrt_section = np.concatenate((xrt_ps, np.flip(xrt_ss, axis=1)), axis=1)

    xyz_ps = np.stack(
        (
            xrt_ps[..., 0],
            xrt_ps[..., 1] * np.cos(xrt_ps[..., 2]),
            xrt_ps[..., 1] * np.sin(xrt_ps[..., 2]),
        ),
        axis=-1,
    )

    xyz_ss = np.stack(
        (
            xrt_ss[..., 0],
            xrt_ss[..., 1] * np.cos(xrt_ss[..., 2]),
            xrt_ss[..., 1] * np.sin(xrt_ss[..., 2]),
        ),
        axis=-1,
    )

    xyz_section = np.stack(
        (
            xrt_section[..., 0],
            xrt_section[..., 1] * np.cos(xrt_section[..., 2]),
            xrt_section[..., 1] * np.sin(xrt_section[..., 2]),
        ),
        axis=-1,
    )

    nj, ni, _ = xrt_section.shape
    nface = 2 * (ni - 1) * (nj - 1)

    # Add on tip if needed
    if close_tip:
        # Extra faces for the tip
        ni_tip = xrt_ps.shape[1]
        nface += 2 * (ni_tip - 2) + 1

    data = np.zeros(nface, dtype=stl.mesh.Mesh.dtype)

    for i in range(ni - 1):
        for j in range(nj - 1):
            # Calculate 1D face indices from 2D node indices
            kl = 2 * (i + (ni - 1) * j)
            ku = kl + 1
            xyz_kl = np.stack(
                (
                    xyz_section[j, i, :],
                    xyz_section[j, i + 1, :],
                    xyz_section[j + 1, i + 1, :],
                )
            )
            xyz_ku = np.stack(
                (
                    xyz_section[j, i, :],
                    xyz_section[j + 1, i + 1, :],
                    xyz_section[j + 1, i, :],
                )
            )
            data["vectors"][kl] = xyz_kl
            data["vectors"][ku] = xyz_ku

    if close_tip:
        kface_st = 2 * (ni - 1) * (nj - 1)

        # Single triangle at LE
        xyz_nose = np.stack(
            (
                xyz_ps[-1, 0, :],
                xyz_ps[-1, 1, :],
                xyz_ss[-1, 1, :],
            )
        )
        data["vectors"][kface_st] = xyz_nose

        for i in range(1, ni_tip - 1):
            xyz_kl = np.stack(
                (
                    xyz_ps[-1, i, :],
                    xyz_ps[-1, i + 1, :],
                    xyz_ss[-1, i + 1, :],
                )
            )

            xyz_ku = np.stack(
                (
                    xyz_ss[-1, i, :],
                    xyz_ss[-1, i + 1, :],
                    xyz_ps[-1, i, :],
                )
            )

            kl = 2 * i - 1 + kface_st
            ku = kl + 1
            data["vectors"][kl] = xyz_kl
            data["vectors"][ku] = xyz_ku

    return data


def post(grid, machine, meanline, _, postdir):
    """write_stl()

    Write machine surface geometry to stl files.

    Generates a triangulated Cartesian stl for each aerofoil row, and x-r csvs for the hub and shroud."""

    # Extract coordinates
    sections, annulus, zcst, Nb, tip, splitters = machine.get_coords()

    # Write annulus lines
    hub, cas = annulus

    hub_name = os.path.join(postdir, "hub.csv")
    shroud_name = os.path.join(postdir, "shroud.csv")

    np.savetxt(hub_name, hub, delimiter=",")
    logger.info(f"Wrote hub xr to  {hub_name}")

    np.savetxt(shroud_name, cas, delimiter=",")
    logger.info(f"Wrote shroud xr to {shroud_name}")

    # Write an stl for each blade
    for iblade, section in enumerate(sections):
        # Skip unbladed rows, e.g. vaneless diffuser
        if section[0] is None:
            logger.info(f"Skipping unbladed row {iblade}")
            continue

        mesh = stl.mesh.Mesh(get_stl_data(section, tip[iblade]))

        stl_name = os.path.join(postdir, f"blade_{iblade}.stl")
        mesh.save(stl_name)
        logger.info(f"Wrote row {iblade} blade xyz to {stl_name}")

    # Write an stl for each splitter
    if splitters:
        for iblade, section in enumerate(splitters):
            # Join the points at leading edge

            if section[0] is None:
                continue

            mesh = stl.mesh.Mesh(get_stl_data(section, tip[iblade]))
            stl_name = os.path.join(postdir, f"splitter_{iblade}.stl")
            mesh.save(stl_name)
            logger.info(f"Wrote row {iblade} splitter xyz to {stl_name}")
