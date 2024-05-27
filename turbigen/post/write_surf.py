"""Write coordinates to an stl file."""

import numpy as np
import os
import turbigen.util
import stl


logger = turbigen.util.make_logger()


def post(grid, machine, meanline, postdir):

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
        # Join the points at leading edge

        xrt_ps = section[0, ...]
        xrt_ss = np.flip(section[1, :, :, :], axis=1)  # Fix the hole in LE?
        xrt_section = np.concatenate((xrt_ps, xrt_ss), axis=1)

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

        mesh = stl.mesh.Mesh(data)

        stl_name = os.path.join(postdir, f"blade_{iblade}.stl")
        mesh.save(stl_name)
        logger.info(f"Wrote row {iblade} blade xyz to {stl_name}")

    # Write an stl for each splitter
    if splitters:
        for iblade, section in enumerate(splitters):
            # Join the points at leading edge

            if section[0] is None:
                continue

            xrt_ps = section[0]
            xrt_ss = np.flip(section[1], axis=1)  # Fix the hole in LE?
            xrt_section = np.concatenate((xrt_ps, xrt_ss), axis=1)

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

            mesh = stl.mesh.Mesh(data)

            stl_name = os.path.join(postdir, f"splitter_{iblade}.stl")
            mesh.save(stl_name)
            logger.info(f"Wrote row {iblade} splitter xyz to {stl_name}")
