"""Write coordinates to an stl file."""

import numpy as np
import os
import turbigen.util
import stl

logger = turbigen.util.make_logger()


def run(grid, settings, machine):
    workdir = settings["workdir"]

    # Extract coordinates
    sections, annulus, zcst, Nb, tip, splitters = machine.get_coords()

    # Write annulus lines
    hub, cas = annulus

    hub_name = os.path.join(workdir, "hub.csv")
    shroud_name = os.path.join(workdir, "shroud.csv")

    np.savetxt(hub_name, hub, delimiter=",")
    logger.info(f"Wrote hub xr to  {hub_name}")

    np.savetxt(shroud_name, cas, delimiter=",")
    logger.info(f"Wrote shroud xr to {shroud_name}")

    # import matplotlib.pyplot as plt
    # from mpl_toolkits import mplot3d
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # fig_xr, ax_xr = plt.subplots()

    # Write an stl for each blade
    for iblade, section in enumerate(sections):
        # Join the points at leading edge

        xrt_ps = section[0, ...]
        # xrt_ss = np.flip(section[1, :, 1:, :], axis=1)  # Do not repeat LE
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

        stl_name = os.path.join(workdir, f"blade_{iblade}.stl")
        mesh.save(stl_name)
        logger.info(f"Wrote row {iblade} blade xyz to {stl_name}")

    # Write an stl for each splitter
    for iblade, section in enumerate(splitters):
        # Join the points at leading edge

        if section[0] is None:
            continue

        # print(len(section))
        # print(type(section))
        # print(type(section[0]))
        # print(type(section[0]))
        # print(len(section[0]))

        xrt_ps = section[0]
        # xrt_ss = np.flip(section[1, :, 1:, :], axis=1)  # Do not repeat LE
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

        stl_name = os.path.join(workdir, f"splitter_{iblade}.stl")
        mesh.save(stl_name)
        logger.info(f"Wrote row {iblade} splitter xyz to {stl_name}")

        # ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
        # r = np.sqrt(mesh.y**2.+mesh.z**2.)
        # ax_xr.scatter(mesh.x, r)
    # ax_xr.axis('equal')
    # scale = mesh.points.flatten()
    # lims = []
    # for ii in range(3):
    # lims.append((xyz_section[...,ii].min(),xyz_section[...,ii].max()))
    # ax.auto_scale_xyz(*lims)
    # ax.axis('equal')
    # ax.view_init(elev=0., azim=0., roll=0)
    # plt.show()

    logger.iter('NOTE: the stl "solver" does not run CFD, only writes coordinates.')
    logger.iter("NOTE: continuing with the inital guess flow field.")
