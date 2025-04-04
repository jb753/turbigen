import numpy as np
import turbigen.grid


def _read_coord(f, nijkb):
    """Read a block coordinate from plot3d file."""
    nijkbf = np.flip(nijkb)
    size = np.prod(nijkb)
    data = np.fromfile(f, count=size, sep=" ").reshape(nijkbf, order="F")
    return np.flip(data.transpose((2, 1, 0)), axis=0)


def read(g_file, bcs_file, Lref=1.0):
    #
    # Process the IGG bcs file with patches
    #
    f = open(bcs_file, "r")

    # First line is number of blocks
    nb = int(f.readline().split()[-1])

    # Loop over blocks
    nijk = np.zeros((nb, 3), dtype=int)
    nblade = np.zeros((nb,), dtype=int)
    patches = []
    for ib in range(nb):
        f.readline()  # Skip the block label

        # Numbers of points
        nijk[ib] = np.flip(np.array([int(n) for n in f.readline().split()]))

        # Number of blades
        nblade[ib] = int(f.readline().split()[-2])

        # Preallocate patches for this block
        patches.append([])

        # Loop over faces
        for iface in range(6):
            # Loop over patches on this face
            npface = int(f.readline())
            for ipface in range(npface):
                # Extract data for this patch
                ln_patch = f.readline().split()
                plabel, ptype = ln_patch[:2]
                psten = np.array(ln_patch[2:6], dtype=int).reshape(2, 2).T - 1

                # Skip solid wall patches
                # We assume walls are solid by default
                if ptype == "SOL":
                    continue

                # Beware k and i are switched
                ind0 = np.array((0, 0), dtype=int)
                indn = np.array((-1, -1), dtype=int)

                # On const k face
                if iface in (0, 5):
                    if iface == 0:
                        ksten = ind0
                    elif iface == 5:
                        ksten = indn
                    assert np.logical_and(0 <= psten[0], psten[0] < nijk[ib, 1]).all()
                    assert np.logical_and(0 <= psten[1], psten[1] < nijk[ib, 0]).all()
                    jsten, isten = psten

                # On const j face
                elif iface in (1, 4):
                    if iface == 1:
                        jsten = ind0
                    elif iface == 4:
                        jsten = indn
                    assert np.logical_and(0 <= psten[0], psten[0] < nijk[ib, 2]).all()
                    assert np.logical_and(0 <= psten[1], psten[1] < nijk[ib, 0]).all()
                    ksten, isten = psten

                # On const i face
                elif iface in (2, 3):
                    if iface == 2:
                        isten = ind0
                    elif iface == 3:
                        isten = indn
                    assert np.logical_and(0 <= psten[0], psten[0] < nijk[ib, 2]).all()
                    assert np.logical_and(0 <= psten[1], psten[1] < nijk[ib, 1]).all()
                    ksten, jsten = psten

                # Reverse indexing of i dimension
                isten[isten >= 0] = nijk[ib, 0] - isten[isten >= 0] - 1
                isten[isten < 0] = -isten[isten < 0] - 1
                if isten[0] > isten[1]:
                    isten = np.flip(isten)

                # Choose patch kind
                if ptype in ("PER", "CON"):
                    patch_now = turbigen.grid.PeriodicPatch(isten, jsten, ksten)
                elif ptype == "INL":
                    patch_now = turbigen.grid.InletPatch(isten, jsten, ksten)
                elif ptype == "OUT":
                    patch_now = turbigen.grid.OutletPatch(isten, jsten, ksten)
                elif ptype == "ROT":
                    patch_now = turbigen.grid.MixingPatch(isten, jsten, ksten)
                elif ptype == "NMB":
                    patch_now = turbigen.grid.NonMatchPatch(isten, jsten, ksten)
                else:
                    raise Exception(f"Unrecognised IGG patch type {ptype}")

                patches[-1].append(patch_now)

    f.close()

    #
    # Process the plot3d g file with coordinates
    #
    f = open(g_file, "r")

    # First line is number of blocks
    nb_g = int(f.readline())
    assert nb == nb_g

    # The next nb lines are ni, nj, nk for each block
    nijk_g = np.array(
        [np.flip([int(n) for n in f.readline().split()]) for ib in range(nb)]
    )
    assert np.allclose(nijk_g, nijk)

    # Now loop over blocks
    blocks = []
    for ib in range(nb):
        # The x, y, z coordinates are stored sequentially
        # But in AutoGrid nomenclature, z is axial

        yb = _read_coord(f, nijk[ib])
        zb = _read_coord(f, nijk[ib])
        xb = _read_coord(f, nijk[ib])

        # Apply scale factor
        xb *= Lref
        yb *= Lref
        zb *= Lref

        # Convert to polars
        rb = np.sqrt(yb**2.0 + zb**2.0)
        tb = np.arctan2(zb, yb)

        # Make a block
        xrtb = np.stack((xb, rb, tb))
        blocks.append(
            turbigen.grid.BaseBlock.from_coordinates(xrtb, nblade[ib], patches[ib])
        )

    f.close()

    g = turbigen.grid.Grid(blocks)

    # Verify that the patch indices are valid
    for p in g.periodic_patches:
        ijk = p.ijk_limits
        nijk = np.reshape(p.block.shape, (3, 1))
        assert (ijk < nijk).all()

    g.match_patches()

    return g
