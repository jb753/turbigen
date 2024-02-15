"""A general multiblock structured grid class."""
import numpy as np
from turbigen import util
import turbigen.fluid
import turbigen.flowfield
import turbigen.marching_cubes
import importlib
from scipy.spatial import KDTree
from scipy.interpolate import interpn
from enum import IntEnum

logger = util.make_logger()


class MatchDir(IntEnum):
    IPLUS = 0
    JPLUS = 1
    KPLUS = 2
    IMINUS = 3
    JMINUS = 4
    KMINUS = 5
    CONST = 6


class BaseBlock(turbigen.flowfield.BaseFlowField):
    """Base block with coordinates only, flow-field and connectivity data."""

    _data_rows = (
        "x",
        "r",
        "t",
        "w",
        "mu_turb",
        "Omega",
    )  # , "Vx", "Vr", "Vt", "P", "T", "w")

    grid = None

    def __str__(self):
        return (
            f"Block({self.label}, xav={self.x.mean()}, rav={self.r.mean()},"
            f" tav={self.t.mean()}"
        )

    def to_perfect(self):
        bnew = PerfectBlock(shape=self.shape)
        bnew.xrt = self.xrt
        bnew.w = self.w
        bnew.Omega = self.Omega
        bnew._metadata.update(self._metadata)
        for patch in bnew.patches:
            patch.block = bnew
        bnew.mu_turb = np.full_like(bnew.w, np.nan)
        return bnew

    def to_real(self):
        bnew = RealBlock(shape=self.shape)
        bnew.xrt = self.xrt
        bnew.w = self.w
        bnew.Omega = self.Omega
        bnew._metadata.update(self._metadata)
        for patch in bnew.patches:
            patch.block = bnew
        bnew.mu_turb = np.full_like(bnew.w, np.nan)
        return bnew

    @classmethod
    def from_coordinates(cls, xrt, Nb, patches=()):
        # Make empty object of correct shape
        block = cls(shape=xrt.shape[1:])
        block.xrt = xrt
        block._metadata = {"Nb": Nb, "patches": patches}
        for p in patches:
            p.block = block
        block.label = None
        return block

    @property
    def w(self):
        return self._get_data_by_key("w")

    @w.setter
    def w(self, val):
        return self._set_data_by_key("w", val)

    @property
    def mu_turb(self):
        return self._get_data_by_key("mu_turb")

    @mu_turb.setter
    def mu_turb(self, val):
        return self._set_data_by_key("mu_turb", val)

    @property
    def Nb(self):
        return self._get_metadata_by_key("Nb")

    @Nb.setter
    def Nb(self, val):
        return self._set_metadata_by_key("Nb", val)

    @property
    def label(self):
        return self._get_metadata_by_key("label")

    @label.setter
    def label(self, val):
        return self._set_metadata_by_key("label", val)

    @property
    def patches(self):
        return self._get_metadata_by_key("patches")

    @patches.setter
    def patches(self, val):
        return self._set_metadata_by_key("patches", val)

    @property
    def npts(self):
        return self.size

    @property
    def pitch(self):
        return 2.0 * np.pi / self.Nb

    @property
    def ni(self):
        return self.shape[0]

    @property
    def nj(self):
        return self.shape[1]

    @property
    def nk(self):
        return self.shape[2]

    def get_wall(self):

        # Preallocate wall indicator with True on boundaries, False interior
        is_wall = np.ones(self.shape, dtype=bool)
        is_wall[1:-1, 1:-1, 1:-1] = False

        # Loop over patches
        for patch in self.patches:
            # Unset wall indicator if patch is not wall
            if type(patch) in NOT_WALL_PATCHES:
                is_wall[patch.get_slice(trim=1)] = False

        return is_wall

    def check_coordinates(self):
        """Raise an error if coordinates are invalid."""

        # No negative radii
        assert (self.r >= 0.0).all()

        # Finite coordinates
        try:
            assert np.isfinite(self.xrt).all()
        except AssertionError:
            print(
                np.nanmean(self.xrt[0]),
                np.nanmin(self.xrt[0]),
                np.nanmax(self.xrt[0].max),
                np.sum(np.isnan(self.xrt[0])),
            )
            print(
                np.nanmean(self.xrt[1]),
                np.nanmin(self.xrt[1]),
                np.nanmax(self.xrt[1].max),
                np.sum(np.isnan(self.xrt[1])),
            )
            print(
                np.nanmean(self.xrt[2]),
                np.nanmin(self.xrt[2]),
                np.nanmax(self.xrt[2].max),
                np.sum(np.isnan(self.xrt[2])),
            )
            raise Exception("Coordinates not finite")

        # No negative cells
        assert (self.vol > 0.0).all()

    def check_wall_distance(self):
        """Raise an error if wall distance is invalid."""

        # No zeros or nans
        assert np.isfinite(self.w).all()

        # No negative distances
        assert (self.w >= 0.0).all()

        # No huge distances
        Lmax = np.max(self.xrrt.ptp(axis=(1, 2, 3)))
        assert (self.w < Lmax).all()

    def get_connected(self, max_depth=10):
        """Return all blocks that are connected to this patch."""
        blocks = [
            self,
        ]
        for _ in range(max_depth):
            for block in blocks:
                for patch in block.patches:
                    if isinstance(patch, PeriodicPatch) or isinstance(
                        patch, PorousPatch
                    ):
                        if patch.match and (patch.match.block not in blocks):
                            blocks.append(patch.match.block)
        return blocks

    def add_patch(self, patch):
        patch.block = self
        self.patches.append(patch)

    def find_patches(self, cls):
        patches = []
        for patch in self.patches:
            if isinstance(patch, cls):
                patches.append(patch)
        return patches

    @property
    def rotating_patches(self):
        return self.find_patches(RotatingPatch)

    @property
    def inlet_patches(self):
        return self.find_patches(InletPatch)

    @property
    def outlet_patches(self):
        return self.find_patches(OutletPatch)

    def interp_from(self, other):
        """Interpolate solution from another block."""

        # TODO - logic to transfer fluid properties should be a method of the
        # RealState or PerfectState, can then remove branch here
        if not isinstance(self, RealBlock):
            self.cp = other.cp
            self.gamma = other.gamma
            self.mu = other.mu
        else:
            self.fluid_name = other.fluid_name

        if self.shape == other.shape:
            # When shapes match exactly, just take a copy
            self.Vxrt = other.Vxrt.copy()
            self.set_rho_u(other.rho, other.u)
        else:
            # Otherwise, interpolate by index

            # Other block relative indexes
            ijkv_other = [np.linspace(0.0, 1.0, n) for n in other.shape]

            # Target block relative indexes
            ijkv = [np.linspace(0.0, 1.0, n) for n in self.shape]
            ijk = np.stack(np.meshgrid(*ijkv, indexing="ij"), axis=-1)

            self.Vx = interpn(ijkv_other, other.Vx, ijk)
            self.Vr = interpn(ijkv_other, other.Vr, ijk)
            self.Vt = interpn(ijkv_other, other.Vt, ijk)
            rho = interpn(ijkv_other, other.rho, ijk)
            u = interpn(ijkv_other, other.u, ijk)
            self.set_rho_u(rho, u)

    def refine(self, k):
        """Make a finer mesh by halving each edge k times."""

        # Store angular velocity and reset later to make sure Omega.ptp() remains 0
        Omega = self.Omega.mean()

        # Input data
        ni, nj, nk = self.shape
        ijk = range(ni), range(nj), range(nk)
        d = np.moveaxis(self._data, 0, -1)

        # Query data
        iqv = np.linspace(0, ni - 1, (ni - 1) * (2**k) + 1)
        jqv = np.linspace(0, nj - 1, (nj - 1) * (2**k) + 1)
        kqv = np.linspace(0, nk - 1, (nk - 1) * (2**k) + 1)
        ijkq = np.moveaxis(np.stack(np.meshgrid(iqv, jqv, kqv, indexing="ij")), 0, -1)

        # Peform interpolation
        dq = interpn(ijk, d, ijkq)
        self._data = np.moveaxis(dq, -1, 0)

        # Adjust patches
        for patch in self.patches:
            pos = patch.ijk_limits >= 0
            patch.ijk_limits[pos] *= 2**k
            patch.ijk_limits[~pos] = ((patch.ijk_limits[~pos] + 1) * 2**k) - 1

        self.Omega = Omega


class PerfectBlock(turbigen.flowfield.PerfectFlowField, BaseBlock):
    _data_rows = ("x", "r", "t", "Vx", "Vr", "Vt", "P", "T", "w", "mu_turb", "Omega")

    def __str__(self):
        return f"Block({self.label})"


class RealBlock(turbigen.flowfield.RealFlowField, BaseBlock):
    _data_rows = ("x", "r", "t", "Vx", "Vr", "Vt", "rho", "u", "w", "mu_turb", "Omega")

    def __str__(self):
        return f"Block({self.label}"


class Grid:
    """A collection of blocks."""

    def __init__(self, blocks):
        self._blocks = blocks
        self._iter_ind = 0

        for block in self._blocks:
            block.grid = self

    def __iter__(self):
        # Use an iterator class so we can do nested iteration
        class GridIter:
            def __init__(self, g):
                self.g = g
                self.i = -1

            def __next__(self):
                self.i += 1
                if self.i >= len(self.g):
                    raise StopIteration
                return self.g[self.i]

        return GridIter(self)

    def __len__(self):
        return len(self._blocks)

    def __getitem__(self, key):
        return self._blocks[key]

    def extend(self, g):
        self._blocks += g._blocks
        self._iter_ind = 0
        for block in self._blocks:
            block.grid = self

    def append(self, b):
        self._blocks.append(b)
        b.grid = self

    def index(self, block):
        return self._blocks.index(block)

    def find_patches(self, cls):
        patches = []
        for block in self:
            for patch in block.patches:
                if isinstance(patch, cls):
                    patches.append(patch)
        return patches

    @property
    def inlet_patches(self):
        return self.find_patches(InletPatch)

    @property
    def outlet_patches(self):
        return self.find_patches(OutletPatch)

    @property
    def mixing_patches(self):
        return self.find_patches(MixingPatch)

    @property
    def porous_patches(self):
        return self.find_patches(PorousPatch)

    @property
    def periodic_patches(self):
        return self.find_patches(PeriodicPatch)

    @property
    def nonmatch_patches(self):
        return self.find_patches(NonMatchPatch)

    def match_patches(self):
        """Connect all pairs of patches that should match together."""

        # Periodics first, then mixing
        for patches in [
            self.periodic_patches,
            self.mixing_patches,
            self.nonmatch_patches,
        ]:
            # Remove existing matches
            for P in patches:
                P.match = None

            if not np.mod(len(patches), 2) == 0:
                raise Exception(f"Wrong number of {type(patches[0])} to match")
            for P1 in patches:
                for P2 in patches:
                    if P1 is P2:  # or P1.match or P2.match:
                        continue
                    elif P1.check_match(P2):
                        break
            for P in patches:
                if P.match is None:
                    raise Exception(
                        "Could not match patch "
                        f"bid={self._blocks.index(P.block)} "
                        f"pid={P.block.patches.index(P)} {P}"
                    )

    @property
    def nrow(self):
        if len(self.mixing_patches) == 0:
            return 1
        else:
            return len(self.mixing_patches) // 2 + 1

    @property
    def ncell(self):
        return sum([b.size for b in self])

    @property
    def row_blocks(self):
        """Split blocks into rows."""

        if self.nrow == 1:
            return [list(self)]
        else:
            blkin = self.inlet_patches[0].block.get_connected()
            blkout = [b for b in self._blocks if b not in blkin]
            return [blkin, blkout]

    def row_index(self, block):
        for irow, row_block in enumerate(self.row_blocks):
            if block in row_block:
                return irow
        raise Exception(f"Could not locate {block} in the row lists")

    def check_coordinates(self):
        for ib, b in enumerate(self):
            try:
                b.check_coordinates()
            except AssertionError:
                raise Exception(f"Coordinate check failed in block {ib} {b}") from None

    def apply_periodic(self):
        """For each pair of periodic patches, set average of conserved quantities."""
        done = []
        for patch in self.periodic_patches:
            if patch in done:
                continue
            i1 = (slice(3, 7, None),) + patch.get_slice()[1:]
            i2 = (slice(3, 7, None),) + patch.match.get_slice()[1:]
            C1 = patch.get_cut()
            C2 = patch.get_match_cut()
            avg = 0.5 * (C1._data[i1] + C2._data[i2])
            patch.block._data[i1] = avg
            patch.match.block._data[i2] = avg

    def apply_rotation(self, row_types, Omega):
        """Set wall rotations."""

        assert len(row_types) == len(Omega)
        assert self.nrow == len(row_types)

        for row_block, row_type, Omegai in zip(self.row_blocks, row_types, Omega):
            for block in row_block:
                block.Omega = Omegai

                if row_type == "stationary":
                    patches = []

                elif row_type == "tip_gap":
                    patches = [
                        RotatingPatch(i=0),
                        RotatingPatch(i=-1),
                        RotatingPatch(j=0),
                        RotatingPatch(k=0),
                        RotatingPatch(k=-1),
                    ]

                elif row_type == "shroud":
                    patches = [
                        RotatingPatch(i=0),
                        RotatingPatch(i=-1),
                        RotatingPatch(j=0),
                        RotatingPatch(j=-1),
                        RotatingPatch(k=0),
                        RotatingPatch(k=-1),
                    ]

                else:
                    raise Exception("Unknown row type %s", row_type)

                for patch in patches:
                    patch.Omega = Omegai
                    block.add_patch(patch)

    def apply_inlet(self, state, Alpha, Beta):
        for patch in self.inlet_patches:
            patch.state = state
            patch.Alpha = Alpha
            patch.Beta = Beta

    def apply_outlet(self, Pout):
        for patch in self.outlet_patches:
            patch.Pout = Pout

    def apply_throttle(self, mdot, Kpid):
        for patch in self.outlet_patches:
            patch.mdot_target = mdot
            patch.Kpid = Kpid

    def update_outlet(self):
        for patch in self.outlet_patches:
            if patch.mdot_target:
                patch.Pout = patch.get_cut().P.mean()

    def check_outlet_choke(self):
        for patch in self.outlet_patches:
            if patch.mdot_target:
                C = patch.get_cut()
                Cm = C.mix_out()[0]
                if Cm.Mam > 1.0:
                    print(
                        f"Warning: outlet Mam={Cm.Mam:.3f} is choked; this can affect"
                        " mass flow continuity."
                    )

    def get_wall_nodes(self):
        """Unstructured coordinates of all points on walls."""

        # Loop over blocks
        xrrt_wall_block = []
        for block in self:

            # Assemble unstructured wall coordinates for this block
            xrtbw = block.xrt[:, block.get_wall()].reshape(3, -1)

            # Replicate by +/- a pitch
            pitch = 2.0 * np.pi / float(block.Nb)
            dxrt = np.zeros_like(xrtbw)
            dxrt[2] = pitch
            xrtbw_rep = np.concatenate((xrtbw - dxrt, xrtbw, xrtbw + dxrt), axis=1)

            # Convert to rt
            xrrtbw = xrtbw_rep + 0.0
            xrrtbw[2] *= xrrtbw[1]

            xrrt_wall_block.append(xrrtbw)

        # Join all blocks together
        xrrt_wall = np.concatenate(xrrt_wall_block, axis=1)

        return xrrt_wall

    def calculate_wall_distance(self):
        """Get distance to nearest wall node for all grid points."""

        # Initialise a kdtree of wall points
        kdtree = KDTree(self.get_wall_nodes().T)

        # Loop over blocks
        for block in self:
            # wmax = 2.0 * np.pi * block.r.max() / block.Nb * 0.1

            block.w = kdtree.query(block.to_unstructured().xrrt.T, workers=-1,)[
                0
            ].reshape(block.shape)

    def apply_guess_meridional(self, Fg):
        """Apply meridional guess from a mean-line object."""

        # Ensure the guess flow field is sane
        Fg.check_flow()

        # Initialise a kdtree of guess points
        xrgT = Fg.xr.T
        kdtree = KDTree(xrgT)

        # Loop over all blocks
        for block in self:
            # Copy fluid props etc.
            block._metadata.update(Fg._metadata)

            # Find indices of nearest guess point to all block points
            xri = block.to_unstructured().xr.T
            ind_nearest = kdtree.query(
                xri,
                workers=-1,
            )[1]

            # Set thermodynamic properties
            rob = Fg.rho[ind_nearest].reshape(block.shape)
            ub = Fg.u[ind_nearest].reshape(block.shape)
            block.set_rho_u(rob, ub)

            # Set velocities
            block.Vxrt = Fg.Vxrt[:, ind_nearest].reshape(block.Vxrt.shape)

            block.mu_turb = np.full_like(block.mu_turb, np.mean(Fg.mu))

    def apply_guess_3d(self, g):
        for block, block_other in zip(self, g):
            block.interp_from(block_other)

    def run(self, settings, machine):
        """Run a solver on the grid, prescribing some settings."""

        # Dynamically import the solver and run

        settings_copy = settings.copy()
        solver_type = settings_copy.pop("type")
        solver = importlib.import_module(f".{solver_type}", package="turbigen.solvers")
        return solver.run(self, settings_copy, machine)

    def unstructured_cut(self, xr_cut):
        """xr_cut has axes [x or r, point on cutting line]."""

        logger.debug("Taking an unstructured cut")
        logger.debug(f"Cut plane xr_cut={xr_cut}")

        # Determine cut plane slope
        xc, rc = xr_cut
        dxrc = np.diff(xr_cut, axis=1)

        bp = []

        def _is_above(xq, rq, xrc):
            xc, rc = xrc
            dxc = np.diff(xc)
            drc = np.diff(rc)
            if not dxc == 0.0:
                return rq >= rc[0] + drc / dxc * (xq - xc[0])
            elif not drc == 0.0:
                return xq >= xc[0] + dxc / drc * (rq - rc[0])
            else:
                raise Exception("Cannot cut with dx and dr both zero")

        # Loop over blocks
        for block in self:
            logger.debug("****")
            logger.debug(
                f"Block xrt = {block.x.mean(), block.r.mean(), block.t.mean()}"
            )

            assert not np.isnan(block.mu_turb).any()

            dist = turbigen.util.signed_distance(xr_cut, block.xr)

            # Identify zero crossings of signed distance
            dsgn = np.abs(np.diff(np.sign(dist), axis=0))

            if np.logical_not(dsgn).all():
                logger.debug("This block is not cut.")
                continue
            logger.debug("Block is intersected.")

            # Determine the i indices on each side of cut
            ni, nj, nk = block.shape

            # BEGIN OLD WAY
            # if side[0, :, :].any():
            #     side = np.logical_not(side)
            # icut = np.argmax(side, axis=0, keepdims=True)
            # logger.debug(f"min icut={icut.min()}")
            # logger.debug(f"shape={block.shape}")
            # # print(np.sum(icut == 0))
            # # print(np.size(icut))
            # if icut.min() <= 1:
            #     logger.debug(f"min icut={icut.min()}, skipping this block")
            #     continue
            # if icut.min() == 0:
            #     jplot = 10
            #     import matplotlib.pyplot as plt
            #     # fig, ax = plt.subplots()
            #     # bplot = block[:, jplot, :].squeeze()
            #     # ax.plot(bplot.r, bplot.rt, "kx")
            #     # plt.savefig("test.pdf")
            # # quit()
            # # Get xr coordinates on either side of cut
            # xi = np.take_along_axis(block.x, icut - 1, axis=0)
            # xip1 = np.take_along_axis(block.x, icut, axis=0)
            # ri = np.take_along_axis(block.r, icut - 1, axis=0)
            # rip1 = np.take_along_axis(block.r, icut, axis=0)
            # END OLD WAY

            icut = np.argmax(dsgn, axis=0, keepdims=True)
            # If there are no zero crossing, then argmax will return icut=0
            # If the first cell is cut, argmax will return icut=1

            # Get xr coordinates on either side of cut
            xrti = np.take_along_axis(block.xrt, icut[None, ...], axis=1)
            xrtip1 = np.take_along_axis(block.xrt, icut[None, ...] + 1, axis=1)

            # Solve for cut fraction along i grid lines
            Dxrti = xrtip1 - xrti

            # Choose if the cut is axial or radial
            if np.abs(dxrc[0]) > np.abs(dxrc[1]):
                drdx = dxrc[1] / dxrc[0]
                frac_cut = (rc[0] - xrti[1] + drdx * (xrti[0] - xc[0])) / (
                    Dxrti[1] - drdx * Dxrti[0]
                )
            else:
                dxdr = dxrc[0] / dxrc[1]
                if np.any((Dxrti[0] - dxdr * Dxrti[1]) == 0):
                    continue
                frac_cut = (xc[0] - xrti[0] + dxdr * (xrti[1] - rc[0])) / (
                    Dxrti[0] - dxdr * Dxrti[1]
                )
            # Remove grid lines with no zero crossing
            frac_cut[icut == 0] = np.nan

            assert not (np.isnan(block._data).any())

            # xi = np.take_along_axis(block.x, icut, axis=0)
            # xip1 = np.take_along_axis(block.x, icut+1, axis=0)
            # ri = np.take_along_axis(block.r, icut, axis=0)
            # rip1 = np.take_along_axis(block.r, icut +1, axis=0)

            # # Solve for cut fraction along i grid lines
            # Dri = rip1 - ri
            # Dxi = xip1 - xi

            # # Choose if the cut is axial or radial
            # logger.debug(f"dxc={dxc}, drc={drc}")
            # if np.abs(dxc) > np.abs(drc):
            #     drdx = drc / dxc
            #     logger.debug(f"This is an r~const cut. drdx={drdx}")
            #     frac_cut = (rc[0] - ri + drdx * (xi - xc[0])) / (Dri - drdx * Dxi)

            # else:
            #     dxdr = dxc / drc
            #     logger.debug(f"This is an x~const cut. dxdr={dxdr}")
            #     frac_cut = (xc[0] - xi + dxdr * (ri - rc[0])) / (Dxi - dxdr * Dri)

            # OLD WAY
            # # Points outside the block will have cut fractions outside unit interval
            # frac_cut[frac_cut < 0.0] = np.nan
            # frac_cut[frac_cut > 1.0] = np.nan
            # NEW WAY
            # frac_cut[icut == 0] = np.nan

            bpi = np.take_along_axis(block._data, icut[None, ...], axis=1)
            bpip1 = np.take_along_axis(block._data, icut[None, ...] + 1, axis=1)
            bp_now = np.squeeze((1.0 - frac_cut) * bpi + frac_cut * bpip1)

            # # Evaluate the flow properties at these cut fractions
            # bp_now = np.zeros((block.nprop, nj, nk))
            # for ind in range(block.nprop):
            #     bpi = np.take_along_axis(block._data[ind], icut, axis=0)
            #     bpip1 = np.take_along_axis(block._data[ind], icut+1, axis=0)
            #     bp_now[ind, :, :] = np.squeeze(
            #         (1.0 - frac_cut) * bpi + frac_cut * bpip1
            #     )

            # Trim nans
            kgood = [0, nk - 1]
            abort = False
            if np.any(np.isnan(bp_now[0])):
                for j in range(nj):
                    xx = bp_now[0, j, :]
                    kgood_now = np.atleast_1d(np.squeeze(np.argwhere(~np.isnan(xx))))
                    if len(kgood_now) == 0:
                        abort = True
                        continue
                    kgood[0] = np.maximum(kgood[0], kgood_now[0])
                    kgood[-1] = np.minimum(kgood[-1], kgood_now[-1])
                # bp_now = np.delete(bp_now, knan, axis=2)
                if abort:
                    continue
                # bp_now = bp_now[:, :, kgood[0] : (kgood[-1] + 1)]
                bp_now = bp_now[:, :, kgood[0] : (kgood[-1])]

            if bp_now.shape[-1] == 1:
                continue

            if np.any(np.isnan(bp_now)):
                for iprop in range(block.nprop):
                    print(
                        f"{block._data_rows[iprop]}:"
                        f" {np.sum(np.isnan(bp_now[iprop]))}/{np.size(bp_now[iprop])}"
                    )
                # import matplotlib.pyplot as plt
                # print(kgood)
                # fig, ax = plt.subplots()
                # m = ax.contourf(frac_cut.squeeze())
                # plt.colorbar(m)
                # plt.show()
                raise Exception("NaNs remain in unstructured cut:")
            # if np.any(np.isnan(bp_now)):
            #     continue
            # assert not np.any(np.isnan(bp_now))

            last_block = block

            bp.append(bp_now)

        # Now join the blocks together
        assert np.ptp([bpi.shape[1] for bpi in bp]) == 0
        bp_tmp = []
        for i, bpi in enumerate(bp):
            if bpi.shape[-1] <= 1:
                continue
            if bpi[2, 0, 0] > bpi[2, 0, -1]:
                bp_now = np.flip(bpi, axis=2)
            else:
                bp_now = np.copy(bpi)
            if i < len(bp):
                bp_now = bp_now[:, :, :-1]
            bp_tmp.append(bp_now)
        bp = bp_tmp

        rtref = [bpi[2, 0, 0] for bpi in bp]
        bp = [bp[i] for i in np.argsort(rtref)]
        bp_all = np.concatenate(bp, axis=2)

        # Insert a singleton i dimension
        bp_all = np.expand_dims(bp_all, 1)

        cut = last_block.empty(shape=bp_all.shape[1:])
        cut._data = bp_all
        cut._metadata = last_block._metadata

        cut.Omega = cut.Omega.mean()

        if not np.isnan(block.xrt).any():
            assert not np.isnan(cut.xrt).any()

        if not np.isnan(block.Vxrt).any():
            assert not np.isnan(cut.P).any()
            assert not np.isnan(cut.T).any()
            assert not np.isnan(cut.Vxrt).any()

        return cut

    def unstructured_cut_marching(self, xr_cut):
        """Take an unstructured cut using marching cubes."""

        triangles = []
        last_block = None
        for block in self:
            # Evaluate signed distance for all points
            dist = turbigen.util.signed_distance(xr_cut, block.xr)

            # Get triangles for this block
            triangles_block = turbigen.marching_cubes.marching_cubes(block._data, dist)

            # Add triangles to the list
            if triangles_block is not None:
                triangles.append(triangles_block)
                last_block = block

        if triangles:
            triangles = np.concatenate(triangles).transpose(2, 0, 1)

            # Now make into a 2D state
            out = last_block.empty(shape=triangles.shape[1:])
            out._data[:] = triangles

            return out

    def cut_blade_sides(self):
        """Nested list of pressure/suction side cuts in each row."""

        # Assuming a H-mesh
        cuts = []

        for i in range(self.nrow):
            ile = None
            ite = None
            for patch in self.periodic_patches:
                this_row = patch.block in self.row_blocks[i]
                same_block = patch.match.block == patch.block
                spans_j = np.allclose(patch.ijk_limits[1], [0, -1])
                spans_i = np.allclose(patch.ijk_limits[0], [0, -1])
                k0 = np.allclose(patch.ijk_limits[2], [0, 0])
                if same_block and spans_j and k0 and not spans_i and this_row:
                    if patch.ijk_limits[0, 0] == 0:
                        ile = patch.ijk_limits[0, 1]
                    elif patch.ijk_limits[0, 1] == -1:
                        ite = patch.ijk_limits[0, 0]

            if not ile or not ite:
                cuts.append(None)
                continue

            # Get both sides
            Ck0 = self[i][ile : (ite + 1), :, None, 0].copy()
            Cnk = self[i][ile : (ite + 1), :, None, -1].copy()
            C = [Ck0, Cnk]

            # Find the side at highest theta
            iu = np.argmax([Ci.t.max() for Ci in C])
            C[iu].t -= self[i].pitch

            cuts.append(C)

        return cuts

    @property
    def is_hmesh(self):
        return len(self) == len(self.row_blocks)

    def cut_blade_surfs(self):
        """O-mesh style cuts for the blades in each row."""

        surfs = []

        if self.is_hmesh:
            row_sides = self.cut_blade_sides()
            for sides in row_sides:
                if sides is None:
                    surfs.append(None)
                else:
                    cut_now = sides[0].concatenate(
                        (sides[0].flip(axis=0), sides[1][1:, ...]), axis=0
                    )
                    surfs.append([cut_now])
        else:

            for row_block in self.row_blocks:

                # Preallocate list for this row
                surfs.append([])

                # Determine full span nj as the modal nj in this row
                nj_vals, nj_counts = np.unique(
                    [b.shape[1] for b in self], return_counts=True
                )
                nj = nj_vals[np.argmax(nj_counts)]

                # Loop over blocks and find o-meshes
                for b in row_block:
                    if (
                        np.allclose(b[0, :, :].xrt, b[-1, :, :].xrt)
                        and b.shape[1] == nj
                    ):
                        surfs[-1].append(b[:, :, None, 0])

        return surfs

    def cut_mid_pitch(self):
        # Assumes H-mesh
        k = self[0].shape[2] // 2
        return [b[:, :, k].squeeze() for b in self]

    def cut_span(self, spf):
        # Find j index nearest to requested span fraction
        jspf = np.argmin(np.abs(self[0].spf[1, :, 1] - spf))
        nj = self[0].shape[1]
        logger.debug(f"Cutting at spf={spf}: jspf={jspf}, nj={nj}")

        bcut = []
        for block in self:
            njb = block.shape[1]
            if njb < nj:
                # This is a tip block
                jtip = jspf - (nj - njb)
                if jtip >= 0:
                    logger.debug(f"Tip block jcut={jtip}")
                    bcut.append(block[:, jtip, :])
                else:
                    logger.debug("Skipping tip block")
            else:
                # This is a normal block
                bcut.append(block[:, jspf, :])
                logger.debug(f"Main block jcut={jspf}")
        return bcut

    def partition(self, N):
        nb = len(self)

        if N == 1:
            procids = [0 for _ in self]
        elif N == nb:
            procids = list(range(0, nb))
        elif N > nb:
            raise Exception(f"Cannot load balance {nb} blocks into {N} partitions!")
        else:
            # Lazy import
            import metis

            # Assemble block sizes and adjacencyectivity
            vertex_weights = (
                np.round(np.array([b.size for b in self]) / self.ncell * 100)
                .astype(int)
                .tolist()
            )
            adjacency = []
            logger.debug("Weights and adjacency for each block:")
            for ib, block in enumerate(self):
                adjacency_now = []
                for patch in block.patches:
                    if isinstance(patch, PeriodicPatch) or isinstance(
                        patch, PorousPatch
                    ):
                        if patch.match:
                            nxblock = patch.match.block
                            nxblockid = self._blocks.index(nxblock)
                            if nxblockid not in adjacency_now:
                                adjacency_now.append(nxblockid)
                adjacency.append(tuple(adjacency_now))
                logger.debug(f"    {vertex_weights[ib]} {adjacency[-1]}")
            G = metis.adjlist_to_metis(adjacency, vertex_weights)
            _, procids = metis.part_graph(G, N)
            procids = np.array(procids)

            # Metis may produce fewer partitions than requested, which results
            # in skipped procids. Shift the procids first so there are no gaps.
            procids_unique = np.unique(procids)
            procids_missing = np.setdiff1d(range(N), procids_unique)
            for pmiss in procids_missing:
                procids[procids >= pmiss] -= 1
            npart = procids.max() + 1

            if not npart == N:
                logger.debug(
                    f"Metis produced {npart} partitions, fewer than target {N}"
                )
                logger.debug(f"Original procids {procids}")

                # Find indices of repeated procids, i.e. those that are not
                # required to form the unique array
                ind_repeat = []
                proc_used = []
                for iiproc, iproc in enumerate(procids):
                    if iproc in proc_used:
                        ind_repeat.append(iiproc)
                    else:
                        proc_used.append(iproc)

                # ind_repeat = np.setdiff1d(range(N), ind_unique)
                logger.debug(f"Indexes of repeats {ind_repeat}")
                procids_add = list(range(npart - 1, N))
                logger.debug(f"procids to be added {procids_add}")

                # Loop over the left-over procids and reassign to repeated procids
                for iipart, ipart in enumerate(procids_add):
                    procids[ind_repeat[iipart]] = ipart
                logger.debug(f"Corrected procids {procids}")

            assert len(np.unique(procids)) == N
            assert (procids >= 0).all()
            assert len(procids) == nb
            assert procids.max() == (N - 1)

            # Sum cells per partition
            ncell_part = np.zeros(N)
            for ib, b in enumerate(self):
                ncell_part[procids[ib]] += b.size
            logger.info(
                "Load-balanced cells per GPU/10^6: "
                f"{np.array2string(ncell_part/1e6,precision=2)}"
            )
            assert (ncell_part > 0.0).all()

        return procids


class Patch:
    """Base class for all patches."""

    @staticmethod
    def _get_indices(ijk):
        if ijk is None:
            st, en = (0, -1)
        else:
            try:
                st, en = ijk
            except TypeError:
                st = ijk
                en = ijk + 1
        return st, en

    def __init__(self, i=None, j=None, k=None, label=None):
        """Select a subset of a block by indices."""

        self.label = label

        # ijk limits are INCLUSIVE
        # because we cannot use an integer to range slice including last element
        self.ijk_limits = np.empty((3, 2), dtype=int)
        for n, ind in enumerate([i, j, k]):
            if ind is None:
                self.ijk_limits[n] = (0, -1)
            else:
                try:
                    self.ijk_limits[n] = ind
                except TypeError:
                    self.ijk_limits[n] = (ind, ind)

        # Disallow volume patches
        assert np.sum(np.diff(self.ijk_limits) == 0) >= 1

        self.block = None

        self.idir = None
        self.jdir = None
        self.kdir = None

    @property
    def ijkdir(self):
        return [self.idir, self.jdir, self.kdir]

    @ijkdir.setter
    def ijkdir(self, value):
        self.idir, self.jdir, self.kdir = value

    def get_slice(self, offset=0, trim=0):
        # Convert inclusive start/end to indices for range slice
        sl = []
        for lim in self.ijk_limits:
            lim_now = lim.copy()

            if lim.ptp() == 0:
                if lim[0] == 0:
                    lim_now += offset
                else:
                    lim_now -= offset
            else:
                lim_now[0] += trim
                lim_now[1] -= trim

            if (lim_now == -1).any():
                sl.append(slice(lim_now[0], None))
            else:
                sl.append(slice(lim_now[0], lim_now[1] + 1))
        return tuple(sl)

    def get_cut(self, offset=0):
        return self.block[self.get_slice(offset)]

    def __str__(self):
        return (
            f"{self.__class__.__name__}(i={self.ijk_limits[0]}, j={self.ijk_limits[1]},"
            f" k={self.ijk_limits[2]}, label={self.label}, block={self.block})"
        )


class PeriodicPatch(Patch):
    """Node-to-node matching periodicity."""

    match = None

    def check_match(self, other, rtol=1e-4):
        return _get_patch_connectivity(self, other, corners_only=False, rtol=rtol)

    def get_match_cut(self, offset=0):
        # We need to establise a permutation order and set of flips that will
        # transform the other coordinates to our indexing
        perm = np.empty(3, dtype=int)
        flip = np.empty(3, dtype=int)

        for n in range(3):
            if self.ijkdir[n] == MatchDir.IPLUS:
                perm[n] = 0
                flip[n] = 0
            elif self.ijkdir[n] == MatchDir.JPLUS:
                perm[n] = 1
                flip[n] = 0
            elif self.ijkdir[n] == MatchDir.KPLUS:
                perm[n] = 2
                flip[n] = 0
            elif self.ijkdir[n] == MatchDir.IMINUS:
                perm[n] = 0
                flip[n] = 1
            elif self.ijkdir[n] == MatchDir.JMINUS:
                perm[n] = 1
                flip[n] = 1
            elif self.ijkdir[n] == MatchDir.KMINUS:
                perm[n] = 2
                flip[n] = 1
            elif self.ijkdir[n] == MatchDir.CONST:
                perm[n] = n
                flip[n] = 0

        perm = np.insert(perm + 1, 0, 0)
        flip = np.where(np.insert(flip, 0, 0))[0]

        Cnx = self.match.get_cut(offset)
        Cnx._data = np.flip(Cnx._data.transpose(perm), axis=flip).copy()

        return Cnx


class PorousPatch(PeriodicPatch):
    """Node-to-node matching periodicity with pressure loss."""

    porous_fac_loss = None

    def check_match(self, other):
        match_coords = super().check_match(other)
        try:
            match_porous = np.isclose(self.porous_fac_loss, other.porous_fac_loss)
        except (AttributeError, TypeError):
            match_porous = False
        return match_coords and match_porous


class MixingPatch(Patch):
    """Connect two reference frames with a mixing plane."""

    match = None
    slide = False

    def check_match(self, other, rtol=1e-6):
        # Slice both the patches
        C = [self.get_cut(), other.get_cut()]

        # Reference length to set meridional tolerance
        Lref = np.max((C[0].x.ptp(), C[0].r.ptp()))

        # Check these cuts satisfy the conditions
        try:
            assert np.diff(self.ijk_limits[0]) == 0
            assert np.diff(other.ijk_limits[0]) == 0
            for Ci in C:
                assert (np.ptp(Ci.xr, axis=-1) < Lref * rtol).all()
        except AssertionError:
            raise Exception(f"Invalid mixing patch indices {self} {other}")

        # Get coordinates of hub and casing on each patch
        # xr has dimensions: [which patch, x or r, hub/casing]
        xr = np.stack([Ci.xr[:, :, (0, -1), :].mean(axis=-1).squeeze() for Ci in C])

        nj = np.array([Ci.shape[1] for Ci in C], dtype=int)

        err = np.abs(np.diff(xr, axis=0).squeeze())
        err_rel = err / Lref

        if err_rel.max() < rtol:
            self.match = other
            other.match = self

            if nj.ptp() == 0:
                dt = np.stack(
                    [np.diff(Ci.t[:, :, (0, -1)], axis=-1).squeeze() for Ci in C]
                )
                if np.allclose(dt[0], dt[1]):
                    self.slide = True
                    other.slide = True

        else:
            return False


class InletPatch(Patch):
    state = None
    rfin = 0.5
    force_type = None
    amplitude = 0.0
    phase = 0.0
    store = None


class InviscidPatch(Patch):
    pass


class OutletPatch(Patch):
    Pout = None
    mdot_target = None
    Kpid = None
    force = False
    amplitude = 0.0
    phase = 0.0


class RotatingPatch(Patch):
    Omega = None


class ProbePatch(Patch):
    pass


class NonMatchPatch(Patch):
    match = None

    def check_match(self, other, rtol=1e-4):
        return _get_patch_connectivity(self, other, corners_only=True, rtol=rtol)

        # # Get the four corners of each patch
        # C = [self.get_cut(), other.get_cut()]
        # xrt = np.stack(
        #     [Ci.xrt.squeeze()[:, [0, 0, -1, -1], [0, -1, 0, -1]] for Ci in C]
        # )

        # # Number of blades  should be equal on both patches
        # nb = np.array([Ci.Nb for Ci in C])
        # if nb.ptp() > 0:
        #     return False

        # # Get coordinates
        # Lref = np.max((C[0].x.ptp(), C[0].r.ptp()))

        # # Cope with circumferential offset by taking mod wrt pitch
        # pitch = 2.0 * np.pi / float(nb[0])
        # xrt[:, 2] = np.mod(xrt[:, 2], pitch)

        # # Sort coordinates in a unique order
        # for xrti in xrt:
        #     xrti[:] = xrti[:, np.argsort(np.prod(xrti, axis=0))]

        # # Test for equality
        # err = np.abs(np.diff(xrt, axis=0).squeeze())
        # err_rel = np.empty_like(err)
        # err_rel[:2] = err[:2, :] / Lref
        # err_rel[2] = err[2, :] / pitch

        # return err_rel.max() < rtol


# Default is that block edges are walls
# So we want to identify patches that are NOT walls
NOT_WALL_PATCHES = [
    InletPatch,
    OutletPatch,
    MixingPatch,
    PeriodicPatch,
    PorousPatch,
    ProbePatch,
]


def _get_patch_connectivity(patch, other, corners_only=False, rtol=1e-4):
    """Patch attributes describing periodic or mixing connectivity."""

    # Get patches and their coordinates and shapes
    p = [patch, other]
    xrt = [pi.get_cut().xrt.copy() for pi in p]
    dijk = [xrti.shape[1:] for xrti in xrt]

    # The patches cannot match if their pitches are different
    pitch = [2.0 * np.pi / pi.block.Nb for pi in p]
    if not np.ptp(pitch) == 0.0:
        return False

    # Cope with circumferential offset by taking mod wrt pitch
    for xrti in xrt:
        xrti[2, ...] = np.mod(xrti[2, ...], pitch[0])
        # We need to be careful at the pitch boundaries. For example, if one point
        # is pitch - tol/2 and its matching point is pitch + tol/2 then they
        # *should* match, but will be in error by whole pitch after modulus.
        # So move any points very close to upper pitch boundary back to zero
        xrti[2, ...][xrti[2, ...] / pitch[0] > (1.0 - rtol)] = 0.0

    # We are going to loop over all possible choices for i/j/kdir
    # and return from this function if the coordinates match.
    # Skip iterations if dir=-1 does not match shape or a direction is repeated.
    # TS3 notations for dirs:
    # -1: current patch is on this face
    # 0: matches i on next patch
    # 1: matches j
    # 2: matches k
    # 3: matches -i
    # 4: matches -j
    # 5: matches -k

    # Begin looping
    for idir in range(-1, 6):
        idirm = np.mod(idir, 3)
        # If we have one i point, patch on i face, idir must be -1
        if dijk[0][0] == 1 and not idir == -1:
            continue
        for jdir in range(-1, 6):
            jdirm = np.mod(jdir, 3)
            # If we have one j point, patch on j face, jdir must be -1
            if dijk[0][1] == 1 and not jdir == -1:
                continue
            for kdir in range(-1, 6):
                kdirm = np.mod(kdir, 3)
                # If we have one k point, patch on k face, kdir must be -1
                if dijk[0][2] == 1 and not kdir == -1:
                    continue

                # Make a permutation order that will convert next patch
                # coordinates to same shape as current patch
                order = np.array(
                    [
                        idirm if idir >= 0 else -1,
                        jdirm if jdir >= 0 else -1,
                        kdirm if kdir >= 0 else -1,
                    ]
                )

                # Skip repeated directions - two dirs cannot match to same dir
                # on the next patch
                if not len(np.unique(order)) == 3:
                    continue

                # Choose location for next patch const face
                assert np.sum(order == -1) == 1
                order[order == -1] = np.setdiff1d([0, 1, 2], order)

                # Which axes of next patch need flipping
                dirs = np.array([idir, jdir, kdir])
                flip = np.where(dirs > 2)[0]

                # Add one to dirs array because xrt contains 3 coordinates on
                # first dim, and apply transpose or flips
                xrt_next = np.flip(
                    xrt[1].copy().transpose(np.insert(order + 1, 0, 0)),
                    axis=tuple(flip + 1),
                )

                if corners_only:
                    # For non-matching patches, we only care about corners
                    xrtc1 = xrt_next.squeeze()[:, (0, 0, -1, -1), (0, -1, 0, -1)]
                    xrtc2 = xrt[0].squeeze()[:, (0, 0, -1, -1), (0, -1, 0, -1)]
                    err = np.abs(xrtc1 - xrtc2)
                else:
                    # For fully matching patches, we expect the shapes to be
                    # compatible, and coordinates to be coincident
                    if not xrt_next.shape == xrt[0].shape:
                        continue
                    err = np.abs(xrt_next - xrt[0])

                # Test for coordinate equality
                dxref = xrt_next[0].ptp()
                drref = xrt_next[1].ptp()
                Lref = np.max((dxref, drref))
                err_rel = np.empty_like(err)
                err_rel[0] = err[0, :] / Lref
                err_rel[1] = err[1, :] / Lref
                err_rel[2] = err[2, :] / pitch[0]

                # Although the TS User Manual says that -1 implies the constant
                # direction, it seems that 6 is the real convention
                dirs[dirs == -1] = 6
                idir6, jdir6, kdir6 = dirs

                # Only error if more than 1 in 1000 points do not match
                if err_rel.max() < rtol:
                    patch.idir = MatchDir(idir6)
                    patch.jdir = MatchDir(jdir6)
                    patch.kdir = MatchDir(kdir6)

                    patch.match = other

                    return True

    return False
