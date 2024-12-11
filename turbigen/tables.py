"""Generate thermodynamic property tables in TS4 format."""

from CoolProp import CoolProp as CP
import numpy as np
from multiprocessing import Pool
from itertools import repeat
from turbigen import util, fluid
from scipy.interpolate import RegularGridInterpolator

PROPERTY_TABLES = [
    (CP.ispeed_sound, CP.DmassUmass_INPUTS),
    (CP.iP, CP.DmassUmass_INPUTS),
    (CP.iT, CP.DmassUmass_INPUTS),
    (CP.iUmass, CP.HmassP_INPUTS),
    (CP.iSmass, CP.DmassUmass_INPUTS),
    (CP.iUmass, CP.PSmass_INPUTS),
    (CP.iDmass, CP.PSmass_INPUTS),
    (CP.iP, CP.HmassSmass_INPUTS),
    (CP.iT, CP.HmassSmass_INPUTS),
    (CP.iCpmass, CP.DmassT_INPUTS),
    (CP.iHmass, CP.SmassT_INPUTS),
    ((CP.iUmass, CP.iP, CP.iDmass), CP.DmassP_INPUTS),
    ((CP.iUmass, CP.iDmass, CP.iP), CP.DmassP_INPUTS),
    ((CP.iHmass, CP.iP, CP.iDmass), CP.DmassP_INPUTS),
    ((CP.iHmass, CP.iDmass, CP.iP), CP.DmassP_INPUTS),
    ((CP.iSmass, CP.iP, CP.iDmass), CP.DmassP_INPUTS),
    ((CP.iSmass, CP.iDmass, CP.iP), CP.DmassP_INPUTS),
    (CP.iP, CP.DmassHmass_INPUTS),
    ((CP.iT, CP.iDmass, CP.iUmass), CP.DmassP_INPUTS),
    ((CP.iT, CP.iUmass, CP.iDmass), CP.DmassP_INPUTS),
    ((CP.iP, CP.iDmass, CP.iUmass), CP.DmassP_INPUTS),
    ((CP.iP, CP.iUmass, CP.iDmass), CP.DmassP_INPUTS),
    (CP.iUmass, CP.DmassT_INPUTS),
]
NTABLE = len(PROPERTY_TABLES)


GRID_TYPES = {
    CP.DmassUmass_INPUTS: (CP.iDmass, CP.iUmass),
    CP.HmassP_INPUTS: (CP.iHmass, CP.iP),
    CP.PSmass_INPUTS: (CP.iP, CP.iSmass),
    CP.HmassSmass_INPUTS: (CP.iHmass, CP.iSmass),
    CP.DmassT_INPUTS: (CP.iDmass, CP.iT),
    CP.SmassT_INPUTS: (CP.iSmass, CP.iT),
    CP.DmassP_INPUTS: (CP.iDmass, CP.iP),
    CP.DmassHmass_INPUTS: (CP.iDmass, CP.iHmass),
}

TABLES_BY_GRID = {
    k: [i for i, (_, grid_type) in enumerate(PROPERTY_TABLES) if grid_type == k]
    for k in GRID_TYPES
}


FLIP_GRID = [CP.PSmass_INPUTS, CP.HmassSmass_INPUTS]


LOG_VARS = [CP.iDmass, CP.iP]

INPUT_VARS = [CP.iDmass, CP.iUmass, CP.iHmass, CP.iP, CP.iSmass, CP.iT]


def _get_limits(fluid_name, smin, smax, Pmin, Tmax):
    """Evaluate min/max values of all input properties given a bounding box.

    Parameters
    ----------
    fluid_name: str
        Name of the working fluid in CoolProp database.
    smin: float
        Minimum entropy.
    smax: float
        Maximum entropy.
    Pmin: float
        Minimum pressure.
    Tmax: float
        Maximum temperature.

    Returns
    -------
    limits: dict
        A dictionary of `(min_val, max_val)` tuples, keyed by CoolProp ID
        number for all input variables.

    """

    abstract_state = CP.AbstractState("HEOS", fluid_name)

    # Evaluate all input variables at four corners
    corners = []

    # min s, min P
    abstract_state.update(CP.PSmass_INPUTS, Pmin, smin)
    corners.append({ivar: abstract_state.keyed_output(ivar) for ivar in INPUT_VARS})

    # max s, min P
    abstract_state.update(CP.PSmass_INPUTS, Pmin, smax)
    corners.append({ivar: abstract_state.keyed_output(ivar) for ivar in INPUT_VARS})

    # max s, max T
    abstract_state.update(CP.SmassT_INPUTS, smax, Tmax)
    corners.append({ivar: abstract_state.keyed_output(ivar) for ivar in INPUT_VARS})

    # min s, max T
    abstract_state.update(CP.SmassT_INPUTS, smin, Tmax)
    corners.append({ivar: abstract_state.keyed_output(ivar) for ivar in INPUT_VARS})

    # Minimum and maximum for each variable
    minimums = {ivar: np.min([c[ivar] for c in corners]) for ivar in INPUT_VARS}
    maximums = {ivar: np.max([c[ivar] for c in corners]) for ivar in INPUT_VARS}

    # Assemble limits
    limits = {ivar: (minimums[ivar], maximums[ivar]) for ivar in INPUT_VARS}

    return limits


def _get_grids(limits, ni, nj):
    """Evaluate 2D grids of input variables needed for TS4.

    Parameters
    ----------
    limits: dict
        Dictionary of `(min_val, max_val)` tuples, keyed by CoolProp ID
        number for all input variables, from :func:`get_limits`.
    ni: int
        Number of points along horizontal dimension.
    nj: int
        Number of points along vertical dimension.

    Returns
    -------
    grids: dict
        Keyed by the CoolProp ID for input pairs, a dictionary of `x` and `y`
        grids each of shape `(nj,ni)`.

    """

    # Loop over the required grid types
    grids = {}
    for grid_type, (x_name, y_name) in GRID_TYPES.items():
        # When CoolProp wants variables in a different order to TS4, swap the
        # numbers of points (because we also swap x and y later)
        if grid_type in FLIP_GRID:
            ny = ni
            nx = nj
        else:
            nx = ni
            ny = nj

        # Use logaritmic spacing if applicable
        if x_name in LOG_VARS:
            x_vec = np.geomspace(*limits[x_name], nx)
        else:
            x_vec = np.linspace(*limits[x_name], nx)
        if y_name in LOG_VARS:
            y_vec = np.geomspace(*limits[y_name], ny)
        else:
            y_vec = np.linspace(*limits[y_name], ny)

        # When CoolProp wants variables in a different order to TS4, swap x and y
        if grid_type in FLIP_GRID:
            x_vec, y_vec = y_vec, x_vec

        # Evaluate the grid
        grids[grid_type] = np.meshgrid(x_vec, y_vec)

    return grids


def _get_points(fluid_name, grid_type, xy):
    """Evaluate thermodynamic properties at a set of xy points for a given grid type.

    Parameters
    ----------
    fluid_name: str
        Name of the working fluid in CoolProp database.
    grid_type: int
        CoolProp ID number for the pair of grid variables.
    xy: (Npts, 2) array
        Grid variables to evaluate at.

    Returns
    -------
    table: (Npts, Ntable_grid) array
        Table properties at the requested grid points. TS4 requires, for each
        grid type, a certain number of dependent variables `Ntable_grid`.

    """
    abstract_state = CP.AbstractState("HEOS", fluid_name)

    # Indices for the tables relavent to current grid type
    itable_now = TABLES_BY_GRID[grid_type]
    ntable_now = len(itable_now)

    # Preallocate
    npts = xy.shape[0]
    table = np.zeros(
        (
            npts,
            ntable_now,
        )
    )

    # Loop over input points
    for k, xyk in enumerate(xy):
        # Set the state to current grid point, flipping x and y if
        # CoolProp's order differs to TS4
        if grid_type in FLIP_GRID:
            xyk[1], xyk[0] = xyk[0], xyk[1]

        # Catch ValueErrors if we cannot evaluate this grid point
        try:
            abstract_state.update(grid_type, *xyk)
            converged = True
        except ValueError:
            # Retry with a very small delta if this grid point does not converge
            # This seems to be needed to overcome very local instability in iterations
            eps = 1e-8
            converged = False
            dx, dy = np.meshgrid((-1, 0, 1), (-1, 0, 1))
            for dxi, dyi in zip(dx.flat, dy.flat):
                dxy = np.array([dxi, dyi]) * eps
                xyk_delta = xyk * (dxy + 1.0)
                try:
                    abstract_state.update(grid_type, *xyk_delta)
                    converged = True
                    break
                except ValueError:
                    continue

        # Now loop over the property tables required for current grid type
        for j, itable in enumerate(itable_now):
            # Get the dependent property for this table
            prop, _ = PROPERTY_TABLES[itable]

            if not converged:
                table[k, j] = np.nan
                continue

            # Catch ValueErrors if we cannot evaluate this property
            try:
                # Basic variables are just integer CoolProp IDs
                if isinstance(prop, int):
                    table[k, j] = abstract_state.keyed_output(prop)
                # Derivatives are a tuple of three CoolProp IDs: (of, wrt, const)
                else:
                    table[k, j] = abstract_state.first_partial_deriv(*prop)
            except ValueError:
                # Placeholder if we cannot evaluate
                table[k, j] = np.nan

    return table


def _get_tables(fluid_name, grids, Nworker=32):
    """Make tables over a set of grids.

    Parameters
    ----------

    fluid_name: str
        Name of the working fluid in CoolProp database.
    grids: dict
        Keyed by the CoolProp ID for input pairs, a dictionary of `x` and `y`
        grids each of shape `(nj,ni)`.
    Nworker: int
        Number of processes for parallel evaluation.

    Returns
    -------
    tables: (Ntable, nj, ni)
        Table properties for all `Ntable` dependent variables required by TS4.

    """

    # Preallocate main table
    nj, ni = list(grids.values())[0][0].shape
    tables = np.zeros((NTABLE, nj, ni))

    # Loop over grids
    for grid_type, grid in grids.items():
        # Get a (2,ni*nj) array of xy points on this grid
        x, y = grid
        xy = np.stack((x.reshape(-1), y.reshape(-1)), axis=-1)

        # Split into chunks for parallel execution
        xy_chunk = np.array_split(xy, Nworker, axis=0)

        # Send to workers
        with Pool(Nworker) as p:
            z = np.concatenate(
                p.starmap(
                    _get_points,
                    zip(repeat(fluid_name), repeat(grid_type), xy_chunk),
                ),
                axis=0,
            )

        # Loop over the tables for this grid type
        for j, itable in enumerate(TABLES_BY_GRID[grid_type]):
            zt = z[:, j].reshape(nj, ni)

            # Fix missing values
            util.replace_nan(x, y, zt, "nearest")

            # Insert into main table
            tables[itable, :, :] = zt

    return tables


def _write_npz(tables, grids, fname):
    """Write out the tables in TS4 format.

    Parameters
    ----------
    tables: (Ntable, nj, ni) array
        Property data for all tables.
    grids: dict
        Keyed by the CoolProp ID for input pairs, a dictionary of `x` and `y`
        grids each of shape `(nj,ni)`.
    fname: str
        Filename to save the tables at.

    """

    # Loop over tables
    tables_out = []
    for itable, (prop, prop_type) in enumerate(PROPERTY_TABLES):
        # Select the grid for this property
        x, y = grids[prop_type]
        z = tables[itable]

        # Concatenate all the tables data into a big list
        tables_out += [x[0, :], y[:, 0], z]

    # Write out the npz
    np.savez(fname, *tables_out)


def make_tables(fluid_name, smin, smax, Pmin, Tmax, ni, table_fname):
    """Given a fluid name and grid size, evaluate and write out tables for TS4.

    Parameters
    ----------
    fluid_name: str
        Name of the working fluid in CoolProp database.
    smin: float
        Minimum entropy.
    smax: float
        Maximum entropy.
    Pmin: float
        Minimum pressure.
    Tmax: float
        Maximum temperature.
    ni: int
        Number of points along one side of grid (same in other direction)
    fname: str
        Filename to save the tables at.

    """

    limits = _get_limits(fluid_name, smin, smax, Pmin, Tmax)
    grids = _get_grids(limits, ni, ni + 1)
    tables = _get_tables(fluid_name, grids)
    _write_npz(tables, grids, table_fname)


def read_ro_u_interpolators(fname):
    """Load the density-internal energy tables from an npz file."""

    dat = np.load(fname)

    itable_P = PROPERTY_TABLES.index((CP.iP, CP.DmassUmass_INPUTS))
    itable_T = PROPERTY_TABLES.index((CP.iT, CP.DmassUmass_INPUTS))

    ro = dat[f"arr_{itable_P*3}"]
    u = dat[f"arr_{itable_P*3+1}"]
    P = dat[f"arr_{itable_P*3+2}"]
    T = dat[f"arr_{itable_T*3+2}"]

    func_P = RegularGridInterpolator((ro, u), P.T, method="nearest")
    func_T = RegularGridInterpolator((ro, u), T.T, method="nearest")

    return func_P, func_T


class TabularState:
    "Encapsulate a working fluid modelled by CFD solver table"

    # Reference pressure and temperature for entropy datum
    Pref = 1e5
    Tref = 300.0
    IS_PERFECT = False

    # indices for the tables we need
    _ITABLE = {
        "T": PROPERTY_TABLES.index((CP.iT, CP.DmassUmass_INPUTS)),
        "P": PROPERTY_TABLES.index((CP.iP, CP.DmassUmass_INPUTS)),
        "a": PROPERTY_TABLES.index((CP.ispeed_sound, CP.DmassUmass_INPUTS)),
        "s": PROPERTY_TABLES.index((CP.iSmass, CP.DmassUmass_INPUTS)),
        "Po": PROPERTY_TABLES.index((CP.iP, CP.HmassSmass_INPUTS)),
        "To": PROPERTY_TABLES.index((CP.iT, CP.HmassSmass_INPUTS)),
        "u": PROPERTY_TABLES.index((CP.iUmass, CP.HmassP_INPUTS)),
        "ro": PROPERTY_TABLES.index((CP.iDmass, CP.PSmass_INPUTS)),
    }

    def __init__(self, tables, fluid_name=None):
        r"""Create an object to represent a tabular working fluid state.

        Parameters
        ----------
        tables: array
            All the tables in TS4 format
        fluid_name: str
            Optionally record the CoolProp name for this fluid.

        """

        self._tables = tables
        self._rho = None
        self._u = None
        self._fluid_name = fluid_name

        # Get interpolators
        self._interp = {}
        for k, itable in self._ITABLE.items():
            ix = 3 * itable
            iy = ix + 1
            iz = ix + 2
            self._interp[k] = RegularGridInterpolator(
                (self._tables[ix], self._tables[iy]),
                self._tables[iz].T,
                method="nearest",
                bounds_error=False,
                fill_value=None,
            )
            assert not np.isnan(self._tables[ix]).any()
            assert not np.isnan(self._tables[iy]).any()
            assert not np.isnan(self._tables[iz]).any()

    @property
    def fluid_name(self):
        return self._fluid_name

    @property
    def T(self):
        return self._interp["T"]((self.rho, self.u))

    @property
    def P(self):
        return self._interp["P"]((self.rho, self.u))

    @property
    def a(self):
        return self._interp["a"]((self.rho, self.u))

    @property
    def s(self):
        return self._interp["s"]((self.rho, self.u))

    @property
    def h(self):
        return self.u + self.P / self.rho

    @property
    def rho(self):
        return self._rho

    @property
    def u(self):
        return self._u

    def set_P_h(self, P, h):
        u = self._interp["u"]((h, P))
        rho = P / (h - u)
        return self.set_rho_u(rho, u)

    def to_stagnation(self, Ma):
        V = Ma * self.a
        ho = self.h + 0.5 * V**2.0
        To = self._interp["To"]((self.s, ho))
        Po = self._interp["Po"]((self.s, ho))
        return TabularStagnationState(ho, To, Po)

    def set_rho_u(self, rho, u):
        """Define state using density and internal energy."""
        self._rho = rho
        self._u = u
        return self

    def copy(self):
        """Make a copy of this state."""
        S = self.__class__(tables=self._tables)
        return S

    def to_real(self):
        """Convert to a CoolProp real gas state."""
        S = fluid.RealState(self.fluid_name)
        return S


class TabularStagnationState:
    def __init__(self, ho, To, Po):
        self.h = ho
        self.T = To
        self.P = Po
