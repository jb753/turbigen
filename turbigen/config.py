"""Object to encapsulate a configuration file."""

from turbigen import util, fluid
from turbigen.exceptions import ConfigError
import turbigen.yaml
from inspect import signature
from scipy.interpolate import griddata
from scipy.spatial import QhullError
from scipy.stats.qmc import LatinHypercube
import numpy as np
import os

logger = util.make_logger()


class Config:
    REAL_INLET_KEYS = set(["Po", "To", "fluid_name"])
    PERFECT_INLET_KEYS = set(["Po", "To", "cp", "gamma", "mu"])
    PITCH_KEYS = set(["Nb", "Co", "DFL", "Cb"])

    def __init__(self, d):
        # Blank configs if not there
        self.mesh = d.get("mesh", {})
        self.solver = d.get("solver", {})
        self.iterate = d.get("iterate", {})
        self.job = d.get("job", {})
        self.database = d.get("database", {})
        self.operating_point = d.get("operating_point", {})
        self.hypercube = d.get("hypercube", {})
        self.post_process = d.get("post_process", {})
        self.install = d.get("install", {})

        if workdir := d["workdir"]:
            self.workdir = os.path.abspath(workdir)
        else:
            self.workdir = None

        self.plot = d.get("plot", False)
        self.wdist = d.get("wdist", True)

        self.inlet = d.get("inlet", {})
        self.mean_line = d.get("mean_line", {})
        self.annulus = d.get("annulus", {})

        # Determine number of rows from length of the list of blade definitions
        bldconf = d.get("blades", [])
        self.nrow = len(bldconf)

        # Collect keys from all of the sections
        # Because e.g. tip might be omitted from stator but included in rotor
        # Exclude section definitions because they will be separate
        all_keys = []
        has_splitter = False
        self.splitter = []
        for b in bldconf:
            if b:
                for k in b.keys():
                    if k == "splitter":
                        has_splitter = True
                    if (k not in ("sections", "splitter")) and (k not in all_keys):
                        all_keys += [
                            k,
                        ]

        # Loop over rows
        self.blades = {k: [] for k in all_keys}
        self.sections = []
        for irow in range(self.nrow):
            if bldrow := bldconf[irow]:
                # Fill in data for bladed rows
                for k in all_keys:
                    self.blades[k].append(bldrow.get(k, None))
                self.sections.append(bldrow.get("sections", None))
                if has_splitter:
                    self.splitter.append(bldrow.get("splitter", None))
            else:
                # Blank data for unbladed rows
                self.sections.append([])
                if has_splitter:
                    self.splitter.append([])
                for k in all_keys:
                    self.blades[k].append(None)

        self.sections = [util.list_of_dict_to_dict_of_list(si) for si in self.sections]
        if has_splitter:
            # print(self.splitter)
            splitter_new = []
            for si in self.splitter:
                if si:
                    splitter_new.append(util.list_of_dict_to_dict_of_list(si))
                else:
                    splitter_new.append(None)
            self.splitter = splitter_new
            # print(self.splitter)
            # quit()

        # Make workdir absolute if specified
        if self.solver.get("workdir"):
            self.solver["workdir"] = os.path.abspath(self.solver["workdir"])

        self.mean_line_type = self.mean_line.pop("type", None)

        if self.mean_line_type.endswith(".py"):
            self.mean_line_type = os.path.abspath(self.mean_line_type)

        if self.install:
            if self.install["type"].endswith(".py"):
                self.install["type"] = os.path.abspath(self.install["type"])

        self.check()

    def check(self):
        self._check_blades()
        self._check_viscosity()
        self._check_inlet()
        self._check_mean_line()
        self._check_iterate()
        # self._check_annulus()

    def _check_blades(self):
        """Ensure blade data OK."""
        nrow = len(self.sections)
        for i in range(nrow):
            if self.sections[i]:
                pitch_keys = [
                    k
                    for k in self.PITCH_KEYS
                    if (self.blades.get(k) and self.blades[k][i])
                ]
                if len(pitch_keys) > 1:
                    raise ConfigError(
                        f"Cannot set pitch twice in row {i} with {pitch_keys}"
                    )
                elif len(pitch_keys) == 0:
                    raise ConfigError(
                        f"No method for setting row {i} blade pitch specified;"
                        f" expecting one of {self.PITCH_KEYS}"
                    )

    def _check_viscosity(self):
        """Make sure the the viscosity specifcation is consistent."""
        if self.inlet.get("fluid_name"):
            if self.blades.get("Re_surf"):
                raise ConfigError("Cannot set viscosity from Reynolds with real inlet")
        else:
            if (
                self.blades.get("Re_surf") and any(self.blades["Re_surf"])
            ) and self.inlet.get("mu"):
                raise ConfigError(
                    "Cannot set viscosity both at inlet and from Reynolds"
                )
            elif (
                not (self.blades.get("Re_surf") and any(self.blades["Re_surf"]))
            ) and (not self.inlet.get("mu")):
                raise ConfigError(
                    "Viscosity not set, either fix inlet mu or a blade Reynolds"
                )

    def _check_inlet(self):
        """Make sure that the inlet data are complete."""
        for k in ["Po", "To"]:
            if not self.inlet[k]:
                raise ConfigError(f'Missing inlet state property "{k}"')
        inkeys = set(self.inlet.keys())
        if fluid_name := self.inlet.get("fluid_name"):
            if not inkeys == self.REAL_INLET_KEYS:
                raise ConfigError(
                    f"Invalid real inlet keys {inkeys}; expecting"
                    f" {self.REAL_INLET_KEYS}"
                ) from None
            try:
                fluid.RealState.from_fluid_name(fluid_name)
            except ValueError:
                raise ConfigError(f'Invalid real fluid name "{fluid_name}"') from None
        elif (gamma := self.inlet["gamma"]) and (cp := self.inlet["cp"]):
            if diffkeys := inkeys.difference(self.PERFECT_INLET_KEYS):
                raise ConfigError(
                    f"Invalid perfect inlet keys {diffkeys}; expecting from"
                    f" {self.PERFECT_INLET_KEYS}"
                ) from None
            if (not (np.isfinite(cp) and cp > 0)) or (
                not (np.isfinite(gamma) and gamma > 0)
            ):
                raise ConfigError(f"Invalid perfect inlet with cp={cp}, gamma={gamma}")
        else:
            raise ConfigError(
                f"Insufficient data for real or perfect inlet: {self.inlet}"
            )

    def _check_mean_line(self):
        """Make sure that the mean-line data is valid."""
        if meanline_type := self.mean_line_type:
            try:
                design = util.load_mean_line(self.mean_line_type)

                sig = signature(design.forward)
                func_params = list(sig.parameters.values())[1:]
                for p in func_params:
                    if (p.default is p.empty) and (self.mean_line.get(p.name) is None):
                        raise ConfigError(
                            f'No value specified for required "{meanline_type}" design'
                            f' parameter "{p.name}", '
                            f"you supplied {list(self.mean_line.keys())}"
                        )
                func_param_names = [p.name for p in func_params]
                for k in self.mean_line:
                    if k == "debug":
                        continue
                    if k not in func_param_names:
                        raise ConfigError(
                            f'Meanline parameter "{k}" is not valid, expecting one of'
                            f" {func_param_names}"
                        )
                func_param_names = [p.name for p in func_params]
            except AttributeError:
                raise ConfigError(f'Invalid mean-line type "{meanline_type}"') from None
        else:
            raise ConfigError("Missing a mean-line type")

    def _check_annulus(self):
        """Validate annulus data"""
        logger.debug(f"Loading annulus {self.annulus.get('type')}")
        Annulus = util.load_annulus(self.annulus.get("type", "Smooth"))
        logger.debug("Getting signature...")
        sig = signature(Annulus)
        for k in self.annulus:
            if k not in sig.parameters and k not in ["type", "debug"]:
                raise ConfigError(f'Invalid annulus design parameter "{k}"')
        for p in list(sig.parameters.values())[3:]:
            if (p.default is p.empty) and (self.annulus.get(p.name) is None):
                raise ConfigError(
                    "No value specified for required annulus design parameter"
                    f' "{p.name}"'
                )

    def _check_iterate(self):
        if mlconf := self.iterate.get("mean_line"):
            for v in mlconf.get("match_tolerance"):
                if v not in self.mean_line:
                    raise Exception(
                        f"Unknown mean-line match variable '{v}', "
                        f"should be one of {list(self.mean_line.keys())}"
                    )

    @classmethod
    def read(cls, yaml_file):
        """Initialise from a yaml configuration file."""
        din = turbigen.yaml.read_yaml(yaml_file)
        return cls(din)

    def to_dict(self):
        """Assemble a nested dictionary for this configuration."""

        d = {
            "inlet": {k: util.to_basic_type(v) for k, v in self.inlet.items()},
            "mean_line": {k: util.to_basic_type(v) for k, v in self.mean_line.items()},
            "annulus": {k: util.to_basic_type(v) for k, v in self.annulus.items()},
        }

        d["mean_line"]["type"] = self.mean_line_type

        d["workdir"] = self.workdir

        if self.iterate:
            d["iterate"] = self.iterate

        if self.plot:
            d["plot"] = self.plot

        if self.mesh:
            d["mesh"] = self.mesh

        if self.job:
            d["job"] = self.job

        if self.solver:
            d["solver"] = self.solver.copy()
            for k in d["solver"]:
                if isinstance(d["solver"][k], dict):
                    d["solver"][k] = {
                        k2: util.to_basic_type(v2) for k2, v2 in d["solver"][k].items()
                    }
                else:
                    d["solver"][k] = util.to_basic_type(d["solver"][k])

        if self.operating_point:
            d["operating_point"] = self.operating_point

        if self.database:
            d["database"] = self.database

        if self.post_process:
            d["post_process"] = self.post_process

        if self.install:
            d["install"] = self.install

        nrow = self.nrow
        d["blades"] = []
        for irow in range(nrow):
            d["blades"].append({})
            for k in self.blades:
                if not self.blades[k][irow] is None:
                    d["blades"][-1][k] = util.to_basic_type(self.blades[k][irow])

            sect_now = self.sections[irow]
            if self.splitter:
                splitter_now = self.splitter[irow]
            if sect_now:
                nsect = len(sect_now["spf"])
                d["blades"][irow]["sections"] = []
                for isect in range(nsect):
                    d["blades"][irow]["sections"].append({})
                    for k in sect_now:
                        d["blades"][irow]["sections"][-1][k] = util.to_basic_type(
                            sect_now[k][isect]
                        )
            if self.splitter:
                if splitter_now:
                    nsect = len(splitter_now["spf"])
                    d["blades"][irow]["splitter"] = []
                    for isect in range(nsect):
                        d["blades"][irow]["splitter"].append({})
                        for k in splitter_now:
                            d["blades"][irow]["splitter"][-1][k] = util.to_basic_type(
                                splitter_now[k][isect]
                            )

        if not self.wdist:
            d["wdist"] = False

        return d

    def write(self, yaml_file, mode="w", runid=None):
        d = self.to_dict()
        if runid is not None:
            d["runid"] = runid
        turbigen.yaml.write_yaml(d, yaml_file, mode)

    def get_inlet(self):
        """Return a State object for the inlet working fluid."""

        if fluid_name := self.inlet.get("fluid_name"):
            So1 = fluid.RealState.from_fluid_name(fluid_name)
        else:
            So1 = fluid.PerfectState.from_properties(
                self.inlet["cp"],
                self.inlet["gamma"],
                self.inlet.get("mu", np.nan),
            )
        So1.set_P_T(self.inlet["Po"], self.inlet["To"])
        return So1

    def copy(self):
        """Return a copy of this configuration."""
        return Config(self.to_dict())

    def interpolate_from_database(self, verbose=False):
        """Fill in dependent variables by interpolation of independents in database."""

        database_file = os.path.abspath(self.database["conf_path"])

        if verbose:
            logger.iter(f"Interpolating from: {database_file}:")

        if not (independent := self.database.get("independent")):
            return None

        dependent = self.database.get("dependent")
        if dependent is None:
            dependent = []

        # Do nothing if database not created yet
        if not os.path.exists(database_file):
            logger.iter("Database not found, skipping interpolation")
            return

        database = read_database(database_file)

        # Deviation
        nsect = [0 if not s else len(s["spf"]) for s in self.sections]
        nrow = len(nsect)
        nq = len(self.sections[np.argmax(nsect)]["qstar_camber"][0])

        # Assemble interpolation data
        npts = len(database)
        nx = len(independent)
        ny = len(dependent)
        x = np.zeros((npts, nx))
        y = np.zeros((npts, ny))
        qstar_camber = [np.zeros((npts, nsecti, nq)) for nsecti in nsect]
        for i, c in enumerate(database):
            for j, v in enumerate(independent):
                if v in (
                    "Co",
                    "Cb",
                    "tip",
                ):
                    # We interpolate based on first row Co
                    # Should generalise to any row index with a config variable
                    x[i, j] = c.blades[v][0]
                else:
                    x[i, j] = c.mean_line[v]
            for j, v in enumerate(dependent):
                y[i, j] = c.mean_line[v]

            for irow in range(nrow):
                if nsect[irow]:
                    qstar_camber[irow][i, :, :] = c.sections[irow]["qstar_camber"]

        # Trim to 100 points to limit slowness
        npts_max = 100
        npts_lim = np.minimum(npts, npts_max)

        # Assemble query points
        xq = np.zeros((1, nx))
        if verbose:
            logger.iter("Independent variables:")
        for j, v in enumerate(independent):
            if v in ("Co", "Cb", "tip"):
                xq[0, j] = self.blades[v][0]
            else:
                xq[0, j] = self.mean_line[v]

            if verbose:
                logger.iter(f"  {v} = {xq[0, j]:.3f}")

        # Normalise independent variables by mean value
        xn = np.mean(x, axis=0, keepdims=True)
        x /= xn
        xq /= xn

        # Perform interpolation
        if verbose:
            logger.iter("Dependent variables:")
        warned = False

        for j, v in enumerate(dependent):
            try:
                yq_now = griddata(
                    x[:npts_lim, :], y[:npts_lim, j], xq, method="linear", rescale=True
                )
                assert not np.isnan(yq_now)

            except (AssertionError, RuntimeError):  # (AssertionError, QhullError):
                if not warned:
                    logger.iter("Falling back to nearest-neighbour")
                    warned = True
                yq_now = griddata(
                    x[:npts_lim, :], y[:npts_lim, j], xq, method="nearest", rescale=True
                )

            self.mean_line[v] = float(yq_now)
            if verbose:
                logger.iter(f"  {v} = {float(yq_now):.3f}")

        for irow in range(nrow):
            for isect in range(nsect[irow]):
                for iq in range(nq):
                    try:
                        qiq = griddata(
                            x[:npts_lim, :],
                            qstar_camber[irow][:npts_lim, isect, iq],
                            xq,
                            method="linear",
                            rescale=True,
                        )
                        assert not np.isnan(qiq).any()
                    except (AssertionError, QhullError):
                        qiq = griddata(
                            x[:npts_lim, :],
                            qstar_camber[irow][:npts_lim, isect, iq],
                            xq,
                            method="nearest",
                            rescale=True,
                        )

                    self.sections[irow]["qstar_camber"][isect][iq] = float(qiq)
                    if iq == 0:
                        logger.iter(
                            f"  inc: irow={irow}, isect={isect} = {float(qiq):.2f}"
                        )
                    if iq == 1:
                        logger.iter(
                            f"  dev: irow={irow}, isect={isect} = {float(qiq):.2f}"
                        )

    def sample_hypercube(self):
        """Return copies of this config sampled over hypercube specified therein."""

        # Define limits
        xmin, xmax = np.column_stack([lim for lim in self.hypercube["limits"].values()])
        Nv = len(self.hypercube["limits"])

        # Draw the samples
        N = self.hypercube["N"]

        if self.hypercube.get("regular"):
            Nside = int(N ** (1.0 / Nv))
            q = (
                np.stack(
                    np.meshgrid(
                        *[np.linspace(0.0, 1.0, Nside) for _ in range(Nv)],
                        indexing="ij",
                    )
                )
                .reshape(Nv, -1)
                .T
            )
        else:
            q = LatinHypercube(d=xmax.shape[0], optimization="random-cd").random(N)

        v = xmin * (1.0 - q) + xmax * q

        # Loop over samples
        call = []
        for vi in v:
            c = self.copy()
            for j, varname in enumerate(self.hypercube["limits"]):
                if varname in ("Co", "Cb", "tip"):
                    c.blades[varname][0] = float(vi[j])
                elif varname in c.install:
                    c.install[varname] = float(vi[j])
                elif varname in c.annulus:
                    c.annulus[varname] = float(vi[j])
                else:
                    c.mean_line[varname] = float(vi[j])

            call.append(c)

        return call

    def sample_hyperfaces(self):
        # Define limits
        xmin, xmax = np.column_stack([lim for lim in self.hypercube["limits"].values()])
        Nv = len(self.hypercube["limits"])

        call = []

        Nedge = self.hypercube["Nedge"]
        fac_edge = self.hypercube.get("fac_edge", (1.0,))
        for f in fac_edge:
            qq = np.stack(
                np.meshgrid(
                    *[np.linspace(1.0 - f, f, Nedge) for _ in range(Nv)], indexing="ij"
                )
            )
            qf = util.hyperfaces(qq).T
            v = xmin * (1.0 - qf) + xmax * qf

            for vi in v:
                c = self.copy()
                for j, varname in enumerate(self.hypercube["limits"]):
                    if varname in ("Co", "Cb", "tip"):
                        c.blades[varname][0] = float(vi[j])
                    else:
                        c.mean_line[varname] = float(vi[j])
                call.append(c)

        return call


def read_database(database_file):
    """Load a list of configs from a database file."""
    confs = []
    runids = []
    for d in turbigen.yaml.read_yaml_list(database_file):
        runids.append(d.pop("runid", 0))
        confs.append(Config(d))
    # Make sure order is consistent
    confs_sorted = [confs[i] for i in np.argsort(runids)]
    return confs_sorted
