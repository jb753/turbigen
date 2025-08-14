"""Classes to update mean-line design using a CFD solution."""

from abc import ABC, abstractmethod
import dataclasses
import numpy as np
from turbigen import util, util_post

logger = util.make_logger()


@dataclasses.dataclass
class IteratorConfig(ABC):
    """Define the interface for an iterator."""

    @abstractmethod
    def update(self, config) -> bool:
        """Apply an in-place iteration update to a config."""
        raise NotImplementedError

    @abstractmethod
    def check(self, config):
        """Ensure that the iterator is correctly configured."""
        raise NotImplementedError

    def interpolate(self, config):
        """Use a fitted design space to set design variables."""
        logger.iter(f"interpolate() is not implemented for {self.__class__.__name__}. ")
        del config

    def get_independent(self, config):
        """Get the changeable independent variables."""
        return ()

    def get_dependent(self, config):
        """Get the targeted dependent variables."""
        raise NotImplementedError

    def get_nominal(self, config):
        """Get the nominal value of target variables."""
        raise NotImplementedError


@dataclasses.dataclass
class Deviation(IteratorConfig):
    relaxation_factor: float = 1.0
    """Multiplier on changes to metal angle."""
    tolerance: float = 1.0
    """Permissible error on exit flow angle [deg]."""
    clip: float = 2.0
    """Maximum change in metal angle per iteration [deg]."""

    def check(self, config):
        """Ensure that the iterator is correctly configured."""
        if self.tolerance <= 0:
            raise ValueError(
                "Could not initialise deviation iterator: tolerance must be positive."
            )

    def get_independent(self, config):
        """Extract mean trailing edge recamber from the configuration.

        We will make the same adjustments over the span, disregarding the
        spanwise distribution, so take the mean value over sections."""
        return np.array([b[0].camber[:, 1].mean() for b in config.blades])

    def get_dependent(self, config):
        """Extract mixed-out exit flow angles from the configuration."""
        return np.array(
            [
                config.mean_line.actual.Alpha_rel[irow * 2 + 1]
                for irow in range(config.nrow)
            ]
        )

    def get_nominal(self, config):
        return np.array(
            [
                config.mean_line.nominal.Alpha_rel[irow * 2 + 1]
                for irow in range(config.nrow)
            ]
        )

    def set_independent(self, config, value):
        """Set the mean trailing edge recamber in the configuration."""
        # Loop over rows
        logger.debug("Dev correction: setting TE recam")
        for recam_new, blade in zip(value, config.blades):
            logger.debug(f"  {recam_new=}")
            recam_old = blade[0].camber[:, 1]
            logger.debug(f"  {recam_old=}")
            # Uniform adjustment across span
            drecam = recam_new.mean() - recam_old.mean()
            blade[0].camber[:, 1] += drecam
            logger.debug(f"  recam_corrected={blade[0].camber[:, 1]}")
            assert np.isclose(blade[0].camber[:, 1].mean(), recam_new.mean())

    def update(self, config) -> bool:
        """Move the metal angle to match the exit flow angle."""

        # Extract angles
        yaw_actual = self.get_dependent(config)
        yaw_target = self.get_nominal(config)

        # Calculate deviation
        dev = yaw_actual - yaw_target

        # Calculate the change to be applied to the current metal angle
        ddev = -np.clip(dev * self.relaxation_factor, -self.clip, self.clip)

        # Apply the change to the metal angle
        recam_TE = self.get_independent(config) + ddev
        self.set_independent(config, recam_TE)

        # Convergence check
        converged = (np.abs(dev) <= self.tolerance).all()

        # Record log data
        log_data = {}
        for irow in range(config.nrow):
            log_data[f"Dev[{irow}]"] = dev[irow]
            log_data[f"DDev[{irow}]"] = ddev[irow]

        return converged, log_data

    def interpolate(self, config):
        """Correct for deviation using a fitted design space."""

        logger.iter("Interpolating deviation correction")

        # Function to extract the blade TE recamber
        def extract_camber(config, irow):
            return config.blades[irow][0].camber[:, 1]

        # Loop over rows
        for irow in range(config.nrow):
            blade = config.blades[irow][0]
            logger.iter(f"Blade {irow}")
            logger.iter(f"  Old TE recamber: {blade.camber[:, 1]}")
            blade.camber[:, 1] = config.design_space.interpolate(
                extract_camber,
                config,
                irow=irow,
            )
            logger.iter(f"  New TE recamber: {blade.camber[:, 1]}")


@dataclasses.dataclass
class DiffusionFactor(IteratorConfig):
    target: dict = dataclasses.field(default_factory=lambda: ({}))
    """Mapping of row index to target diffusion factor."""

    K: float = 0.25
    """Factor to scale DF error to relative Nb change."""

    spf: float = 0.5
    """Span fraction at which to calculate diffusion factor."""

    tolerance: float = 0.01
    """Relative tolerance on blade number changes."""

    clip: float = 0.05
    """Largest relative change in blade number per iteration."""

    def check(self, config):
        """Ensure that the iterator is correctly configured."""
        for irow, DF_target in self.target.items():
            if DF_target <= 0:
                raise ValueError(
                    "Could not initialise diffusion factor iterator: target DF must be positive."
                )
            if irow > config.nrow:
                raise ValueError(
                    f"Could not initialise diffusion factor iterator: row index {irow} out of range."
                )

    def update(self, config):
        # Initialise convergence flag
        converged = True
        log_data = {}

        # Loop over the rows we want to match
        for irow, DF_target in self.target.items():
            # Calculate the diffusion factor from CFD
            _, Mas = util_post.get_isen_mach(
                config.grid,
                config.get_machine(),
                config.mean_line.actual,
                irow,
                self.spf,
            )
            Mas_TE = 0.5 * (Mas[0] + Mas[-1]).item()
            Masf = Mas / Mas_TE
            DF_actual = Masf.max() - 1.0

            # Calculate Nblade adjustment
            dNb_rel = np.clip(
                -self.K * (1.0 - DF_actual / DF_target), -self.clip, self.clip
            )
            log_data[f"DF[{irow}]"] = DF_actual
            log_data[f"DNb_rel[{irow}]"] = dNb_rel

            # Adjust nblade
            config.nblade[irow].adjust(dNb_rel)

            # Check for convergence
            if np.abs(dNb_rel) > self.tolerance:
                converged = False

        return converged, log_data


@dataclasses.dataclass
class Incidence(IteratorConfig):
    relaxation_factor: float = 0.05
    """Multiplier on changes to metal angle."""

    tolerance: float = 20.0
    """Permissible error on local incidence angle [deg]."""

    target: float = 0.0
    """Target value of local incidence angle [deg]."""

    clip: float = 2.0
    """Maximum change in metal angle in one step [deg]."""

    def check(self, config):
        """Ensure that the iterator is correctly configured."""
        pass

    def update(self, config):
        """Move the metal angle to match the target local incidence angle."""

        log_data = {}
        converged = True

        for irow, row in enumerate(config.blades):
            # Skip unbladed rows
            if not row:
                continue

            # Find flow and metal angles
            config.apply_recamber()
            chi = util.incidence_unstructured(
                config.grid,
                config.get_machine(),
                config.mean_line.nominal,
                irow,
                row[0].spf,
            )
            config.undo_recamber()

            # Calculate the incidence angle wrt target
            inc = np.atleast_1d(np.diff(chi[0], axis=0).squeeze())
            inc -= self.target

            # Check for convergence
            if np.abs(inc).max() > self.tolerance:
                converged = False

            # Metal angle change
            dinc = np.clip(inc * self.relaxation_factor, -self.clip, self.clip)

            # Apply to the config
            row[0].camber[:, 0] += dinc

            # Save log data
            imax = np.argmax(np.abs(inc))
            log_data[f"Inc[{irow}]"] = inc[imax]
            log_data[f"DInc[{irow}]"] = dinc[imax]

        return converged, log_data

    def interpolate(self, config):
        """Use a fitted design space to set LE recamber."""

        logger.iter("Interpolating incidence correction")

        # Function to extract the LE recamber
        def extract_camber(config, irow):
            return config.blades[irow][0].camber[:, 0]

        # Loop over rows
        for irow in range(config.nrow):
            blade = config.blades[irow][0]

            logger.iter(f"Blade {irow}")
            logger.iter(f"  Old LE recamber: {blade.camber[:, 0]}")
            blade.camber[:, 0] = config.design_space.interpolate(
                extract_camber,
                config,
                irow=irow,
            )
            logger.iter(f"  New LE recamber: {blade.camber[:, 0]}")


@dataclasses.dataclass
class MeanLine(IteratorConfig):
    """Settings for mean-line iteration."""

    relaxation_factor: float = 0.5
    """Factor controlling size of changes."""

    tolerance: dict = dataclasses.field(default_factory=lambda: ({}))
    """Mapping of design variable name to tolerance for convergence."""

    def check(self, config):
        """Ensure that the iterator is correctly configured."""
        for vname, vtol in self.tolerance.items():
            if vname not in config.mean_line.design_vars:
                raise ValueError(
                    f"Could not initialise mean_line iterator: design variable {vname} "
                    f"not found in design variables, should be one of {list(config.mean_line.design_vars.keys())}."
                )
            if vtol <= 0:
                raise ValueError(
                    f"Could not initialise mean_line iterator: tolerance for {vname} must be positive."
                )

    def update(self, config) -> bool:
        """Relax nominal mean-line guesses towards CFD."""

        # Initialise convergence flag
        converged = True
        log_data = {}

        # Loop over the design variables we want to match
        for vname, vtol in self.tolerance.items():
            # Get the nominal and CFD values for this design variable
            var_nom = np.atleast_1d(config.mean_line.design_vars[vname])
            var_cfd = np.atleast_1d(config.mean_line_actual[vname])

            # Calculate the error and new value
            var_new = (
                var_cfd * self.relaxation_factor
                + (1.0 - self.relaxation_factor) * var_nom
            )

            # Calculate the change to be applied to the nominal values in config
            dvar = var_new - var_nom

            # We have not converged if the err tolerance is exceeded
            err = np.abs(var_nom - var_cfd).max()
            if err > vtol:
                converged = False

            # Reduce (1,) vectors to scalars
            if var_nom.size == 1:
                var_nom = var_nom.item()
                var_cfd = var_cfd.item()
                var_new = var_new.item()
                dvar = dvar.item()

            # Assign back to the configuration
            config.mean_line.design_vars[vname] = var_new

            # Log the data
            try:
                nv = len(var_nom)
                for iv in range(nv):
                    kiv = f"{vname}[{iv}]"
                    log_data[kiv] = var_cfd[iv]
                    log_data["D" + kiv] = dvar[iv]  # Change
            except TypeError:
                log_data[vname] = var_cfd  # Value
                log_data["D" + vname] = dvar  # Change

        return converged, log_data

    def interpolate(self, config):
        """Use a fitted design space to set loss guess."""

        # Function to extract mean-line quantity of interest
        # Although the converged samples have actual mean line,
        # the query config does not have actual mean line yet.
        # So fall back to the nominal value if actual is unavailable
        def extract_mean_line(config, vname):
            v_actual = config.mean_line_actual.get(vname)
            v_nom = config.mean_line.design_vars[vname]
            return v_actual if v_actual is not None else v_nom

        # Loop over the design variables we want to match
        for vname in self.tolerance:
            logger.iter(f"Interpolating mean-line design variable {vname}")

            # Get nominal value
            var_nom = config.mean_line.design_vars[vname]
            logger.iter(f"  Nominal value: {var_nom}")

            # Interpolate this variable from the design space
            var_interp = config.design_space.interpolate(
                extract_mean_line, config, vname=vname
            )
            logger.iter(f"  New value: {var_interp}")

            # Assign back to the configuration
            # Ensure that the shape matches
            assert np.shape(var_interp) == np.shape(var_nom)
            config.mean_line.design_vars[vname] = var_interp


@dataclasses.dataclass
class Repeat(IteratorConfig):
    """Settings for repeating stage."""

    relaxation_factor: float = 0.5
    """Factor controlling size of changes."""

    To_frac: float = 0.5
    """Fraction of varation in To to pass upstream."""

    rtol: float = 0.005
    """Relative tolerance for convergence of Po and To."""

    atol: float = 0.01
    """Absolute tolerance for convergence of angles."""

    dAlpha_max: float = 16.0
    """Clip the variations in yaw."""

    dBeta_max: float = 8.0
    """Clip the variations in pitch."""

    dTo_max: float = 0.1
    """Clip the variations in To."""

    dPo_max: float = 0.1
    """Clip the variations in Po."""

    Dchord: float = 0.5
    """Number of chord lengths downstream of last TE to extract profile."""

    def check(self, config):
        del config

    def update(self, config) -> bool:
        """Pass the outlet profiles upstream."""

        # Get a cut plane after last TE
        xr_cut = config.annulus.get_offset_planes(self.Dchord)[-1]
        C = config.grid.unstructured_cut_marching(xr_cut).interpolate_to_structured()

        # Mix out to uniformity to get reference state
        Cm = C.mix_out()[0]

        # Pitchwise mass-average the boundary condition quantities
        # Mass flow rate per unit meridional area
        mdot = C.pitchwise_integrate(C.rhoVm)
        Po = C.pitchwise_integrate(C.rhoVm * C.Po) / mdot
        To = C.pitchwise_integrate(C.rhoVm * C.To) / mdot
        Alpha = C.pitchwise_integrate(C.rhoVm * C.Alpha) / mdot
        Beta = C.pitchwise_integrate(C.rhoVm * C.Beta) / mdot

        # Assemble into a matrix
        spf = C.spf.mean(axis=2).squeeze()
        profiles = np.stack([Po, To, Alpha, Beta], axis=1)[0]
        nj = len(spf)

        # Subtract the meanline values
        avg = np.array([Cm.Po, Cm.To, Cm.Alpha, Cm.Beta])
        profiles -= avg[:, None]

        # Normalise Po and To
        profiles[0] /= Cm.Po
        profiles[1] /= Cm.To

        # Control temperature build up by only passing
        # a fraction of the To variation upstream
        profiles[1] *= self.To_frac

        # Apply clipping to the normalised profiles
        clip = [self.dPo_max, self.dTo_max, self.dAlpha_max, self.dBeta_max]
        for i in range(4):
            profiles[i] = np.clip(profiles[i], -clip[i], clip[i])

        # Apply to the config object
        inlet = config.inlet

        # No previous inlet, initialise
        if inlet.spf is None:
            inlet.spf = spf
            inlet.profiles = np.zeros((4, nj))

        # Compare with the previous inlet

        # Interpolate the previous profiles to the new span fraction
        profiles_old = np.stack(
            [np.interp(spf, inlet.spf, inlet.profiles[i]) for i in range(4)],
        )

        # Calculate To errors
        err = np.max(np.abs(profiles[1] - profiles_old[1]))

        # Apply relaxation factor to get new profiles
        rf = self.relaxation_factor
        rf1 = 1.0 - rf
        inlet.profiles = rf * profiles + rf1 * profiles_old
        inlet.spf = spf

        return bool(err < self.rtol), {"Repeat_dTo": err}

    def interpolate(self, config):
        """Use a fitted design space to set repeating profiles."""
        # logger.iter("NOT interpolating repeating profiles")
        # return
        logger.iter("Interpolating repeating profiles")

        # Define a new span fraction vector
        # Clustered towards the endwalls
        spf_new = config.inlet.spf = util.cluster_cosine(50)

        # Function to extract profiles from the design space at target spf
        def extract_profile(config, ivar):
            return np.interp(spf_new, config.inlet.spf, config.inlet.profiles[ivar])

        # Loop over profile variables and interpolate each from the design space
        # Apply clipping and relaxation to the normalised profiles
        clip = [self.dPo_max, self.dTo_max, self.dAlpha_max, self.dBeta_max]
        config.inlet.profiles = np.full((4, len(spf_new)), np.nan)
        for ivar in range(4):
            var = config.design_space.interpolate(extract_profile, config, ivar=ivar)
            var_clip = np.clip(var, -clip[ivar], clip[ivar])
            config.inlet.profiles[ivar] = var_clip * self.relaxation_factor
