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

    def update(self, config) -> bool:
        """Move the metal angle to match the exit flow angle."""
        converged = True

        log_data = {}
        for irow in range(config.nrow):
            # Get the actual and target exit flow angles
            yaw_actual = config.mean_line.actual.Alpha_rel[irow * 2 + 1]
            yaw_target = config.mean_line.nominal.Alpha_rel[irow * 2 + 1]
            # Calculate deviation
            dev = yaw_actual - yaw_target
            # Convergence check
            if (np.abs(dev) > self.tolerance).any():
                converged = False
            # Calculate the change to be applied to the current metal angle
            ddev = -np.clip(dev * self.relaxation_factor, -self.clip, self.clip)

            # Apply a uniform change at all sections
            config.blades[irow][0].q_camber[:, 1] += ddev

            # Record log data
            log_data[f"Dev[{irow}]"] = dev
            log_data[f"DDev[{irow}]"] = ddev

        return converged, log_data

    # def interpolate(self, config):
    #     """Correct for deviation using a fitted design space."""
    #
    #     # We need to interpolate q_camber[:, 1] at config
    #     dspace = config.design_space
    #
    #     # Function to extract the blade camber
    #     def extract_camber(config, irow, isect):
    #         return config.blades[irow][0].q_camber[isect, 1]
    #
    #     # Loop over rows
    #     for irow in range(config.nrow):
    #         blade = config.blades[irow][0]
    #         # Loop over sections
    #         for isect in range(blade.nsect):
    #             logger.iter(f"Interpolating blade {irow} section {isect}")
    #             logger.iter(f"Old camber: {blade.q_camber[isect, 1]}")
    #             blade.q_camber[isect, 1] = dspace.interpolate(
    #                 extract_camber,
    #                 [
    #                     config,
    #                 ],
    #                 irow=irow,
    #                 isect=isect,
    #             ).item()
    #             logger.iter(f"New camber: {blade.q_camber[isect, 1]}")


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

    tolerance: float = 10.0
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
            chi = util.incidence_unstructured(
                config.grid,
                config.get_machine(),
                config.mean_line.nominal,
                irow,
                row[0].spf,
            )

            # Calculate the incidence angle wrt target
            inc = np.atleast_1d(np.diff(chi[0], axis=0).squeeze())
            inc -= self.target

            # Check for convergence
            if np.abs(inc).max() > self.tolerance:
                converged = False

            # Metal angle change
            dinc = np.clip(inc * self.relaxation_factor, -self.clip, self.clip)

            # Apply to the config
            row[0].q_camber[:, 0] += dinc

            # Save log data
            imax = np.argmax(np.abs(inc))
            log_data[f"Inc[{irow}]"] = inc[imax]
            log_data[f"DInc[{irow}]"] = dinc[imax]

        return converged, log_data


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


@dataclasses.dataclass
class Repeat(IteratorConfig):
    """Settings for repeating stage."""

    relaxation_factor: float = 0.5
    """Factor controlling size of changes."""

    To_frac: float = 0.5
    """Fraction of varation in To to pass upstream."""

    rtol: float = 0.001
    """Relative tolerance for convergence of Po and To."""

    atol: float = 0.01
    """Absolute tolerance for convergence of angles."""

    dAlpha_max: float = 20.0
    """Clip the variations in yaw."""

    dBeta_max: float = 10.0
    """Clip the variations in pitch."""

    dTo_max: float = 0.1
    """Clip the variations in To."""

    dPo_max: float = 0.1
    """Clip the variations in Po."""

    def check(self, config):
        del config

    def update(self, config) -> bool:
        """Pass the outlet profiles upstream."""

        # Cut the outlet patch
        C = config.grid.outlet_patches[0].get_cut()

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
            inlet.profiles = profiles
            err = np.max(np.abs(profiles[1]))

        # Compare with the previous inlet
        else:
            # Interpolate the previous profiles to the new span fraction
            profiles_old = np.stack(
                [np.interp(spf, inlet.spf, inlet.profiles[i]) for i in range(4)],
            )

            # Calculate To errors
            err = np.max(np.abs(profiles[1] - profiles_old[1]))

            # Apply relaxation factor
            rf = self.relaxation_factor
            rf1 = 1.0 - rf
            inlet.profiles = rf * profiles + rf1 * profiles_old

        inlet.spf = spf

        return bool(err < self.rtol), {"Repeat_dTo": err}
