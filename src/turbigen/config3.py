"""Config class using ember data structures."""

import dataclasses

import turbigen.fluid
import turbigen.inlet
import turbigen.dspace
import turbigen.job
import turbigen.iterators
import turbigen.yaml_utils
import turbigen.meanline_new
import turbigen.annulus
import turbigen.blade
import turbigen.nblade

import numpy as np

import sys

from typing import List

from pathlib import Path

import logging

logger = logging.getLogger("turbigen")


@dataclasses.dataclass
class TurbigenConfig:
    """Top level configuration class for turbigen.

    A run is uniquely defined by an instance of this class.

    """

    work_dir: Path
    """Directory in which to store run data."""

    fluid: turbigen.fluid.FluidConfig
    """Equation of state."""

    inlet: turbigen.inlet.InletConfig
    """Inflow boundary conditions."""

    mean_line: turbigen.meanline_new.MeanLineConfig
    """Settings for the mean-line designer."""

    plug_dir: Path = None
    """Directory in which to store run data."""

    iterate: List[turbigen.iterators.IteratorConfig] = dataclasses.field(
        default_factory=list
    )
    """Iterators to modify the configuration after running."""

    annulus: turbigen.annulus.AnnulusDesigner = None
    """Settings for the annulus designer."""

    blades: List[List[turbigen.blade.BladeDesigner]] = dataclasses.field(
        default_factory=list
    )
    """Settings for the blade designers."""

    nblade: List[turbigen.nblade.BladeNumberConfig] = dataclasses.field(
        default_factory=list
    )
    """Settings for blade number selection."""

    design_space: turbigen.dspace.DesignSpace = None
    """Design space sampling and mapping."""

    job: turbigen.job.BaseJob = None
    """Queue job submission."""

    def __post_init__(self):
        """Ensure correct types after init."""

        self.work_dir = Path(self.work_dir).absolute()
        self.plug_dir = Path(self.plug_dir).absolute() if self.plug_dir else None

        self.inlet = turbigen.inlet.InletConfig(**self.inlet)

        self.fluid = turbigen.fluid.FluidConfig.from_dict(self.fluid)
        self.mean_line = turbigen.meanline_new.MeanLineConfig.from_dict(self.mean_line)
        self.annulus = turbigen.annulus.AnnulusDesigner.from_dict(self.annulus)

        # Set up the blade designers
        blades = []
        for row in self.blades:
            # Check for no splitters
            if not isinstance(row, list):
                row = [
                    row,
                ]
            blades.append([])
            for blade in row:
                blades[-1].append(turbigen.blade.BladeDesigner(**blade))
        self.blades = blades

        self.nblade = [
            turbigen.nblade.BladeNumberConfig.from_dict(nb) for nb in self.nblade
        ]

    def to_dict(self):
        """Convert the config to a dictionary."""

        # Built-in dataclasses method gets us most of the way there
        data = dataclasses.asdict(self)

        # Now convert any nested objects with to_dict methods
        for k in ["mean_line", "fluid", "annulus"]:
            obj = getattr(self, k)
            if not obj:
                continue
            data[k] = obj.to_dict()

        # Convert the blade designer to a dictionary
        data["blades"] = []
        for row in self.blades:
            if len(row) == 1:
                data["blades"].append(row[0].to_dict())
            else:
                data["blades"].append([])
                for blade in row:
                    data["blades"][-1].append(blade.to_dict())

        return data

    def save(self, fname):
        """Write out to a file."""
        turbigen.yaml_utils.write_yaml(self.to_dict(), fname)

    def get_geometry(self):
        """Get the annulus and blade geometry."""

        # Annulus design
        logger.info("Designing annulus...")

        if not self.annulus:
            logger.error("No annulus defined, quitting.")
            sys.exit(0)

        self.annulus.setup_annulus(self.mean_line.nominal)
        logger.info(f"{self.annulus}")

        # Blade design
        logger.info("Designing blades...")

        if not self.blades:
            logger.error("No blades defined, quitting.")
            sys.exit(0)

        for irow, row in enumerate(self.blades):
            # Set meridional locations
            for blade in row:
                blade.set_streamsurface(self.annulus.xr_row(irow))
                print(f"set streamsurface {irow}")

    def get_nblade(self):
        Nb = np.full((len(self.blades),), 0, dtype=int)
        for irow, row in enumerate(self.blades):
            # Set number of blades using main blade
            Nb[irow] = np.round(
                self.nblade[irow].get_blade_number(
                    self.mean_line.nominal.get_row(irow), row[0]
                )
            )
        return Nb

    def apply_recamber(self):
        # Apply recamber to the blades
        for irow, row in enumerate(self.blades):
            for blade in row:
                blade.apply_recamber(self.mean_line.nominal.get_row(irow))

    def check_pitch_chord(self, s_cm_lim=(0.2, 4.0)):
        # Warn if blade spacings are too narrow or wide
        rref = 0.5 * (
            self.mean_line.nominal.r_rms[::2] + self.mean_line.nominal.r_rms[1::2]
        )
        s = 2.0 * np.pi * rref / self.get_nblade()
        s_cm = s / self.annulus.chords(0.5)[1:-1:2]
        if np.any(s_cm < s_cm_lim[0]):
            logger.warning(
                "WARNING: narrow blade spacings may cause problems with meshing"
            )
        if np.any(s_cm > s_cm_lim[1]):
            logger.warning(
                "WARNING: large blade spacings may cause problems with meshing"
            )

    def get_gaps(self):
        """Return non-dimensional tip gaps as fraction of span."""

        # Relative gaps from blade definition
        gap_span = np.full((self.mean_line.n_row,), 0.0)
        chord = self.annulus.chords(0.5)[1::2]
        span = self.mean_line.nominal.span
        span = 0.5 * (span[::2] + span[1::2])  # Average span for each row
        for irow, row in enumerate(self.blades):
            # Choose reference length
            if row[0].tip_ref == "span":
                gap_span[irow] = row[0].tip
            elif row[0].tip_ref == "chord":
                gap_span[irow] = row[0].tip * chord[irow] / span[irow]
            elif row[0].tip_ref == "absolute":
                gap_span[irow] = row[0].tip / span[irow]
            else:
                logger.error(
                    f"Unknown tip reference length {row[0].tip_ref}, quitting."
                )
                sys.exit(1)

        return gap_span
