import logging
import os
from dataclasses import dataclass
from pathlib import Path

import turbigen_ref.grid
import turbigen_ref.util
from turbigen_ref.solvers.base import BaseSolver

logger = logging.getLogger("turbigen")


@dataclass
class Config(BaseSolver):
    """Settings with default values for Plot3D export."""

    _name = "Plot3D"

    fname: str = "mesh.p3d"

    workdir: Path = None


def run(grid, conf, _):
    output_file_path = os.path.join(conf.workdir, conf.fname)
    logger.info(f"PLOT3D solver writing out the grid to {output_file_path}")
    turbigen_ref.grid.write_plot3d(grid, output_file_path)
