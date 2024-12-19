import turbigen.util
import turbigen.grid
import os
from pathlib import Path
from dataclasses import dataclass
from turbigen.solvers.base import BaseSolver

logger = turbigen.util.make_logger()


@dataclass
class Config(BaseSolver):
    """Settings with default values for Plot3D export."""

    _name = "Plot3D"

    fname: str = "mesh.p3d"

    workdir: Path = None


def run(grid, conf, _):
    output_file_path = os.path.join(conf.workdir, conf.fname)
    logger.info(f"PLOT3D solver writing out the grid to {output_file_path}")
    turbigen.grid.write_plot3d(grid, output_file_path)
