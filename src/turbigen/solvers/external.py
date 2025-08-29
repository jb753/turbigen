import subprocess
import pickle
import os
import tempfile
from dataclasses import dataclass
from turbigen.solvers.base import BaseSolver
import turbigen.util
from pathlib import Path

logger = turbigen.util.make_logger()


@dataclass
class External(BaseSolver):
    """
    External CFD Solver Interface

    This solver communicates with an external CFD solver process via pipes,
    sending the grid structure and receiving the converged solution.
    """

    # Override base attributes
    _name = "external"

    script_path: str = "test_receiver.py"
    """Path to the external solver script"""

    def run(self, grid, machine=None, workdir=None):
        """Run the external solver by sending grid data through named pipes"""

        logger.info(f"Starting external solver: {self.script_path}")

        # Convert script_path to absolute and check it exists
        script_path = Path(self.script_path).resolve()
        if not script_path.is_file():
            raise FileNotFoundError(f"External solver script not found: {script_path}")

        # Create temporary named pipes
        with tempfile.TemporaryDirectory() as tmpdir:
            input_pipe = os.path.join(tmpdir, "input_pipe")
            output_pipe = os.path.join(tmpdir, "output_pipe")

            # Create the named pipes
            os.mkfifo(input_pipe)
            os.mkfifo(output_pipe)

            logger.info(f"Created named pipes: {input_pipe}, {output_pipe}")

            # Start the external process with pipe names as arguments
            process = subprocess.Popen(
                ["python", self.script_path, input_pipe, output_pipe],
            )

            # Send grid object via input pipe
            logger.info("Sending grid object to external solver...")
            with open(input_pipe, "wb") as f:
                pickle.dump(grid, f)

            # Receive the processed grid back via output pipe
            logger.info("Receiving processed grid from external solver...")
            with open(output_pipe, "rb") as f:
                processed_grid = pickle.load(f)

            # Wait for process to complete
            logger.info("Waiting for external solver to finish...")
            stderr = process.communicate()[1]

            if process.returncode != 0:
                logger.error(
                    f"External solver failed with return code: {process.returncode}"
                )
                logger.error(f"stderr: {stderr.decode()}")
                raise RuntimeError("External solver execution failed")

            # Update the original grid with results
            for i, block in enumerate(grid):
                if i < len(processed_grid):
                    processed_block = processed_grid[i]
                    # Copy conserved variables and other flow data
                    if hasattr(processed_block, "conserved"):
                        block.set_conserved(processed_block.conserved)

        logger.info("External solver completed successfully")

    def restart(self):
        return self

    def robust(self):
        return self
