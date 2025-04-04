"""The main routine that runs a config file."""

from timeit import default_timer as timer
from turbigen import util
import numpy as np

logger = util.make_logger()


def run(conf):
    """Run a configuration file.

    This will do the following:
        1. Get inlet state;
        2. Design the meanline;
        3. Design the annulus;
        4. Design the blades;
        5. Generate the mesh;
        6. Run the flow solver;
        7. Post-process the results.
        8. Mutate the conf object according to the specified iterators.

    """

    times = []
    times.append(timer())

    # Record the input configuration
    conf.save()

    # Inlet state
    logger.debug("Getting inlet state...")
    So1 = conf.inlet.get_inlet()
    logger.info(f"Inlet: {So1}")

    # Mean-line design
    times.append(timer())
    mean_line_nominal = conf.mean_line.get_mean_line(So1)
    times.append(timer())
    logger.debug(f"Mean-line design took {np.diff(times)[-1]:.1f}s")
    logger.info(mean_line_nominal)

    # Check mean-line design for problems
    logger.info("Checking mean-line conservation...")
    if not mean_line_nominal.check():
        mean_line_nominal.show_debug()
        raise Exception(
            "Mean-line conservation checks failed, have printed debugging information"
        ) from None
    logger.info("Checking mean-line inversion...")
    conf.mean_line.check_backward(mean_line_nominal)
    mean_line_nominal.warn()

    # Annulus design
    logger.info("Designing annulus...")
    times.append(timer())
    annulus = conf.annulus.get_annulus(mean_line_nominal)
    times.append(timer())

    # Blade design
    logger.info("Designing blades...")
    for irow, row in enumerate(conf.blades):
        for blade in row:
            blade.apply_recamber(mean_line_nominal)
            blade.set_streamsurface(annulus.xr_row(irow))
