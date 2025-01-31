"""Find all examples and confirm that they will run without errors."""

import os, sys, shutil
from turbigen import util, main, config, run
from tempfile import mkdtemp
import pytest
import logging


# Look for examples directory above this script
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
RUN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")

# Get all config yamls
example_confs = util.find(EXAMPLE_DIR, "yaml")

logger = util.make_logger()
logger.setLevel(level=logging.INFO)


@pytest.mark.parametrize("conf_yaml", example_confs)
def test_example(conf_yaml, usegpu):
    if os.path.basename(conf_yaml).startswith('include'):
        return
    print("*********************")
    print(f"Running example: {conf_yaml}")
    print("*********************")

    # Load an example config
    c = config.Config.read(conf_yaml)

    workdir = c.workdir
    if not usegpu:
        c.solver["skip"] = True
        c.wdist = False

    # Delete workdir if it exists
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)

    # Don't run a job in a job
    if c.job:
        return

    # # Run the config
    # if not c.solver.get("type"):
    #     with pytest.raises(SystemExit):
    #         run.run(c)
    #     success = True
    # else:
    #     success = run.run(c)

    assert run.run(c)

    # Delete if successful
    shutil.rmtree(workdir)


# If called as a script then test all examples
if __name__ == "__main__":
    if len(sys.argv) == 1:
        for conf in example_confs:
            test_example(conf)
    else:
        test_example(sys.argv[1])
