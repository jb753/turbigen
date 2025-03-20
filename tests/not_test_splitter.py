"""Find all examples and confirm that they will run without errors."""

import os, sys, shutil
from turbigen import util, main, config, run
from tempfile import mkdtemp
import pytest
import logging

# Look for examples directory above this script
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CONF_FILE = os.path.join(TEST_DATA_DIR, "cascade_splitter.yaml")
TAR_FILE = os.path.join(TEST_DATA_DIR, "cascade_splitter.tar.gz")
print(TAR_FILE)

logger = util.make_logger()
logger.setLevel(level=logging.INFO)


def test_splitter():

    tmp_dir = mkdtemp()

    # Extract the mesh tarball
    os.system(f"tar xf {TAR_FILE} --directory={tmp_dir}")

    # Load an example config
    c = config.Config.read(CONF_FILE)

    c.workdir = tmp_dir
    c.solver["skip"] = True
    c.mesh["gbcs_path"] = os.path.join(tmp_dir, "mesh")
    c.wdist = False

    assert run.run(c)

    # Delete if successful
    shutil.rmtree(tmp_dir)


# If called as a script then test all examples
if __name__ == "__main__":
    test_splitter()
