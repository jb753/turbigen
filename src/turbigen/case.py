"""Reading and writing a case: a config, and what running it produced.

Both live in one file, under one document, with the run's answer under a single
``result:`` key beside the config's own. One file because comparing an achieved
efficiency against the design that asked for it should be one load, not two;
one *document* because a second YAML document would break `yaml.safe_load`,
`yq` and every other tool that assumes one.

That does not put results on the config. :class:`~turbigen.config.Config`
stays frozen and knows nothing about them --- it is this module that returns
two objects from one file. A file is not an object.
"""

import logging
from pathlib import Path

import ember.yaml_util

from turbigen import include, plugins
from turbigen.config import Config
from turbigen.result import Result

logger = logging.getLogger("turbigen")

RESULT_KEY = "result"
"""Top-level key holding a run's answer, beside the config's own keys."""


def read(path, design=True):
    """Read a case file, returning its config and its result.

    Parameters
    ----------
    path : Path or str
        File to read.
    design : bool
        Whether to design the machine, so that :attr:`Result.nominal` works.
        On by default, since comparing nominal against actual is the point of
        keeping the two together. Turn it off when scraping many files and only
        the achieved state is wanted --- at the cost that the result's entropy
        and internal energy are then measured from the config's datum rather
        than the design's, so they are not comparable with those of a file read
        the other way. Everything else, being dimensional, is.

    Returns
    -------
    config : Config
    result : Result or None
        None if the file holds no ``result:`` key, which is the case for any
        config that has not been run.

    """
    path = Path(path)
    plugins.discover(path.parent)

    data = include.read(path)
    result_data = data.pop(RESULT_KEY, None)

    config = Config.from_dict(data)

    if result_data is None:
        return config, None

    machine = config.design() if design else None

    # The stored state is dimensional -- pressure, temperature, velocity -- so
    # it can be read against any equation of state and come back unchanged.
    # Which one is chosen decides only what datum the entropy and internal
    # energy of the result are measured from, and it has to be the one the
    # nominal mean line carries, or comparing the two would compare numbers
    # taken from different zeros. Without a machine there is nothing to compare
    # against and no design to reference, so the config's own fluid is used.
    fluid = machine.mean_line.fluid if machine else config.fluid.eos()
    result = Result.from_dict(result_data, fluid, machine=machine)

    return config, result


def write(path, config, result=None):
    """Write `config`, and `result` if there is one, to a single file."""
    path = Path(path)

    data = config.to_dict()
    if result is not None:
        data[RESULT_KEY] = result.to_dict()

    ember.yaml_util.write_yaml(data, path)
