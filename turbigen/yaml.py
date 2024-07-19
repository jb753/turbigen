"""Functions for reading and writing YAML files."""

import yaml
import gzip
import re
import numpy as np


# Allow dumping of numpy float64 to yaml
def represent_float(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:float", str(data))


yaml.representer.SafeRepresenter.add_representer(np.float64, represent_float)


# Allow dumping int to yaml
def represent_int(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:int", str(data))


yaml.representer.SafeRepresenter.add_representer(np.int64, represent_int)


# Allow dumping np.ndarray as a list to yaml
def represent_ndarray(dumper, data):
    return dumper.represent_list(data.tolist())


yaml.representer.SafeRepresenter.add_representer(np.ndarray, represent_ndarray)


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise Exception(f'Config: duplicate key "{key}"')
            mapping.append(key)
        return super().construct_mapping(node, deep)


PATTERN = """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$"""


def read_yaml(fname):
    """Read a dictionary from file."""

    # Patch YAML loader to get scientific notation correct
    loader = UniqueKeyLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            PATTERN,
            re.X,
        ),
        list("-+0123456789."),
    )

    # Read the YAML
    with open(fname, "r") as f:
        config = yaml.load(f, Loader=loader)

    return config


def read_yaml_list(fname):
    """Read a list of dictionaries from YAML file."""
    # Patch YAML loader to get scientific notation correct
    loader = UniqueKeyLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(PATTERN, re.X),
        list("-+0123456789."),
    )
    # Read the YAML
    with open(fname, "r") as f:
        config = list(yaml.load_all(f, Loader=loader))

    return config


def write_yaml(d, fname, mode="w"):
    """Write a dictionary to file."""
    with open(fname, mode) as f:
        yaml.safe_dump(d, f, explicit_start=True, explicit_end=True)


def write_yaml_compressed(d, fname):
    """Write a dictionary to compressed file."""
    with gzip.open(fname, "wt") as f:
        yaml.safe_dump(d, f, explicit_start=True, explicit_end=True)
