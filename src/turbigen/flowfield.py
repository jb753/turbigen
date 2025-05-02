"""Classes to represent flow fields."""

import numpy as np
import turbigen.base
import turbigen.fluid
import turbigen.yaml
from multiprocessing import Pool
import os


def make_mean_line(rrms, A, Omega, Vxrt, S):
    """Assemble a perfect or real mean-line data structure from input states."""
    try:
        S = S[0].stack(S)
    except AttributeError:
        pass
    if isinstance(S, turbigen.fluid.PerfectState):
        ml_class = PerfectMeanLine
    elif isinstance(S, turbigen.fluid.RealState):
        ml_class = RealMeanLine
    else:
        raise Exception(f"Unknown fluid class {type(S)}")
    return ml_class.from_states(rrms, A, Omega, Vxrt, S)


def make_mean_line_from_flowfield(A, F, Ds_mix=0.0):
    """Assemble a perfect or real mean-line data structure from input states."""
    if isinstance(F, PerfectFlowField):
        ml_class = PerfectMeanLine
    elif isinstance(F, RealFlowField):
        ml_class = RealMeanLine
    else:
        raise Exception(f"Unknown fluid class {type(F)}")
    ml = ml_class.from_states(F.r, A, F.Omega, F.Vxrt, F, F.Nb)
    ml.Ds_mix = Ds_mix
    ml._metadata.pop("patches")
    ml._metadata.pop("Nb")
    return ml


class BaseFlowField(
    turbigen.base.StructuredData, turbigen.base.Kinematics, turbigen.base.Composites
):
    def check_flow(self):
        assert np.isfinite(self.Vxrt).all()
        assert np.isfinite(self.P).all()
        assert np.isfinite(self.T).all()
        assert (self.P > 0.0).all()
        assert (self.T > 0.0).all()


class PerfectFlowField(turbigen.fluid.PerfectState, BaseFlowField):
    """Flow and thermodynamic properties of a perfect gas."""

    _data_rows = (
        "x",
        "r",
        "t",
        "Vx",
        "Vr",
        "Vt",
        "rho",
        "u",
        "Omega",
    )

    @classmethod
    def from_properties(cls, xrt, Vxrt, PT, cp, ga, mu, Omega):
        # Make an empty class
        F = cls(np.shape(xrt)[1:])

        # Insert our data
        F.cp, F.gamma, F.mu, F.Omega = cp, ga, mu, Omega
        F.set_P_T(*PT)
        F.Vxrt = Vxrt
        F.xrt = xrt

        return F


class PerfectMeanLine(turbigen.base.MeanLine, PerfectFlowField):
    """Encapsulate the mean-line flow and geometry of a turbomachine."""

    _data_rows = ("x", "r", "A", "Vx", "Vr", "Vt", "rho", "u", "Omega", "Nb")


class RealFlowField(turbigen.fluid.RealState, BaseFlowField):
    """Flow and thermodynamic properties of a perfect gas."""

    _data_rows = (
        "x",
        "r",
        "t",
        "Vx",
        "Vr",
        "Vt",
        "rho",
        "u",
        "Omega",
    )


class RealMeanLine(turbigen.base.MeanLine, RealFlowField):
    """Encapsulate the mean-line flow and geometry of a turbomachine."""

    _data_rows = ("x", "r", "A", "Vx", "Vr", "Vt", "rho", "u", "Omega", "Nb")


def mean_line_from_dict(d):
    if d["class"] == "PerfectMeanLine":
        return PerfectMeanLine.from_dict(d)
    elif d["class"] == "RealMeanLine":
        return RealMeanLine.from_dict(d)
    else:
        raise Exception(f"Unrecognised mean line class {d['class']}")


def read_mean_line_database(database_file):
    """Load a list of mean_lines from a database file."""
    # Initialise the objects in parallel
    Nworker = os.cpu_count()
    with Pool(Nworker) as p:
        ml = p.map(mean_line_from_dict, turbigen.yaml.read_yaml_list(database_file))
    return ml
