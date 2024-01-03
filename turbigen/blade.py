"""Classes for generating blade coordinates.

The purpose of these objects is to evaluate x/r/t coordinates on blade sections
at specified spanwise locations.

"""

import turbigen.util
from scipy.interpolate import interp1d
import numpy as np


def _is_above(xrq, xrc):
    xq, rq = xrq.reshape(2, 1, -1)
    xc, rc = xrc.reshape(2, -1, 1)
    dxc = np.diff(xc, axis=0)
    drc = np.diff(rc, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        above_r = rq >= rc[:-1] + drc / dxc * (xq - xc[:-1])
        below_x = xq < xc[:-1] + dxc / drc * (rq - rc[:-1])

    above = np.where(dxc > drc, above_r, below_x).all(axis=0).reshape(xrq.shape[1:])
    return above


class SampledBlade:
    """Make a blade surface from sampled coordinates."""

    def __init__(self, spf, xrt_upper, xrt_lower):
        """Initialise a sampled blade row with surface coordinates.

        Parameters
        ----------
        spf : (nj,) array
            Span fractions for each spanwise grid point [-].
        xrt_upper : (3, ni_upper, nj) array
            Coordinates of the upper surface [m, m, rad].
        xrt_lower : (3, ni_lower, nj) array
            Coordinates of the lower surface [m, m, rad].

        """

        # Store input data
        nj = len(spf)
        self._xrt = [
            np.reshape(xrt_upper, (3, -1, nj)),
            np.reshape(xrt_lower, (3, -1, nj)),
        ]
        self._spf = np.reshape(spf, nj)

        self._interpolators = [
            interp1d(spf, xrt, fill_value="extrapolate") for xrt in self._xrt
        ]

    def evaluate_section(self, spfq):
        """Sample upper and lower surface coordinates at a constant span fraction."""
        return [interp(spfq) for interp in self._interpolators]

    def surface_length(self, spf):
        """Suction surface length."""
        xrtu, xrtl = self.evaluate_section(spf)
        xrrtu = np.stack((*xrtu[:2],) + (xrtu[1] * xrtu[2],))
        xrrtl = np.stack((*xrtl[:2],) + (xrtl[1] * xrtl[2],))
        Lu = turbigen.util.arc_length(xrrtu)
        Ll = turbigen.util.arc_length(xrrtl)
        return np.maximum(Lu, Ll)

    def get_coords(self, nspf=10, nchord=100, flip_theta=False):
        """3-D coordinates for this blade row in AutoGrid-style format.

        Parameters
        ----------
        nspf : int
            Number of sections in radial direction.
        nchord : int
            Number of chordwise points along each surface.

        Returns
        -------
        xrt : (2, nspf, nchord, 3) array
            Axial, radial, angular coordinates for this blade. `xrt[0]` is the
            upper surface, with highest theta, `xrt[1]` the lower surface.

        """

        xrt = np.stack(
            [self._section(spf, nchord) for spf in np.linspace(0.0, 1.0, nspf)]
        ).transpose(1, 0, 3, 2)

        if flip_theta:
            xrt[:, :, :, 2] *= -1.0

        return xrt

    def write_yaml(self, fname):
        """Save this blade to a yaml file."""
        d = {
            "spf": self._spf.tolist(),
            "xrt_upper": self._xrt[0].tolist(),
            "xrt_lower": self._xrt[1].tolist(),
        }
        turbigen.util.write_yaml(d, fname)

    @classmethod
    def from_yaml(cls, fname):
        return cls(**turbigen.util.read_yaml(fname))
