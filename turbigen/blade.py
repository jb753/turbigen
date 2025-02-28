from abc import abstractmethod
from turbigen import util
import numpy as np
import dataclasses
import turbigen.nblade
import turbigen.camber
import turbigen.thickness
import turbigen.thickness


@dataclasses.dataclass
class BladeDesigner:
    """Store design variables for a blade."""

    spf: np.ndarray
    """Span fractions to define sections on len(spf) = nsect"""

    q_thick: np.ndarray
    """Thickness design variables (nsect, nqthick)"""

    q_camber: np.ndarray
    """Camber design variables (nsect, nqcamber)"""

    camber_type: turbigen.camber.BaseCamber = "quartic"
    """Which method to generate the camber line."""

    thick_type: turbigen.thickness.BaseThickness = "taylor"
    """Which method to generate the thickness distribution."""

    tip: float = 0.0
    """Tip clearance as fraction of span."""

    vortex_expon: float = -1.0
    """Spanwise swirl distribution, Vt ~ r**vortex_expon."""

    def __post_init__(self):
        # self.number = util.init_subclass_by_signature(
        #     turbigen.nblade.BladeNumberConfig, self.number
        # )

        self.camber_type = util.get_subclass_by_name(
            turbigen.camber.BaseCamber, self.camber_type
        )

        self.thick_type = util.get_subclass_by_name(
            turbigen.thickness.BaseThickness, self.thick_type
        )

        # Check the dimensions of the design variables
        self.spf = np.reshape(self.spf, -1)
        nsect = len(self.spf)
        self.q_thick = np.atleast_2d(self.q_thick)
        self.q_camber = np.atleast_2d(self.q_camber)
        assert (
            self.q_thick.shape[0] == nsect
        ), f"Wrong number of sections for thickness, expected {nsect}, got {self.q_thick.shape[0]}"
        assert (
            self.q_camber.shape[0] == nsect
        ), f"Wrong number of sections for camber, expected {nsect}, got {self.q_camber.shape[0]}"

    def to_dict(self):
        # Built-in dataclasses method gets us most of the way there
        data = dataclasses.asdict(self)

        # Convert the camber and thickness types to strings
        data["camber_type"] = util.camel_to_snake(self.camber_type.__name__)
        data["thick_type"] = util.camel_to_snake(self.thick_type.__name__)

        # Convert ndarray to list
        data["spf"] = data["spf"].tolist()
        data["q_thick"] = data["q_thick"].tolist()
        data["q_camber"] = data["q_camber"].tolist()

        return data

    def apply_recamber(self, mean_line):
        """Convert the stored recamber angles to local tanchi.

        Initially q_camber[:2] are the recamber angles at the LE and TE.
        The flow angles at the mean radius are extracted from the input mean line.
        The variation with radius is given by the stored vortex_expon.
        On exit from this function, q_camber[:2] are the local tanchi values.

        """

    def undo_recamber(self, mean_line):
        """Convert the stored tanchi back to recamber angles."""
