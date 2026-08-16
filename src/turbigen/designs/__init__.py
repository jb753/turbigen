"""Mean-line designs shipped with turbigen.

Importing this package registers them. Registration happens by defining the
class, so this file exists to make that explicit rather than leaving it to
whichever module happens to be imported first.
"""

from turbigen.designs.axial_turbine import AxialTurbine
from turbigen.designs.turbine_cascade import TurbineCascade

__all__ = ["AxialTurbine", "TurbineCascade"]
