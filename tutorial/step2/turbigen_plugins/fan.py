from turbigen.design import MeanLineDesign


class Fan(MeanLineDesign):
    """A single-row axial fan."""

    type: str = "fan"
    n_row: int = 1

    DPo: float
    """Stagnation pressure rise across the rotor [Pa]."""

    mdot: float
    """Mass flow rate [kg/s]."""

    phi: float
    """Inlet flow coefficient [--]."""

    psi: float
    """Stage loading coefficient [--]."""

    htr: float
    """Inlet hub-to-tip ratio [--]."""

    eta_tt: float
    """Total-to-total isentropic efficiency [--]."""

    Po1: float = 1e5
    """Inlet stagnation pressure [Pa]."""

    To1: float = 300.0
    """Inlet stagnation temperature [K]."""

    def forward(self, fluid):
        """Return a mean line built from this design's variables."""
        raise NotImplementedError("Implement the forward method")

    def backward(self, ml):
        """Return the design variables represented by mean line `ml`."""
        raise NotImplementedError("Implement the backward method")
