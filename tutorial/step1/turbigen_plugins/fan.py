from turbigen.design import MeanLineDesign


class Fan(MeanLineDesign):
    """A single-row axial fan."""

    type: str = "fan"
    n_row: int = 1

    def forward(self, fluid):
        """Return a mean line built from this design's variables."""
        raise NotImplementedError("Implement the forward method")

    def backward(self, ml):
        """Return the design variables represented by mean line `ml`."""
        raise NotImplementedError("Implement the backward method")
