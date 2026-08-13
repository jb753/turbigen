"""Post-processing.

A post-processor is a :class:`~turbigen2.node.Node`, so it is configured like
any other part of the file and needs no registration or bespoke serialisation.
It *returns* figures rather than drawing into a shared document:

```python
class Post(Node):
    def report(self, config, result) -> list[Figure]
```

That is what lets one be run alone in a notebook, lets the caller decide
whether the figures become a PDF, and lets a test assert on a figure without
touching the filesystem. The package this replaces threads a `PdfPages` through
every processor, so none of those are possible.

It takes the config as well as the result, because the most useful plots
compare design intent against what was achieved, and intent lives in the
config. That is safe here only because a `Config` is frozen: the hazard in the
old interface is not reading the config but that its post-processors call
`config.apply_recamber()` and `config.undo_recamber()`, mutating the geometry
from inside a plot.
"""

import logging
from typing import ClassVar

import numpy as np

from turbigen2.node import Node

logger = logging.getLogger("turbigen")


class Post(Node):
    """Base for post-processors."""

    def report(self, config, result):
        """Return figures describing `result`.

        Parameters
        ----------
        config : Config
            The configuration that was run, for design intent.
        result : Result
            What designing and running produced.

        Returns
        -------
        list of matplotlib.figure.Figure
            Empty if there was nothing to plot, which is not an error.

        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement report(self, config, result)"
        )


class AnnulusPlot(Post):
    """Meridional view of the annulus."""

    type: ClassVar[str] = "annulus"

    m_cut: tuple = ()
    """Normalised meridional positions at which to draw a cut plane."""

    show_axis: bool = False
    """Draw the axis of rotation."""

    def report(self, config, result):
        annulus = result.machine.annulus
        if annulus is None:
            logger.info("No annulus was designed, skipping the annulus plot.")
            return []

        # Imported here so that turbigen2 can be used without a display, and
        # without paying for matplotlib when nothing is being plotted.
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(layout="constrained")
        ax.axis("off")
        ax.axis("equal")

        # Sample the hub and casing straight from the annulus. Everything a
        # meridional view needs is a view on evaluate_xr, so nothing is added
        # to Annulus to support this: shapes wanted by one consumer belong in
        # that consumer.
        m = np.linspace(0.0, annulus.mmax, annulus.n_segment * 50 + 1)
        xr_hub = annulus.evaluate_xr(m, 0.0)
        xr_cas = annulus.evaluate_xr(m, 1.0)

        # A cut plane is the hub-to-casing line at one meridional position
        for m_cut in self.m_cut:
            ax.plot(*annulus.evaluate_xr(m_cut, [0.0, 1.0]), "-", color="C0")

        ax.plot(*xr_hub, "k-")
        ax.plot(*xr_cas, "k-")

        if self.show_axis:
            ax.plot(xr_hub[0, (0, -1)], np.zeros(2), "k-.")

        ax.set_title("Annulus")
        return [fig]
