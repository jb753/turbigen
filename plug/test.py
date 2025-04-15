import dataclasses
import numpy as np
from turbigen import util
import matplotlib.pyplot as plt
import turbigen.post
import turbigen.base
import warnings


# @dataclasses.dataclass
# class Contour(turbigen.post.BasePost):
#     variable: str = "Ys"
#     """Which variable to plot."""
#
#     coord: str = "spf"
#     """Mapping of row index to span fraction(s) to plot."""
#
#     value: float = 0.5
#     """How many points away from the wall."""
#
#     irow_ref: int = 0
#     """Which row to use for reference quantities."""
#
#     N_passage: int = 2
#     """Repeat in the circumferential direction."""
#
#     cmap: str = "plasma"
#     """matplotlib colormap to use."""
#
#     def post(self, config, pdf):
#         """Plot contours over a plane."""
#
#         try:
#             for val in self.value:
#                 self.contour(val, config, pdf)
#         except TypeError:
#             # If value is not iterable, plot a single contour
#             self.contour(self.value, config, pdf)
#
#     def contour(self, val, config, pdf):
#         if self.coord == "spf":
#             # Span fraction cut
#             # Cut and repeat each row separately
#             xrc = config.annulus.get_span_curve(val)
#             Crow = config.grid.cut_span_unstructured(xrc)
#             Crow = [Ci.repeat_pitchwise(self.N_passage) for Ci in Crow]
#
#             # Combine the rows
#             C = turbigen.base.concatenate(Crow)
#
#         else:
#             # Get an xr curve describing the cut plane.
#             if self.coord == "x":
#                 xrc = np.array([[val, val], [0.1, 1.0]])
#             elif self.coord == "r":
#                 xrc = np.array([[-1.0, 1.0], [val, val]])
#             elif self.coord == "m":
#                 xrc = config.annulus.get_cut_plane(val)[0]
#             else:
#                 raise Exception(f"Invalid coord={self.coord}")
#             C = config.grid.unstructured_cut_marching(xrc)
#
#             C = C.repeat_pitchwise(self.N_passage)
#
#         # Centre theta on zero
#         C.t -= 0.5 * (C.t.min() + C.t.max())
#
#         # Matplotlib style triangulate
#         C_tri, triangles = C.get_mpl_triangulation()
#
#         # Get the coordinates to plot
#         if self.coord == "x":
#             c = C_tri.yz
#         elif self.coord == "r":
#             c = C_tri.rt, C_tri.x
#         elif self.coord == "spf":
#             # Now generate a mapping from xr to meridional distance
#             mp_from_xr = config.annulus.get_mp_from_xr(val)
#             c = mp_from_xr(C_tri.xr), C_tri.t
#         elif self.coord == "m":
#             if np.ptp(C_tri.r) > np.ptp(C_tri.x):
#                 c = C_tri.yz
#             else:
#                 c = C_tri.rt, C_tri.r
#         else:
#             raise Exception(f"Invalid coord={self.coord}")
#
#         # Extract meanline reference row
#         if self.coord == "m":
#             irow_ref = int(val / 2 - 1)
#             row = config.mean_line.actual.get_row(irow_ref)
#         else:
#             row = config.mean_line.actual
#
#         # Get the variable
#         v = turbigen.post.calculate_nondim(C_tri, row, self.variable)
#         # levels = clipped_levels(v)
#
#         # Setup figure
#         _, ax = plt.subplots(layout="constrained")
#         ax.set_title(f"{self.variable} at {self.coord}={val:.3g}")
#
#         # It seems that we have to pass triangles as a kwarg to tricontour,
#         # not positional, but this results in a UserWarning that contour
#         # does not take it as a kwarg. So catch and hide this warning.
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             cm = ax.tricontourf(
#                 *c,
#                 v,
#                 # levels,
#                 triangles=triangles,
#                 cmap=self.cmap,
#                 linestyles="none",
#             )
#         cm.set_edgecolor("face")
#         ax.set_aspect("equal")  # Ensures equal scaling
#         ax.set_adjustable("box")  # Ensures equal scaling
#         ax.axis("off")
#
#         # Make the colorbar
#         label = turbigen.post.LABELS.get(self.variable, self.variable)
#         plt.colorbar(cm, label=label, shrink=0.8)
#
#         # Finish this row
#         pdf.savefig()
#         plt.close()
#
#
# def clipped_levels(x, N=11, thresh=0.01):
#     xmin = util.qinv(x, thresh)
#     xmax = util.qinv(x, 1.0 - thresh)
#     dx = np.round((xmax - xmin) / 11, decimals=1)
#     xmin = np.round(xmin / dx) * dx
#     xmax = np.round(xmax / dx) * dx
#     xlev = np.arange(xmin, xmax + dx, dx)
#     return xlev

