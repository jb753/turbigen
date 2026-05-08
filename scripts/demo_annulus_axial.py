"""Interactive demo: axial annulus with Beta=0, increasing span."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from turbigen.new_geometry import Annulus

NROW = 2
NPTS = 300
NSPF = 5


def make_annulus(span_ratio, AR_chord, AR_gap):
    span0 = 0.1
    return Annulus(
        r_mid=0.5 * np.ones(NROW * 2),
        span=span0 * np.linspace(1.0, span_ratio, NROW * 2),
        Beta=np.zeros(NROW * 2),
        AR_chord=AR_chord * np.ones(NROW),
        AR_gap=AR_gap * np.ones(NROW + 1),
    )


def draw(ax, ann):
    ax.cla()
    ax.set_xlabel("x / m")
    ax.set_ylabel("r / m")
    m = np.linspace(0.0, ann.nseg, NPTS)
    ax.plot(*ann.evaluate(m, 0.0), "k-", lw=1.5)
    ax.plot(*ann.evaluate(m, 1.0), "k-", lw=1.5)
    for spf in np.linspace(0.0, 1.0, NSPF)[1:-1]:
        ax.plot(*ann.evaluate(m, spf), "b-", lw=0.6, alpha=0.5)
    for mi in range(1, ann.nseg):
        xr_h = ann.evaluate(mi, 0.0)
        xr_c = ann.evaluate(mi, 1.0)
        ls = "-" if mi % 2 == 1 else "--"
        ax.plot([xr_h[0], xr_c[0]], [xr_h[1], xr_c[1]], color="gray", ls=ls, lw=0.8)
    ax.set_aspect("equal")


init = dict(span_ratio=2.0, AR_chord=2.0, AR_gap=1.0)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.22)
draw(ax, make_annulus(**init))

sliders = [
    (fig.add_axes([0.15, 0.14, 0.70, 0.03]), "Span ratio", 1.0, 4.0, init["span_ratio"]),
    (fig.add_axes([0.15, 0.09, 0.70, 0.03]), "AR chord",   0.5, 5.0, init["AR_chord"]),
    (fig.add_axes([0.15, 0.04, 0.70, 0.03]), "AR gap",     0.5, 3.0, init["AR_gap"]),
]
sl_span, sl_arc, sl_gap = [Slider(a, lbl, lo, hi, valinit=v) for a, lbl, lo, hi, v in sliders]


def update(_):
    try:
        draw(ax, make_annulus(sl_span.val, sl_arc.val, sl_gap.val))
    except Exception as e:
        ax.set_title(str(e))
    fig.canvas.draw_idle()


for sl in (sl_span, sl_arc, sl_gap):
    sl.on_changed(update)

plt.show()
