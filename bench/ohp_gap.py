"""Rerun the passage mesher's tip gap alone, on a main mesh built once.

The first run of a case meshes its row up to the gap, and caches what
:meth:`OHPassage._tip_gap` is given (the blade, the row's layout, the planes
in the gap and their finished main blocks) in `bench/.ohp_gap_cache/`. Every
later run loads them and builds only the gap, so a change to the gap's
construction can be tried in a second or two.

For each block of the gap the worst cell corner, and where, is printed, with
the count of folded cells (signed area against the rest of the block). With
`-o` a PDF is written of the plane at the top of the blade and the one with
the worst cell: the whole gap, then a close-up of the worst cell, the O block's
inner face and the H block's corners marked.

Mesher options after the case, `key=value`, are applied to the gap only: the
cached main mesh is not rebuilt. Options that change the main passage, such
as `tip_match`, go with `--main key=value`, repeatable: they are applied to
the whole mesh, and each set of them has a cache of its own.

A new algorithm can be tried without touching the mesher by patching it in
from a module of your own, `--patch mymod.py`, which is run before the gap is
built and may replace any function of :mod:`turbigen.ohpmesh` (for example
`ohpmesh._gap_H = my_gap_H`).

Usage::

    uv run python bench/ohp_gap.py                         # sweep3_0001
    uv run python bench/ohp_gap.py sweep3_0004 -o gap.pdf
    uv run python bench/ohp_gap.py sweep3_0001 tip_front_lean=30 --tip-only
    uv run python bench/ohp_gap.py --patch try_pairing.py -o gap.pdf
    uv run python bench/ohp_gap.py --main tip_match=2 -o gap.pdf
    uv run python bench/ohp_gap.py --remesh       # after changing the main mesher
"""

import argparse
import copy
import gzip
import logging
import pickle
import runpy
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ohp_mesh  # noqa: E402

import turbigen.ohpmesh as ohp  # noqa: E402
from turbigen.config import Config  # noqa: E402

CACHE = HERE / ".ohp_gap_cache"

MIN_ANGLE = 20.0
"""Cells with a worst corner under this, degrees, are shaded in the plots."""


class _Captured(Exception):
    """Raised to stop meshing once the gap's inputs are captured."""


def _coerce(mesher, key, text):
    """Value `text` of option `key`, of the type of its current value."""
    now = getattr(mesher, key)
    if isinstance(now, bool):
        return text.lower() in ("1", "true", "yes")
    if now is None:
        try:
            return float(text)
        except ValueError:
            return text
    return type(now)(text)


def _mesher(doc, overrides):
    """The case's mesher with options `overrides` of `key=value` applied."""
    doc = copy.deepcopy(doc)
    mesher = Config.from_dict(doc).mesh
    for item in overrides:
        key, text = item.split("=", 1)
        if not hasattr(mesher, key):
            raise SystemExit(f"OHPassage has no option {key!r}")
        doc["mesh"][key] = _coerce(mesher, key, text)
    return Config.from_dict(doc).mesh


def _capture(doc, overrides):
    """Mesh the case up to its tip gap; return the arguments of `_tip_gap`."""
    config = Config.from_dict(copy.deepcopy(doc))
    machine = config.design()
    mesher = _mesher(doc, overrides)
    held = {}
    assemble = ohp.OHPassage.assemble

    def capture(self, machine, i_row, layout, planes, per_plane, jtip=None):
        if jtip is None:
            return assemble(self, machine, i_row, layout, planes, per_plane, jtip)
        held.update(
            blade=machine.rows[i_row].blade,
            layout=layout,
            planes=planes[jtip:],
            per_plane=per_plane[jtip:],
            i_row=i_row,
        )
        raise _Captured

    ohp.OHPassage.assemble = capture
    try:
        mesher.forward(machine, mesher.wall_spacing(machine))
    except _Captured:
        pass
    finally:
        ohp.OHPassage.assemble = assemble
    if not held:
        raise SystemExit("no row of this case has a tip gap")
    return held


def _load(name, doc, main, remesh):
    """The gap's inputs for case `name` with main-mesh options `main`, from
    the cache or meshed afresh."""
    tag = "".join(f"_{item.replace('=', '-')}" for item in sorted(main))
    path = CACHE / f"{name}{tag}.pkl.gz"
    if path.exists() and not remesh:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.perf_counter()
    held = _capture(doc, main)
    print(f"meshed {name} up to the gap in {time.perf_counter() - t0:.1f} s")
    CACHE.mkdir(exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(held, f)
    print(f"cached in {path}")
    return held


def _folded(B):
    """Cells of block `B` (2, ni, nk) whose signed area is against the
    majority of the block's, or zero."""
    p = B.transpose(1, 2, 0).astype(float)
    d1 = p[1:, 1:] - p[:-1, :-1]
    d2 = p[:-1, 1:] - p[1:, :-1]
    area = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    sign = np.sign(np.sum(np.sign(area)))
    return area * sign <= 0.0


def _steps(B, axis, periodic=False):
    """Ratio of each edge of block `B` (2, ni, nk) along `axis` (0 for `i`,
    1 for `k`) to the next, either way up, (ni - 1, nk) or (ni, nk - 1)
    edges less one along `axis`; round the cut too if `periodic` in `i`."""
    d = np.linalg.norm(np.diff(B, axis=axis + 1), axis=0)
    if periodic and axis == 0:
        d = np.concatenate((d[-1:], d), axis=0)
    r = np.diff(np.log(np.maximum(d, 1e-300)), axis=axis)
    return np.exp(np.abs(r))


def _report(gap):
    """Print the worst corner, worst spacing steps along `i` and `k`, and
    folds of each gap block over its planes; return (j, label, i, k, angle)
    of the worst cell of all."""
    worst_all = None
    for ib, label in enumerate(("tipO", "tipH")):
        worst, n_fold = (90.0, None), 0
        steps = [(1.0, None), (1.0, None)]
        for j, xy in enumerate(gap):
            B = np.asarray(xy[ib], dtype=float)
            a = ohp._cell_angles(B)
            n_fold += int(_folded(B).sum())
            if a.min() < worst[0]:
                worst = (float(a.min()), (j, *np.unravel_index(np.argmin(a), a.shape)))
            for axis in (0, 1):
                st = _steps(B, axis, periodic=ib == 0)
                if st.max() > steps[axis][0]:
                    loc = np.unravel_index(np.argmax(st), st.shape)
                    steps[axis] = (float(st.max()), (j, *loc))
        j, i, k = worst[1]
        shape = np.asarray(gap[0][ib]).shape[1:]
        where = [
            f"{v:5.2f}" + (f" (plane {w[0]} i={w[1]} k={w[2]})" if w else "")
            for v, w in steps
        ]
        print(
            f"  {label:5s} {shape[0]:4d} x {shape[1]:3d}  worst {worst[0]:5.1f} deg "
            f"at plane {j} i={i} k={k}, folded {n_fold}; "
            f"step along i {where[0]}, along k {where[1]}"
        )
        if worst_all is None or worst[0] < worst_all[4]:
            worst_all = (j, label, i, k, worst[0])
    return worst_all


def _plot(gap, corners, planes, worst, out):
    """Pages of the tip plane and the worst plane: whole gap, close-up."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection, PolyCollection

    def lines(B, colour, ax):
        p = B.transpose(1, 2, 0)
        segs = list(p) + list(p.transpose(1, 0, 2))
        ax.add_collection(LineCollection(segs, colors=colour, linewidths=0.3))

    def shade(B, mask, colour, ax):
        p = B.transpose(1, 2, 0)
        c = np.stack([p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]], axis=2)
        if mask.any():
            ax.add_collection(PolyCollection(c[mask], facecolors=colour, lw=0))

    j_worst, label, i_w, k_w, _ = worst
    pages = [(0, "top of blade")]
    if j_worst != 0:
        pages.append((j_worst, "worst cell"))
    with PdfPages(out) as pdf:
        for j, what in pages:
            O, H = (np.asarray(B, dtype=float) for B in gap[j])
            spf = planes[j]["cmap"].spf
            B_w = O if label == "tipO" else H
            centre = B_w[:, i_w : i_w + 2, k_w : k_w + 2].reshape(2, -1).mean(1)
            for zoom in (False, True):
                fig, ax = plt.subplots(figsize=(11, 8))
                for B, colour in ((O, "C0"), (H, "C1")):
                    shade(B, ohp._cell_angles(B) < MIN_ANGLE, "orange", ax)
                    shade(B, _folded(B), "black", ax)
                    lines(B, colour, ax)
                inner = O[:, :, 0]
                ax.plot(*inner, "k-", lw=0.6)
                ax.plot(*inner[:, list(corners)], "rs", ms=4)
                for c, n in zip(corners, "ABCD"):
                    ax.annotate(n, inner[:, c], color="r", fontsize=9)
                ax.plot(*H[:, :, 0], "g.", ms=1.5)
                ax.plot(*H[:, :, -1], "m.", ms=1.5)
                if zoom:
                    h = 8 * np.linalg.norm(
                        np.ptp(
                            B_w[:, i_w : i_w + 2, k_w : k_w + 2].reshape(2, -1), axis=1
                        )
                    )
                    h = max(h, 1e-6)
                    ax.set_xlim(centre[0] - h, centre[0] + h)
                    ax.set_ylim(centre[1] - h, centre[1] + h)
                    ax.plot(*centre, "rx", ms=8)
                else:
                    ax.autoscale_view()
                ax.set_aspect("equal")
                ax.set_xlabel("m'")
                ax.set_ylabel("theta")
                worst_here = min(ohp._cell_angles(B).min() for B in (O, H))
                ax.set_title(
                    f"plane {j} ({what}), spf {spf:.4f}: worst corner "
                    f"{worst_here:.1f} deg; O blue, H orange; H k=0 green, k=-1 magenta"
                    + ("; close-up" if zoom else "")
                )
                pdf.savefig(fig)
                plt.close(fig)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "case", nargs="?", default="sweep3_0001", choices=ohp_mesh.CASES
    )
    parser.add_argument("options", nargs="*", help="mesher options key=value")
    parser.add_argument("-o", "--out", help="write plots of the gap to this PDF")
    parser.add_argument(
        "--main",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="mesher option for the whole mesh, cached separately; repeatable",
    )
    parser.add_argument("--remesh", action="store_true", help="rebuild the main mesh")
    parser.add_argument("--tip-only", action="store_true", help="only the tip plane")
    parser.add_argument("--patch", help="Python file run first to patch ohpmesh")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug log")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    logging.basicConfig(level=logging.WARNING)
    if args.verbose:
        logging.getLogger("turbigen").setLevel(logging.DEBUG)

    doc = ohp_mesh.CASES[args.case]()
    if any(item.startswith("tip_match=") for item in args.options):
        parser.error("tip_match changes the main mesh: pass it with --main")
    held = _load(args.case, doc, args.main, args.remesh)
    if args.patch:
        runpy.run_path(args.patch, init_globals={"ohpmesh": ohp})
    mesher = _mesher(doc, args.main + args.options)

    planes, per_plane = held["planes"], held["per_plane"]
    if args.tip_only:
        planes, per_plane = planes[:1], per_plane[:1]
    t0 = time.perf_counter()
    gap, corners, k_cut = mesher._tip_gap(
        held["blade"], held["layout"], planes, per_plane
    )
    dt = time.perf_counter() - t0
    print(
        f"{args.case} {' '.join(args.main + args.options)}: "
        f"{'matched all round' if held['layout'].symmetric else 'non-matching side'}, "
        f"{len(gap)} planes in {dt:.2f} s, "
        f"corners {corners}, rear cut k={k_cut}"
    )
    worst = _report(gap)
    if args.out:
        _plot(gap, corners, planes, worst, args.out)


if __name__ == "__main__":
    main()
