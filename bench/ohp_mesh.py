"""Time the passage mesher and measure the meshes it makes, against a baseline.

Meshes a fixed set of cases one after another, so the timings are not
competing for the machine, and records for each the time, the cells, and
what has gone wrong with these meshes before: the worst cell corner of each
block, the steps in spacing along and across each block, along the blade,
and across the outer face of the O block round it. Then compares them with
`ohp_mesh_baseline.json` beside this file and flags anything worse.

The cases are the axial turbine example meshed by the passage mesher with
its clearance and without, the cascade example, and the two stages of a
design sweep whose high-turning rotors are the hardest tip gaps it has met.

Usage::

    uv run python bench/ohp_mesh.py              # compare with the baseline
    uv run python bench/ohp_mesh.py --update     # and then replace it
    uv run python bench/ohp_mesh.py axial_gap    # only the cases named
"""

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from turbigen.config import Config
from turbigen.ohpmesh import _cell_angles

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"
BASELINE = HERE / "ohp_mesh_baseline.json"

SECTIONS = ("mean_line", "annulus", "blades", "fluid", "inlet_profile", "mesh")
"""The parts of a config that meshing reads."""

WORSE_ANGLE = 1.0
"""Drop in a worst cell corner [deg] counted as a regression."""

WORSE_STEP = 1.05
"""Factor of rise in a worst step in spacing counted as a regression."""

WORSE_TIME = 1.25
"""Factor of rise in meshing time counted as a regression."""


def _load(path):
    doc = yaml.safe_load(Path(path).read_text())
    return {k: doc[k] for k in SECTIONS if doc.get(k) is not None}


def _axial(gap):
    """The axial turbine example, by the passage mesher, which needs round
    trailing edges; with its rotor's clearance or without."""
    doc = _load(EXAMPLES / "axial_turbine.yaml")
    for blade in doc["blades"]:
        for section in blade["sections"]:
            section["thickness"] = dict(section["thickness"], round_TE=True)
        if not gap:
            blade["tip_span"] = 0.0
    doc["mesh"] = {"type": "ohp", "yplus": doc["mesh"]["yplus"]}
    return doc


CASES = {
    "axial": lambda: _axial(gap=False),
    "axial_gap": lambda: _axial(gap=True),
    "cascade": lambda: _load(EXAMPLES / "turbine_cascade_ohp.yaml"),
    "sweep3_0001": lambda: _load(HERE / "cases" / "sweep3_0001.yaml"),
    "sweep3_0004": lambda: _load(HERE / "cases" / "sweep3_0004.yaml"),
}


def _cartesian(xrt):
    xrt = np.asarray(xrt, dtype=float)
    r, t = xrt[..., 1], xrt[..., 2]
    return np.stack((xrt[..., 0], r * np.cos(t), r * np.sin(t)), axis=-1)


def _steps(d, axis):
    """Ratio of each cell length `d` to the next along `axis`, either way up."""
    n = d.shape[axis]
    r = np.take(d, range(1, n), axis) / np.take(d, range(n - 1), axis)
    return np.maximum(r, 1.0 / r)


def _worst_corner(block):
    rt = block.r * block.t
    return min(
        float(_cell_angles(np.stack((block.x[:, j, :], rt[:, j, :]))).min())
        for j in range(block.shape[1])
    )


def _block_steps(block):
    xyz = _cartesian(block.xrt)
    return [
        float(_steps(np.linalg.norm(np.diff(xyz, axis=a), axis=-1), a).max())
        for a in range(3)
    ]


def _wall_step(O):
    """Worst step in spacing along the blade, round the cut too."""
    d = np.linalg.norm(np.diff(_cartesian(O.xrt)[:, :, 0], axis=0), axis=-1)
    return float(_steps(np.concatenate((d[-1:], d, d[:1])), 0).max())


def _outer_face_steps(O, row):
    """Ratio of the O block's last cell off the blade to the cell beyond it in
    the next block, at each node of its outer face that another block shares."""
    P = _cartesian(O.xrt)
    p = P[:, :, -1].reshape(-1, 3)
    a = (P[:, :, -1] - P[:, :, -2]).reshape(-1, 3)
    out = np.full(len(p), np.nan)
    for block in row:
        if block is O:
            continue
        for shift in (-1, 0, 1):
            xrt = block.xrt.astype(float).copy()
            xrt[..., 2] += shift * O.pitch
            C = _cartesian(xrt)
            dist, idx = cKDTree(C.reshape(-1, 3)).query(p)
            hit = np.nonzero(dist < 1e-7 * np.abs(p).max())[0]
            ijk = np.stack(np.unravel_index(idx[hit], C.shape[:3]), axis=1)
            for axis in range(3):
                for step in (-1, 1):
                    q = ijk.copy()
                    q[:, axis] += step
                    ok = (q[:, axis] >= 0) & (q[:, axis] < C.shape[axis])
                    v = C[tuple(q[ok].T)] - C[tuple(ijk[ok].T)]
                    av = a[hit[ok]]
                    cos = np.sum(v * av, 1) / (
                        np.linalg.norm(v, axis=1) * np.linalg.norm(av, axis=1)
                    )
                    along = cos > 0.7
                    out[hit[ok][along]] = np.linalg.norm(
                        v[along], axis=1
                    ) / np.linalg.norm(av[along], axis=1)
    r = np.maximum(out, 1.0 / out)
    return r[np.isfinite(r)]


def measure(name):
    """Mesh case `name` and return what was measured of it."""
    config = Config.from_dict(copy.deepcopy(CASES[name]()))
    machine = config.design()
    t0 = time.perf_counter()
    grid = config.mesh.mesh(machine)
    t_mesh = time.perf_counter() - t0

    blocks = {}
    for block in grid:
        steps = _block_steps(block)
        blocks[block.label] = {
            "worst_corner": _worst_corner(block),
            "step_i": steps[0],
            "step_j": steps[1],
            "step_k": steps[2],
            "folded": int((block.vol <= 0.0).sum()),
        }
    rows = {}
    for i_row, row in enumerate(grid.rows):
        (O,) = [b for b in row if b.label.endswith("_O")]
        face = _outer_face_steps(O, row)
        rows[f"row{i_row}"] = {
            "wall_step": _wall_step(O),
            "outer_face_step": float(face.max()),
            "outer_face_over_1.5": float(np.mean(face > 1.5)),
        }

    main = [v for k, v in blocks.items() if "tip" not in k]
    gap = [v for k, v in blocks.items() if "tip" in k]
    summary = {
        "t_mesh": t_mesh,
        "n_cell": int(sum(np.prod(np.array(b.shape) - 1) for b in grid)),
        "folded": sum(v["folded"] for v in blocks.values()),
        "worst_corner_main": min(v["worst_corner"] for v in main),
        "worst_corner_gap": min((v["worst_corner"] for v in gap), default=None),
        "step_ik": max(max(v["step_i"], v["step_k"]) for v in blocks.values()),
        "wall_step": max(v["wall_step"] for v in rows.values()),
        "outer_face_step": max(v["outer_face_step"] for v in rows.values()),
    }
    return {"summary": summary, "blocks": blocks, "rows": rows}


# Summary quantities compared with the baseline, and whether bigger is worse.
COMPARED = {
    "t_mesh": ("time", True),
    "n_cell": ("count", None),
    "folded": ("count", True),
    "worst_corner_main": ("angle", False),
    "worst_corner_gap": ("angle", False),
    "step_ik": ("step", True),
    "wall_step": ("step", True),
    "outer_face_step": ("step", True),
}


def _worse(kind, bigger_is_worse, now, then):
    if now is None or then is None or bigger_is_worse is None:
        return False
    if kind == "angle":
        return now < then - WORSE_ANGLE
    if kind == "step":
        return now > then * WORSE_STEP
    if kind == "time":
        return now > then * WORSE_TIME
    return now > then


def _fmt(value, kind):
    if value is None:
        return "-"
    return str(value) if kind == "count" else f"{value:.3g}"


def compare(results, baseline):
    """Print each case's summary beside the baseline's; return how many
    quantities came out worse."""
    n_worse = 0
    for name, res in results.items():
        then = baseline.get(name, {}).get("summary", {})
        print(f"\n{name}")
        for key, (kind, bigger_is_worse) in COMPARED.items():
            now, was = res["summary"][key], then.get(key)
            worse = _worse(kind, bigger_is_worse, now, was)
            n_worse += worse
            print(
                f"  {key:20s} {_fmt(now, kind):>10s}   baseline {_fmt(was, kind):>10s}"
                f"{'   WORSE' if worse else ''}"
            )
    return n_worse


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "cases", nargs="*", help=f"any of {', '.join(CASES)}; all if none"
    )
    parser.add_argument("--update", action="store_true", help="replace the baseline")
    args = parser.parse_args()
    unknown = set(args.cases) - set(CASES)
    if unknown:
        parser.error(f"no such case: {', '.join(sorted(unknown))}")
    cases = args.cases or list(CASES)
    # The polish overflows on the cells it is unfolding, harmlessly; a cell it
    # leaves folded is counted instead.
    warnings.simplefilter("ignore", RuntimeWarning)

    results = {}
    for name in cases:
        print(f"Meshing {name}...", flush=True)
        results[name] = measure(name)

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    n_worse = compare(results, baseline)
    if args.update:
        baseline.update(results)
        BASELINE.write_text(json.dumps(baseline, indent=1, sort_keys=True) + "\n")
        print(f"\nUpdated {BASELINE.name}.")
    elif n_worse:
        print(f"\n{n_worse} quantities worse than the baseline.")
    return 1 if n_worse and not args.update else 0


if __name__ == "__main__":
    sys.exit(main())
