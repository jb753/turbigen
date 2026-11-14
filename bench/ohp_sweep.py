"""Mesh a spread of single compressor and turbine rows with the passage mesher.

Draws stage designs from a scrambled Sobol' sequence over flow coefficient,
loading, reaction, hub-to-tip ratio, aspect ratio and a loading coefficient
that sets the pitch-to-chord ratio, separately for compressors and turbines.
Each design gives two cases, its rotor and its stator, as single rows of the
`stage_row` mean line under a free vortex. Rotors have a clearance of one
per cent of span; stators none. Every row is held at the same surface
Reynolds number, :data:`RE_SURF`, by its viscosity, so the wall spacing for
the y+ asked is the same fraction of the blade throughout.

Every case is meshed in its own process, several at once, with a time limit,
so a hang or a crash in one costs only that case. Each case gets a directory
under the output directory holding its config, its log and a `result.json`,
and the run ends with a table of all of them. A case passes if no cell is
folded and the worst cell corner of every block is above :data:`MIN_ANGLE`.

The sequence is fixed by its seed, and draws are skipped where either row
turns the flow past :data:`ANGLE_MAX` at any span under the free vortex, or a
compressor row fails the de Haller limit, so asking for more points per machine extends the
same set of cases rather than replacing it.

Usage::

    uv run python bench/ohp_sweep.py                  # 4 designs per machine
    uv run python bench/ohp_sweep.py -n 16 -j 24      # more, more at once
    uv run python bench/ohp_sweep.py comp_002_rotor   # only the cases named
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

logger = logging.getLogger("ohp_sweep")

HERE = Path(__file__).resolve().parent

MIN_ANGLE = 20.0
"""Worst cell corner a passing mesh may have [deg]."""

DH_MIN = 0.7
"""Smallest de Haller number a compressor row may have [--]."""

ANGLE_MAX = 75.0
"""Largest flow angle, either row, either edge, any span [deg]."""

N_DRAW = 1024
"""Sobol' points drawn per machine, before filtering, to take cases from."""

FLUID = {"type": "perfect", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5}

MACHINES = {
    "comp": {
        "bounds": {
            "phi": (0.4, 0.8),
            "psi": (0.25, 0.5),
            "Lam": (0.5, 0.9),
            "htr": (0.5, 0.95),
            "AR": (1.0, 3.0),
            "DFL": (0.35, 0.5),
        },
        "Mb": 0.5,
        "thickness": {
            "type": "taylor",
            "R_LE": 0.01,
            "t_max": 0.08,
            "m_tmax": 0.4,
            "t_TE": 0.01,
            "tanwedge": 0.05,
            "round_TE": True,
        },
    },
    "turb": {
        "bounds": {
            "phi": (0.4, 1.0),
            "psi": (-2.5, -1.0),
            "Lam": (0.0, 0.6),
            "htr": (0.5, 0.95),
            "AR": (1.0, 3.0),
            "Co": (0.55, 0.8),
        },
        "Mb": 0.3,
        "thickness": {
            "type": "taylor",
            "R_LE": 0.04,
            "t_max": 0.15,
            "m_tmax": 0.35,
            "t_TE": 0.02,
            "tanwedge": 0.18,
            "round_TE": True,
        },
    },
}

SPF_SECTIONS = (0.1, 0.5, 0.9)
"""Span fractions of the blade sections."""

TIP_SPAN = 0.01
"""Rotor clearance as a fraction of span."""

RE_SURF = 5e5
"""Surface-length Reynolds number every row is held at, by its viscosity."""


def acceptable(machine, p):
    """Whether both rows of design `p` design, and within the limits.

    Checked on the rows themselves, designed as the mesher will get them: the
    de Haller number of a compressor row from its mean line, and the metal
    angles of either row at hub, mid-span and tip against :data:`ANGLE_MAX`.
    """
    from turbigen.config import Config

    for rotor in (True, False):
        cfg = Config.from_dict(config(machine, p, rotor))
        try:
            machine_row = cfg.design()
        except Exception as exc:
            logger.info(f"{machine} {p}: does not design, {exc}")
            return False
        ml = machine_row.mean_line
        if machine == "comp" and cfg.mean_line.backward(ml)["DH"] < DH_MIN:
            return False
        chi = np.array(geometry(machine_row)["chi"])
        if np.abs(chi).max() > ANGLE_MAX:
            return False
    return True


def designs(machine, n, seed):
    """The first `n` acceptable designs of the Sobol' sequence for `machine`."""
    spec = MACHINES[machine]
    names = list(spec["bounds"])
    lo, hi = np.array(list(spec["bounds"].values())).T
    sobol = qmc.Sobol(len(names), scramble=True, seed=seed)
    x = qmc.scale(sobol.random(N_DRAW), lo, hi)
    out = []
    for row in x:
        p = {k: float(v) for k, v in zip(names, row)}
        if not acceptable(machine, p):
            continue
        out.append(p)
        if len(out) == n:
            break
    return out


def config(machine, p, rotor):
    """The config for one row of design `p`."""
    spec = MACHINES[machine]
    section = {
        "dchi_LE": 0.0,
        "dchi_TE": 0.0,
        "camber": {"type": "quadratic", "aft_loading": 0.0},
        "thickness": spec["thickness"],
    }
    count = (
        {"type": "DFL", "DFL": p["DFL"]}
        if "DFL" in p
        else {"type": "Co", "Co": p["Co"]}
    )
    blade = {
        "count": count,
        "sections": [dict(section, spf=s) for s in SPF_SECTIONS],
    }
    if rotor:
        blade["tip_span"] = TIP_SPAN
    return {
        "fluid": FLUID,
        "mean_line": {
            "type": "stage_row",
            "phi": p["phi"],
            "psi": p["psi"],
            "Lam": p["Lam"],
            "Mb": spec["Mb"],
            "htr": p["htr"],
            "rotor": rotor,
        },
        "annulus": {"type": "aspect_ratio", "AR_row": [p["AR"]], "AR_gap": [1.0, 1.0]},
        "blades": [blade],
        "mesh": {"type": "ohp", "yplus": 30.0},
    }


def hold_Re(doc):
    """Set the viscosity of config `doc` so its row is at :data:`RE_SURF`.

    The Reynolds number is exactly proportional to one over viscosity at fixed
    geometry, so one step is enough, as in the `Re_surf` iterator.
    """
    from turbigen.config import Config

    machine = Config.from_dict(doc).design()
    Re = float(machine.Re_surf()[0])
    doc["fluid"] = dict(doc["fluid"], mu=doc["fluid"]["mu"] * Re / RE_SURF)
    return doc


def cases(n, seed):
    """Every case, by name: its config and the design it came from."""
    out = {}
    for machine in MACHINES:
        for i, p in enumerate(designs(machine, n, seed)):
            for rotor in (True, False):
                name = f"{machine}_{i:03d}_{'rotor' if rotor else 'stator'}"
                doc = hold_Re(config(machine, p, rotor))
                out[name] = {"doc": doc, "design": p}
    return out


#
# One case, in its own process
#


def geometry(machine):
    """What the blade looks like, to read the results against."""
    row = machine.rows[0]
    blade = row.blade
    ml = machine.mean_line
    chi = np.array([blade.evaluate_chi(s) for s in (0.0, 0.5, 1.0)]).reshape(3, 2)
    stagger = np.degrees(np.arctan(np.mean(np.tan(np.radians(chi)), axis=1)))
    span = machine.annulus.evaluate_span([1, 2])
    r_mid = float(np.mean(ml.r))
    cm = float(blade.evaluate_chord(0.5))
    s_cm = row.pitch * r_mid / cm
    return {
        "n_blade": int(row.n_blade),
        "chi": chi.tolist(),
        "chi_hub": chi[0].tolist(),
        "chi_mid": chi[1].tolist(),
        "chi_tip": chi[2].tolist(),
        "stagger_hub": float(stagger[0]),
        "stagger_mid": float(stagger[1]),
        "stagger_tip": float(stagger[2]),
        "s_c": float(s_cm * np.cos(np.radians(stagger[1]))),
        "span_ratio": float(span[1] / span[0]),
        "tip_gap": float(row.tip_gap),
    }


def run_one(folder, plot_all=False):
    """Design and mesh the case in `folder`, writing `result.json` there."""
    sys.path.insert(0, str(HERE))
    from ohp_mesh import measure_grid

    from turbigen.clusterfunc.exceptions import ClusteringException
    from turbigen.config import Config

    warnings.simplefilter("ignore", RuntimeWarning)
    folder = Path(folder)
    doc = yaml.safe_load((folder / "config.yaml").read_text())
    result = {"stage": "design"}
    try:
        cfg = Config.from_dict(doc)
        machine = cfg.design()
        result["geometry"] = geometry(machine)
        result["stage"] = "mesh"
        mesher = cfg.mesh
        t0 = time.perf_counter()
        # What Mesher.mesh does, short of the volume check and the wall
        # distance, so that a folded mesh is kept to be measured and drawn
        spacing = mesher.wall_spacing(machine)
        grid = mesher.forward(machine, spacing)
        grid.set_L_ref(mesher.L_ref(machine))
        grid.set_fluid(machine.mean_line.fluid)
        t_mesh = time.perf_counter() - t0
        result["stage"] = "measure"
        result.update(measure_grid(grid, t_mesh))
        result["stage"] = "done"
        s = result["summary"]
        worst = [s["worst_corner_main"], s["worst_corner_gap"]]
        worst = min(w for w in worst if w is not None)
        why = []
        if s["folded"]:
            why.append(f"{s['folded']} folded")
        if worst <= MIN_ANGLE:
            bad = [
                k for k, v in result["blocks"].items() if v["worst_corner"] <= MIN_ANGLE
            ]
            why.append(f"corner {worst:.1f} deg in {','.join(bad)}")
        result["status"] = "fail" if why else "pass"
        result["reason"] = "; ".join(why)
        if why or plot_all:
            result["stage"] = "plot"
            from ohp_plot import plot_grid

            spf, tip_j = mesher._span_grid(machine, spacing)
            plot_grid(
                grid,
                machine.annulus,
                spf,
                folder / "debug.pdf",
                MIN_ANGLE,
                tip_j,
                title=folder.name,
            )
            result["stage"] = "done"
    except (Exception, ClusteringException) as exc:
        # A clustering failure is a BaseException, so named
        result["status"] = "error"
        result["reason"] = f"{result['stage']}: {type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    (folder / "result.json").write_text(json.dumps(result, indent=1) + "\n")


#
# The driver
#


def launch(folder, timeout, plot_all=False):
    """Run one case in a child process; return its result."""
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    t0 = time.perf_counter()
    with open(folder / "log.txt", "w") as log:
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", str(folder)]
                + (["--plot-all"] if plot_all else []),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = None
    path = folder / "result.json"
    if path.exists():
        result = json.loads(path.read_text())
    elif code is None:
        result = {"status": "timeout", "reason": f"over {timeout:.0f} s"}
    else:
        result = {"status": "crash", "reason": f"exit code {code}, see log.txt"}
    result["t_wall"] = time.perf_counter() - t0
    return result


def _cell(v, fmt):
    return "-" if v is None else format(v, fmt)


def table(all_cases, results):
    head = (
        f"{'case':18s} {'status':8s} {'phi':>5s} {'psi':>6s} {'Lam':>5s} "
        f"{'htr':>5s} {'AR':>5s} {'s/c':>5s} {'stag':>6s} {'twist':>6s} "
        f"{'Nb':>4s} {'main':>6s} {'gap':>6s} {'t':>6s}  reason"
    )
    lines = [head]
    for name, res in results.items():
        p = all_cases[name]["design"]
        g = res.get("geometry", {})
        s = res.get("summary", {})
        twist = g["stagger_tip"] - g["stagger_hub"] if "stagger_tip" in g else None
        lines.append(
            f"{name:18s} {res['status']:8s} {p['phi']:5.2f} {p['psi']:6.2f} "
            f"{p['Lam']:5.2f} {p['htr']:5.2f} {p['AR']:5.2f} "
            f"{_cell(g.get('s_c'), '5.2f')} {_cell(g.get('stagger_mid'), '6.1f')} "
            f"{_cell(twist, '6.1f')} {_cell(g.get('n_blade'), '4d')} "
            f"{_cell(s.get('worst_corner_main'), '6.1f')} "
            f"{_cell(s.get('worst_corner_gap'), '6.1f')} "
            f"{_cell(s.get('t_mesh'), '6.1f')}  {res.get('reason', '')}"
        )
    counts = {}
    for res in results.values():
        counts[res["status"]] = counts.get(res["status"], 0) + 1
    lines.append("")
    lines.append(", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help="only the cases named")
    parser.add_argument("-n", type=int, default=4, help="designs per machine")
    parser.add_argument("-j", "--jobs", type=int, default=16, help="cases at once")
    parser.add_argument("--seed", type=int, default=0, help="Sobol' scramble seed")
    parser.add_argument("--timeout", type=float, default=900.0, help="[s] per case")
    parser.add_argument(
        "--out", type=Path, default=HERE.parent / "ohp_sweep", help="output dir"
    )
    parser.add_argument(
        "--plot-all", action="store_true", help="debug plots of passing cases too"
    )
    parser.add_argument(
        "--mesh",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a mesher setting on every case, YAML value; repeatable",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_one(args.worker, args.plot_all)
        return 0

    all_cases = cases(args.n, args.seed)
    sys.path.insert(0, str(HERE))
    import ohp_mesh

    for item in args.mesh:
        key, _, value = item.partition("=")
        for case in all_cases.values():
            ohp_mesh.set_option(case["doc"]["mesh"], key, yaml.safe_load(value))
    unknown = set(args.cases) - set(all_cases)
    if unknown:
        parser.error(f"no such case: {', '.join(sorted(unknown))}")
    names = args.cases or list(all_cases)

    folders = {}
    for name in names:
        folder = args.out / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "result.json").unlink(missing_ok=True)
        case = all_cases[name]
        (folder / "config.yaml").write_text(
            yaml.safe_dump(case["doc"], sort_keys=False)
        )
        (folder / "design.json").write_text(json.dumps(case["design"], indent=1))
        folders[name] = folder

    print(f"Meshing {len(names)} cases, {args.jobs} at once, into {args.out}")
    results = {}
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {
            pool.submit(launch, f, args.timeout, args.plot_all): name
            for name, f in folders.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            res = results[name] = fut.result()
            print(
                f"  {name:18s} {res['status']:8s} {res['t_wall']:6.0f} s  "
                f"{res.get('reason', '')}",
                flush=True,
            )
    results = {name: results[name] for name in names}

    text = table(all_cases, results)
    print("\n" + text)
    (args.out / "summary.txt").write_text(text + "\n")
    (args.out / "summary.json").write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
