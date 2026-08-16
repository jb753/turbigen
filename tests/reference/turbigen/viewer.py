import fcntl
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger("turbigen")

_INDEX_HTML = """\
<div class="container-fluid" id="target"> </div>
<script src="https://cdn.jsdelivr.net/npm/dbslice/build/dbslice.min.js"></script>
<script>
    dbslice.start( "target" , "session.json");
</script>
"""

_SESSION = {
    "title": "turbigen",
    "metaDataConfig": {
        "metaDataUrl": "metaData.json",
        "metaDataCsv": False,
        "generateTaskIds": False,
        "taskIdRoot": "case_",
        "taskIdFormat": "d",
        "setLabelsToTaskIds": False,
    },
    "uiConfig": {
        "plotTasksButton": True,
        "saveTasksButton": False,
        "replaceTasksNameWith": "Cases",
    },
    "plotRows": [],
}


def _make_histogram(key: str) -> dict:
    return {
        "plotType": "cfD3Histogram",
        "data": {"property": key},
        "layout": {
            "title": key,
            "colWidth": 3,
            "height": 300,
            "highlightTasks": True,
        },
    }


def _make_trimesh_row(title: str, url_template: str) -> dict:
    return {
        "title": title,
        "plots": [],
        "ctrl": {
            "plotType": "threeTriMesh",
            "fetchData": {
                "urlTemplate": url_template,
                "buffer": True,
                "tasksByFilter": True,
                "maxTasks": 4,
            },
            "layout": {"colWidth": 3, "height": 350},
        },
    }


_CONVERGENCE_ROW = {
    "title": "Convergence history",
    "plots": [
        {
            "plotType": "d3LineSeries",
            "fetchData": {
                "urlTemplate": "run_${taskId}/convergence_err_mdot.json",
                "maxTasks": 10,
                "tasksByFilter": True,
                "autoFetchOnFilterChange": True,
                "dataFilterType": "lineSeriesFromLines",
            },
            "layout": {
                "title": "Mass flow error",
                "colWidth": 3,
                "height": 300,
                "highlightTasks": True,
                "cSet": "converged",
            },
        },
        {
            "plotType": "d3LineSeries",
            "fetchData": {
                "urlTemplate": "run_${taskId}/convergence_work.json",
                "maxTasks": 10,
                "tasksByFilter": True,
                "autoFetchOnFilterChange": True,
                "dataFilterType": "lineSeriesFromLines",
            },
            "layout": {
                "title": "Work",
                "colWidth": 3,
                "height": 300,
                "highlightTasks": True,
                "cSet": "converged",
            },
        },
        {
            "plotType": "d3LineSeries",
            "fetchData": {
                "urlTemplate": "run_${taskId}/convergence_loss.json",
                "maxTasks": 10,
                "tasksByFilter": True,
                "autoFetchOnFilterChange": True,
                "dataFilterType": "lineSeriesFromLines",
            },
            "layout": {
                "title": "Loss",
                "colWidth": 3,
                "height": 300,
                "highlightTasks": True,
                "cSet": "converged",
            },
        },
    ],
}


def _bootstrap_viewer(directory: Path, config=None):
    """Write index.html and session.json into directory if not already present."""
    index = directory / "index.html"
    if not index.exists():
        index.write_text(_INDEX_HTML)

    session_path = directory / "session.json"
    if not session_path.exists():
        session = dict(_SESSION)
        session["plotRows"] = list(session["plotRows"])

        ds = config and getattr(config, "design_space", None)
        if ds:
            keys = list(ds.independent.mean_line.keys())
            if keys:
                session["plotRows"].append(
                    {
                        "title": "Design variables",
                        "plots": [_make_histogram(k) for k in keys],
                    }
                )

        if config:
            session["plotRows"].append(
                {
                    "title": "Performance",
                    "plots": [_make_histogram(k) for k in ("eta_ts", "eta_tt")],
                }
            )
            session["plotRows"].append(_CONVERGENCE_ROW)
            mas_plots = []
            for irow in range(config.nrow):
                spf = config.blades[irow][0].spf
                spfi = spf[len(spf) // 2]
                mas_plots.append(
                    {
                        "plotType": "d3LineSeries",
                        "fetchData": {
                            "urlTemplate": f"run_${{taskId}}/Mas_row_{irow}_spf_{spfi}.json",
                            "maxTasks": 10,
                            "tasksByFilter": True,
                            "autoFetchOnFilterChange": True,
                            "dataFilterType": "lineSeriesFromLines",
                        },
                        "layout": {
                            "title": f"Row {irow + 1}",
                            "colWidth": 3,
                            "height": 300,
                            "highlightTasks": True,
                            "cSet": "converged",
                        },
                    }
                )
            session["plotRows"].append(
                {
                    "title": "Surface isentropic Mach",
                    "plots": mas_plots,
                }
            )
            for irow in range(config.nrow):
                i_exit = irow * 2 + 1
                session["plotRows"].append(
                    _make_trimesh_row(
                        f"Row {irow + 1} exit loss",
                        f"run_${{taskId}}/cut_{i_exit}_Yp.tm3",
                    )
                )

        session_path.write_text(json.dumps(session, indent=2))


def record_metadata(config):
    """Record case metadata into metaData.json, safe for concurrent processes."""
    flat = {}
    for k, v in config.design_vars_actual.items():
        if v is None:
            continue
        if np.isscalar(v) or np.ndim(v) == 0:
            flat[k] = float(v)
        else:
            for i, vi in enumerate(np.asarray(v).ravel()):
                flat[f"{k}{i + 1}"] = float(vi)

    categorical = {"converged": config.converged}
    record = {
        **flat,
        **categorical,
        "label": config.work_dir.name,
        "taskId": config.task_id,
    }

    directory = config.work_dir.parent
    path = directory / "metaData.json"
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        content = f.read()
        if content.strip():
            payload = json.loads(content)
            payload["data"].append(record)
        else:
            payload = {
                "header": {
                    "continuousProperties": list(flat.keys()),
                    "categoricalProperties": list(categorical.keys()),
                },
                "data": [record],
            }
        f.seek(0)
        f.truncate()
        json.dump(payload, f, indent=4)

    _bootstrap_viewer(directory, config)
    if config.grid:
        _write_Mas_json(config)


def _write_Mas_json(config):
    import turbigen.util_post
    from turbigen.post import calculate_nondim
    import ember.cut

    cuts = turbigen.util_post.cut_blade_surfs(config.grid, offset=0)

    for irow in range(config.nrow):
        spfrow = config.blades[irow][0].spf
        if spfrow is None:
            continue
        ml_row = config.mean_line.actual.get_row(irow)
        C = cuts[irow][0]

        for spfi in spfrow:
            xrc = config.annulus.get_span_curve(spfi)
            Ci = ember.cut.structured_meridional(C, xrc.T)[0]

            y = calculate_nondim(Ci, ml_row, "Mas")

            i_stag = turbigen.util_post.get_i_stag(Ci)
            zeta = turbigen.util_post.get_zeta(Ci)
            zeta_stag = zeta - zeta[i_stag]
            zeta_stag -= zeta_stag[np.argmin(y)]
            zeta_max = zeta_stag.max(axis=0)
            zeta_min = np.abs(zeta_stag.min(axis=0))
            zeta_norm = zeta_stag.copy()
            zeta_norm[zeta_norm < 0.0] /= zeta_min
            zeta_norm[zeta_norm > 0.0] /= zeta_max
            x = np.abs(zeta_norm)

            records = [
                {"x": float(xi), "y": float(yi)} for xi, yi in zip(x.ravel(), y.ravel())
            ]
            fname = config.work_dir / f"Mas_row_{irow}_spf_{spfi}.json"
            fname.write_text(json.dumps(records))
            logger.info(f"Written {fname.name}")
