# Step 3: TS3 write side — port `_run` input-writing to the ember `TS3Writer`

## Context

The turbigen `ts3` solver was broken against the deleted `turbigen.grid` API and
is being migrated onto ember piece by piece. Done so far: Step 1 (purge), Step 2
(result read-back via `ember.ts3.read_conserved`/`read_mu_turb`). The write side
is still stubbed — `_run` raises `NotImplementedError` (ts3.py:258-264), so no
real solve can run.

This step writes the TS3 input file from the (ember) grid + config by driving
`ember.ts3.TS3Writer` directly. The headline requirement: the config carries
**`av` and `bv` override dicts** for any TS3 variable not exposed as a
named/typed dataclass field, so power users can set anything without a code
change — while common knobs keep their typed, documented fields (which drive
`robust()`, the Sphinx docs, and the derived `nstep_save_start`).

## Config surface (D′ hybrid)

The grid arriving at `run()` is already an `ember.grid.Grid` (config.py:849).

**Keep typed fields** (already present, classified against `ember.ts3` defaults):
- av-backed: `cfl, dampin, facsecin, ilos, nchange, viscosity_law, nstep, rfmix,
  sfin` — forwarded by name via `writer.set_av(...)`.
- bv-backed: `fmgrid` — forwarded per block via `writer.set_bv(bid, ...)`.
- turbigen-domain: `nstep_avg`, `soft_start`, `nstep_soft`.

**Changes to the field set (per user):**
- **Drop `tvr`** — obsolete; the grid now carries `mu_turb` directly, so seeding
  `trans_dyn_vis` from a viscosity-ratio guess is no longer needed.
- **Drop `Lref_xllim` and `xllim`** — the mixing-length limit is now baked into
  `wdist` upstream (via `Grid.calculate_wdist(limit_pitch=...)`), and the
  `TS3Writer` defaults no longer clamp (`xllim=1e6`, `xllim_free=0.0`;
  ember/ts3.py:287-291). So turbigen sets neither field nor any `xllim` bv; the
  grid arriving at `run()` already carries the limit in `wdist`. Users wanting a
  manual cap can still use the `bv` override dict (`bv: {<bid>: {xllim: ...}}`).
- **Keep `rfin`** — applied to inlet patches as a `pv` (see below).

**Add two override dicts:**
```python
av: dict = None   # av[name] = value, applied with writer.set_av(**av)
bv: dict = None   # bv[bid][name] = value, applied with writer.set_bv(bid, **bv[bid])
```
`bv` already exists as a field; `av` is new. Both default `None` (→ `{}`).

### Overlap policy (per user: error on overlap)

Before writing, raise a clear error if any name set by a **non-default typed
field** also appears in the matching override dict — i.e. a typed av-field name
in `av`, or a typed bv-field name in any `bv[bid]`. "Non-default" is determined
by comparing the instance value to the dataclass field default
(`dataclasses.fields`). This forbids two competing sources for one variable.
(Note: `robust()`/`restart()` set typed fields to non-defaults, so an `av` dict
duplicating e.g. `cfl` will correctly error after `robust()` too.)

## Design — `_write_input(grid, ts3_config)` (replaces the `_run` stub)

New module-level helper in `turbigen/src/turbigen/solvers/ts3.py`, called from
`_run` before `_execute`. Mirrors the structure of `ember.ts3.write_ts3` but
drives `TS3Writer` so turbigen can layer its av/bv:

1. `writer = ember.ts3.TS3Writer(); writer.get(grid, strict=True)` — populates
   av/bv/bp/pa/pv from the grid (fluid props, coords, conserved, patches). This
   also writes total energy on Turbostream's zero datum (`roe`) automatically —
   `get_blocks` now does the re-expression internally (ember/ts3.py:887-899), so
   `_write_input` needs no `roe` handling. **`strict=True`**: turbigen's contract
   is to hand TS3 a complete, runnable grid, so a missing flow field, fluid, or
   periodic connectivity is a bug and must fail at write time (this also enforces
   that the grid carries `wdist`, which holds the baked-in mixing-length limit).
2. **typed av**: `writer.set_av(**{name: getattr(cfg, name) for name in _AV_FIELDS})`.
3. **derived av**: `nstep_save_start = nstep − nstep_avg`; `restart=1` (ember
   default already, set explicitly for clarity).
4. **typed bv** (`fmgrid`): per block, `writer.set_bv(bid, fmgrid=cfg.fmgrid)`.
   No xllim handling — the writer's non-clamping defaults stand and the limit is
   already in `wdist`.
5. **rfin (inlet pv)**: for each inlet patch, `writer.set_pv(bid, pid,
   rfin=cfg.rfin)` — a new ember seam mirroring `set_av`/`set_bv` (see Critical
   files / Open questions).
6. **raw overrides (after typed, validated for overlap first)**:
   `writer.set_av(**(cfg.av or {}))`; for each `bid`, `writer.set_bv(bid,
   **(cfg.bv or {}).get(bid, {}))`. `set_av`/`set_bv` already validate unknown
   names, type-cast, and reject NaN (ts3.py:724-768).
7. `writer.check(); writer.write(workdir/"input.hdf5");
   writer.write_probe_meta(...)`.

`_AV_FIELDS` / `_BV_FIELDS` are small module-level frozensets naming which typed
fields map where, used by both the forwarding and the overlap check.

### `robust()` / `restart()`

Unchanged in spirit; `robust()` already only touches typed fields that still
exist (`ilos, dampin, facsecin, sfin, cfl, fmgrid, soft_start`). Remove any
reference to dropped fields (none currently reference `tvr`/`Lref_xllim`).

## Critical files

- `ember/src/ember/ts3.py` — add `TS3Writer.set_pv(bid, pid, **kwargs)` mirroring
  `set_av`/`set_bv` (set-before-get guard, type-cast, NaN-reject). Validation:
  the valid pv names differ by patch kind and there is no `DEFAULT_PV`, so accept
  only keys **already present** in `self.pv[bid][pid]` (populated by
  `get_patches` per kind); reject unknown names. Cast to the existing value's
  type.
- `turbigen/src/turbigen/solvers/ts3.py` — drop `tvr`, `Lref_xllim`, `xllim`
  fields, add `av` field; add `_write_input`, `_AV_FIELDS`/`_BV_FIELDS`, the
  overlap check; replace the `_run` stub body with `_write_input(grid, cfg)` then
  `_execute` + `_read_hdf5`.

## Verification

In-process, no Turbostream binary (write the input file, read it back with the
Step-2 readers / `ember.ts3.read_ts3`). Reuse the grid fixtures from
`turbigen/tests/test_ts3_solver.py` (`_make_solved_grid`) and the ember
`_make_grid`/round-trip patterns.

Extend `turbigen/tests/test_ts3_solver.py`:
1. **write→read av/bv round-trip**: build a grid+cfg, `_write_input` to a tmp
   workdir, reopen `input.hdf5` (h5py) and assert representative av (`cfl`,
   `nstep`, derived `nstep_save_start`, `restart`) and bv (`fmgrid`) match the
   config.
2. **rfin pv**: assert `rfin_pv` on an inlet patch in the file equals `cfg.rfin`.
3. **raw override applies**: `cfg.av = {"sfin_sa": 0.1}` (a name with no typed
   field) → assert `sfin_sa_av` in the file equals 0.1; `cfg.bv = {0: {"fmgrid":
   0.0}}`… but that overlaps `fmgrid` → instead use a non-typed bv name.
4. **overlap errors**: `cfg.av = {"cfl": 0.2}` (typed field) raises; same for a
   typed bv name in `cfg.bv`.
5. **`_run` no longer raises NotImplementedError**: monkeypatch `_execute` and
   `_read_hdf5` to no-ops and assert `_run` writes `input.hdf5`.

Also add an ember-side test for `set_pv` (reject unknown name, type-cast,
set-before-get guard), alongside the existing `set_av`/`set_bv` tests.

Run: `cd turbigen && uv run python -m pytest tests/test_ts3_solver.py -q`.

## Resolved decisions

- **zero-datum `roe`**: no longer a question — `get_blocks` writes `roe` on the
  TS3 datum internally (ember/ts3.py:887-899), so `_write_input` does nothing.
- **`rfin` seam**: add `TS3Writer.set_pv` to ember (decided). With the zero-datum
  and xllim pokes both gone, a direct `writer.pv[...]` reach would be the *only*
  such hack left and no longer matches upstream `write_ts3` (which now pokes the
  writer nowhere). `set_pv` keeps the av/bv/pv seam uniform; validate against
  existing keys (no `DEFAULT_PV`, valid names are per-patch-kind).

## Out of scope (later)

- Step 4: log-parsing regression test + prune the 4 unused regexes
  (`re_ncycle`, `re_nstep_cycle`, `re_nstep_save_start`, `re_eta`).
- Step 5: execution/orchestration (`_execute`, HPC group/SLURM) — HPC-only,
  validated on-cluster.
- Probe `.dat` post-run caching (commented-out TODO in `_run`).
