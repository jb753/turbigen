# turbigen2 command-line interface

Plan for the CLI. **`design`, `mesh` and `run` are implemented.** The other verbs are
specified here so the shape is settled before they are written; each is marked
with its status.

## Why it looks like this

The existing `turbigen.main` dispatches subcommands by hand on `sys.argv[1]`,
so `turbigen --help` documents only the default command and the other two are
invisible. More seriously, it expresses *iteration* as a flag on the run
(`-I/--no-iteration`), which means there is no single unit to call twice: the
whole pipeline is written out in both branches of one `if`, and a second
unreachable copy of both branches sits at `main.py:371-463`, ninety-three lines
that have already drifted from the live ones.

Three principles follow from that, and from the fact that a turbigen2 `Config`
is frozen and holds no results.

**The verbs truncate the pipeline.** Each stage implies the ones before it, so
stopping early is a different verb rather than a flag that disables part of a
larger one. This also absorbs two existing flags that are really truncations in
disguise: `-S/--no-solve` means "everything but the solve", and `--mesh SPF`
means "build the mesh, plot it, and `sys.exit(0)` from inside the pipeline".

**Iteration composes runs; it is not a mode of one.** `iterate` calls `run` in a
loop, updating the design between calls. Keeping it separate is what forces
`run` to be a reusable unit and prevents the duplication above from recurring.

**Output and diagnostics are different channels.** Results go to stdout;
diagnostics go to stderr and, when there is somewhere to put it, a log file.
The existing code routes ordinary progress through `logger.warning` so that it
survives the level being lowered during iteration, which leaves genuine
warnings — `check_round_trip` reporting that `backward` omits a design
variable, or a mean line with relative flow angles approaching 90 degrees —
indistinguishable from the startup banner.

## Verbs

| verb | does | output dir | status |
|---|---|---|---|
| `design` | mean line, then annulus and blade geometry | optional | **implemented** |
| `mesh` | `design` + grid generation | optional | **implemented** |
| `run` | `mesh` + CFD + mix-out | **required** | **implemented** |
| `iterate` | repeated `run`, updating the design between calls | **required** | deferred |

The cut between `design` and `mesh` is where external tools and side effects
begin: everything up to and including blade geometry is pure computation on
numpy arrays.

## Output directories

An output directory is a property of the *verb*, not of the config file. It
does not appear in the YAML.

`design` and `mesh` are pure enough to run with nowhere to write, which is the
common case when experimenting with a design or driving it from a notebook.
`run` and `iterate` produce artefacts and require `-o`.

A *scratch* directory is a separate idea from an output directory: an
implementation detail of a tool, ephemeral and cleaned up. AutoGrid already
works this way (`autogrid/autogrid.py:496` calls `mkdtemp`), it is only
parented on a caller-supplied directory. Policy: scratch goes under
`<out>/mesh/` when there is an output directory and a temporary directory when
there is not; on failure, log the path and do not delete it.

## Shared options

Carried by a parent parser so every verb accepts them.

```
CONFIG_YAML                 configuration file
-o, --out DIR               write results here; a '*' is replaced by the next
                            free number, as in run_* -> run_0
                            (required for run and iterate)
-s, --set KEY=VALUE         override a config value; dotted key, integer
                            segments index into lists, value parsed as YAML;
                            repeatable
-v, --verbose               more diagnostics on stderr
-q, --quiet                 suppress results on stdout
-V, --version               print version and exit
```

`--set` is applied to the raw dict before `Config.from_dict`, so a mistyped key
is caught by the strict unknown-key check rather than silently ignored — an
improvement over the current behaviour, where the override is merged into a
dict nobody validates.

## Plugins

There is no flag and no config key. A `turbigen_plugins/` directory is found by
walking up from the directory holding the config file, as git looks for `.git`,
taking the first one that exists and stopping at the filesystem root. The
directory used is logged, so that both "why did this design load" and "why did
it not" are answerable from `-v` or the log file.

Anchoring the search on the config file rather than the working directory is
what makes a case directory portable: `case.yaml` and `turbigen_plugins/` can
be copied anywhere together. It also means a config written into an output
subdirectory re-runs unchanged, because walking up from `out_0/` reaches the
same plugin directory as the run that produced it. Recording a path in the
written config --- absolute, relative, or copied alongside --- turns out to be
unnecessary.

Directories owned by another user are skipped with a warning. The search
reaches every ancestor of the config file, which on a shared filesystem may
include directories the user does not control, and importing Python found there
would be running someone else's code.

Discovery runs only when a config is loaded, in `Config.from_file` and in the
CLI. Not on import, and not from the working directory. The registry's only job
is mapping a `type:` string to a class, so code that instantiates a design
directly --- a notebook, a test --- never needs it. And because a class
registers itself when its body executes, an ordinary `import` already
registers; `load_plugins(dir)` exists only for files that are not importable in
the normal way.

### Deliberately not carried over

`--edit`, which opens the config in `$EDITOR` before running. It can be made to
work ephemerally, via a temporary file, but it covers much the same ground as
`--set` while being the non-reproducible half: `--set` leaves a record of what
was changed in the command line, whereas an editor session leaves none unless
`-o` happens to capture the result. Easy to add back if it is missed.

## `design` — implemented

```
turbigen2 design case.yaml
turbigen2 design case.yaml -o run_* -s mean_line.psi=1.8
```

1. Discover and load plugins, walking up from the config file's directory.
2. Read the YAML into a dict.
3. Apply `--set` overrides.
4. Build the `Config` (strict: unknown keys and missing required fields raise).
5. `config.design()` — returns a `Machine`, stores nothing.
6. Print the machine tables to stdout.
7. If `-o` was given: create the directory, write the resolved config to
   `config.yaml`, and tee diagnostics to `log_turbigen2.txt`.
8. Exit 0.

Nothing is written and no directory is created without `-o`. The resolved
config includes every default, so an archived file reproduces its machine even
if a default later changes.

## `mesh` — implemented

```
turbigen2 mesh case.yaml
turbigen2 mesh case.yaml -o run_* -s mesh.resolution_factor=0.5
```

`design`, then `config.mesh.mesh(machine)`, then the machine tables and a block
summary. The steps and the `-o` behaviour are otherwise identical, and the same
config file serves both verbs: a `mesh:` section is simply ignored by `design`.

The grid itself is **not** written. How a mesh is serialised is a property of
the solver that will read it, so it belongs to `run`; until then `mesh` is for
checking that a design meshes and for seeing how large it comes out.

Still to add: `--plot SPF`, to render the mesh at a span fraction rather than
the old `--mesh SPF` flag that exits from inside the pipeline. It draws from
the returned grid, so it is a flag on the verb and not a field on the mesher.

## `run` — implemented

```
turbigen2 run case.yaml -o run_*
```

`prepare(config)` -- design, mesh, boundary conditions, initial guess -- then
solve. `prepare` is shared with `mesh` rather than written out again, so the
grid `mesh` hands back for inspection is the one `run` actually solves. The two
drifting apart is precisely what happened to `turbigen.main`.

`--out` is required, because a run produces artefacts worth keeping.

**Exit 2 when the solver did not converge**, with everything written anyway: a
diverged run is exactly the one whose output someone needs to look at, so
failing must not also discard the evidence. A separate code from 1 lets a
script driving a sweep tell "the solver did not converge" from "the config was
wrong" without parsing the log.

Convergence is divergence only for now. `ConvergenceHistory.check_convergence`
disables its residual-decay and residual-slope criteria at their defaults and
always checks divergence, so calling it bare is exactly that, and is the call
that grows thresholds later without a signature change.

The solution is then mixed out at each design station into `Result.actual`, and
written into the same `config.yaml` under a `result:` key. A failed mix-out is
logged and the run still writes everything else: a march that will not reduce
to a mean line is exactly the one whose output someone needs to look at.

## `iterate` — deferred

Specified only enough to keep the shape honest.

`iterate` owns only the loop, the numbered subdirectories, the convergence
table, and the collapse of intermediate directories at the end:

```python
guess = None
for i in range(max_iter):
    result = run(config, base / f"{i:03d}", guess=guess)
    guess = result.grid
    config, converged = update_all(config, result)
    if converged:
        break
```

Because `Config` is frozen, this is a fold: each iteration's config is a
distinct value that can be serialised and diffed for free, rather than the
current arrangement where iterators mutate one config in place and the CLI
copies `config.yaml` out of each numbered directory afterwards to reconstruct
the history.

That also changes what an iterator is. Today `iterator.update(config)` mutates
— `Deviation.set_independent` writes into `config.blades[i][0].camber` — and
`config.py:477` has to force `Incidence` to the front of the iterator list
because iterators that change geometry invalidate the grid for those that
follow. Against a frozen config an iterator becomes
`update(config, result) -> config`, and the ordering hazard goes with it.

## Entry point

```toml
[project.scripts]
turbigen2 = "turbigen2.cli:main"
```

A separate console script from `turbigen`, so the experiment can be installed
alongside the existing tool without displacing it.

No `sys.excepthook` is installed at all. `main()` catches exceptions around the
command and reports them as a message with exit code 1, showing the traceback
only under `-v`. Importing the CLI therefore has no effect on the process,
unlike `turbigen.main:35`, which replaces the global excepthook at module
scope.

## Testing

The CLI is thin and the work underneath it is already covered, so the tests
target the parts unique to the command line:

- `design` with no `-o` creates nothing — assert the working directory is
  unchanged;
- `design -o` writes a config that reads back equal to the one used;
- `--set` reaches the design, and a mistyped `--set` key fails;
- an unknown verb, a missing file, and a bad config each exit non-zero with a
  message on stderr rather than a traceback;
- `-q` silences stdout while leaving the exit code intact.
