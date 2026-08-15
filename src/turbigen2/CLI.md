# turbigen2 command-line interface

Plan for the CLI, now that every verb it specifies is implemented: `design`,
`mesh`, `run` and `iterate`.

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

**Everything a run says is one stream.** Results and diagnostics alike go
through the logging system, to stderr and, when there is somewhere to put it, a
log file — so the console and `log_turbigen2.txt` are the same transcript in
the same order. Nothing here is meant to be piped: the artefacts of a run are
its files, and stdout is left clean for a machine-readable mode should one ever
be wanted.

Results are ordinary `INFO` records rather than a level of their own. The
existing code routes them through `logger.warning` so that they survive the
level being raised during iteration, which leaves genuine warnings —
`check_round_trip` reporting that `backward` omits a design variable, or a mean
line with relative flow angles approaching 90 degrees — indistinguishable from
the startup banner. Here `--quiet` raises the level of the console handler
alone, so quietening a run does not also blank its record.

## Verbs

| verb | does | output dir | status |
|---|---|---|---|
| `design` | mean line, then annulus and blade geometry | optional | **implemented** |
| `mesh` | `design` + grid generation | optional | **implemented** |
| `run` | `mesh` + CFD + mix-out | **required** | **implemented** |
| `iterate` | repeated `run`, updating the design between calls | **required** | **implemented** |

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
-v, --verbose               more diagnostics on the console
-q, --quiet                 console shows warnings and errors only; a log
                            file written under --out still records it all
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
6. Log the machine tables, as every other stage logs what it produced.
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

`--restart` re-plots a previous run:

```
turbigen2 mesh out_0000/config.yaml --restart -o out_0000        # in place
turbigen2 mesh case.yaml --restart out_0000/restart.npz -o replot_*
```

The stored field goes back onto the grid and the standard plots describe it,
with no solve. Nothing needs to be kept beyond the config and the restart file
that `run` already writes, because re-designing and re-meshing costs seconds
against the minutes of the march it stands in for --- which is also why the
grid is not worth serialising.

Bare `--restart` reads `restart.npz` from the `--out` directory, which makes
re-plotting a run in place a flag rather than a path to type. Naming a file
still wins, so a field can come from anywhere.

The convergence history is looked for beside the *field*, not in the output
directory, so a restart named from elsewhere brings its own and a re-plot draws
the same pages the run did. A run writes it to `conv.cnv` --- ember's own
format, which is a pickle, so reading it is guarded: a history that will not
load costs the report its convergence page and nothing else.

Writing a report back into a run's own directory must not cost that run its
answer, so the resolved config is left alone when the file already holds
exactly it. Rewriting would replace the `result:` block --- the mixed-out mean
line, and whether it converged --- with the empty one a `mesh` produces.

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

## `iterate` — implemented

```
turbigen2 iterate case.yaml -o iter_* --max-iter 6
```

A mean line is a set of assumptions the CFD then contradicts: flow leaves a
blade less turned than the metal, arrives at an incidence, and loses more than
was allowed for. `iterate` measures each mismatch, corrects the design, and
solves again.

It owns only the loop. Each iteration is an ordinary `run` in a directory of
its own, chained so that iteration *k+1* starts from *k*'s flow field:

```
iter_0000/  config.yaml (with result: and its errors), restart.npz, conv.cnv, post.pdf
iter_0001/  ...
final -> iter_0001
```

**Every iteration is kept**, and `final` is a symlink to the last. The
alternative — the existing `main.py` copies the converged iteration over the
base directory and deletes the rest — destroys exactly the data that would let
a later fit predict these corrections instead of iterating for them.

### What an iterator is

Three layers, so that the physics, the arithmetic and any future learning stay
apart:

- an **iterator** declares the design variables it owns
  (`unknowns`/`with_unknowns`), measures the error they should null (`error`),
  and carries `gain`, `clip` and `tolerance`;
- the **stepper** assembles every iterator's knobs into one flat table, solves
  `B dx = -e` for the step, clips it, and decides convergence against the
  declared tolerances;
- an **estimator** owns which designs a knob can be predicted from, and how —
  see [Starting from designs already run](#starting-from-designs-already-run).

Because the iterators disappear into that table, a better step rule replaces
`iterate.step` alone and touches no iterator. That has already happened once:
the rule began as `u -= gain * e`.

`B` starts as the diagonal the gains already assert — `u -= gain * e` is a
Newton step under exactly that assumption — and gains a rank-one **Broyden**
update for every move the run has already paid for, so the off-diagonal terms
are learned from the trajectory rather than assumed away. With no history the
step is arithmetically identical to the old rule, so a first iteration is never
worse than it was; on a lower-triangular system of two knobs, which is the
structure a row feeding the next produces, twelve iterations become four.

The work is done in units of each knob's own tolerance. Degrees of recamber and
a loss coefficient otherwise share one Euclidean norm in the update, and a
least-change update in that norm would spend itself entirely on whichever
variable carried the larger numbers.

Only numbers are remembered: `converge` keeps each iterate's knobs and errors,
never the `Result`, which holds a live grid — gigabytes on a real machine, to
read a few dozen floats.

Three guards, each earning its place from something measured. An update is
skipped when the knob moved less than a quarter of a tolerance-equivalent step,
because the slope a secant infers has error of order `noise/du` and the same
deviation slope read +1.27 from a 200-step march and +0.3 from a 50-step one. An
ill-conditioned `B` falls back to the gains rather than propagating a NaN into a
design. And the clip stays, as the trust bound: a flat response — incidence
against leading-edge recamber is known to go flat and then flip — makes `B`
singular and the step large, and the clip reduces that to the old behaviour
rather than a wild excursion. There is no line search, because a rejected step
would cost a whole CFD solve.

`gain` carries the sign of the local sensitivity as well as its size: it is
negative for `incidence`, whose error falls as its knob rises. Both signs are
checked against CFD rather than assumed.

Knobs are disjoint by construction, so two iterators claiming one design
variable is refused at assembly, and applying them in any order gives the same
config. Because `Config` is frozen, each iteration's config is a distinct value
that serialises and diffs for free — where the existing `Deviation.
set_independent` writes into `config.blades[i][0].camber` in place, and
`config.py:477` has to force `Incidence` to the front of the list because
iterators that change geometry invalidate the grid for those that follow. That
ordering hazard does not exist here.

### Errors are recorded by every run

`error()` reads a solved grid, so it is measured inside `run` — always, whether
or not anything is iterating — and stored under `result: error:`. The incidence
onto a row and the exit angle it achieved are observations of the flow worth
keeping for their own sake, and recording them from the first run is what makes
an archive of design-and-mismatch pairs accumulate before anything reads it.

### The three ported

| name | knob | error |
|---|---|---|
| `deviation` | mean `dchi_TE` of each row | achieved exit `Alpha_rel` minus design |
| `incidence` | mean `dchi_LE` of each row | flow angle ahead of the leading edge minus metal angle, at one span fraction |
| `mean_line` | named mean-line design variables | design value minus what `backward()` recovers from the solution |

`DiffusionFactor` and `Repeat` are not ported: the first steps relatively and
moves blade count, which changes the mesh; the second's knob is a spanwise
profile.

### Starting from designs already run

The knobs start wherever the file leaves them, which is a waste when similar
machines have already been solved. An optional `database:` key names a glob of
finished case files, and the iterators start from a blend of the nearest:

```yaml
database:
  path: ../runs/**/config.yaml
```

That is the whole configuration. The design variables to measure distance in
are not declared but deduced --- whatever the finished runs *differ in*, less
the knobs themselves, which an iterator names through a third method,
`paths()`. The blend is inverse distance weighting, so it cannot return a
recamber no converged design ever used, and it decays to a copy of the single
nearest run rather than needing a minimum sample count.

It happens once, before iteration 0, and nothing else reads the key: `run` and
`design` are unaffected, and `converge` and `step` do not know it exists. A
run is a sample if it converged *and* its stored errors are within tolerance,
so the per-iteration directories of an earlier `iterate` are matched by the
glob and correctly ignored.

See [Starting from designs already
run](ARCHITECTURE.md#starting-from-designs-already-run) for why deduced rather
than declared, and why not the polynomial surrogate this replaces.

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
- `-q` silences the console while leaving the exit code, and the log file,
  intact.
