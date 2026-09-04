# turbigen command-line interface

Plan for the CLI, now that every verb it specifies is implemented: `design`,
`report`, `run`, `iterate` and `batch`.

## Why it looks like this

The existing `turbigen.main` dispatches subcommands by hand on `sys.argv[1]`,
so `turbigen --help` documents only the default command and the other two are
invisible. More seriously, it expresses *iteration* as a flag on the run
(`-I/--no-iteration`), which means there is no single unit to call twice: the
whole pipeline is written out in both branches of one `if`, and a second
unreachable copy of both branches sits at `main.py:371-463`, ninety-three lines
that have already drifted from the live ones.

Three principles follow from that, and from the fact that a turbigen `Config`
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
log file — so the console and `log_turbigen.txt` are the same transcript in
the same order. Nothing here is meant to be piped: the artefacts of a run are
its files, and stdout is left clean for a machine-readable mode should one ever
be wanted.

Results are ordinary `INFO` records rather than a level of their own. The
existing code routes them through `logger.warning` so that they survive the
level being raised during iteration, which leaves genuine warnings —
`_check_round_trip` reporting that `backward` omits a design variable, or a mean
line with relative flow angles approaching 90 degrees — indistinguishable from
the startup banner.

There is no `--quiet`. It was there to make a loud run bearable, but the one
verb that is genuinely too loud is `iterate` — tens of runs, each printing its
tables in full — and that quietens the console by *logger name* on its own,
which is strictly better: it keeps warnings, and it keeps the log file
complete. Everything else is one run's worth of output, and a shell already
knows how to redirect it.

## Verbs

| verb | does | writes | status |
|---|---|---|---|
| `design` | mean line, then annulus and blade geometry | **never** | **implemented** |
| `report` | `design` + grid + any stored field, drawn | `post.pdf`, `output.yaml` | **implemented** |
| `run` | `report` + CFD + mix-out | everything | **implemented** |
| `iterate` | repeated `run`, updating the design between calls | everything | **implemented** |
| `chic` | `iterate`, then repeated `run` off design to the limit | everything | **implemented** |
| `batch` | write configs covering a design space, running none | a `batch_NNNN` | **implemented** |

The cut between `design` and `report` is where external tools and side effects
begin: everything up to and including blade geometry is pure computation on
numpy arrays, and `design` is the verb you can run anywhere without
consequence.

**A verb is named for what you get, not for the stage it stops at.** The verb
before `run` used to be called `mesh`, and could not produce a mesh: the grid is
never serialised, because how a mesh is serialised is a property of the solver
that will read it. Its only durable output was a PDF, and its documented main
use was re-plotting a finished run --- so it was a report verb named after an
implementation detail of getting there.

## Where output goes

**Beside the config it was given, as `output.yaml`.** Nothing to derive and
nothing to type:

```
hiload/input.yaml -> hiload/{output.yaml, restart.npz, conv.cnv, post.pdf,
                             log_turbigen.txt}
```

Two rules make that safe, and each removes a check that would otherwise be
needed.

**The written name is never a name we were given.** A config file is therefore
never a candidate for being overwritten. This matters more than it sounds: what
a run writes is the *resolved* config with every default expanded, so writing
it over a hand-kept file would turn twenty commented lines into two hundred and
lose every comment to the safe loader.

The rule used to hold by construction, because nothing read an `output.yaml`
and wrote one back. `report` does both, so it is now enforced instead: **no
verb will run on a file named `output.yaml`**, `design` included. An
`output.yaml` only exists because some other config was run, and that config is
still there --- `run` leaves the file it was handed, whatever it was called,
and `batch`, `iterate` and `chic` write an `input.yaml` into every directory
they invent. So there is always another file naming the same directory, and
wanting this one is not a case to support.

One flat rule rather than an exemption for `design`, which writes nothing and
could read one safely. An exception is one more thing to remember, and it would
buy only the ability to type a name that always has an equivalent beside it.

**One directory is one run.** Artefact names are fixed, so two configs run in
one directory would collide; avoiding that is the user's business, and
`batch` writes a directory per member precisely so that it never happens by
accident.

`design` never writes and needs no flag saying so; `report` always writes,
because a report with nowhere to put its figures is not a thing anyone wants.
There is no `--write`: it was a boolean left over from `-o DIR`, and once the
location stopped being a choice it carried nothing.

**Only a verb with an answer writes a `result:`.** `report` writes
`output.yaml` too --- it is the only way to get a resolved config without
paying for a solve, `design` having promised to write nothing --- but the
config half and the answer half are governed separately.

A report reaches an answer only when the `restart.npz` beside it is stamped as
the solution to the design it just resolved. The stamp is a digest of the
sections that decide what a field *is*: fluid, mean line, annulus, blades,
mesh, operating point, inlet profile. `solver` is out, because it decides how
the answer was reached rather than what it is, and raising the step count must
not invalidate a good field.

With a match, the report mixes out and records exactly what the run recorded.
Without one --- a moved design, an unstamped field, no history to judge
convergence by --- it has no answer of its own, and then **it never removes the
one already there**: it writes the config alone if there is nothing to lose,
and otherwise leaves the file untouched and says so.

Note where the stamp is *not* checked. Applying a field asks whether it is a
useful place to start, and the answer is usually yes even when the design has
moved: `iterate` chains each pass off the last, `chic` walks its operating
point along, `database` warm-starts from a neighbour. A mismatch is the normal
case there. Only the report asks the strict question, because only the report
writes an answer it did not itself march to.

### Replacing an answer, and `-o`

**Anything that would replace a recorded answer refuses**, whatever the number
of targets. `-f` is the honest spelling of "yes, replace it".

There used to be a count in this rule: one target overwrote silently, several
refused. The reason given was that a batch is cluster hours whose loss is
discovered a day later, while one re-run is recoverable --- but how many paths
are on the command line is a poor proxy for how much is at stake, and it is not
what "did I mean all of these" measures either. It also made `--force`
advertised by verbs that do not have it, `design` among them, which is a dead
end rather than a safeguard.

The flat rule only works because **`-o DIR` runs a config somewhere new**:

```
turbigen run case/input.yaml -o runs/v2
```

The config is copied into the workdir as `input.yaml`, with its includes
expanded and any `--set` applied, and the run happens there. A variant
therefore goes somewhere that has no answer in it, and refusing to overwrite
stays rare enough to be worth doing --- which is what makes `-f` mean something
when you do type it.

**`-o` moves the directory; it does not split it.** The copy becomes the
target, so config and output stay together and no verb learns about the flag.
That colocation is load-bearing: `report` finds `restart.npz` beside the
config, plugin discovery walks up from it, and `iterate` and `chic` write an
`input.yaml` into every directory they invent. A flag that put output somewhere
else would break all three. What is written is the *document* --- what you
asked for, not what it expands to --- because the expanded version is
`output.yaml`, sitting next to it.

On `run`, `iterate` and `chic` only. `design` writes nothing, `batch` numbers
its own directories, and a report into an empty directory would have no field
to find. Several targets with `-o` refuse, one directory being one run.

This is `workdir:` returning as a flag rather than a config key, which is the
same split `--queue` makes: the file says how a thing is done, the command line
says where and whether.

**A `%` in the last part of the path is the next free number:**

```
turbigen run case/input.yaml -o runs/v%      # runs/v0000, then runs/v0001
```

One spelling, not the `%`-versus-`*` pair the old package took, where which of
them numbered a run was a thing to remember rather than work out. It goes where
the `%` is, so `x%_hot` numbers in the middle, and more than one, or one
outside the last component, is refused rather than guessed at.

Numbering carries on from the highest that exists rather than counting how many
there are, so a run deleted from the middle cannot let the next one take a
later one's number. That is the property `batch` needed first, and both now
call the same `next_numbered_dir`.

Worth seeing what this does to the rule above: a numbered workdir is free by
construction, so it can never hold an answer and there is nothing for `-f` to
be needed for. Numbering and `-f` are the two ways of not losing a run, and
asking for one means never reaching for the other.

`-f` is still wanted for the third thing a workdir can already hold. A config
sitting there with no answer beside it yet --- an unrun batch member, or
something being drafted --- is nobody's to replace, so `-o` pointed at one
refuses. Documents are compared rather than text, so re-running the same config
into the same directory after a failure is silent, which is the case where
insisting would be pure noise.

**A run that fails keeps its workdir**, holding the config it tried and the log
saying how far it got, and a numbered one keeps its number. The transcript of a
failure is the most useful thing in that directory at that moment, and deleting
it to keep the numbering tidy would throw the evidence away for the sake of the
filing. A number is cheap.

### Batches are numbered, not named

`batch` writes into the next free `batch_NNNN` beside its datum config.
Numbering carries on from the highest that exists rather than counting how many
there are, so a deleted batch in the middle does not make the next one
overwrite a later one, and nothing is ever written into an existing batch.

No `-o` here. A batch is many runs, so a single workdir would mean nothing, and
the rule is the same one every other verb follows --- output goes beside the
input. What that costs is sending a batch to a different filesystem, which is a
real loss, and the constraint `run` imposed on output a thousand times larger
until `-o` gave it a way out: put the config where you want the output, or name
where you want it copied to.

The numbering is the same `next_numbered_dir` that `-o %` uses, with
`batch_` for a prefix.

It also makes the layout record which datum produced which batch. Nothing else
does: `--continue` cannot tell that the datum or its bounds changed between
batches, and nothing is written at a batch root to say what generated it.

A *scratch* directory is a separate idea again: an implementation detail of a
tool, ephemeral and cleaned up. AutoGrid already works this way
(`autogrid/autogrid.py:496` calls `mkdtemp`), it is only parented on a
caller-supplied directory. Policy: scratch goes under `<out>/mesh/` when there
is an output directory and a temporary directory when there is not; on failure,
log the path and do not delete it.

## Shared options

Carried by a parent parser so every verb accepts them.

```
CONFIG_YAML...              one or more configuration files; several run one
                            after another, or submit together with --queue
-s, --set KEY=VALUE         override a config value; dotted key, integer
                            segments index into lists, value parsed as YAML;
                            repeatable
-v, --verbose               more diagnostics on the console
-V, --version               print version and exit
```

And on `run` and `iterate`, the two that spend real time and can lose real
answers:

```
--force                     overwrite an existing output.yaml when several
                            config files are given
-Q, --queue                 submit to the queue the job: section names,
                            instead of running here
--restart [NPZ]             start the solve from a stored field
```

Everything else that could be a flag is a *value*, and values live in the
config where `-s` can reach them: `-s iterate.max_iter=6`, `-s job.hours=12`. The
division is between **actions**, which must be typed deliberately and must
never be latent in a file, and **values**, which belong in the file so that an
archived case records what it was run under. `--max-iter` was on the wrong side
of that line and has moved.

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

## Includes

A top-level `include:` names files whose keys are merged underneath this one's,
so a site's `job:`, a standard `solver:` and a family's `mesh:` are written once
and shared.

```yaml
include: [../site/cluster.yaml, solver.yaml]
mean_line: {...}
solver:
  n_step: 100          # keeps the rest of the included solver block
```

**One rule.** Later beats earlier, the including file beats everything it
includes, and mappings merge exactly one level deep. Depth one is where it
stops on purpose: it covers overriding one setting of an included block, while
refusing to splice two mappings that declare different `type:` keys into a node
with `psi` from one design and `span` from another. Lists replace wholesale,
because merging `blades:` by index never gives the row you meant.

**Files are found relative to whoever named them**, never the working
directory, and an include named by an included file resolves against *that*
file's directory. Same anchoring as plugin discovery and for the same reason: a
case directory, its `turbigen_plugins/` and its fragments copy anywhere
together, and the same file cannot design two different machines depending on
where you invoked it from. The package this replaces tries the working
directory first and warns when that shadows something.

Includes may include, depth-first. A diamond — two files including a third —
resolves; a loop raises naming the chain.

**Ambiguity is refused, not resolved**, in the two places it arises. A key
written twice in one file raises, at any depth, because every YAML loader keeps
the last one and says nothing, so the earlier is a setting that goes quietly
missing. And two files in the *same* `include:` list both setting a top-level
key raises, because equals have no precedence worth relying on — where a file
overriding what it includes is a hierarchy, and is what the merge depth is for.

The old package wrote the duplicate-key check and then wired it to
`read_yaml_list` alone, so no config has ever been checked by it.

**An included `result:` is dropped**, with a debug line. Including a finished
`output.yaml` to inherit its design is a fair thing to want; inheriting the
answer it achieved is not, and `database` decides what counts as a sample by
reading `result:`, so one inherited answer would poison every warm start that
globs it.

Resolution happens before `--set`, so an override is the last word and applies
to the assembled document rather than to whichever fragment defined the key.
`include:` is popped on the way in, so it never reaches the strict unknown-key
check — and what a run writes is the expanded document, because an archived
case records what it ran rather than pointing at files that may since have
changed.

### Deliberately not carried over

`--edit`, which opens the config in `$EDITOR` before running. It can be made to
work ephemerally, via a temporary file, but it covers much the same ground as
`--set` while being the non-reproducible half: `--set` leaves a record of what
was changed in the command line, whereas an editor session leaves none unless
`-o` happens to capture the result. Easy to add back if it is missed.

## `design` — implemented

```
turbigen design case/input.yaml
turbigen design case/input.yaml -s mean_line.psi=1.8
```

1. Discover and load plugins, walking up from the config file's directory.
2. Read the YAML into a dict.
3. Apply `--set` overrides.
4. Build the `Config` (strict: unknown keys and missing required fields raise).
5. `config.design()` — returns a `Machine`, stores nothing.
6. Log the machine tables, as every other stage logs what it produced.
7. Exit 0.

**Nothing is written, ever.** This is the verb to run while changing a number
and watching the tables move, so it must be safe to point anywhere. Anything
worth keeping comes from `report`, whose output is its point.

## `report` — implemented

```
turbigen report case/input.yaml            # draw what the case supports
turbigen report hiload/input.yaml          # re-plot a finished run
turbigen report hiload/iter_0003/input.yaml   # or one pass of an iterated one
```

`design`, then the grid if the config says how, then any `restart.npz` a
previous run left beside the config, then `post.pdf` and `output.yaml`. One
command, no flags, whatever depth of case you point it at:

| the config has | the report has |
|---|---|
| a mean line | nothing to draw, so no PDF --- but the resolved config |
| an annulus and blades | the geometry pages |
| a `mesh:` section | those, plus the grid |
| a field beside it | those, plus the flow and the convergence page |
| a field that is *this* design's | those, plus the answer recorded |

Each standard processor draws nothing when what it needs is absent, so there is
no mode to select. **Re-plotting a finished run is the same command as plotting
a fresh one** — no `--restart`, because a report of a run that has a field
always wants it and there is nothing else it could mean.

The grid itself is **not** written. How a mesh is serialised is a property of
the solver that will read it, so it belongs to `run`. Re-designing and
re-meshing to put a stored field back costs seconds against the minutes of the
march it stands in for, which is why serialising it would not repay the
trouble.

`output.yaml` is, and it is the only way to get a resolved config onto disk
without paying for a solve. Whether it carries a `result:` depends on the stamp
on the field beside it, and a report never removes an answer it cannot
reproduce — see [Where output goes](#where-output-goes).

Re-plotting therefore names the *input*, not the `output.yaml` the run wrote:
what a run writes is not a file to hand back. Every directory a run happened in
holds one, `iterate` and `chic` writing an `input.yaml` into the directories
they invent so that a single pass or a single operating point stays addressable
on its own.

The convergence history is looked for beside the field, in `conv.cnv` ---
ember's own format, which is a pickle, so reading it is guarded: a history that
will not load costs the report its convergence page and nothing else.

Still to add: `--plot SPF`, to render the mesh at a span fraction rather than
the old `--mesh SPF` flag that exits from inside the pipeline. It draws from
the grid the verb already has.

## `run` — implemented

```
turbigen run hiload/input.yaml
turbigen run hiload/input.yaml -o runs/v2       # a variant, somewhere new
turbigen run hiload/input.yaml -o runs/v%       # the next free runs/vNNNN
turbigen run batch_0000/*/input.yaml            # serially, here
turbigen run batch_0000/*/input.yaml --queue    # as one submission
```

`prepare(config)` -- design, mesh, boundary conditions, initial guess -- then
solve. `prepare` is shared with `report` rather than written out again, so the
grid a report draws is the one `run` actually solves. The two
drifting apart is precisely what happened to `turbigen.main`.

Everything lands beside the config, so there is nothing to name -- or beside
the copy `-o` makes, which is how a variant is run without replacing the answer
already there. A bare `--restart` still means the field beside the config *you
named*, not the workdir being created, so continuing from what you have while
writing somewhere new is the two flags together and nothing more.

Several config files run one after another, which is the bash loop this saves
you writing, or become one submission with `--queue`.

**An exception stops the whole invocation, and the targets behind it do not
run.** A config that will not load, a design that will not close, a mesh that
cannot be built: these say the set is wrong rather than that one member of it
is unlucky, and stopping while the message is on the screen beats burying it
under the thirty that followed. A solve that merely diverges is different --
that is an exit 2, and the next target still runs, because a diverged march is
an answer about that design rather than a reason to doubt the rest. Resume by
fixing the config and running the rest; the finished ones refuse to be redone,
which is what makes that safe.

`--queue` does not work this way. It submits every path without loading any but
the first, so the same command queued gets answers for the good members. That
is the difference between validating locally and handing work to a scheduler,
not an inconsistency waiting to be ironed out.

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
written into `output.yaml` under a `result:` key. A failed mix-out is
logged and the run still writes everything else: a march that will not reduce
to a mean line is exactly the one whose output someone needs to look at.

## `iterate` — implemented

```
turbigen iterate hiload/input.yaml
turbigen iterate hiload/input.yaml -s iterate.max_iter=6
```

A mean line is a set of assumptions the CFD then contradicts: flow leaves a
blade less turned than the metal, arrives at an incidence, and loses more than
was allowed for. `iterate` measures each mismatch, corrects the design, and
solves again.

It owns only the loop. Each iteration is an ordinary `run` in a directory of
its own, chained so that iteration *k+1* starts from *k*'s flow field:

A design that **settles** ends up reading as a `run` directory, because that is
what it now is:

```
hiload/input.yaml
hiload/output.yaml     the converged answer, restart.npz, conv.cnv, post.pdf
hiload/iter_0000/      input.yaml, output.yaml, conv.cnv
hiload/iter_0001/      input.yaml
```

The last iteration's artefacts are **moved** to the root, so `output.yaml`
means "what this run achieved" whichever verb produced it and a database glob,
a script reading a result and a `--restart` need not know whether a design took
one solve or six. Its own directory is left holding the config that produced
it, that answer having become the run's.

The iterations before it keep their config, their answer and their march --- a
few kilobytes each, and the only record of how the design moved --- and lose
the flow field and the report drawn from it, which are the megabytes. Nothing
reads those again: `database` filters an unsettled iteration out by definition,
`chic` reads only the config it was given, and no code globs `iter_*` at all.
Both are rebuilt by re-running that iteration's `input.yaml`, which is what
makes this a tidy-up rather than a loss, and which was not true until each
iteration recorded its own config.

A design that **does not settle keeps every iteration whole, and nothing is
promoted.** That is exactly when the history is what you came to look at. It
also gives the root a meaning worth having: **an `output.yaml` there means this
design converged**, rather than "here is wherever the iteration happened to
stop". Restarting from an unsettled run means naming the iteration, which is
the honest position when there is no converged field to point at.

Moved rather than copied or linked. A copy is megabytes duplicated and two
files free to disagree. A symlink avoids both --- and this was three symlinks
for a while --- but needs a filesystem that has them, and leaves one answer
reachable by two paths, which `database` duly counted twice: the settled
iteration is precisely the one that survives its filters. Nothing turbigen
writes is a symlink now.

The alternative at the other extreme — the existing `main.py` copies the
converged iteration over the base directory and deletes the rest — destroys
exactly the data that would let a later fit predict these corrections instead
of iterating for them. Keeping the yamls keeps that.

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

The clip **scales** the step rather than clipping each knob separately. The
difference only shows when more than one knob is over its limit, and then it is
the difference between a shorter step and a different one: clipping each
component projects the step onto a corner of the box, keeping the sign pattern
of the direction and none of its shape. That cycles. Two runs of a two-knob
`loading` iterator sat in a period-2 orbit doing exactly this — alternating
steps of `+(0.1, -0.1)` and `-(0.1, +0.1)`, each overshoot provoking the
opposite corner, the design returning to where it had been two iterations
before — and Broyden could not escape it, every move being collinear with the
last. Scaling keeps the direction; the binding knob still moves exactly its
clip, and the others move less than they asked for, which is what going the
right way costs.

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
| `loading` | two interior Bernstein camber coefficients of one row | where the suction peak sits and the duty-normalised leading-edge Mach number, minus what was asked for |
| `peak_Ma` | the circulation coefficient of one row | peak Mach number over the trailing edge value, minus what was asked for |

`DiffusionFactor` and `Repeat` are not ported: the first steps relatively and
moves blade count, which changes the mesh; the second's knob is a spanwise
profile.

### Shaping the loading, and why two knobs

`deviation` and `incidence` correct the *ends* of a blade. `loading` corrects
what happens in between — where the suction peak sits and how hard the leading
edge accelerates — by moving the interior coefficients of a `bernstein` camber
line, whose endpoint coefficients are pinned so the metal angles stay put and
the three cannot fight.

Three knobs, and the number is physics rather than convenience. At a fixed duty
the area enclosed by the isentropic Mach loop is the blade circulation, which
the pitch sets; a camber line with pinned ends redistributes that area along the
chord without changing it. So of the three numbers describing a two-line
shape — a front value, a peak value, and where the peak sits — one is spoken for
at fixed blade count, and only **two are reachable**.

Targeting two and letting the third float is what this first did, and it does
not work: the level and the shape are bound by that same constraint, so an
uncontrolled level drifts and drags the shape with it. Sweeping the circulation
coefficient over 0.6, 0.7 and 0.8 at one fixed pair of shape targets, only the
value the targets had been calibrated at converged; the outer two diverged or
went flat, because the blade's natural peak position moves with loading
(0.638, 0.610, 0.571 at those three) and two camber coefficients cannot drag it
back.

So the circulation coefficient becomes the third lever and the peak height the
third target, which lifts the constraint and makes the system square.

**Two members, not one, because a gain carries one sign.** `gain` states the
sign of a knob's sensitivity as well as its size, and these knobs disagree: the
peak rises with the circulation coefficient where the shape targets fall with
the camber coefficients. Folded into a single iterator, one scalar gain drove
the count the wrong way at every iteration — `Co` walked from 0.70 to 0.57
while the peak it was meant to raise fell with it. Split into `loading` and
`peak_Ma`, each declares the sign it has, and the stepper merges them into one
table so nothing about the coupling is lost. They are configured together and
are of little use apart.

The three targets are `zeta_peak`; `fac_front`, which is
`Ma(zeta_front) / Ma_TE * Ma_2 / Ma_1` — Clark's third parameter, referred to
the trailing edge because that is a mean-line quantity fixed by the duty, and
carrying the `Ma_2 / Ma_1` factor so the same number means the same style of
leading edge across rows of different duty; and `fac_peak = Ma_peak / Ma_TE`,
which is one more than the diffusion factor `metrics: diffusion_factor` records
— the same measurement, through `turbigen.loading`, so a report cannot
contradict what a design was iterated onto.

That metric reports both peaks. `Mas_max` and `zeta_max` come from a maximum of
the data and exist for every blade; `zeta_peak`, `fac_front` and `fac_peak` come
from the fit and are NaN on a blade that accelerates all the way to its trailing
edge and so has no interior peak to place. `DF` is built from the maximum, so
every blade still gets one.

Moving blade count is what stopped `DiffusionFactor` being ported, because it
changes the mesh. It still does — the mesher sizes the grid from the pitch, so
`Co` 0.70 and 0.75 mesh at 225 and 209 streamwise nodes on the example
cascade — which puts a floor on `atol_peak` but does not prevent the loop, since
every iteration remeshes and restarts by index-space interpolation anyway. The
integer blade count is the smaller worry it looks: on that cascade one blade is
0.36% of `Co`, far finer than any step taken, though it scales as `1/N_blade`
and would matter on a row with forty.

The window starts at `zeta_front`, a fifth of the way along the surface by
default rather than a tenth: two straight lines have to be a fair description of
what they are fitted to, and a window reaching inside the sharp acceleration
round the nose is not one — on a cascade measured here the fitted front line and
the data disagreed by 0.08 in Mach fraction starting at 0.1, and by 0.03
starting at 0.2.

The measurement is the surface distribution `post: surface` draws, through the
same functions, so what is iterated to is what the report shows — and the report
overlays the target on it. Where the peak sits comes from a least-squares fit of
two straight lines meeting at a breakpoint, not from an argmax: the peak then
falls out as the intersection of two lines each fitted over many points, which
is what keeps it steady on the flat-topped profiles that are a design style
rather than a pathology.

### Starting from designs already run

The knobs start wherever the file leaves them, which is a waste when similar
machines have already been solved. An optional `database:` key names a glob of
finished case files, and the iterators start from a blend of the nearest:

```yaml
database:
  path: ../runs/**/output.yaml
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

## `--queue` — implemented

Where work executes is orthogonal to how far down the pipeline it goes, so it
is a flag on `run` and `iterate` rather than a verb of its own. `design <
report < run < iterate` truncate the pipeline; a queue does not sit anywhere in that
order.

```yaml
job:
  type: slurm
  hours: 4.0
  partition: ampere
  gres: "gpu:1"
```

```
turbigen run batch_0000/*/input.yaml -Q
turbigen run batch_0000/*/input.yaml -Q -s job.hours=12
turbigen iterate hiload/input.yaml -Q -s job.type=tsp
turbigen batch datum/input.yaml -n 32 -Q
```

**The key says how, the flag says whether.** That is the whole difference from
the package this replaces, which submits whenever a `job:` key is present, so a
run re-execs itself and every entry point needs `--no-job` to break the
recursion. Here the escape hatch is not typing `-Q`.

**Per-invocation overrides need no new syntax**, `-s job.hours=12` being the
same mechanism as every other override, so there are no queue flags beyond
`-Q` itself and nothing new to remember.

`type: slurm` submits every target as one array; `type: tsp` queues them
through task-spooler.

On `run` and `iterate`, `-Q` submits the targets you named, as the verb you
typed, carrying this invocation's options. On `batch` it submits the members
it has just written, as the verb the datum implies, carrying none — see
[`batch`](#batch--implemented). A third backend is a plugin class registering its own
`type:`, needing no change here. See [Where a run
executes](ARCHITECTURE.md#where-a-run-executes).

Nothing wraps `tsp -l`, `squeue` or `scancel`: they are better than anything we
would put in front of them.

## `chic` — implemented

```
turbigen chic case/input.yaml
```

Converge the design, then hold its geometry fixed and step the back pressure
until a point will not converge — halving the step and coming back at it from
the last good field, until the limit is pinned to `chic.step_min`.

```yaml
chic:
  step: 0.05        # DP_adjust per point
  step_min: 0.01    # the resolution of the answer
  max_points: 20
```

**Whether it iterates first is inferred, never asked for.** A case whose stored
`result:` says the design has converged *and* its iterators are inside their
tolerances sweeps straight away; anything else gets the design converged first.
That is the same two-part test `database` uses to decide whether a finished run
counts as a sample, so "this design is finished" means one thing package-wide.
Both branches are logged with their reason.

`--restart` keeps meaning only *here is the field*. Overloading it to also
truncate the pipeline would make one flag mean two things depending on the verb,
where `run --restart` truncates nothing.

### Sweeping a design converged earlier

```
mkdir chic && cp case/output.yaml chic/input.yaml
turbigen chic chic/input.yaml --restart case/restart.npz
```

A directory of its own because one directory is one run. The copied config
carries the converged `result:`, so the design phase is skipped. Both files
being at the root of `case` is what a settled design leaves; if they are not
there, it did not settle, and there is no converged design to sweep.

Note the copy is renamed on the way in. `output.yaml` is not a name turbigen
will read back, so adopting one as an input is a thing you do deliberately.

Two things to know. `DP_adjust` scales the **converged** design's pressure
change, not the original file's — `iterate` relaxes `Ys` onto what the CFD
achieved, so the two differ, and the same `DP_adjust` under `chic` and under a
bare `run` of the file you first wrote are not the same back pressure. And the
inference cannot see an override: `-s mean_line.psi=1.8` invalidates the stored
result, and the logged reason is the only warning you get.

### The geometry cannot move

The sweep calls `run` once per point and never enters the iterate loop, so
nothing calls `with_unknowns` and no recamber can happen — by construction, not
by a rule about which iterators are allowed. Their errors are still *measured*
at every point, because that is what `solve` does whether anything is iterating
or not, so watching incidence grow as the machine is throttled costs nothing.

### What it finds

Where a **steady solver** stops converging. That is not the surge line: real
stall is unsteady. The report says so rather than leaving it to be assumed.

Near the peak of a characteristic `dPR/dmdot` tends to zero, so a pressure
bracket of a given width spans a wide range of mass flow — the limiting
*pressure* is resolved to `step_min` and the limiting *mass flow* it implies
rather less, which is unfortunate given the mass flow is the number of
interest.

The stable side of the characteristic needs no feedback and is already a batch:

```yaml
batch:
  values:
    operating_point.DP_adjust: [-0.10, -0.05, 0.0]
```

so `chic` covers exactly the part that cannot be written that way. Exits 0 if
any point converged — a point that refused is the answer here, not a failure.

## `batch` — implemented

```
turbigen batch datum/input.yaml -n 32
```

One verb for a set of related runs, whichever question you are asking of the
design. It writes one config per point and runs none of them.

**Two ways to say what varies**, and they are the two questions anyone asks.
`bounds:` gives a box to fill quasi-randomly, which is what builds an archive
`database:` can interpolate in:

```yaml
batch:
  seed: 0
  bounds:
    mean_line.psi: [1.2, 2.0]
    mean_line.phi2: [0.5, 1.0]
```

`values:` names the points outright, and the batch is every combination of
them — the parameter study you run to see a trend:

```yaml
batch:
  values:
    mean_line.psi: [1.2, 1.4, 1.6]
```

They are mutually exclusive. Mixing a grid over one variable with a fill over
the rest is a real thing to want, but it makes `-n`, the index space and
`--continue` all mean something new, so it waits for a case that needs it.

Paths are spelled as `node.flatten` writes them, the same as
`database.variables`, so a design variable is named identically wherever it
appears. `-n` and `--continue` are properties of the invocation and so are flags;
`seed` describes the space and so lives in the file. Both flags are **refused**
against `values:`: the count is the product of what it names, and a finite
product has no tail to carry on from.

**Why this is a verb and not a shell loop.** Output goes beside the config it
was given, so `for psi in 1.2 1.4 1.6; do turbigen run datum.yaml -s
mean_line.psi=$psi; done` writes three designs into one directory and keeps the
last. `check_clobber` cannot catch it either, seeing only the targets of one
invocation where a loop is N of them. A batch gives each point a directory, and
bakes its values into the member config, so the study is in the files rather
than in shell history.

**Nothing is run unless you ask.** Bare `batch` writes configs and stops, so
inspecting them before spending cluster hours costs nothing. `-Q` submits:

```
turbigen batch datum/input.yaml -n 32            # write only
turbigen batch datum/input.yaml -n 32 -Q         # write, then submit
```

**A grid is reachable from the command line**, which is what makes a study a
one-liner. The whole mapping has to be replaced rather than one entry of it:
`parse_path` splits a `--set` key on dots and these keys contain their own, so
`-s batch.values.mean_line.psi=[...]` would build a four-deep nest instead.

```
turbigen batch datum.yaml -s 'batch.values={mean_line.psi: [1.2, 1.4, 1.6]}' -Q
```

### Why `batch:` and not something else

The section names a set of related runs, however they are chosen, and it is
also what the verb writes — `batch_NNNN` — so key, verb and artefact are one
word. It was `sample:` while drawing from a box was all it did; `sample` names
one of the two modes and cannot cover naming points outright.

`sweep:` reads better and is **reserved**: sweep and lean are blade stacking
terms, and a config with a study `sweep:` beside a geometry `sweep:` is a
collision that cannot be undone once files exist. Also rejected: `study:`
(means examine, not emit, so it fails as a verb), `matrix:`, `plan:`, `vary:`,
`doe:` and `space:`.

"Sample" keeps its other meaning untouched — in `database:`, a *sample* is a
finished run that converged and is within tolerance. The rename frees the word
for that rather than colliding with it.

**The verb is inferred**: `iterate` when the datum has an `iterate:` section,
`run` when it does not, logged either way. Inferred rather than asked for,
because the depth of a design is already set by what the config contains — and
because getting it wrong is quiet. A batch submitted as `run` builds an archive
`database` reads back as **empty**: `_sample` takes a run only if it converged
*and* `iterate.converged` holds on its stored errors, and a single solve of a
freshly drawn design will not put those inside their tolerances. Closing that
gap is what iterating is for.

The submitted jobs carry **no options**, unlike `run -Q` and `iterate -Q`. The
members are new files written from the already-overridden datum, so a `-s` is
in them; forwarding it again would be redundant, and a `-s batch.*` would
re-create on each member the key `_strip` deliberately removed.

The two-step still works, and is what to use when you want the other verb:

```
turbigen run $(turbigen batch datum/input.yaml -n 32)/*/input.yaml -Q
```

### Sobol', not Latin hypercube

The old `DesignSpace` uses `LatinHypercube`. An LHS stratifies each axis into N
equal bins, which gives exact 1-D marginal coverage — but the stratification is
*defined by N*. A subset is not an LHS and a superset is not an LHS of N+k, so
growing a database means regenerating it and either discarding runs already
paid for or abandoning the property.

A Sobol' sequence has a fixed order in which **any prefix is space-filling**.
The first 32 points are a good design; the first 64 are a good design
*containing* the first 32. Extending is "take the next 32".

That is what this consumer needs. IDW cares about fill distance — how far a
query sits from its nearest sample — not marginal uniformity, so the LHS
advantage is the one `database.py` never uses and the LHS drawback is the one
it meets on the first extension. Balance properties hold at powers of two, so
`-n` defaults to 32 and warns otherwise.

### Batches and indices

The batch directory is *when you asked*; the file name is *which point in the
sequence*:

```
datum/input.yaml
datum/batch_0000/0000/input.yaml … 0031/input.yaml
datum/batch_0001/0032/input.yaml … 0063/input.yaml
```

Members are numbered by their **global sequence index**, not their position in
the batch, and each gets a directory of its own, because one directory is one
run: that is what gives every member an `output.yaml` to be run into, rather
than thirty-two of them sharing one.

So one glob — `database: {path: ../../batch_*/*/output.yaml}` — sees every batch
as one archive without knowing there were several, and it matches only the
members that have finished, an unrun one having an `input.yaml` and nothing
else. Nothing is ever written into an existing batch.

A batch is many designs and hours of solving to come, so it is numbered
rather than named and never written into twice. Because the directory cannot
then be given in advance, `batch` prints it on **stdout** — the
machine-readable channel this document reserves — so
`BATCH=$(turbigen batch datum/input.yaml)` works. No other verb writes to
stdout.

### Extending

```
turbigen batch datum/input.yaml -n 32 --continue
```

`--continue` needs no argument because the datum config already names the
family: its own directory is where the batches are, so it reads the highest
member index they hold and carries on. The points it draws are
bit-identical to the tail of one longer batch, which is the property that made
this Sobol' and not a Latin hypercube.

There was also a `--from INDEX`, stating the number that `--continue` reads.
Two mutually exclusive flags for one concept, and the one you would type is the
one that does not make you look the number up first.

`--continue` reads directory names only, so it cannot tell that `case.yaml` or its
bounds changed between batches — the datum is yours to keep, and nothing is
written at the batch root to say what generated it. The resolved bounds are
logged into each batch's own `log_turbigen.txt`.

### Points that will not design

A corner of the box will not design: `solve_for` fails to converge, or
`_check_round_trip` refuses. Designing costs no CFD, so each point is screened
as it is drawn and a failure is skipped — found for nothing now, instead of one
wasted cluster job at a time.

A skipped index is **never retried**: the point is deterministic given the
seed, so it would fail identically. That is why the emitted set is not
contiguous and why members carry their sequence index rather than a count. A
box that is mostly infeasible stops after a bounded number of attempts rather
than spinning.

Whole numbers are refused as `bounds:`. Rounding a continuous draw collapses
neighbours into duplicate designs, which `database._predict` then averages as
repeat runs; and the obvious integer, blade count, changes the mesh. `values:`
allows them, the reason for the ban being a property of drawing: three named
blade counts cannot collide.

A named point that will not design is **warned** rather than noted, unlike a
drawn one. You asked for that point by name, so its absence from the batch is
news; the batch is still written, and only fails if nothing designs at all.

## Entry point

```toml
[project.scripts]
turbigen = "turbigen.cli:main"
```

A separate console script from `turbigen`, so the experiment can be installed
alongside the existing tool without displacing it.

No `sys.excepthook` is installed at all. `main()` catches exceptions around the
command, logs the traceback, and exits 1. Every failure gets its traceback,
whatever it is: most are raised from a config file or a plugin, and the file
and line say which of the user's own lines to look at. A `Type: message`
summary reads tidily for the errors raised deliberately against user input, but
those cannot be told apart by type from the ones that mean something is broken
-- both arrive as `ValueError` -- so suppressing the trace for one suppresses
it for the other. `-v` sets the logging level and says nothing about how errors
print. Importing the CLI has no effect on the process, unlike
`turbigen.main:35`, which replaces the global excepthook at module scope.

## Testing

The CLI is thin and the work underneath it is already covered, so the tests
target the parts unique to the command line:

- `design` creates nothing — assert the config's directory is unchanged;
- `run` writes a config that reads back equal to the one used, and leaves the
  input file byte-identical;
- `report` of a run records the same answer the run did, and replaces none it
  cannot reproduce — neither with `--set` moving the design out from under the
  stored field, nor with an unstamped field of unknown provenance;
- `iterate` and `chic` leave every subdirectory runnable, each holding the
  design it solved;
- no verb, `design` included, will run on a file named `output.yaml`, and the
  refusal names the config beside it;
- a stamped field still restarts onto a design that has moved, which is what
  every chained restart depends on;
- `--set` reaches the design, and a mistyped `--set` key fails;
- an unknown verb, a missing file, and a bad config each exit non-zero, and a
  failure prints its traceback on stderr with or without `-v`;
- nothing but `batch` writes to stdout.
