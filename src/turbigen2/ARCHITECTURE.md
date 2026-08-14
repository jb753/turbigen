# turbigen2 architecture

How the pieces fit together. See `CLI.md` for the command line, which sits on
top of this.

## Two kinds of object

**Config nodes** describe what to build. They are frozen dataclasses deriving
from `Node`, they appear in the YAML file, and they serialise both ways. A
`Fluid`, a `MeanLineDesign`, an `AnnulusDesign`.

**Results** are what building produces. They are not `Node`s: they never appear
in a config file, have no `type`, and need no serialisation. A `MeanLine`, an
`Annulus`, and the `Machine` that collects them.

Keeping these apart is the point of the whole design. The old config conflated
them — `TurbigenConfig` held `nominal` and `actual` mean lines alongside the
design variables, and `AnnulusDesigner.forward` wrote its fitted splines onto
the designer object, so the config *was* the result and could only ever hold
one.

## The stage interface

Every design stage exposes the same three methods:

```python
stage.design(upstream) -> result      # framework: validates, calls forward
stage.forward(upstream) -> result     # the author writes this
stage.backward(result) -> dict        # the author writes this, where invertible
```

| stage | `forward` | `backward` |
|---|---|---|
| `MeanLineDesign` | `(Fluid) -> MeanLine` | `(MeanLine) -> dict`, required: it supplies `solve_for`'s residual |
| `AnnulusDesign` | `(MeanLine) -> Annulus` | absent; a spline fit does not invert to design variables |

`forward` **returns** its result rather than filling one in. The mean line
could work either way, because a `MeanLine` is a fixed-size Block that can be
allocated up front, but an `Annulus` cannot: its result is a set of fitted
curves, not a container. Returning everywhere means one rule rather than two.

Mean-line designs get `self.allocate(fluid)` for the boilerplate — it checks
`n_row`, builds the `MeanLine`, and sets the equation of state — so a design
reads:

```python
def forward(self, fluid):
    ml = self.allocate(fluid)
    ...
    return ml
```

`allocate` follows ember's own vocabulary: `Block.__init__` documents itself as
"Allocate a structured grid block".

## Machine

```python
@dataclass(frozen=True)
class Machine:
    mean_line: MeanLine
    annulus: Annulus | None = None
```

The result of designing. Named after `turbigen.geometry.Machine`, which
collects the same sort of thing for the mesher, but widened: ours holds the
mean line too, and eventually blade count, tip gaps and splitters, which are
design outputs currently reassembled on demand by `config.get_machine()`. The
two are not interchangeable while both exist.

It is deliberately not called `Design`, which would collide with the `...Design`
suffix used for *config* classes, and `machine = config.design()` reads better
than `design = config.design()`.

## Config

```python
class Config(Node):
    fluid: Fluid
    mean_line: MeanLineDesign
    annulus: AnnulusDesign | None = None

    def design(self) -> Machine       # runs every configured geometry stage
```

There is no `upto` argument and no per-stage method on `Config`. The CLI verbs
are separate methods (`design`, later `mesh`, `run`), so there is no stop point
to parameterise; and the depth of a design is set by what the config contains,
since a config with no `annulus` designs only a mean line. Per-stage access for
debugging already exists through the stage objects themselves —
`config.annulus.design(mean_line)` — so methods on `Config` would be a third
spelling of the same operation.

## Annulus

One design only: **fixed axial chord, with a merge weight.**

The existing package has four annulus classes in 1035 lines, which are a 2x2 of
{chord specification} x {merged or not}. Counting every YAML in the repo:
`merged_fixed_axial_chord` 78 uses, `fixed_axial_chord` 14, `smooth` about 26
including the silent default, `merged` **zero**. Two of the four parameters
unique to `Smooth` — `rcout_offset` and the boolean `smooth` — are used zero
times.

Merging is not a type: `merge_weight` is a continuous blend in `[0, 1]` that
defaults to `0.0`, and at zero the result is identical to the unmerged fit. So
`fixed_axial_chord` is `merged_fixed_axial_chord` with `merge_weight=0`, and
the distinction costs an entire duplicated class — the old code says as much,
noting that the fitting and blending in `MergedFixedAxialChord` are "identical"
to `Merged`'s, while subclassing `AnnulusDesigner` directly rather than reusing
them.

That leaves the aspect-ratio chord specification (`AR_chord`/`AR_gap`) as the
only real casualty. It is genuinely used in the old package, and can be added
later as a second design if wanted; it is not needed to prove the architecture.

```python
class FixedAxialChord(AnnulusDesign):
    type: ClassVar[str] = "fixed_axial_chord"

    cx_row: tuple       # axial chord of each row [m], (n_row,)
    cx_gap: tuple       # axial chord of each gap [m], (n_row + 1,)
    nozzle_ratio: float = 1.0
    merge_weight: float = 0.0
```

Unlike `MeanLineDesign`, an annulus design declares no `n_row`: it is generic
over row count, which comes from the mean line handed to `forward`. A useful
check that the reserved-name promotion in `Node` is not over-applied.

`Annulus` holds the fitted PCHIP curves and the geometry read off them:
`evaluate_xr`, `nrow`, `nseg`, `mmax`, the hub/casing/mid/rms radii, `Am`,
`htr`, `x_rms`, `chords` and `to_string`.

The mesh-facing helpers on the old designer — `get_cut_plane`,
`get_offset_planes`, `get_interfaces`, `get_mp_from_xr`, `xr_row`,
`get_span_curve` — are deliberately not ported. They exist to serve meshing,
and belong with the `mesh` verb.

### A result class exposes the minimal complete interface

For `Annulus` that is `evaluate_xr(m, spf)`, plus the station geometry read off
it. Shapes wanted by one consumer belong in that consumer.

This is easy to breach without noticing. Porting the annulus plot first added
`get_cut_plane` and `get_coords` back onto `Annulus`, and both turned out to be
pure reshapes of `evaluate_xr` returning byte-identical values — adding no
knowledge, only a layout. Worse, `get_coords` was shaped "in AutoGrid format"
by its own docstring, so a mesh writer's data layout was pulled into the
geometry class for a line plot to transpose back. The plot is shorter without
them.

The test for whether a helper belongs: does it *compute* something, or only
re-lay-out something already computed? `chords()` integrates arc length per
segment, so it belongs. Those two did not. This is how the old
`AnnulusDesigner` reached 1035 lines: each consumer added the view it wanted,
and none of them were the annulus's business.

## Blades

A `BladeDesign` describes one row; designing it against a row of the mean line
and a row of the annulus produces a `Row`, holding the `Blade` it is a row of.

```python
class BladeDesign(Node):
    sections: tuple[Section, ...]
    count: BladeCount
    tip_span: float = 0.0
    tip_chord: float = 0.0
    tip_metre: float = 0.0
    vortex_exponent: float = -1.0
    theta_offset: float = 0.0
    m_stack: float = 0.5

    def forward(self, mean_line_row, stream_surface) -> Row
```

This is the stage where the config-as-its-own-result problem is worst. The old
`BladeDesigner` gets to its geometry by mutating itself three times:
`set_streamsurface` writes the annulus and a derived thickness scale onto it,
so nothing can be evaluated before it has been called; `apply_recamber`
overwrites the first two columns of the camber array in place, turning recamber
angles into metal angles; and an `is_recambered` flag guards the second, which
post-processors toggle on and off around plots. All three disappear here for
the same reason: a metal angle, a tip gap in metres and a stream surface are
functions of the design *and* the mean line, so they belong to the result.

### Named fields, and one more config/result split

`q_camber` and `q_thick` are positional vectors --- `q_thick[3]` is
`kappa_max`, `q_camber[2]` is the aft-loading factor --- dispatched by
`util.get_subclass_by_name`, the third string-dispatch scheme that `Node`
retires. As fields they document themselves, and the shape assertions in
`__post_init__` go away, because a `Section` carries its own parameters and so
cannot disagree in length with a parallel array.

Camber splits the same way the rest of the package does, one level down. A
`CamberDesign` is the *shape* between the end angles, `chi_hat(m)`; a
`CamberLine` is that shape placed between metal angles it learned from the mean
line. The recamber angles sit on the `Section`, not on the camber, because they
are what is added to the flow angle rather than anything about the curve.

Thickness needs no such split: it is always normalised by meridional chord, so
`thick_ref`, the `_thick_scale` it produced and `BaseThickness.scale` are all
gone. No YAML in the repo ever set it to anything else.

### The stage takes a row, not an index

```python
blades = tuple(
    design.design(mean_line.row(i), annulus.row(i))
    for i, design in enumerate(self.blades)
)
```

Passing an `Annulus` and a row index would copy the convention that row `i`
occupies `m = 2i + 1 ... 2i + 2` into every blade design anyone writes, when it
is the annulus that defines what `m` means. So `Annulus.row` returns a
`StreamSurface`: the coordinate map restricted to one row, `m` running 0 at the
leading edge to 1 at the trailing edge, plus the meridional chord --- which
`set_streamsurface` used to recompute by arc length even though
`Annulus.chords` already had it.

### Blade number is paired with the blade, not merged into it

Two separate questions, and the old package got one right and one wrong.

**In the config**, `nblade` was a second top-level list indexed by row, so a row
and its count could get out of step; `config.get_nblade()` calls `sys.exit(1)`
when they do. Here it is a `count` field on the row's design, so there is
nothing to keep in step.

**In the result**, though, the old package was right that a count is not part of
a shape. How a blade is shaped says nothing about how many of them there are,
and no consumer wants both: the mesher reads `evaluate_section` and `chi` off
the shape, and `n_blade` and `tip_gap` off the row, never both from one object.
So designing a row gives a `Row` holding a `Blade`:

```python
class Row:
    blade: Blade      # the shape of one blade
    n_blade: int
    tip_gap: float    # how it sits in the annulus, not what it looks like
```

Paired rather than parallel is what keeps the old package's insight without its
failure mode. There is no second list to fall out of step, and a different count
over the same geometry is `dataclasses.replace(row, n_blade=...)` rather than a
redesign.

It also removes a half-built object. Counting needs the geometry, because a
circulation coefficient is set against surface length --- but the geometry never
needs the count, so the shape can be finished first and the count read off it.
An earlier version built the blade twice, passing one with `n_blade=None` into
the counting rule; that intermediate was the same "cannot be used until
something else has happened" defect as `set_streamsurface`, and it disappears
once the two concerns are separate objects.

### Tip clearance is one number and its reference

`tip: 0.01` with `tip_ref: span` becomes `tip_span: 0.01`. Three fields, all
defaulting to zero, at most one set --- so the reference is named by which
field carries the number, and there is no second field to contradict the first.
Resolution is one line of `forward` instead of `get_gaps`'s three-way branch
and its `sys.exit`. A family of `TipGap` nodes was considered and rejected: it
is four classes to choose a divisor.

### What a Blade exposes

`evaluate_section`, `chi`, `surface_length`, `chord`, and the `m_stack` and
`theta_offset` fixed at design time. `n_blade` and `tip_gap` are on the `Row`,
not here, for the reason above. By the test the annulus set, three things stay
with their consumers: `get_coords` is an
AutoGrid-format reshape and belongs with the mesher, `get_nose` and
`get_LE_cent` are used only by `post/plot_nose.py`, and `get_pitch_chord` and
`blade_table` are a report.

Only the camber and thickness types in use are ported --- every YAML in the
repo uses `quadratic` and `taylor`, and the three `thick_type: Impeller` uses
name a class that does not exist in the package. `Quartic`, `Taylor` and
`TaylorQuadratic` camber, the `DFL` blade count and splitters (which appear
only in `old-examples/`) are additions, not blockers, exactly as `AR_chord` is
for the annulus.

## Meshing

A `Mesher` turns a `Machine` into an ember `Grid`. The result is ember's, not
ours, so there is no `Mesh` class for a `...Design` suffix to pair with and the
family keeps the name the class it replaces already had.

```python
class Mesher(Node):
    yplus: float = 30.0

    def mesh(self, machine) -> Grid       # framework
    def forward(self, machine, spacing)   # the author writes this
```

### The framework method earns its keep here

For a pure design stage, `design` only validates and delegates. `mesh` does
real work at both ends, and all of it is shared:

* **on the way in**, the wall spacings implied by `yplus` --- a surface
  Reynolds number per row, a flat-plate skin friction, a friction velocity and
  hence a viscous length, returned as a `WallSpacing` of hub, casing and
  per-row surface sizes in metres;
* **on the way out**, the four steps that make a fresh grid usable: set the
  reference length, check every cell volume, compute wall distance, report the
  size.

In the package this replaces every one of those sits in `config.setup_mesh`,
which then calls `make_grid(workdir, machine, dhub, dcas, dsurf, Omega)` --- six
arguments, five of them derivable from the machine. A second mesher would have
to be handed the same five, and any caller can forget the finishing steps or
get them out of order. Here a mesher author writes `forward` and gets the rest.

Two things follow from putting the spacings on the base. The `yplus`
calculation is no longer on the *config*, where it had no business being, and
it is no longer broken: it needs `MeanLine.ref`, which was deleted in 948516a,
so `config.calculate_d_wall` has been raising `AttributeError` for anything
that called it. `ref` is restored here, unchanged --- note that it is an *area*
criterion, and so picks the inlet of a row whose annulus opens out even when
the flow accelerates through it.

`WallSpacing` is a result, not a `Node`: computed from the machine, never
written to a file.

### H needs no adapter

`turbigen.geometry.Machine` is not ported. It exists only to hand five
attributes to a mesher, and `turbigen2.Machine` already carries all of them ---
`row.n_blade` and `row.tip_gap` were the two the blade port moved off their own
parallel arrays and onto the row. The mesher reads the annulus through
`evaluate_xr`, `chords` and `span`, and the rows through `evaluate_section` and
`chi` on their blades, and nothing else.
`Annulus.span(m)` is the one method the mesh verb turned out to need; the five
mesh-facing helpers deferred when the annulus was ported are still unused.

### What did not come across

The mesh mathematics is carried over unchanged --- it never knew about the
config --- but four things are dropped:

* **Plotting.** A `plot` field that called `plt.show()` from inside a mesh
  routine, a `_plot_grid` helper, a module-level matplotlib import, and a
  `plot=` argument threaded down into `add_cusp`. Mesh figures belong to a
  post-processor and to the verb's `--plot`, not to the mesher.
* **`_log_ram`**, at seven call sites.
* **`recluster`**, which no configuration sets and which cannot work: its
  branch calls `_get_mlim` on a *list* of blades, so it raises `AttributeError`
  if it is ever reached.
* **`slip_cusp` and `slip_annulus`**, patch-type swaps that no configuration
  sets. With `slip_annulus` gone, `_add_annulus_patches` turned out to be a
  no-op --- its only other branch computes an index and then has its two
  statements commented out --- so it went too, and with it the `row_meta` that
  existed to feed it.

## Initial guess

A grid leaves the mesher as geometry with reference scales and no flow in it.
`guess.apply(grid, machine)` writes one: circumferentially uniform, taken from
the mean line along the annulus mid-span, and applied by ember's
`apply_guess_meridional`, which is a nearest-neighbour search in the meridional
plane and so needs no topology matching.

It is a free function, and the `mesh` verb calls it, so the grid that verb
hands back is the one a solver would start from and plotting it shows what will
actually be solved. Nothing of ours passes between the two halves --- the guess
is an ember `Block` and the target an ember `Grid` --- so there is no guess
class, and no family until a second strategy earns one.

Two things are easy to get wrong here, and both fail silently.

**The guess block must carry the grid's own fluid.** `apply_guess_meridional`
does `block.set_fluid(block_guess.fluid)` on every block it touches, so a guess
built with the mean line's fluid replaces the scales and datum the mesher set.

**And its state must be transferred datum-independently.** Conserved energy is
measured from the datum where internal energy is zero, so copying a mean line's
conserved variables into a block whose fluid has a different datum reinterprets
them --- by about a hundred kelvin, for a datum moved as far as
`referenced_fluid` moves it, while the result looks perfectly well formed.
Pressure, temperature and velocity carry across; `conserved` does not.

`to_quasi3d` is not ported. It builds a guess with a pitchwise pressure
difference from the blade loading and a radial-equilibrium correction, and the
physics is reasonable, but it was **commented out** in the package this
replaces (`config.py:846-849`) and its ember counterpart
`Grid.apply_guess_quasi3d` has never been called either. Two hundred lines that
have never run are not a port, they are a rewrite; worth revisiting once there
is a solver to time it against.

## Restart guesses

A converged flow field is carried into the next run as an initial guess. The
package this replaces pickles the conserved arrays to a `.pkl.gz` beside the
config, which is version-brittle, opaque, and separable from the config it
belongs to.

Instead the guess goes **in the config file**, as base64 of a compressed,
decimated conserved array. Measured on a 1M-node block with realistic
small-scale content, 19.2 MB raw:

| scheme | in YAML | of full | RMS err | max err | wall err |
|---|---|---|---|---|---|
| full float32 | 14.8 MB | 100% | — | — | — |
| **decimate x2 in i, j and k** | **1.9 MB** | **13%** | 2.6e-3 | 2.8e-2 | 1.1e-2 |
| decimate x3 | 0.6 MB | 4% | 3.6e-3 | 4.0e-2 | 1.2e-2 |
| full, scaled float16 | 5.3 MB | 36% | 1.6e-4 | 3.8e-4 | 2.9e-4 |

Three decisions come out of that.

**Decimate by two in all three directions.** It is what takes the blob from
disqualifying to tolerable. A quarter of a percent RMS error is immaterial in
something the solver is about to iterate away, and the errors barely worsen
from x2 to x4 while the size drops sevenfold, so x2 is the conservative end of
a shallow curve.

**Byte-shuffle before gzip.** Transposing so that byte 0 of every float sits
together, then byte 1, puts exponents and high mantissa bytes where they
compress. It is about fifteen lines, lossless, 21% smaller than plain gzip and
2.5x faster to encode.

**Use `resample` and `interp_from`; write no interpolation.** Both halves
already exist in ember. `Grid.resample(0.5)` decimates, and
`Grid.interp_from(src)` restores onto the current grid, wrapping the Fortran
kernel `ember.fortran.map_coordinates_3d`.

Neither should be reimplemented, and the reason is `_interp_coords`: it is not
a linspace. It collects critical indices --- patch boundaries *as well as*
endpoints --- from both the source and the target, requires the counts to
match, and maps between each consecutive pair separately so those locations
land exactly. That is what keeps a mixing plane at a mixing plane and a
periodic boundary at a periodic boundary through an interpolation. A
hand-rolled `linspace(0, n_src - 1, n_tgt)` is correct only for a single block
with no interior patches, and silently smears patch boundaries otherwise.
`resample` preserves the same critical indices on the way down, and updates
the patch indices on the block it returns.

`interp_from` also packs `mu_turb` alongside the five conserved variables into
one kernel call, asserts that trilinear interpolation created no new extrema,
asserts the result has finite positive temperature, and works in dimensional
form so that differing reference scales between source and target are handled
--- which matters here, because the mesher sets `V_ref` and `rho_ref` on the
grid from each design, so a guess from the previous iteration is
non-dimensionalised differently.

Two things follow that shrink the scheme further. Because interpolation is in
**index space**, no coordinates need storing: the guess is conserved variables
and `mu_turb`, nothing else. And index space maps leading edge to leading edge,
which is the right behaviour for a design iteration that has recambered a blade
while keeping the topology --- exactly when a restart guess is worth having.

So the only genuinely new code is the byte shuffle and the base64 encode and
decode. On the way out: `grid.resample(0.5)`, then shuffle, gzip, base64. On
the way back: decode into a `Block` at the decimated shape with the fluid set,
then `grid.interp_from(...)`.

Full arrays stay in memory and in `to_dict`; decimation happens only on the way
to a file, so nothing in the object model knows about it.

Two things to keep in view. Wall error is about four times the RMS error and
does not improve with gentler decimation, because dropping every other
wall-normal point on a stretched mesh loses the near-wall profile regardless.
If preserving a converged boundary layer turns out to matter, resample `i` and
`k` only and keep `j` intact: a quarter rather than an eighth, still under
4 MB, with the wall exact.

Note also that plain float16 is *not* an option for conserved variables:
`rhoe` reaches 3.8e5 against a float16 maximum of 65504 and silently becomes
`inf`. It only works with a per-variable scale factor stored alongside.

### YAML implementation

Use `ember.yaml_util`, not `turbigen.yaml_utils`. It exposes the same
`read_yaml`/`write_yaml`, but is backed by libyaml's `CSafeLoader`/
`CSafeDumper` where the turbigen one still uses the pure-Python loaders. On a
multi-megabyte scalar that is the difference between 7.7 s and 0.15 s to dump.
It also carries the numpy and `Path` representers already, so switching drops a
dependency on the old package rather than adding one.

## Result, and nominal against actual

Running a machine produces a `Result`:

```python
@dataclass(frozen=True)
class Result:
    machine: Machine                 # geometry, and the flow it was designed for
    actual: MeanLine | None = None   # mixed out from the CFD solution
    grid: Grid | None = None
    converged: bool = False

    @property
    def nominal(self) -> MeanLine:
        return self.machine.mean_line
```

**Only the flow has a nominal and an actual.** There is no actual annulus: the
annulus you designed is the annulus, and CFD does not produce a different one.
The same holds for blades --- the deviation iterator changes a blade, but that
yields a new *design* for the next iteration, not an actual version of the
current one. So geometry appears once and the mean line appears twice, which is
what the data actually looks like.

**There are two states, not three.** It is tempting to distinguish what was
requested (`config.mean_line`'s fields) from what the design achieved
(`backward(nominal)`), since `check_round_trip` only asserts they agree to
0.5%. They do not need separating: `solve_for` raises if it cannot hit its
targets and `check_round_trip` raises if the round trip fails, so a nominal
mean line that exists *is* the requested design. The gap is an assertion
threshold, not something to plot. It is the designer's job to refuse a design
it cannot achieve.

Note that this is also the naming trap the old config fell into. There,
`config.mean_line_actual` (a dict) sat alongside `config.mean_line.actual` (a
MeanLine), because "actual" was attached to a config object rather than kept
beside the design it is compared with. Here `result.nominal` and
`result.actual` sit at the same level, one step away from the `Machine`.

## Post-processing

A post-processor is a `Node` like any other config object, so `type:` dispatch,
defaults and round-tripping come free, and the four processors the old config
inserts imperatively into `post_process` in `__post_init__` are simply
configured.

```python
class Post(Node):
    def report(self, config, result) -> Iterable[Figure]
```

**It takes the config as well as the result**, because the most valuable plot
in the system compares design intent against reality, and intent lives in the
config. That is safe here in a way it is not today: the hazard in the existing
`post(config, pdf)` is not reading the config but that `post.py` calls
`config.apply_recamber()` and `config.undo_recamber()` --- a plot mutating the
geometry and putting it back. Against a frozen `Config` that cannot happen.

What a post-processor has no business with is the plumbing: `work_dir` (the CLI
decides where output goes), `solver`, `job`, `cut_offset`. Those describe how
the run was executed, not what was intended.

**It returns figures rather than drawing into a shared PDF.** In the old code
`pdf` is threaded through every processor, appearing 21 times, so none can be
run alone, output cannot be redirected, and nothing can be tested without a
`PdfPages`. Returning figures means one processor can be run in a notebook, the
CLI decides whether they become a PDF, and a test can assert on a figure
without touching the filesystem --- which is also what makes plotting work in
the ephemeral mode where nothing is written.

Failures should raise, with a `--keep-going` opt-out. Today
`post_process_all` catches everything, prints a traceback to stderr rather than
the log, and carries on, so a broken plot leaves a silently incomplete PDF.

Two things to settle before porting. **Exports are not plots**: `Metadata`,
`write_cuts`, `write_ibl` and `write_stl` produce files, not figures, and
probably want their own verb rather than a `report` that returns nothing. And
**the recamber toggling** inside the current post-processors needs
understanding before it is removed --- if a plot genuinely needs recambered
geometry then `Machine` should carry both forms rather than have plots switch
between them.

### The post/ directory

`src/turbigen/post/` holds thirteen modules, about 1200 lines, that **cannot be
imported**. There is no `__init__.py` and `post.py` shadows the directory, so
`import turbigen.post` resolves to the module and `turbigen.post.spanwise`
raises `ModuleNotFoundError`. All thirteen are tracked in git, and some were
written against a live API --- `spanwise.py` and `plot_nose.py` use
`mean_line.get_row(irow)` and `[::2]` indexing.

Establish whether these are abandoned or a half-finished refactor before
porting anything. Otherwise they will either be ported by accident or block the
port while someone works out whether they matter.

## What dies with this

`util.BaseDesigner` is a third implementation of the design-variables-in-a-dict
pattern, alongside the mean-line one already replaced and the config's own. Its
only subclass is `AnnulusDesigner`, so porting the annulus retires it: the
signature introspection in `check_design_vars`, the `_supplied_design_vars`
marker for arguments that come from upstream rather than the file, and the
hand-written `from_dict`/`to_dict` all become dataclass fields and the `Node`
protocol.
