# turbigen2 architecture

How the pieces fit together. See `CLI.md` for the command line, which sits on
top of this.

## Two kinds of object

**Config nodes** describe what to build. They are frozen dataclasses deriving
from `Node`, they appear in the YAML file, and they serialise both ways. A
`Fluid`, a `MeanLineDesign`, an `AnnulusDesign`.

**Results** are what building produces. They are not `Node`s: they have no
`type`, and they never appear among a config's own keys. A `MeanLine`, an
`Annulus`, and the `Machine` that collects them.

Most of them need no serialisation at all, being reproducible from the config
that made them. The exception is a run's answer, which is not --- see
[Storing what a run achieved](#storing-what-a-run-achieved).

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
are separate methods (`design`, later `report`, `run`), so there is no stop point
to parameterise; and the depth of a design is set by what the config contains,
since a config with no `annulus` designs only a mean line. Per-stage access for
debugging already exists through the stage objects themselves —
`config.annulus.design(mean_line)` — so methods on `Config` would be a third
spelling of the same operation.

## Annulus

Two designs — **fixed axial chord** and **aspect ratio** — over one body, with
a merge weight on both.

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

That leaves the chord specification as the one axis that is a real type, and it
has two values. `Merged` and `MergedFixedAxialChord` in the old package are the
*same algorithm*: diff their `forward` methods and the only divergence is how
one number per segment is arrived at, either

```python
Ds = span_avg / AR          # aspect ratio
Ds = cx / cos_Beta_avg      # axial chord
```

after which the control points, the arc-length iteration, the duct extensions,
the nozzle scaling and both PCHIP fits are identical. So the two designs share
a body and differ by a method returning that one array:

```python
class PchipAnnulus(AnnulusDesign):        # no type: not selectable
    nozzle_ratio: float = 1.0
    merge_weight: float = 0.0

    def segment_lengths(self, span_avg, cos_Beta_avg)   # the author writes this
    def forward(self, mean_line) -> Annulus             # shared, ~50 lines

class FixedAxialChord(PchipAnnulus):
    type: ClassVar[str] = "fixed_axial_chord"
    cx_row: tuple       # axial chord of each row [m], (n_row,)
    cx_gap: tuple       # axial chord of each gap [m], (n_row + 1,)

class AspectRatio(PchipAnnulus):
    type: ClassVar[str] = "aspect_ratio"
    AR_row: tuple       # span to meridional chord of each row [--], (n_row,)
    AR_gap: tuple       # span to meridional chord of each gap [--], (n_row + 1,)
```

**The hook returns arc length, not axial length.** Arc length is what both the
parameterisation and the duct extensions are measured in, so the base takes
`Dx = Ds * cos_Beta_avg` in one line. `FixedAxialChord` therefore round-trips
its own chords through a divide and a multiply, which is a one-ulp change, four
orders below the tolerance the equivalence test against the old package holds
to.

**Not a field on one class.** `tip_span`/`tip_chord`/`tip_metre` set the
precedent for naming a unit by which field carries the number, but that works
because the three are scalars that sum, with zeros elsewhere. These are arrays
of unequal length that cannot be summed, so one class would mean four optional
fields and an exactly-one-of-each-pair check, and it would leave a class named
`fixed_axial_chord` accepting aspect ratios. A second `type:` breaks no
existing file.

**`AR_row`, not the old `AR_chord`.** It pairs with `cx_row`/`cx_gap`, and the
old name reads as the aspect ratio of the chord when it means the aspect ratio
of the row.

### Aspect ratio is the one that works radially

`Ds = span_avg / AR` never divides by `cos(Beta)`, so a segment at 90 degrees
gives a finite chord and `Dx = 0`. An axial chord cannot describe such a
segment at all — the arc length it implies is infinite — which is why old
`Smooth` needed `AR = NaN` as an escape for radial machines, and why nothing
here does.

`Smooth` itself is not what was ported. Its ninety lines of `root_scalar` and
`minimize` x-offset iteration, its `rcout_offset`, its `smooth` boolean and its
negative-AR "choose the length that smooths the curvature" branch all exist
because it fits through `MeridionalLine.smooth()` rather than in arc-length
space. `Merged` reaches the same targets by construction. A negative aspect
ratio is now rejected when the config is read, rather than quietly meaning
something else.

The arc length a fitted curve actually has agrees with the target to about
0.1%, not exactly: the fixed-point iteration matches each segment's *share* of
the total, so a segment's own length is a fixed point of the fit rather than a
constraint on it. The axial positions of the stations are exact. Both are
asserted, so the gap is a recorded property rather than something rediscovered
later as a bug.

Unlike `MeanLineDesign`, an annulus design declares no `n_row`: it is generic
over row count, which comes from the mean line handed to `forward`. A useful
check that the reserved-name promotion in `Node` is not over-applied.

`Annulus` holds the fitted PCHIP curves and the geometry read off them:
`evaluate_xr`, `nrow`, `nseg`, `mmax`, the hub/casing/mid/rms radii, `Am`,
`htr`, `x_rms`, `chords` and `to_string`.

The mesh-facing helpers on the old designer — `get_cut_plane`,
`get_offset_planes`, `get_interfaces`, `get_mp_from_xr`, `xr_row`,
`get_span_curve` — are deliberately not ported. They exist to serve meshing,
and belong with the mesher.

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

It is a free function, and `prepare` calls it, so the grid a `report` draws is
the one a solver would start from and plotting it shows what will actually be
solved. Nothing of ours passes between the two halves --- the guess
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

## Storing what a run achieved

A design is reproducible from its config; a CFD answer is not. So the mixed-out
mean line is written into the **same file** as the config, under one `result:`
key beside the config's own:

```yaml
mean_line: {...}
solver: {...}
result:
  converged: true
  actual:
    P: [95185., 76842.]
    T: [295.6, 287.4]
    ...
```

One file, because comparing an achieved efficiency against the design that
asked for it should be one load and not two. One *document*, because a second
YAML document breaks `yaml.safe_load` and everything built on it, and the file
is read far more often by scripts than by us.

This does not put results back on the config. `Config` stays frozen and unaware
of them; it is `case.read` that returns two objects from one file. A file is not
an object, and that distinction is what the old config lost.

**State is stored, never derived quantities.** `eta_tt`, `PR_tt` and the whole
`backward()` dict are recomputed from the reconstructed mean line, so an
archived file cannot hold a number that no longer matches the definition that
produced it. The package this replaces stored the derived `design_vars_actual`
dict instead.

**And dimensionally.** `MeanLine.STATE` is eight quantities --- `P`, `T`, the
three velocity components, `r`, `Am` and `Omega`. Not the conserved variables,
whose energy is measured from a fluid's datum: rebuilt against a different one
they are silently reinterpreted, which on a realistic design is a hundred
kelvin. The same trap caught the initial guess, and is why `MeanLine.to_dict`
and `from_dict(data, fluid)` live on the mean line rather than on whatever
writes one --- both the choice of quantities and the datum rule are its own
knowledge. Four of the twelve data keys it inherits are deliberately never set
and raise on read, so this is not something a caller can work out by looking.

### Why not store the flow field and re-derive

Tempting, since the restart guess already carries one. But it is lossy in
exactly the quantity of interest: the guess is decimated by two per direction,
and the table below puts the wall error at 1.1e-2, while mixed-out efficiency is
an entropy flux dominated by the near-wall profile. It is also not free ---
re-deriving means re-meshing, interpolating, cutting and mixing, every time
someone wants to plot. The stored state is about 2 KB against the restart's
1.9 MB, and it is exact. They are complements: a restart is an *input* to the
next run, `actual` is *this* run's answer.

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

`ember.yaml_util` is used throughout, not `turbigen.yaml_utils`. The two are
interchangeable --- same `read_yaml`/`write_yaml`, same numpy and `Path`
representers, same patched resolver so that `1e5` reads as a float rather than
a string --- and they were checked to read and write byte-identical results on
a real config before the switch.

The difference is that ember's is backed by libyaml's
`CSafeLoader`/`CSafeDumper` where the turbigen one still uses the pure-Python
loaders, and that using it drops a dependency on the old package rather than
adding one. `turbigen.util` is now the only part of it turbigen2 still needs.

Speed was the original argument --- 7.7 s against 0.15 s to dump a
multi-megabyte scalar --- but that measurement is what settled where the
restart field goes, rather than which loader to use: see below.

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

## Starting from designs already run

An iteration begins wherever the file leaves its knobs, and then spends CFD on
walking them to the answer. When similar machines have already been solved that
is wasteful, so a `database:` key names a glob of finished case files and the
iterators start from a blend of the nearest.

```yaml
database:
  path: ../runs/**/output.yaml
```

Three things are different from `dspace.py`, which does this today.

**Nothing is declared.** The old `IndependentConfig` asks for the independent
variables in the file, with limits for each. Here they are deduced: flatten
every sample with `node.flatten`, and a design variable is any leaf they
*differ in*. The range to normalise by is the range they cover, so there are no
limits to state either, and a leaf they all agree on drops out rather than
dividing by zero.

Two guards make that safe. The candidates are restricted to the design subtree
--- `fluid`, `mean_line`, `annulus`, `blades` --- so a machine run at a finer
mesh is the same design rather than a different one. And the knobs themselves
are excluded, because a recamber is what the blend *predicts* and using it as
an input would place a query by the number it is trying to supply.

**An iterator declares what it owns.** That exclusion cannot be done by name:
`unknowns` calls a knob `dchi_TE[0]`, a mean over sections, while its leaves
are `blades[0].sections[*].dchi_TE`. So `Iterator` has a third method beside
the two an author already writes:

```python
def paths(self, config) -> set[str]   # config leaves this iterator moves
```

It raises like its siblings rather than defaulting to nothing, because an
iterator whose leaves went unnamed would have its knobs silently read as design
variables. The obvious alternative --- perturbing every knob through
`with_unknowns` and seeing which leaves move --- infers ownership from side
effects and assumes the perturbation is always a real change. It survives as
the *test* that a declaration matches what is written, which is where an
inference of that kind belongs.

`node.flatten` is the single definition of how a path is spelled, so the
declaration and the candidate set cannot drift apart in spelling while agreeing
in content.

**Inverse distance weighting, not a polynomial surrogate.** The sample count is
the binding constraint: a total-order cubic in eight design variables is 165
terms against perhaps fifteen finished runs, which is why the old fit needed
`order_max`, `frac_dof` and a train/test split to stop it overfitting, and why
it could still return an unmeshable blade outside the sample hull. IDW has no
order, no basis and no conditioning, and it *cannot* leave the hull: the answer
is a convex combination of knobs that converged, so no clip is needed to keep a
warm start meshable.

It also decays gracefully, which is why there is no minimum sample count to
configure --- the old `nsample_min_interp: 8` guarded against the polynomial,
not against this. Two samples is a blend of two converged designs; one is a
copy of it, reached through the same code path, because a lone sample makes
every column's range zero and so puts every query on top of it. Repeats of one
design are averaged rather than resolved by whichever the glob sorted first.

The price is that IDW carries no trend: its gradient is zero at every sample,
and far from all of them it decays to their mean. It interpolates between
designs already run rather than extrapolating beyond them. For a starting point
that Broyden then refines, bounded beats accurate.

**What makes a run a sample** is read from its `result:`, not from where it
sits. It must have converged, and `iterate.converged` must hold on the errors
it stored --- an intermediate iteration is a converged march whose recambers
are still on their way somewhere. The old code filters by directory depth
instead (`dspace.py:322`), counting parents to exclude per-iteration
subdirectories, which never checks whether a run finished and breaks under any
other layout. Filtering on the data is what makes a bare `**` glob correct.

The run being started is excluded explicitly, or a design whose own output is
under the glob would be started from itself and predict its own answer.

Sampling a design space is **not** here. The old `DesignSpace` owns both; they
are separate questions and separate modules --- see below.

## Choosing what to run next

`database` reads finished runs; `batch` writes the configs that become them.
The old `DesignSpace` is both at once, which is why it needs a `basedir`, a
sampler, a seed and a target count alongside the fit.

They point opposite ways, and that is the whole reason to split them.
**Reading deduces**: the design variables are whatever the runs differ in, the
ranges whatever they cover. **Writing must be told**, because an empty archive
differs in nothing. So `batch:` declares what varies, and that is not a
relapse into `IndependentConfig` --- a design of experiments is a statement of
intent, where a warm start is an observation.

**Two ways of saying what varies, one verb.** `bounds:` fills a box
quasi-randomly, which is what an archive worth interpolating in needs;
`values:` names the points and runs every combination of them, which is the
parameter study. They differ only in how the points are chosen --- everything
after that, a directory per member with its values baked in, a numbered batch,
one array submission, is the same --- so they are a field apart rather than a
verb apart. The section is `batch:` and not `sample:` because sampling names
only the first of the two; `sweep:` is reserved for blade stacking.

**Sobol', not a Latin hypercube.** An LHS stratifies each axis into N equal
bins, and the stratification is *defined by N*: a subset is not an LHS and a
superset is not an LHS of N+k, so growing an archive means regenerating it and
discarding runs already paid for. Any prefix of a Sobol' sequence is
space-filling, so extending is taking the next N. For this consumer that
settles it, because IDW cares about fill distance rather than marginal
uniformity --- the LHS advantage is the one `database` never uses, and its
drawback is the one met on the first extension.

**Points are screened before they are written.** Designing costs no CFD, so a
corner where `solve_for` cannot converge is found at emit time rather than one
wasted cluster job at a time. A skipped point is never retried, being
deterministic given the seed, so the emitted set has gaps --- which is why a
member carries its *sequence index* rather than a position in the batch, and
why an extension can pick up in the right place.

**A member carries no `batch:` key.** It is one design, not a space; left in,
batching a member would expand one design into another N. Nothing is written to
record what generated a batch either: the datum config is the user's to keep,
and turbigen2 accumulating a copy is the sort of state the rebuild exists to
remove. The resolved bounds are logged into the batch's own log file, and the
batch sits in the datum's own directory, so the layout says what the file does
not.

**A member is a directory, not a file.** One directory is one run, which is
what gives every member an `output.yaml` of its own rather than thirty-two of
them sharing one set of fixed artefact names. It also makes the glob that reads
the archive back name `output.yaml`, so it matches only the members that
finished.

**Whole numbers are refused.** Rounding a continuous draw collapses neighbours
into duplicate designs, which `_predict` then averages as repeat runs of one
design --- which is what they are. The obvious integer, blade count, also
changes the mesh, which is why `DiffusionFactor` was not ported.

## Where a run executes

A `Job` is a config node chosen by `type:`, like everything else:

```yaml
job:
  type: slurm
  hours: 4.0
  partition: ampere
  gres: "gpu:1"
```

```python
class Job(Node):
    def submit(self, tasks, verb, options) -> list[str]   # framework
    def forward(self, tasks, verb, options) -> list[str]  # the author writes this
```

A family rather than a flag per backend, for the reason every other family
exists here: `--slurm`, `--tsp` and `-j` would be a second dispatch mechanism
beside the one the package already has, and a hand-rolled flag set cannot be
extended by a plugin --- which is exactly what a `type: pbs` needs to be.

### Submission is never implied

The key says *how* to submit; `--queue` says *whether*. The package this
replaces submits whenever the key is present, so `turbigen config.yaml`
re-execs itself as `turbigen --no-job config.yaml` inside a job and exits from
the middle of the pipeline. Every entry point then needs the negative flag to
break the recursion, and `main.py` carries it three times.

One positive flag makes the recursion one level deep and the escape hatch the
absence of a flag rather than the presence of one. Per-invocation overrides
need no new syntax either, `-s job.hours=12` being the same mechanism as every
other override.

### A task is a config path, and nothing else

The verb and its options are shared by every task in a submission, so only the
config varies. That is what lets a SLURM array put the varying part in a
`tasks.txt` and the fixed part in the script once:

```
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" tasks.txt)
turbigen2 run "$CONFIG" -s mean_line.psi=1.8
```

**The array indexes lines, not directories.** `job.py:186` requires "a
consecutive range of numbered directories", which the batches `batch` writes
are not: a point that will not design is skipped and never retried, so the
indices have gaps by construction. Indexing a file removes the constraint
entirely, along with any assumption about what a config is called or where it
sits.

### Zero or empty means unstated

Every `Slurm` field defaults to zero or empty and is left out of the script
when it is, so sbatch's own `SBATCH_ACCOUNT`, `SBATCH_PARTITION` and
`SBATCH_TIMELIMIT` still apply. A cluster that already sets those in a profile
needs nothing in the file but `type: slurm`, and nothing about the site has to
be repeated in every case.

`hold_on_fail` is not ported. It traps a failure, starts a detached tmux and
`sleep 36h` to hold the node for a debug shell; a run already writes its
evidence and exits 2.

### Local queueing is task-spooler, not ours

`type: tsp` shells to Debian's `task-spooler`: slots, job ids, listing,
cancellation, and a `-N` that says how many slots one job occupies. The package
this replaces hand-rolls the same thing in 340 lines --- a flock'd text file, a
PID file, SIGHUP cancel-all, a systemd unit template and a `--follow` that
execs `journalctl` --- and still cannot express a job wanting four cores.

Nothing wraps `tsp -l` or `squeue`. They are better than anything we would put
in front of them, so the turbigen2 surface is `--queue` and the `job:` key.

### A partition is not part of the design

Nothing here is read by any design stage, and `database.SUBTREE` already
restricts design variables to `fluid`, `mean_line`, `annulus` and `blades`, so
a `job:` key cannot be mistaken for one. It rides along into an archived
`output.yaml` and into every batch member, which is useful rather than
harmful: a member that carries its own `job:` can be submitted without being
told anything, and re-running an archive elsewhere is `-s job.partition=...`.

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
