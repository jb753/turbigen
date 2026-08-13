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
`get_span_curve` — are deliberately not ported yet. They exist to serve
meshing, and belong with the `mesh` verb.

## What dies with this

`util.BaseDesigner` is a third implementation of the design-variables-in-a-dict
pattern, alongside the mean-line one already replaced and the config's own. Its
only subclass is `AnnulusDesigner`, so porting the annulus retires it: the
signature introspection in `check_design_vars`, the `_supplied_design_vars`
marker for arguments that come from upstream rather than the file, and the
hand-written `from_dict`/`to_dict` all become dataclass fields and the `Node`
protocol.
