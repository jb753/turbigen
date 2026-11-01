"""The whole config file, as one object.

A config is a tree of :class:`~turbigen.node.Node` values, so loading a file
is one recursive call and writing it back is its mirror image. It is frozen and
holds no results: designing returns a mean line rather than storing one, so a
config never accumulates state from a run.
"""

import dataclasses
from pathlib import Path

import ember.yaml_util

from turbigen import include, plugins
from turbigen.annulus import AnnulusDesign
from turbigen.batch import Batch
from turbigen.bconds import InletProfile, OperatingPoint
from turbigen.blade import BladeDesign
from turbigen.chic import Chic
from turbigen.database import Database
from turbigen.design import MeanLineDesign
from turbigen.fluid import Fluid
from turbigen.iterate import Iteration
from turbigen.job import Job
from turbigen.machine import Machine
from turbigen.mesh import Mesher
from turbigen.metric import Metric
from turbigen.node import Node
from turbigen.post import Post
from turbigen.solver import Solver
from turbigen.spanwise import SpanwiseDesign, VortexExpon


class Config(Node):
    """A complete turbigen case."""

    fluid: Fluid
    """Working fluid and its equation of state."""

    mean_line: MeanLineDesign
    """Mean-line design."""

    annulus: AnnulusDesign | None = None
    """Annulus design. Omit it to design the mean line alone."""

    blades: tuple[BladeDesign, ...] = ()
    """Blade designs, one per row. Omit them to design the annulus alone."""

    spanwise: SpanwiseDesign = VortexExpon()
    """How the flow the blades are designed against varies from hub to
    casing. A free vortex if omitted."""

    mesh: Mesher | None = None
    """Mesh generation. Only needed by the commands that make a grid."""

    solver: Solver | None = None
    """Flow solver. Only needed by the commands that solve."""

    operating_point: OperatingPoint | None = None
    """Where to run the machine, as a departure from its design point. Read by
    no design stage: the same geometry runs at every point of its
    characteristic, so this changes the boundary conditions and nothing else."""

    inlet_profile: InletProfile | None = None
    """What feeds the machine, if not a uniform flow.

    Beside the operating point rather than inside it, because the same profile
    applies at every point of a characteristic --- a rig's intake or the stage
    upstream does not change because you moved along the map. Nested, a `batch`
    over `operating_point.DP_adjust` would copy the whole profile into every
    member."""

    restart: str | None = None
    """Flow field to start the march from, a ``restart.npz`` written by a
    previous run, instead of the meridional guess.

    A path that resolves as an ``include:`` does: relative to the file that
    names it, and written relative to wherever turbigen writes the config
    next, so a case and its field move together. Absolute in memory. Set from
    the command line with ``-s restart=PATH``, relative to the working
    directory, or ``-s restart=null`` to start from the guess. A field named
    but not there is an error rather than a quiet start from the guess.

    In a config, the field this run *starts* from. In an ``output.yaml`` or an
    ``input.yaml`` turbigen wrote, the field the run it records *started* from,
    or null for the guess --- a record, which may name a field since pruned.
    Seeds only the first march of ``iterate`` and ``chic``: each later one
    starts from the one before. Not stamped, so changing it neither re-meshes
    nor disowns a stored field. ``report`` ignores it, taking its field from
    beside the config."""

    iterate: Iteration = Iteration()
    """Closing the loop between the design and its CFD. Only needed by the
    command that iterates, but the errors it names are measured by every run
    that solves."""

    database: Database | None = None
    """Finished runs to start the iterators from, instead of from whatever this
    file says. Read once, by the command that iterates."""

    chic: Chic | None = None
    """How to sweep a characteristic to its stability limit. Read only by the
    command that sweeps, which holds the geometry fixed and moves the operating
    point alone."""

    batch: Batch | None = None
    """Design variables to vary, for covering a space with runs. Read only by
    the command that writes a batch, and stripped from the configs it emits: a
    member is one design, not a design of experiments."""

    post_process: tuple[Post, ...] = ()
    """Post-processors to run. Nothing is added implicitly: what the config
    asks for is what runs."""

    metrics: tuple[Metric, ...] = ()
    """Quantities to measure from the solved field and record under
    ``result: metrics:``. Nothing is measured unless the config asks for it."""

    job: Job | None = None
    """Where to execute, when ``--queue`` asks for it. Read by no design stage:
    a partition is a property of where you are, not of the machine."""

    def __post_init__(self):
        # Checked against the blades, and its defaults written out, as soon as
        # it is read: the knots are the blade sections, so only here are both
        # in hand
        if self.blades:
            n_sections = [len(blade.sections) for blade in self.blades]
            object.__setattr__(self, "spanwise", self.spanwise.filled(n_sections))
        self.iterate.validate(self)

    @classmethod
    def from_file(cls, path):
        """Read a config from a YAML file.

        User-defined designs are discovered relative to the file, since
        resolving the ``type:`` keys it contains is exactly what needs the
        registry populated. `from_dict` does no discovery: it has no path to
        anchor a search on, and a caller working from a dict is already in
        Python and can import whatever it needs.
        """
        path = Path(path)
        plugins.discover(path.parent)

        data = include.read(path)
        # A run writes its answer into the same file under `result:`. Ignore it
        # here: this returns a config. `turbigen.case.read` returns both.
        data.pop("result", None)

        return cls.from_dict(data)

    def to_file(self, path):
        """Write this config to a YAML file, defaults included."""
        path = Path(path)
        ember.yaml_util.write_yaml(include.relative(self.to_dict(), path.parent), path)

    def design(self) -> Machine:
        """Return the machine this config describes.

        Every configured stage runs. There is no stop point to pass: the depth
        of a design is set by what the config contains, so a config with no
        annulus designs only a mean line.

        This is the only verb that lives here, because it is the only one that
        is a property of the config alone. Anything combining a config with a
        result -- meshing a machine, running a grid -- is a function of both,
        and `config.mesh.mesh(machine)` already says it without a second
        spelling on `Config`.
        """
        mean_line = self.mean_line.design(self.fluid.eos())

        annulus = None
        if self.annulus is not None:
            annulus = self.annulus.design(mean_line)
            # The inviscid endwalls where the design asks, whichever kind
            # made it
            if len(self.annulus.m_inviscid) != 4:
                raise ValueError(
                    f"annulus.m_inviscid={self.annulus.m_inviscid} needs four "
                    "positions: hub and casing at the inlet, then at the outlet."
                )
            annulus = dataclasses.replace(
                annulus, m_inviscid=tuple(float(m) for m in self.annulus.m_inviscid)
            )

        rows = ()
        if self.blades:
            if annulus is None:
                raise ValueError("Blades need an annulus to sit in.")
            if len(self.blades) != mean_line.n_row:
                raise ValueError(
                    f"Expected one blade per row, but got {len(self.blades)} "
                    f"blades for {mean_line.n_row} rows."
                )
            rows = tuple(
                blade.design(
                    mean_line[:, i_row],
                    annulus.extract_row(i_row),
                    self.spanwise,
                    i_row,
                )
                for i_row, blade in enumerate(self.blades)
            )

        machine = Machine(
            mean_line=mean_line, annulus=annulus, rows=rows, spanwise=self.spanwise
        )
        # Rows swept so far into a gap that no cut fits between them fail here,
        # not when the solution is first reduced
        if rows:
            machine.check_cuts()
        self.iterate.check(machine)
        return machine
