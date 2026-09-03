"""Camber, thickness, and how blade sections stack into a row.

A :class:`BladeDesign` is the blade row specification as it appears in the
input file under :ref:`blades: <config-blades>`, but is not in itself a shape:
it defines recamber angles measured from a flow that is not known until the
mean line is built. Combining a :class:`BladeDesign` with a
:class:`~turbigen.meanline.MeanLine` and a :class:`~turbigen.annulus.RowAnnulus`
produces a :class:`Row` with a full blade shape, a blade count, and a tip gap.

The :doc:`/tutorial` builds a blade in a complete design; this page documents
the classes in more detail. The mean line a blade turns against is documented
at :doc:`/meanline` and the annulus it sits in at :doc:`/annulus`.


.. _blade-shapes:

Built-in shapes
^^^^^^^^^^^^^^^

A :class:`SectionDesign` names a span fraction, a recamber at each end, and a
camber and a thickness design. A :class:`BladeDesign` holds one or more sections
in increasing span fraction, together with a blade count, a tip clearance, a swirl
distribution and a stacking position. Each of these is again a design paired
with a result, the same split the rest of :program:`turbigen` makes at machine
level:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Design (config)
     - Combined with
     - Result
   * - :class:`~turbigen.camber.CamberDesign`
     - leading and trailing metal angles
     - :class:`~turbigen.camber.CamberLine`
   * - :class:`~turbigen.thickness.ThicknessDesign`
     - nothing; normalised by chord
     - evaluated directly, no result type
   * - :class:`SectionDesign`, :class:`BladeDesign`
     - a mean-line row, a :class:`~turbigen.annulus.RowAnnulus`
     - :class:`Row` --- a :class:`Blade`, ``n_blade``, ``tip_gap``

:class:`~turbigen.camber.Quadratic` and :class:`~turbigen.camber.Bernstein` are
the built-in camber shapes and :class:`~turbigen.thickness.Taylor` the built-in
thickness distribution; all are documented in the sections below.

The number of blades comes from a :class:`BladeCount` rule on the design:

* :class:`FixedCount` (``Nb``) states it directly;
* :class:`Circulation` (``Co``) sets it from a circulation coefficient and the
  surface length;
* :class:`DiffusionFactor` (``DFL``) sets it from the Lieblein diffusion factor
  and the chord.

A blade has one tip clearance, stated as a fraction of span
(:attr:`~BladeDesign.tip_span`), a fraction of meridional chord
(:attr:`~BladeDesign.tip_chord`) or an absolute length
(:attr:`~BladeDesign.tip_metre`); giving more than one is an error.


.. _blade-process:

Design process
^^^^^^^^^^^^^^

:program:`turbigen` converts each entry of :ref:`blades: <config-blades>` into
a row :class:`BladeDesign` and, once the mean line and annulus are built, passes
them :meth:`~BladeDesign.design` which:

#. Collects the recambers :attr:`~SectionDesign.dchi_LE` and
   :attr:`~SectionDesign.dchi_TE` of each section, together with the mean-line
   row and the :attr:`~BladeDesign.vortex_exponent` the local flow angle they
   are measured off is evaluated with. The two are added when a metal angle is
   asked for rather than here, so a blade defined by one section still varies
   over the span with the vortex distribution;
#. Camber and thickness designs, the recambers, the stacking
   position and the blade rotation are collected across all sections into a
   :class:`Blade` for each row;
#. The :class:`BladeCount` rule reads the finished :class:`Blade` to fix the
   number of blades.

.. _blade-evaluate:

Evaluating a blade
^^^^^^^^^^^^^^^^^^

A :class:`Blade` is addressed by span fraction, and every geometric quantity is
read off it with an ``evaluate_`` method that interpolates the section designs
field by field, extrapolating beyond the end sections. The metal angles are the
exception: only the recamber is interpolated, and the flow angle it is added to
is evaluated at the span fraction asked for.
:meth:`~Blade.evaluate_section` gives the axial, radial and angular coordinates
of the two surfaces; :meth:`~Blade.evaluate_chi` the leading and trailing metal
angles; and :meth:`~Blade.evaluate_chord` and
:meth:`~Blade.evaluate_surface_length` the meridional chord and the longer
surface length used by the count rules.
"""

import dataclasses
import logging
from typing import ClassVar

import numpy as np

import turbigen.util
from turbigen.annulus import RowAnnulus
from turbigen.camber import CamberDesign, CamberLine
from turbigen.meanline import MeanLine
from turbigen.node import Node
from turbigen.thickness import ThicknessDesign

logger = logging.getLogger("turbigen")


def _Alpha_rel(mean_line_row, spf, vortex_exponent):
    """Relative flow angles at span fractions, for a vortex distribution.

    Parameters
    ----------
    mean_line_row : MeanLine
        Inlet and outlet stations of one blade row, shape (2,).
    spf : array_like
        Span fractions, shape (n_section,).
    vortex_exponent : float
        Spanwise swirl distribution, with tangential velocity varying as
        radius to this power.

    Returns
    -------
    Alpha_rel : ndarray, shape (n_section, 2)
        Relative flow angle at each span fraction, at inlet and outlet [deg].

    """
    ml = mean_line_row
    spf = np.asarray(spf, dtype=float).reshape(-1, 1)
    r = ml.r_hub * (1.0 - spf) + ml.r_cas * spf
    Vt = ml.Vt * (r / ml.r) ** vortex_exponent
    Vt_rel = Vt - ml.Omega * r
    return np.degrees(np.arctan(Vt_rel / ml.Vm))


def _interpolate(nodes, spf_sections, spf):
    """Interpolate like-typed Nodes field-wise onto a span fraction.

    Every section must use the same design, because there is no meaning to
    blending a quadratic camber line into a quartic one. The parameters of one
    design are interpolated linearly, extrapolating beyond the end sections.
    """
    cls = type(nodes[0])
    if any(type(node) is not cls for node in nodes):
        raise ValueError(
            f"Every section must use the same design to interpolate between, "
            f"got {sorted({type(node).__name__ for node in nodes})}."
        )

    names = [field.name for field in dataclasses.fields(cls)]
    if len(nodes) == 1 or not names:
        return nodes[0]

    values = np.array([[getattr(node, name) for name in names] for node in nodes])
    interpolated = turbigen.util.interp1d_linear_extrap(spf_sections, values)(spf)

    try:
        return cls(**dict(zip(names, interpolated.reshape(-1))))
    except ValueError as err:
        # Interpolating between two valid sections can land on an invalid one,
        # since nothing constrains the path between them. Say where, or the
        # message describes parameters that appear nowhere in the config file.
        raise ValueError(
            f"Interpolating the {cls.__name__} sections onto spf={spf} gives a "
            f"section that is not valid: {err}"
        ) from err


class SectionDesign(Node):
    """One spanwise section of a blade.

    The :doc:`/blade` page covers how the sections stack into a shape.
    """

    spf: float
    """Span fraction this section is defined at, 0 at hub and 1 at casing."""

    dchi_LE: float
    """Recamber of the leading edge from the local flow angle [deg]."""

    dchi_TE: float
    """Recamber of the trailing edge from the local flow angle [deg]."""

    camber: CamberDesign
    """Shape of the camber line between the end angles."""

    thickness: ThicknessDesign
    """Thickness distribution, normalised by meridional chord."""


class BladeCount(Node):
    """Base for rules setting the number of blades in a row.

    The implementations sit after :class:`BladeDesign`, since each reads a
    finished :class:`Blade` to work.
    """

    def count(self, mean_line_row, blade) -> int:
        """Return the number of blades for `blade` in `mean_line_row`."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement count(self, mean_line_row, blade)"
        )


class BladeDesign(Node):
    """Design variables for one blade row.

    The :doc:`/blade` page covers how the sections, count and clearance become
    a shape.
    """

    sections: tuple[SectionDesign, ...]
    """Spanwise sections, in increasing span fraction."""

    count: BladeCount
    """How many blades this row has."""

    tip_span: float = 0.0
    """Tip clearance as a fraction of span [--]."""

    tip_chord: float = 0.0
    """Tip clearance as a fraction of meridional chord [--]."""

    tip_metre: float = 0.0
    """Tip clearance as an absolute length [m]."""

    vortex_exponent: float = -1.0
    """Spanwise swirl distribution, with tangential velocity varying as radius
    to this power. The default is a free vortex."""

    theta_offset: float = 0.0
    """Rotate the whole blade through this angle [rad]."""

    m_stack: float = 0.5
    """Normalised meridional position the sections are stacked at [--]."""

    def __post_init__(self):
        if not self.sections:
            raise ValueError("A blade needs at least one section.")

        spf = np.array([section.spf for section in self.sections])
        if np.any(np.diff(spf) <= 0.0):
            raise ValueError(
                f"Blade sections must be in increasing span fraction, got {list(spf)}."
            )

        # Sections are interpolated field-wise, which is only meaningful
        # between like designs. Caught here so that a config that cannot work
        # fails when it is read, rather than part-way through a design.
        for attribute in ("camber", "thickness"):
            designs = {
                type(getattr(section, attribute)).__name__ for section in self.sections
            }
            if len(designs) > 1:
                raise ValueError(
                    f"Every section must use the same design to interpolate "
                    f"between, but the {attribute} designs are {sorted(designs)}."
                )

        # One clearance, named by which reference length it is measured
        # against, so there is no separate reference to disagree with it.
        tips = {
            "tip_span": self.tip_span,
            "tip_chord": self.tip_chord,
            "tip_metre": self.tip_metre,
        }
        set_tips = sorted(name for name, value in tips.items() if value)
        if len(set_tips) > 1:
            raise ValueError(
                f"A blade has one tip clearance, but {set_tips} were all given."
            )

    def design(self, mean_line_row, row_annulus):
        """Return the blade this design describes.

        Parameters
        ----------
        mean_line_row : MeanLine
            Inlet and outlet stations of this row, shape (2,).
        row_annulus : RowAnnulus
            The annulus within this row, from :meth:`Annulus.extract_row`.

        """
        spf = np.array([section.spf for section in self.sections])

        # The recambers are carried as they were configured, not resolved into
        # metal angles here: the flow angle they are measured from varies over
        # the span, and a blade defined by one section would freeze it at that
        # section's own span if the sum were taken now.
        dchi = np.array(
            [(section.dchi_LE, section.dchi_TE) for section in self.sections]
        )

        # Tip clearance in metres. Whichever reference length was used, the
        # other terms are zero, which __post_init__ has already ensured.
        span = float(np.mean(mean_line_row.span))
        tip_gap = (
            self.tip_span * span + self.tip_chord * row_annulus.chord + self.tip_metre
        )

        blade = Blade(
            row_annulus=row_annulus,
            spf=spf,
            dchi=dchi,
            mean_line_row=mean_line_row,
            vortex_exponent=self.vortex_exponent,
            cambers=tuple(section.camber for section in self.sections),
            thicknesses=tuple(section.thickness for section in self.sections),
            m_stack=self.m_stack,
            theta_offset=self.theta_offset,
        )

        # Read at the endwalls, which is where the vortex distribution is most
        # extreme and so where an unbuildable angle appears first. A blade with
        # sections short of the endwalls has none of its own there, so this is
        # a statement about what will be meshed rather than about what was
        # written in the config file.
        chi = np.array([blade.evaluate_chi(0.0), blade.evaluate_chi(1.0)])
        if np.any(np.abs(chi) > 80.0):
            logger.warning(f"WARNING: high blade angles may hinder meshing, chi={chi}")

        # The shape is complete before anything is counted, which is what lets
        # the count be read off it rather than written onto it. A circulation
        # coefficient is set against surface length, so counting needs a shape
        # -- but a shape never needs a count.
        return Row(
            blade=blade,
            n_blade=self.count.count(mean_line_row, blade),
            tip_gap=tip_gap,
        )


class FixedCount(BladeCount):
    """Directly specify the number of blades."""

    type: ClassVar[str] = "Nb"

    Nb: int
    """Number of blades [--]."""

    def count(self, mean_line_row, blade):
        del mean_line_row, blade
        return int(self.Nb)


class Circulation(BladeCount):
    """Set the number of blades using a circulation coefficient.

    The circulation coefficient is the blade circulation over an ideal one
    that carries the exit velocity along the whole suction surface and
    stagnated flow along the pressure surface, Coull and Hodson (2013) eqn.
    (21). A typical value is 0.7; the loss correlations it was fitted against
    begin to extrapolate above about 0.8.

    Coull and Hodson write it for an axial row. What is evaluated here is the
    generalisation to a changing radius, Kaufmann (2020) eqn. (F.6).
    """

    type: ClassVar[str] = "Co"

    Co: float
    """Circulation coefficient [--]."""

    spf: float = 0.5
    """Span fraction to take the surface length from."""

    def count(self, mean_line_row, blade):
        ml = mean_line_row

        # Ratios across the row
        VmR = ml.Vm[1] / ml.Vm[0]
        RR = ml.r[1] / ml.r[0]
        tanAlpha = turbigen.util.tand(ml.Alpha)
        tanAlpha_rel = turbigen.util.tand(ml.Alpha_rel)
        cosAlpha_rel = turbigen.util.cosd(ml.Alpha_rel)

        # Circulation from the change in angular momentum, split into the part
        # due to a change in radius and the part due to a change in swirl. For
        # an irrotational inlet flow the two together are the circulation bound
        # to the blade, the centrifugal term being the relative eddy that a
        # changing radius sweeps out; on an axial row it drops out entirely.
        centrifugal = (1.0 - RR**2.0) * (tanAlpha[0] - tanAlpha_rel[0])
        tangential = tanAlpha_rel[0] - RR * VmR * tanAlpha_rel[1]

        # Normalise by inlet or outlet dynamic head, whichever is the larger
        A_flow = ml.Am * cosAlpha_rel
        total_in = cosAlpha_rel[0] * (centrifugal + tangential)
        total_out = cosAlpha_rel[1] / VmR * (centrifugal + tangential)
        total = total_in if A_flow[1] / A_flow[0] > 1.0 else total_out

        # Pitch that delivers the requested circulation coefficient
        pitch = np.abs(self.Co / total) * blade.evaluate_surface_length(self.spf)

        # The coefficient is written in the inlet pitch, so it is the inlet
        # radius that turns a pitch into a count; the count being the same all
        # the way through then fixes the pitch at every other radius.
        return int(np.round(2.0 * np.pi * ml.r[0] / pitch).item())


class DiffusionFactor(BladeCount):
    """Set the number of blades using the Lieblein diffusion factor."""

    type: ClassVar[str] = "DFL"

    DFL: float
    """Lieblein diffusion factor [--]. A typical value is 0.45; the flow
    separates above about 0.6."""

    spf: float = 0.5
    """Span fraction to take the chord from."""

    def count(self, mean_line_row, blade):
        ml = mean_line_row

        # Pitch to true chord ratio, Dixon and Hall eqn. (3.32)
        V1, V2 = ml.V_rel
        DVt = np.abs(np.diff(ml.Vt_rel).item())
        excess = self.DFL + V2 / V1 - 1.0
        if excess < 0.0:
            raise ValueError(
                f"A velocity ratio V2/V1={V2 / V1} is too low for a diffusion "
                f"factor DFL={self.DFL}; they must satisfy DFL + V2/V1 > 1."
            )
        s_c = 2.0 * V1 / DVt * excess

        # Stagger, assuming a quadratic camber line, resolves the true chord
        # onto the meridional chord the blade reports
        tanAlpha_rel = turbigen.util.tand(ml.Alpha_rel)
        stagger = np.arctan(0.5 * (tanAlpha_rel[0] + tanAlpha_rel[1]))
        pitch = s_c / np.cos(stagger) * blade.evaluate_chord(self.spf)

        r_ref = np.mean(ml.r)
        return int(np.round(2.0 * np.pi * r_ref / pitch).item())


@dataclasses.dataclass(frozen=True, eq=False)
class Blade:
    """The shape of one blade.

    Frozen, like every other result: it is what the design produced against a
    mean line, and nothing downstream has any business changing it.
    """

    row_annulus: RowAnnulus = dataclasses.field(repr=False)
    """The annulus within this row."""

    spf: np.ndarray = dataclasses.field(repr=False)
    """Span fraction of each section, shape (n_section,)."""

    dchi: np.ndarray = dataclasses.field(repr=False)
    """Recamber of each section off the local flow angle, shape (n_section, 2) [deg]."""

    mean_line_row: MeanLine = dataclasses.field(repr=False)
    """Inlet and outlet stations of this row, shape (2,).

    Held so that the metal angle can be resolved wherever the blade is asked
    for one. A metal angle is a function of the design and the mean line
    together, and the flow angle half of that sum varies over the span --- on a
    rotor by tens of degrees --- so it cannot be reduced to a per-section
    number without losing the variation between the sections and beyond them.
    """

    vortex_exponent: float = dataclasses.field(repr=False)
    """Spanwise swirl distribution the flow angle is evaluated with [--]."""

    cambers: tuple = dataclasses.field(repr=False)
    """Camber shape of each section."""

    thicknesses: tuple = dataclasses.field(repr=False)
    """Thickness distribution of each section."""

    m_stack: float
    """Normalised meridional position the sections are stacked at [--]."""

    theta_offset: float
    """Angle the whole blade is rotated through [rad]."""

    @property
    def n_section(self):
        """Number of sections this blade was defined by."""
        return len(self.spf)

    def _get_cam_thick(self, spf):
        """Return the camber line and thickness at span fraction `spf`.

        Interpolated linearly between the sections, extrapolating beyond the
        end ones.
        """
        camber = _interpolate(self.cambers, self.spf, spf)
        thickness = _interpolate(self.thicknesses, self.spf, spf)

        return CamberLine(
            camber, *turbigen.util.tand(self.evaluate_chi(spf))
        ), thickness

    def evaluate_chi(self, spf):
        """Return the metal angles at the leading and trailing edges [deg].

        The recamber the sections declare, applied to the flow angle *here*:
        the vortex distribution is evaluated at `spf` rather than at the
        sections, so a blade defined by one section still turns with the span
        it stands in.
        """
        if self.n_section == 1:
            dchi = self.dchi[0]
        else:
            dchi = turbigen.util.interp1d_linear_extrap(self.spf, self.dchi)(
                spf
            ).reshape(-1)

        chi = _Alpha_rel(self.mean_line_row, [spf], self.vortex_exponent)[0] + dchi

        # Checked here rather than at design time, because here is where a span
        # fraction is named: the sections do not bound the angles any more, and
        # a blade is meshed and cut at spans no section was written at.
        if np.any(np.abs(chi) > 90.0):
            raise ValueError(
                f"Cannot set a blade angle over 90 degrees, chi={chi} at spf={spf}."
            )

        return chi

    def evaluate_section(self, spf, nchord=10000, m=None):
        """Return coordinates of the upper and lower surfaces at `spf`.

        Parameters
        ----------
        spf : float
            Span fraction to take the section at.
        nchord : int
            Number of chordwise points, if `m` is not given.
        m : array_like, optional
            Normalised chordwise positions along the camber line.

        Returns
        -------
        xrt_upper, xrt_lower : ndarray, shape (3, n)
            Axial, radial and angular coordinates of each surface. The upper
            surface is at the higher angular coordinate.

        """
        camber, thickness = self._get_cam_thick(spf)

        if m is None:
            m = turbigen.util.cluster_cosine(nchord)

        dydm = camber.dydm(m)
        chi = np.arctan(dydm)
        tau = thickness.thick(m)

        # Offsets for thickness perpendicular to the camber line
        Dm = -tau * np.sin(chi)
        Dy = tau * np.cos(chi)

        mu = m + Dm
        ml = m - Dm

        # The surfaces overhang the camber line at the ends, so rescale onto
        # the row so that the aerofoil, not its camber line, spans leading to
        # trailing edge.
        mcam_LE = np.min((mu.min(), ml.min()))
        mcam_TE = np.max((mu.max(), ml.max()))
        mcam_ptp = mcam_TE - mcam_LE
        mu_LTE = (mu - mcam_LE) / mcam_ptp
        ml_LTE = (ml - mcam_LE) / mcam_ptp
        mcam = (m - mcam_LE) / mcam_ptp
        chord = turbigen.util.arc_length(self.row_annulus.evaluate_xr(mcam, 0.5))

        # Meridional coordinates of the upper, lower and camber lines
        xru = self.row_annulus.evaluate_xr(mu_LTE, spf)
        xrl = self.row_annulus.evaluate_xr(ml_LTE, spf)
        xr = self.row_annulus.evaluate_xr(mcam, spf)

        # Project the camber angle onto the annulus
        theta = turbigen.util.cumtrapz0(dydm / xr[1], mcam * chord)

        # Stack the sections, then rotate the whole blade
        theta -= np.interp(self.m_stack, mcam, theta)
        theta += self.theta_offset

        # Angular offsets to the surfaces, at the mean radius between the
        # camber line and each surface
        dtu = Dy * chord / xr[1]
        drtu = dtu * 0.5 * (xr[1] + xru[1])
        drtl = -dtu * 0.5 * (xr[1] + xrl[1])

        xrtu = np.stack((*xru, (theta * xru[1] + drtu) / xru[1]))
        xrtl = np.stack((*xrl, (theta * xrl[1] + drtl) / xrl[1]))

        return xrtu, xrtl

    def evaluate_surface_length(self, spf):
        """Return the length of the longer of the two surfaces at `spf` [m]."""
        xrtu, xrtl = self.evaluate_section(spf)
        lengths = [
            turbigen.util.arc_length(np.stack((*xrt[:2], xrt[1] * xrt[2])))
            for xrt in (xrtu, xrtl)
        ]
        return np.maximum(*lengths)

    def evaluate_chord(self, spf):
        """Return the meridional length of the camber line at `spf` [m]."""
        xr = np.stack(self.evaluate_section(spf)).mean(axis=0)[:2]
        return turbigen.util.arc_length(xr)


@dataclasses.dataclass(frozen=True, eq=False)
class Row:
    """A number of blades installed in an annulus.

    Separate from the shape it holds, because the two are independent: how a
    blade is shaped says nothing about how many of them there are, and every
    consumer wants one or the other, never both. That separation is also what
    makes the shape constructible in one go, since counting reads a shape but a
    shape never reads a count. Paired rather than parallel, though, so a row and
    its count cannot get out of step.
    """

    blade: Blade = dataclasses.field(repr=False)
    """Shape of one blade."""

    n_blade: int
    """Number of blades in this row [--]."""

    tip_gap: float
    """Tip clearance [m]. A property of how the blade sits in its annulus,
    rather than of its shape."""
