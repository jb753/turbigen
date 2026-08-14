"""Blade geometry.

A :class:`BladeDesign` is a config node describing one blade row; designing it
against a row of the mean line and a row of the annulus produces a
:class:`Blade`, which evaluates aerofoil sections.

The design is frozen and the result holds everything a mean line was needed to
work out --- metal angles, blade number, tip gap. The package this replaces
reaches the same place by mutating the designer three times over:
``set_streamsurface`` writes the annulus and a thickness scale onto it,
``apply_recamber`` overwrites the recamber angles in place with metal angles
behind an ``is_recambered`` flag, and post-processors toggle that flag on and
off around plots. None of it is possible here.

Blade number lives on the design rather than in a parallel top-level list, so a
row and its count cannot get out of step.
"""

import dataclasses
import logging
from typing import ClassVar

import numpy as np

import turbigen.util
from turbigen2.camber import CamberDesign, CamberLine
from turbigen2.node import Node
from turbigen2.thickness import ThicknessDesign

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
    return cls(**dict(zip(names, interpolated.reshape(-1))))


class Section(Node):
    """One spanwise section of a blade."""

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
    """Base for rules setting the number of blades in a row."""

    def count(self, mean_line_row, blade) -> int:
        """Return the number of blades for `blade` in `mean_line_row`."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement count(self, mean_line_row, blade)"
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
    """Set the number of blades using a circulation coefficient."""

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
        # due to a change in radius and the part due to a change in swirl
        centrifugal = (1.0 - RR**2.0) * (tanAlpha[0] - tanAlpha_rel[0])
        tangential = tanAlpha_rel[0] - RR * VmR * tanAlpha_rel[1]

        # Normalise by inlet or outlet dynamic head, whichever is the larger
        A_flow = ml.Am * cosAlpha_rel
        total_in = cosAlpha_rel[0] * (centrifugal + tangential)
        total_out = cosAlpha_rel[1] / VmR * (centrifugal + tangential)
        total = total_in if A_flow[1] / A_flow[0] > 1.0 else total_out

        # Pitch that delivers the requested circulation coefficient
        pitch = np.abs(self.Co / total) * blade.surface_length(self.spf)

        r_ref = np.mean(ml.r)
        return int(np.round(2.0 * np.pi * r_ref / pitch).item())


class BladeDesign(Node):
    """Design variables for one blade row."""

    sections: tuple[Section, ...]
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

    def forward(self, mean_line_row, stream_surface):
        """Return the blade this design describes.

        Parameters
        ----------
        mean_line_row : MeanLine
            Inlet and outlet stations of this row, shape (2,).
        stream_surface : StreamSurface
            The annulus within this row, from :meth:`Annulus.row`.

        """
        spf = np.array([section.spf for section in self.sections])

        # Recamber onto the local flow angles. This happens once, here: a metal
        # angle is a function of the design and the mean line together, so it
        # is a property of the result and never of the design.
        dchi = np.array(
            [(section.dchi_LE, section.dchi_TE) for section in self.sections]
        )
        chi = _Alpha_rel(mean_line_row, spf, self.vortex_exponent) + dchi
        if np.any(np.abs(chi) > 90.0):
            raise ValueError(f"Cannot set a blade angle over 90 degrees, chi={chi}.")
        if np.any(np.abs(chi) > 80.0):
            logger.warning(f"WARNING: high blade angles may hinder meshing, chi={chi}")
        tanchi = turbigen.util.tand(chi)

        # Tip clearance in metres. Whichever reference length was used, the
        # other terms are zero, which __post_init__ has already ensured.
        span = float(np.mean(mean_line_row.span))
        tip_gap = (
            self.tip_span * span
            + self.tip_chord * stream_surface.chord
            + self.tip_metre
        )

        parts = dict(
            stream_surface=stream_surface,
            spf=spf,
            tanchi=tanchi,
            cambers=tuple(section.camber for section in self.sections),
            thicknesses=tuple(section.thickness for section in self.sections),
            m_stack=self.m_stack,
            theta_offset=self.theta_offset,
            tip_gap=tip_gap,
        )

        # Counting blades needs the geometry, because a circulation coefficient
        # is set against surface length. So the blade is built, counted, then
        # built again holding its count, rather than having a count written
        # onto it afterwards.
        n_blade = self.count.count(mean_line_row, Blade(n_blade=None, **parts))
        return Blade(n_blade=n_blade, **parts)

    def design(self, mean_line_row, stream_surface):
        """Return the blade this design describes."""
        return self.forward(mean_line_row, stream_surface)


class Blade:
    """A designed blade row."""

    def __init__(
        self,
        stream_surface,
        spf,
        tanchi,
        cambers,
        thicknesses,
        m_stack,
        theta_offset,
        tip_gap,
        n_blade,
    ):
        self._surface = stream_surface
        self._spf = np.asarray(spf, dtype=float)
        self._tanchi = np.asarray(tanchi, dtype=float)
        self._cambers = tuple(cambers)
        self._thicknesses = tuple(thicknesses)
        self.m_stack = float(m_stack)
        """Normalised meridional position the sections are stacked at [--]."""
        self.theta_offset = float(theta_offset)
        """Angle the whole blade is rotated through [rad]."""
        self.tip_gap = float(tip_gap)
        """Tip clearance [m]."""
        self.n_blade = n_blade
        """Number of blades in this row [--]."""

    def __repr__(self):
        return f"Blade(n_section={len(self._spf)}, n_blade={self.n_blade})"

    @property
    def n_section(self):
        """Number of sections this blade was defined by."""
        return len(self._spf)

    def section(self, spf):
        """Return the camber line and thickness at span fraction `spf`.

        Interpolated linearly between the sections, extrapolating beyond the
        end ones.
        """
        camber = _interpolate(self._cambers, self._spf, spf)
        thickness = _interpolate(self._thicknesses, self._spf, spf)

        if self.n_section == 1:
            tanchi = self._tanchi[0]
        else:
            tanchi = turbigen.util.interp1d_linear_extrap(self._spf, self._tanchi)(
                spf
            ).reshape(-1)

        return CamberLine(camber, *tanchi), thickness

    def chi(self, spf):
        """Return the metal angles at the leading and trailing edges [deg]."""
        camber, _ = self.section(spf)
        return camber.chi((0.0, 1.0))

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
        camber, thickness = self.section(spf)

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
        chord = turbigen.util.arc_length(self._surface.xr(0.5, mcam))

        # Meridional coordinates of the upper, lower and camber lines
        xru = self._surface.xr(spf, mu_LTE)
        xrl = self._surface.xr(spf, ml_LTE)
        xr = self._surface.xr(spf, mcam)

        # Project the camber angle onto the stream surface
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

    def surface_length(self, spf):
        """Return the length of the longer of the two surfaces at `spf` [m]."""
        xrtu, xrtl = self.evaluate_section(spf)
        lengths = [
            turbigen.util.arc_length(np.stack((*xrt[:2], xrt[1] * xrt[2])))
            for xrt in (xrtu, xrtl)
        ]
        return np.maximum(*lengths)

    def chord(self, spf):
        """Return the meridional length of the camber line at `spf` [m]."""
        xr = np.stack(self.evaluate_section(spf)).mean(axis=0)[:2]
        return turbigen.util.arc_length(xr)
