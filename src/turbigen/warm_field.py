"""Seed a fresh grid from a converged neighbour's flow field.

The meridional guess in :mod:`turbigen.guess` is uniform across span and pitch
and imposes the design exit swirl and pressure everywhere: no boundary layers,
no wakes, and the turning applied as a smooth meridional ramp with no knowledge
of where along the chord the blade does the work. For a marginal design the
transient from that guess overshoots into a full-span suction-side separation
and the march runs away before it can relax.

:func:`turbigen.database.nearest_field` finds the design-space nearest converged
run. This applies that run's field to the new grid --- index space, leading edge
to leading edge, exactly as a chained restart does --- and then moves its levels
onto the mean line the new design was drawn for: the neighbour's converged
*structure* kept, its *levels* corrected.

The correction is measured rather than assumed. Index-space interpolation puts
the neighbour's field on *different* geometry, so what the neighbour's own
mixed-out mean line says is no longer what the grid holds; :mod:`turbigen.mixout`
is run on the interpolated field to find out what it actually is, and the
perturbation is the difference between that and the nominal.

It is a perturbation, not a ratio. A ratio is unbounded wherever the quantity it
divides by passes through zero --- which tangential velocity does at an axial
exit, in the middle of the operating range --- and multiplying scales gradients,
wake depth and boundary-layer shape along with the level, when only the level is
meant to move. Five quantities are perturbed, against five state variables:
entropy, stagnation enthalpy, the two meridional velocity components and angular
momentum. Pressure and temperature do not appear. The thermodynamic state is
rebuilt from entropy and stagnation enthalpy *after* the velocity is known, so it
comes out consistent with the velocity rather than being scaled beside it.

Stagnation enthalpy rather than rothalpy, though rothalpy is the quantity that is
actually conserved through a rotor. Rothalpy is defined with the station's own
shaft speed, so a perturbation ramped across an interrow gap would interpolate
between two quantities of different meaning; stagnation enthalpy has one
definition everywhere, the shaft speed drops out of the reconstruction entirely,
and there is no stator-or-rotor branch. Nothing here exploits the conservation
property anyway, the level being set station by station regardless.

The perturbation is a field in :math:`(x, r)`, applied by nearest-neighbour
search on the same densified meridional polyline the cold guess uses. That is
what makes it independent of block topology: there is no assumption that a row
is one block, or that the streamwise index runs the way a ramp would need it to.
"""

import dataclasses
import logging

import numpy as np
from scipy.spatial import KDTree

from turbigen import guess, mixout, restart

logger = logging.getLogger("turbigen")

PERTURBED = ("s", "ho", "Vx", "Vr", "rVt")
"""The quantities a seed corrects, and the order they are reported in.

Five of them against the five state variables of a compressible flow, so the
correction is exactly determined: entropy and stagnation enthalpy fix the
thermodynamics once the velocity is fixed, and the velocity is fixed by its two
meridional components and its angular momentum.
"""


@dataclasses.dataclass(frozen=True)
class Seed:
    """A converged neighbour's field, to start a march from.

    Built by :func:`turbigen.database.nearest_field` and applied by
    :meth:`apply` in place of :func:`turbigen.guess.apply`.

    A path and nothing else. The neighbour's own achieved mean line used to be
    carried alongside, to divide by; it is not, because it describes the field
    on the neighbour's grid and :meth:`apply` needs to know what the field is on
    *this* one.
    """

    field: object
    """Path to the neighbour's ``restart.npz``."""

    def apply(self, grid, machine):
        """Write the corrected neighbour field into `grid`, in place.

        Parameters
        ----------
        grid : ember.grid.Grid
            A meshed grid carrying the meridional guess, which is overwritten.
        machine : Machine
            The design the grid was meshed from; its nominal mean line is the
            target the field's levels are moved onto.

        """
        restart.apply(grid, self.field)

        # What the neighbour's field *is*, now that it sits on this geometry.
        # Cut at the design stations and contracted to the design areas, so it
        # is directly comparable with the nominal mean line station for station.
        measured, _ = mixout.mean_line(grid, machine)

        delta = _station_delta(machine.mean_line.flat, measured.flat)

        logger.debug(
            "Field seed perturbation, inlet to outlet: "
            + "  ".join(
                f"d{name} {np.array2string(delta[name], precision=3)}"
                for name in PERTURBED
            )
        )

        _apply_delta(grid, machine, delta)

        # Measured again, because the correction is not exact: mixing out is
        # nonlinear and momentum-weighted, so a uniform perturbation does not
        # reproduce itself exactly in the mixed-out value. What is left is the
        # one number that says how good the seed is, and it costs another set
        # of cuts to know it.
        check, _ = mixout.mean_line(grid, machine)
        residual = _station_delta(machine.mean_line.flat, check.flat)

        logger.info(
            f"Corrected the seed field onto {machine.mean_line.n_row} row(s), "
            "leaving "
            + "  ".join(
                f"d{name} {np.abs(residual[name]).max():.3g}" for name in PERTURBED
            )
        )


def _station_delta(nominal, measured):
    """Return what carries `measured` onto `nominal`, station by station.

    Both are flat :class:`~turbigen.meanline.MeanLine` views in streamwise
    order, of the same shape, in the same rotating frame and on the same fluid
    datum --- :func:`turbigen.mixout.mean_line` builds the measured one as a
    copy of the nominal --- so every difference below is like for like.

    Angular momentum rather than tangential velocity, each side taken at its own
    radius: the measured radius is the mass-mean radius of the cut and the
    nominal is the design radius, and the difference between the two is real.
    """
    return {
        "s": np.asarray(nominal.s) - np.asarray(measured.s),
        "ho": np.asarray(nominal.ho) - np.asarray(measured.ho),
        "Vx": np.asarray(nominal.Vx) - np.asarray(measured.Vx),
        "Vr": np.asarray(nominal.Vr) - np.asarray(measured.Vr),
        "rVt": np.asarray(nominal.r) * np.asarray(nominal.Vt)
        - np.asarray(measured.r) * np.asarray(measured.Vt),
    }


def _interpolator(machine, delta):
    """Return a densified meridional point cloud of `delta`, and its tree.

    The stations are where :func:`turbigen.mixout.mean_line` cut, which is a
    blade-chord offset into the adjacent gap and *not* the integer meridional
    stations :func:`turbigen.guess.meridional` uses --- the perturbation has to
    be anchored where the thing it was measured from was measured.

    Densified along arc length and searched by nearest neighbour, which is what
    :meth:`ember.grid.Grid.apply_guess_meridional` does, so that the seed and
    the cold guess agree on where a row begins and ends. That method cannot be
    used directly: it overwrites where this adds, entropy and stagnation
    enthalpy differences are not a state that a carrier block could hold, and
    its blade-passage velocity snap would flatten exactly the turning the
    neighbour's field was fetched for.
    """
    # Mid-span of each cut plane. `cut_planes` returns hub and casing, and
    # `Annulus.evaluate_xr` is linear in span fraction, so the mean of the two
    # is the mid-span point without evaluating the annulus again.
    xr = np.array(
        [plane.mean(axis=0) for plane in machine.annulus.cut_planes()], dtype=float
    )

    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(xr, axis=0), axis=1))]
    )
    fine = np.linspace(0.0, arc[-1], (len(xr) - 1) * guess.REFINE_FACTOR + 1)

    xr_fine = np.stack([np.interp(fine, arc, xr[:, i]) for i in (0, 1)], axis=-1)
    delta_fine = {name: np.interp(fine, arc, value) for name, value in delta.items()}

    return KDTree(xr_fine), delta_fine


def _apply_delta(grid, machine, delta):
    """Add `delta` to every block of `grid`, in place."""
    tree, delta_fine = _interpolator(machine, delta)

    for block in grid:
        _, near = tree.query(
            np.stack([block.x.reshape(-1), block.r.reshape(-1)], axis=-1)
        )
        d = {
            name: value[near].reshape(block.shape) for name, value in delta_fine.items()
        }

        # Every read before the first write. Stagnation enthalpy is built from
        # the velocity, so reading it back after the velocity has moved would
        # measure the new field against the wrong datum and leave something
        # entirely plausible and quietly wrong.
        Vx = np.asarray(block.Vx)
        Vr = np.asarray(block.Vr)
        Vt = np.asarray(block.Vt)
        r = np.asarray(block.r)
        s = np.asarray(block.s)
        h = np.asarray(block.h)

        # Formed here rather than read from the block, so that the
        # reconstruction below inverts exactly the definition used to make it.
        ho = h + 0.5 * (Vx**2 + Vr**2 + Vt**2)

        Vx = Vx + d["Vx"]
        Vr = Vr + d["Vr"]
        Vt = Vt + d["rVt"] / r
        ho = ho + d["ho"]
        s = s + d["s"]

        # Shaft speed appears nowhere: a stator and a rotor are the same four
        # lines, because stagnation enthalpy is measured in the stationary
        # frame whichever frame the block is solved in.
        h = ho - 0.5 * (Vx**2 + Vr**2 + Vt**2)

        # State first, then velocity, as `guess.meridional` sets them: the
        # block stores conserved variables, so a density that moves after the
        # momentum has been set would reinterpret it.
        block.set_h_s(h, s)
        block.set_Vx(Vx)
        block.set_Vr(Vr)
        block.set_Vt(Vt)
