"""A single turbine blade row at fixed inlet stagnation conditions."""

from typing import ClassVar

import numpy as np

import ember.set_iterative
from turbigen.design import MeanLineDesign


class TurbineCascade(MeanLineDesign):
    """A single turbine blade row at fixed inlet stagnation conditions."""

    type: ClassVar[str] = "turbine_cascade"
    n_row: ClassVar[int] = 1

    span: tuple[float, float]
    """Annulus span at inlet and outlet [m]."""

    Alpha: tuple[float, float]
    """Yaw angle at inlet and outlet [deg]."""

    Ma2: float
    """Outlet Mach number [--]."""

    Ys: float
    """Pseudo entropy loss coefficient [--], see :meth:`loss_coefficient`."""

    htr: float = 0.95
    """Outlet hub-to-tip ratio [--]."""

    Po1: float = 1e5
    """Inlet stagnation pressure [Pa]."""

    To1: float = 300.0
    """Inlet stagnation temperature [K]."""

    #
    # SHARED DEFINITIONS
    #
    # Written once and called from both directions. A formula duplicated
    # between forward and backward is free to drift, and has.
    #

    @staticmethod
    def entropy_rise(Ys, ao1, To1):
        """Entropy rise implied by a pseudo loss coefficient.

        The inverse of :meth:`loss_coefficient`.
        """
        return Ys * (0.5 * ao1**2) / To1

    def loss_coefficient(self, ml):
        """Pseudo entropy loss coefficient of a mean line.

        Non-dimensionalised on inlet stagnation conditions,
        ``Ys = (s2 - s1) * To1 / (0.5 * ao1**2)``. The inverse of
        :meth:`entropy_rise`.
        """
        inlet = ml.inlet
        return (ml.outlet.s - inlet.s) * inlet.To / (0.5 * inlet.ao**2)

    #
    # DESIGN
    #

    def forward(self, fluid):
        ml = self.allocate(fluid)

        span = np.asarray(self.span, dtype=float)
        Alpha = np.asarray(self.Alpha, dtype=float)
        if span.shape != (2,) or Alpha.shape != (2,):
            raise ValueError(
                f"turbine_cascade needs span and Alpha of length 2, got "
                f"{span.shape} and {Alpha.shape}."
            )

        # Inlet stagnation state, on a scratch station so the mean line itself
        # is untouched until we have something to write.
        stag = ml.inlet.empty()
        stag.set_P_T(self.Po1, self.To1)
        ho1, s1, ao1 = stag.h, stag.s, stag.a

        s2 = s1 + self.entropy_rise(self.Ys, ao1, self.To1)

        # Outlet state from stagnation enthalpy, entropy, Mach and angles
        ember.set_iterative.set_ho_s_Ma_Alpha_Beta(
            ml.outlet, ho1, s2, self.Ma2, Alpha[1], Beta=0.0
        )
        ml.outlet.set_span_htr(span[1], self.htr)

        # Conserve mass to fix the inlet state
        rhoVx1 = ml.outlet.rhoVx * span[1] / span[0]
        ember.set_iterative.set_ho_s_rhoVm_Alpha_Beta(
            ml.inlet, ho1, s1, rhoVx1, Alpha[0], Beta=0.0
        )

        # Inlet annulus shares the outlet mid radius; hub-to-tip may differ
        ml.inlet.set_span_r_mid(span[0], ml.outlet.r_mid)

        return ml

    def backward(self, ml):
        flat = ml.flat
        return {
            # Design variables
            "span": flat.span,
            "Alpha": flat.Alpha,
            "Ma2": ml.outlet.Ma,
            "Ys": self.loss_coefficient(ml),
            "htr": ml.outlet.htr,
            "Po1": ml.inlet.Po,
            "To1": ml.inlet.To,
            # Diagnostics
            "PR_ts": ml.PR_ts,
            "eta_ts": ml.eta_ts,
        }
