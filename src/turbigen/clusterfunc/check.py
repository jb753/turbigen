import numpy as np
from turbigen.clusterfunc.exceptions import ClusteringException
import turbigen.clusterfunc.util


def unit_single(x, Dmin, Dmax, ERmax, rtol=1e-9):
    """Verify that unit clustering vector obeys the desired limits."""

    # Check x

    if not x[0] == 0.0:
        raise ClusteringException(f"Normalised start value x[0]={x[0]}, expected 0.0.")

    if not np.isclose(x[-1], 1.0, rtol=rtol):
        raise ClusteringException(f"Normalised end value x[-1]={x[-1]}, expected 1.0.")

    if not ((x >= 0.0) & (x <= 1.0 + rtol)).all():
        raise ClusteringException(
            "Normalised values outside unit interval, "
            f"min(x)={x.min()}, max(x)={x.max()}."
        )

    # Check dx

    dx = np.diff(x)
    if not (dx > 0.0).all():
        raise ClusteringException(
            f"Normalised spacing min(dx)={dx.min()}, expected all > 0.0."
        )

    if not np.isclose(dx[0], Dmin, rtol):
        raise ClusteringException(
            f"Normalised start spacing dx[0]={dx[0]}, expected {Dmin}"
        )

    if not (dx <= Dmax).all():
        raise ClusteringException(
            f"Normalised spacing max(dx)={dx.max()} exceeds target {Dmax}."
        )

    # Check expansion ratio

    ER = dx[1:] / dx[:-1]

    if not (ER <= ERmax).all():
        raise ClusteringException(
            f"Expansion ratio max(ER)={ER.max()} exceeds target {ERmax}."
        )

    if not (ER >= (1.0 - rtol)).all():
        raise ClusteringException(
            f"Expansion ratio min(ER)={ER.min()} less than unity."
        )


def unit_symmetric(x, Dmin, Dmax, ERmax, rtol=1e-9):
    # Check x

    if not x[0] == 0.0:
        raise ClusteringException(f"Normalised start value x[0]={x[0]}, expected 0.0.")

    if not np.isclose(x[-1], 1.0, rtol=rtol):
        raise ClusteringException(f"Normalised end value x[-1]={x[-1]}, expected 1.0.")

    if not ((x >= 0.0) & (x <= 1.0 + rtol)).all():
        raise ClusteringException(
            "Normalised values outside unit interval, "
            f"min(x)={x.min()}, max(x)={x.max()}."
        )

    # Check dx

    dx = np.diff(x)

    if not (dx > 0.0).all():
        raise ClusteringException(
            f"Normalised spacing min(dx)={dx.min()}, expected all > 0.0."
        )

    if not np.isclose(dx[0], Dmin, rtol=rtol):
        raise ClusteringException(
            f"Normalised start spacing dx[0]={dx[0]}, expected {Dmin}"
        )

    if not np.isclose(dx[-1], Dmin, rtol=rtol):
        raise ClusteringException(
            f"Normalised end spacing dx[-1]={dx[0]}, expected {Dmin}"
        )

    # Check expansion ratio

    # Evaluate and correct for the shrinking part
    ER = dx[1:] / dx[:-1]
    Na, Nb = turbigen.clusterfunc.util.split_cells(len(x))

    ER1 = ER.copy()
    ER1[ER1 < 1.0] = 1.0 / ER1[ER1 < 1.0]

    if not (ER1 <= ERmax).all():
        raise ClusteringException(
            f"Expansion ratio max(ER)={ER1.max()} exceeds target {ERmax}."
        )

    if not (ER1 >= 1.0 - rtol).all():
        raise ClusteringException(
            f"Expansion ratio min(ER)={ER1.min()} less than unity."
        )
