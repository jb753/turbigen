"""Building a POD basis for modal inlet profiles.

A :class:`turbigen.bconds.Pod` profile is coefficients of modes tabulated in a
file. This makes that file, from the exit profiles of runs already solved ---
typically every converged run of a sweep, cut with
:func:`turbigen.iterate.exit_deficit` --- so that the next sweep's repeating
stages are fitted in the shapes the last one actually produced.

A basis is built once and then fixed: the coefficients written against it mean
nothing in another. Nothing here runs during a sweep.
"""

import hashlib
from pathlib import Path

import numpy as np

from turbigen import bconds

FORMAT_VERSION = 1
"""Written into every basis, for whoever reads one later."""


def build_basis(spf, profiles, n_mode, trim=1):
    """Return the arrays of a POD basis for `profiles`.

    Each profile has its span-weighted mean removed, since a profile carries no
    level, and the modes are the right singular vectors of the profiles scaled
    by the square root of :func:`turbigen.bconds.face_weights` --- so they are
    orthonormal in the span integral, the same inner product the fit uses.

    Parameters
    ----------
    spf : (n_spf,) array
        Span fractions every profile is sampled at, hub to casing.
    profiles : dict
        ``(n_run, n_spf)`` array per column name, for any of
        :data:`turbigen.bconds.InletProfile.COLUMNS`.
    n_mode : int
        Modes to keep per column, most energetic first.
    trim : int
        Stations dropped at each wall before decomposing, matching
        :data:`turbigen.iterate.WALL_TRIM`.

    Returns
    -------
    dict
        ``spf`` running from exactly 0 to exactly 1; an ``(n_mode, n_spf + 2)``
        array per column, its end values held out to the walls; and
        ``singular_<column>``, every singular value, for choosing an order.

    """
    spf = np.asarray(spf, dtype=float)
    inner = slice(trim, spf.size - trim)
    stations = spf[inner]
    if stations[0] <= 0.0 or stations[-1] >= 1.0:
        raise ValueError(
            "The stations kept must lie strictly inside the span, so that the "
            "basis can be closed at spf 0 and 1."
        )

    root_weight = np.sqrt(bconds.face_weights(stations))
    closed = np.concatenate([[0.0], stations, [1.0]])
    closed_weight = bconds.face_weights(closed)

    arrays = {"spf": closed}
    for name, values in profiles.items():
        if name not in bconds.InletProfile.COLUMNS:
            raise ValueError(f"{name} is not an inlet profile column.")
        X = np.asarray(values, dtype=float)[:, inner]
        weight = root_weight**2
        X = X - ((X @ weight) / weight.sum())[:, None]

        _, singular, Vt = np.linalg.svd(X * root_weight, full_matrices=False)
        if n_mode > Vt.shape[0]:
            raise ValueError(
                f"{n_mode} {name} modes asked for, but {Vt.shape[0]} profiles "
                f"can give at most that many."
            )
        modes = Vt[:n_mode] / root_weight

        # Held flat to the walls, then the level the closing stations add is
        # removed again, so the table passes the no-level check it is read by.
        modes = np.hstack([modes[:, :1], modes, modes[:, -1:]])
        modes -= ((modes @ closed_weight) / closed_weight.sum())[:, None]

        # A singular vector's sign is arbitrary; fix it so rebuilding from the
        # same runs gives the same file.
        largest = modes[np.arange(n_mode), np.argmax(np.abs(modes), axis=1)]
        modes *= np.sign(largest)[:, None]

        arrays[name] = modes
        arrays[f"singular_{name}"] = singular

    return arrays


def write_basis(path, arrays, **provenance):
    """Write `arrays` from :func:`build_basis` to `path`, returning its SHA-256.

    `provenance` is stored as strings beside the modes --- where the profiles
    came from, which runs --- and is never read back to evaluate anything.
    """
    path = Path(path)
    extra = {key: np.asarray(str(value)) for key, value in provenance.items()}
    extra["version"] = np.asarray(str(FORMAT_VERSION))
    with open(path, "wb") as stream:
        np.savez(stream, **arrays, **extra)
    return hashlib.sha256(path.read_bytes()).hexdigest()
