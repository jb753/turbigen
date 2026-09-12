"""
.. _shapespace:

Shape space
^^^^^^^^^^^

The arithmetic that :doc:`camber lines </blade>` and thickness distributions
have in common, as free functions over plain arrays. Nothing here is a
:class:`~turbigen.node.Node` and nothing here holds state: a shape is a
:class:`~turbigen.camber.CamberDesign` or a
:class:`~turbigen.thickness.ThicknessDesign`, and this is what those are
written in terms of.

A curve is written against a normalised meridional coordinate ``m``, zero at
the leading edge and one at the trailing edge. *Shape space* is the curve
``tau`` left after dividing a half-thickness by the class function
``sqrt(m) * (1 - m)``, which is what carries the square-root nose and the
closing trailing edge; :func:`thickness_from_tau` and
:func:`tau_from_thickness` move between the two. :cite:`Taylor2016` writes a
cubic in that space and Clark (2019) a Bernstein polynomial, but both are the
same transform, and the two ends of it are physical: the value at the leading
edge is fixed by the leading edge radius (:func:`tau_LE`) and the value at the
trailing edge by the trailing edge thickness and wedge angle
(:func:`tau_TE`).

Two different linear terms appear, and they are not the same thing:

* the trailing edge thickness enters *physical* thickness, as the ``t_TE / 2``
  ramp added outside the class function by :func:`thickness_from_tau`;
* a curve written as a straight line plus a Bernstein perturbation carries a
  ramp in *shape space*, which stays at the call site rather than living here.
  There is no function for it, because evaluation is linear and the line is
  exactly representable in the basis: adding the line to the result of
  :func:`evaluate_bernstein` gives the same curve as adding it to the
  coefficients, and doing it to the result keeps the coefficients purely the
  perturbation. That matters because :func:`elevate_bernstein` leaves the end
  coefficients alone, so a perturbation pinned at zero stays pinned however it
  is elevated --- and a leading edge radius or a wedge angle written into the
  ends of a curve cannot drift when something moves the interior.
"""

import math

import numpy as np


def validate_domain(m):
    """Check that a normalised meridional coordinate lies in [0, 1]."""
    if np.any(np.asarray(m) < 0.0) or np.any(np.asarray(m) > 1.0):
        raise ValueError("Meridional distance m must be in the range [0, 1].")


def evaluate_bernstein(coeff, m):
    """Return the Bernstein polynomial with coefficients `coeff` at `m`.

    Parameters
    ----------
    coeff : array_like, shape (order + 1,)
        Every coefficient of the curve, ends included --- an order `n` curve
        takes `n + 1` of them. The first and last are the values at `m = 0`
        and `m = 1`.
    m : array_like
        Normalised meridional positions to evaluate at.

    Returns
    -------
    ndarray or float
        The curve at each position, scalar if `m` was one.

    """
    coeff = np.asarray(coeff, dtype=float)
    if coeff.ndim != 1 or len(coeff) < 2:
        raise ValueError(
            f"A Bernstein curve needs order + 1 coefficients, at least two, "
            f"got {coeff.shape}."
        )

    m = np.asarray(m, dtype=float)
    scalar = m.ndim == 0
    m = np.atleast_1d(m)

    n = len(coeff) - 1
    binom = np.array([math.comb(n, k) for k in range(n + 1)], dtype=float)

    # Powers by repeated multiplication rather than by `**`. Every exponent
    # from zero to `n` is wanted, so each one is the last times another factor
    # and `np.power` is being asked to rediscover that on every row --- three
    # to six times the cost, for a result that differs only in the last bit.
    # A mesh evaluates this a thousand times over ten thousand chordwise
    # points, which is where the difference shows up.
    powers_m = np.empty((n + 1, m.size))
    powers_1m = np.empty((n + 1, m.size))
    powers_m[0] = 1.0
    powers_1m[0] = 1.0
    one_m = 1.0 - m
    for k in range(1, n + 1):
        powers_m[k] = powers_m[k - 1] * m
        powers_1m[k] = powers_1m[k - 1] * one_m

    basis = binom[:, None] * powers_m * powers_1m[::-1]

    value = coeff @ basis
    return value[0] if scalar else value


def elevate_bernstein(coeff, order):
    """Return `coeff` rewritten in the basis of degree `order`.

    An exact identity, not a fit: the elevated coefficients describe the same
    curve, written in a basis with room for more of them. The end
    coefficients are carried over untouched, so a curve pinned at its ends
    stays pinned --- which is what lets a low-order design be read as a
    high-order one and then perturbed further.

    Sampling the low-order curve and solving a matrix problem to match points
    would only approximate this, and get worse-conditioned as the order grows.

    Parameters
    ----------
    coeff : array_like, shape (n_low + 1,)
        Every coefficient of the curve, ends included.
    order : int
        Degree to raise the curve to, at least its own.

    Returns
    -------
    ndarray, shape (order + 1,)
        The same curve, in the higher basis.

    """
    b = np.asarray(coeff, dtype=float)
    if b.ndim != 1 or len(b) < 2:
        raise ValueError(
            f"A Bernstein curve needs order + 1 coefficients, at least two, "
            f"got {b.shape}."
        )

    n_low = len(b) - 1
    if order < n_low:
        raise ValueError(
            f"Cannot elevate an order {n_low} curve to order {order}; "
            f"degree elevation only raises the order."
        )

    for _ in range(order - n_low):
        n = len(b) - 1
        i = np.arange(n + 2)
        left = np.concatenate(([0.0], b))
        right = np.concatenate((b, [0.0]))
        t = i / (n + 1)
        b = t * left + (1.0 - t) * right

    return b


def control_m(order):
    """Return where each control point of a shape-space curve moves thickness most.

    A Bernstein basis function peaks at ``k / order``, but a coefficient in
    shape space is not what a blade is made of: the thickness it produces is
    the curve times the class function ``sqrt(m) * (1 - m)``, which vanishes at
    both ends and peaks at a third of chord. Multiplying by it kills the
    boundary peaks that ``B_0`` and ``B_order`` have and drags every interior
    one toward a third. Maximising

    .. math::
        \\sqrt{m}(1 - m) \\, B_{k,n}(m)
        \\;\\propto\\; m^{k + 1/2} (1 - m)^{n - k + 1}

    gives ``(k + 1/2) / (order + 3/2)``, strictly inside `(0, 1)` for every `k`
    including the ends --- so the first and last control points act just inside
    the nose and just ahead of the trailing edge rather than on them. That they
    do nothing *at* the ends is correct: half-thickness is zero at `m = 0` for
    any nose radius, since the radius lives in the curvature of the square-root
    nose rather than in a thickness value, and is `t_TE / 2` at `m = 1`
    whatever the wedge angle does.

    **Where a coefficient moves the thickness, which is a proxy for where it
    moves the loading.** A thickness bump accelerates the flow over it, but a
    pressure field is not local, so the two peaks need not coincide exactly.
    Checked against a finished section rather than against the formula alone:
    perturbing one coefficient displaces the surface most at this `m` to four
    decimal places, so the annulus lift, the leading-to-trailing edge rescaling
    and the perpendicular thickness offset do not move it.

    Parameters
    ----------
    order : int
        Degree of the curve, which has `order + 1` control points.

    Returns
    -------
    ndarray, shape (order + 1,)
        Normalised meridional position of each control point's peak influence.

    """
    if order < 1:
        raise ValueError(f"A curve needs a degree of at least one, got {order}.")

    return (np.arange(order + 1) + 0.5) / (order + 1.5)


def thickness_from_tau(m, tau, t_TE):
    """Return half-thickness from a curve `tau` in shape space.

    The class function ``sqrt(m) * (1 - m)`` puts a square-root nose on the
    leading edge and closes the trailing edge; the trailing edge thickness is
    then added back as a linear ramp, so that half of it is left at `m = 1`.

    Parameters
    ----------
    m : array_like
        Normalised meridional positions.
    tau : array_like
        Shape space curve at those positions.
    t_TE : float
        Trailing edge thickness, the total due to both sides [--].

    """
    m = np.asarray(m, dtype=float)
    return np.sqrt(m) * (1.0 - m) * np.asarray(tau, dtype=float) + m * t_TE / 2.0


def tau_from_thickness(m, t, t_TE):
    """Return shape space from a half-thickness, inverting :func:`thickness_from_tau`.

    How a thickness distribution measured off an existing aerofoil is read
    into the parametrisation. Undefined at both ends, where the class
    function vanishes and every shape space curve gives the same
    half-thickness: ask :func:`tau_LE` and :func:`tau_TE` for those instead,
    which is where the leading edge radius and the wedge angle come from.

    Parameters
    ----------
    m : array_like
        Normalised meridional positions, strictly between 0 and 1.
    t : array_like
        Half-thickness at those positions.
    t_TE : float
        Trailing edge thickness, the total due to both sides [--].

    """
    m = np.asarray(m, dtype=float)
    validate_domain(m)
    if np.any(m <= 0.0) or np.any(m >= 1.0):
        raise ValueError(
            "Shape space is undefined at the ends of the chord, where the "
            "class function vanishes; take tau_LE and tau_TE from the "
            "leading edge radius and the wedge angle instead."
        )

    return (np.asarray(t, dtype=float) - m * t_TE / 2.0) / (np.sqrt(m) * (1.0 - m))


def tau_LE(R_LE):
    """Return the shape space value at the leading edge, for a radius `R_LE`."""
    return np.sqrt(2.0 * R_LE)


def R_LE_from_tau(tau):
    """Return the leading edge radius of a curve that starts at `tau` [--]."""
    return tau**2.0 / 2.0


def tau_TE(t_TE, tanwedge):
    """Return the shape space value at the trailing edge.

    Parameters
    ----------
    t_TE : float
        Trailing edge thickness, the total due to both sides [--].
    tanwedge : float
        Tangent of the trailing edge wedge angle [--].

    """
    return t_TE + tanwedge


def tanwedge_from_tau(tau, t_TE):
    """Return the wedge angle tangent of a curve that ends at `tau` [--]."""
    return tau - t_TE
