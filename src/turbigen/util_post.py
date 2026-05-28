"""Functions for post processing, without plotting."""

import numpy as np
import ember.block
import ember.block_util
import ember.cut
import ember.patch
from turbigen import util

import resource
import logging

logger = logging.getLogger("turbigen")


def _log_ram(label):
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    logger.debug(f"RAM [{label}]: {rss_gb:.2f} GB")


def get_zeta(block):
    """Calculate arc length along i-gridlines.

    Computes cumulative arc length along the i-direction (axis 0) for each
    j-k gridline. The arc length starts at zero at i=0 and increases along
    the i-direction.

    Parameters
    ----------
    block : ember.block.Block
        Block with initialized x, r, t coordinates

    Returns
    -------
    ndarray, shape (ni, nj, nk)
        Cumulative arc length along each i-gridline, with zeta=0 at i=0

    Notes
    -----
    Arc length is calculated in Cartesian (x, y, z) coordinates where:
    - y = r * sin(t)
    - z = r * cos(t)
    """
    # Convert cylindrical to Cartesian coordinates
    x = block.x
    y = block.r * np.sin(block.t)
    z = block.r * np.cos(block.t)

    # Stack coordinates: shape (3, ni, nj, nk)
    xyz = np.stack((x, y, z))

    # Calculate differences along i-direction (axis=1 in xyz array)
    dxyz = np.diff(xyz, n=1, axis=1) ** 2.0

    # Sum squared differences and take sqrt to get segment lengths
    # Shape: (1, ni-1, nj, nk)
    ds = np.sqrt(np.sum(dxyz, axis=0, keepdims=True))

    # Cumulative sum with initial zero
    # Insert 0 at beginning along i-direction, shape: (1, ni, nj, nk)
    zeta = np.insert(np.cumsum(ds, axis=1), 0, 0.0, axis=1)

    # Remove leading dimension and return: shape (ni, nj, nk)
    return zeta[0]


def get_i_stag(block, xrt_LE=None):
    """Find i-index of stagnation point for each j-line in a 2D block.

    Locates the stagnation point by finding pressure maxima near the leading
    edge of each spanwise (j) gridline. Uses rotary static pressure to account
    for centrifugal pressure gradients in rotating frames.

    Parameters
    ----------
    block : ember.block.Block
        2D block (shape (ni, nj)) with initialized flow field
    xrt_LE : ndarray, shape (3,), optional
        Geometric leading-edge coordinates ``[x, r, t]``. When supplied,
        the LE search window is centred on the grid index nearest this
        point (per j-line), instead of on the arc-length midpoint of the
        cut. This is more robust on blades with strong PS/SS asymmetry.

    Returns
    -------
    ndarray, shape (nj,)
        i-index of stagnation point for each j-line

    Raises
    ------
    ValueError
        If block is not 2D (ndim != 2)
    """
    if block.ndim != 2:
        raise ValueError(
            f"Can only find stagnation point on 2D cuts; "
            f"this block has shape {block.shape}"
        )

    P = block.P_rot

    # Get arc length and normalize to [-1, 1] on each j-line
    zeta = get_zeta(block)
    z = zeta / np.ptp(zeta, axis=0) * 2.0 - 1.0

    _, nj = block.shape[:2]

    # Per-j-line window centre in normalised arc length
    if xrt_LE is not None:
        dx = block.xrt[:, :, 0] - xrt_LE[0]
        dr = block.xrt[:, :, 1] - xrt_LE[1]
        dt = block.xrt[:, :, 2] - xrt_LE[2]
        # r-weight the angular component since xrt third coord is angle
        r_avg = 0.5 * (block.xrt[:, :, 1] + xrt_LE[1])
        d2 = dx**2 + dr**2 + (r_avg * dt) ** 2
        i_nose = np.argmin(d2, axis=0)
        z_nose = z[i_nose, np.arange(nj)]
    else:
        z_nose = np.zeros((nj,))

    half_window = 0.1

    i_stag = np.full((nj,), 0, dtype=int)
    for j in range(nj):
        z_centre = z_nose[j]

        # Calculate pressure gradient
        dP = np.diff(P[:, j])

        # Find indices of downward zero crossings (pressure maxima)
        izj = np.where(np.diff(np.sign(dP[:-2])) < 0.0)[0] + 1

        # Only keep maxima close to leading edge
        izj = izj[np.abs(z[izj, j] - z_centre) < half_window]

        if len(izj):
            i_stag[j] = izj[np.argmax(P[izj, j])]
        elif xrt_LE is not None:
            # Restrict fallback to the same window when nose is supplied
            mask = np.abs(z[:, j] - z_centre) < half_window
            idx = np.where(mask)[0]
            i_stag[j] = idx[np.argmax(P[idx, j])]
        else:
            i_stag[j] = np.argmax(P[:, j])

    return i_stag


def get_zeta_stag(block, i_stag):
    """Sub-cell stagnation arc-length via parabolic fit in zeta.

    Refines the integer stagnation index from ``get_i_stag`` to a
    continuous arc-length coordinate by fitting a parabola through the
    three points ``(i-1, i, i+1)`` of rotary static pressure versus
    arc-length on each j-line, and locating its vertex.

    Parameters
    ----------
    block : ember.block.Block
        2D block (shape (ni, nj)) with initialized flow field.
    i_stag : ndarray, shape (nj,)
        Integer stagnation index per j-line, as returned by ``get_i_stag``.

    Returns
    -------
    ndarray, shape (nj,)
        Sub-cell stagnation arc-length per j-line, in the same units as
        ``get_zeta(block)``.
    """
    P = block.P_rot
    zeta = get_zeta(block)
    ni, nj = block.shape[:2]

    # Clamp so i-1, i+1 stay in range
    i = np.clip(i_stag, 1, ni - 2)
    j = np.arange(nj)

    z0, z1, z2 = zeta[i - 1, j], zeta[i, j], zeta[i + 1, j]
    p0, p1, p2 = P[i - 1, j], P[i, j], P[i + 1, j]

    d01 = z1 - z0
    d12 = z2 - z1
    s01 = (p1 - p0) / d01
    s12 = (p2 - p1) / d12
    curv = (s12 - s01) / (z2 - z0)
    slope_mid = 0.5 * (s01 + s12)

    # Fall back to the node when the triple is not concave-down
    bad = curv >= 0.0
    delta = np.where(bad, 0.0, -slope_mid / (2.0 * curv))

    # Keep the vertex inside the bracket
    delta = np.clip(delta, -d01, d12)

    return z1 + delta


def get_isen_mach(
    grid,
    machine,
    meanline,
    irow,
    spf,
    offset=0,
):
    """Extract blade surface pressure distribution from a grid.

    Parameters
    ----------
    grid : Grid
        Grid object containing full flowfield solution.
    machine :
        Machine geometry object.
    meanline :
        Meanline object containing reference pressures.
    irow : int
        Row index to extract.
    spf: float
        Span fraction within the row to extract.
    offset : int
        Number of cells away from blade surface.
    use_rot: bool
        Use rotary static pressure to take out centrifugal effects.

    Returns
    -------
    zeta_norm: (ni,) array
        Surface distance normalised by total surface length on each surface.
        This is a looped array which goes from TE to LE and back again.
        The final point is repeated to close the loop.
    Mas: (ni,) array
        Isentropic Mach number around the blade surface.

    """

    # Extract reference entropy
    s1 = meanline.get_row(irow).s[0]

    # Get blade surface and slice at span fraction
    surf = grid.cut_blade_surfs(offset)[irow][0]
    xr_spf = machine.ann.get_span_curve(spf)
    C = surf.meridional_slice(xr_spf)

    # Isentropic from inlet entropy to local static
    Cs = C.copy().set_P_s(C.P, s1)
    hs = Cs.h
    ho = C.ho_rel
    # Ensure ho > hs
    dh = ho - hs
    hs += np.min(dh)
    Vs = np.sqrt(2.0 * np.maximum(ho - hs, 0.0))
    Mas = Vs / C.a

    # Ensure that minimum Mach number is not negative
    zeta_stag = C.zeta_stag.copy()
    zeta_stag -= zeta_stag[np.argmin(Mas)]

    # Normalise to [-1, 1]
    zeta_max = zeta_stag.max(axis=0)
    zeta_min = np.abs(zeta_stag.min(axis=0))
    zeta_norm = zeta_stag.copy()
    zeta_norm[zeta_norm < 0.0] /= zeta_min
    zeta_norm[zeta_norm > 0.0] /= zeta_max

    return zeta_norm, Mas


def get_pressure_distribution(
    grid,
    machine,
    meanline,
    irow,
    spf,
    offset=0,
    use_rot=False,
    normalise=True,
):
    """Extract blade surface pressure distribution from a grid.

    Parameters
    ----------
    grid : Grid
        Grid object containing full flowfield solution.
    machine :
        Machine geometry object.
    meanline :
        Meanline object containing reference pressures.
    irow : int
        Row index to extract.
    spf: float
        Span fraction within the row to extract.
    offset : int
        Number of cells away from blade surface.
    use_rot: bool
        Use rotary static pressure to take out centrifugal effects.
    normalise: true
        If true normalise returned zeta to [-1, 1]. Starts at -1, rises through
        zero at the stagnation point, and carries on increasing to 1 at
        the trailing edge. If false, just return the raw arc length.


    Returns
    -------
    zeta_norm: (ni,) array
        Surface distance normalised by total surface length on each surface.
        This is a looped array which goes from TE to LE and back again.
        The final point is repeated to close the loop.
    Cp: (ni,) array
        Pressure coefficient distribution around the blade surface.

    """

    # Extract reference pressures
    meanline_row = meanline.get_row(irow)
    Po1 = meanline_row.Po_rel[0]
    # if use_rot:
    #     P1, P2 = meanline_row.P_rot
    # else:
    P1, P2 = meanline_row.P

    # Get blade surface and slice at span fraction
    surf = grid.cut_blade_surfs(offset)[irow][0]
    xr_spf = machine.ann.get_span_curve(spf)
    surf = surf.meridional_slice(xr_spf)

    # Get surface distance and pressure
    # if use_rot:
    #     P = surf.P_rot
    # else:
    P = surf.P

    # Extract surface distance and normalise to [-1, 1]
    zeta_stag = surf.zeta_stag
    zeta_max = zeta_stag.max(axis=0)
    zeta_min = np.abs(zeta_stag.min(axis=0))
    zeta_norm = zeta_stag.copy()
    if normalise:
        zeta_norm[zeta_norm < 0.0] /= zeta_min
        zeta_norm[zeta_norm > 0.0] /= zeta_max

    # Choose compressor or turbine non-dimensionalisation
    if P2 > P1:
        # Compressor
        Cp = (P - Po1) / (Po1 - P1)
    else:
        # Turbine
        Cp = (P - Po1) / (Po1 - P2)

    return zeta_norm, Cp


def get_diffusion_factor(
    grid,
    machine,
    meanline,
    irow,
    spf,
):
    """Calculate diffusion factor for a blade in the machine."""

    zeta_norm, Cp = get_pressure_distribution(grid, machine, meanline, irow, spf)

    # Calculate diffusion factor
    # DF = (Vmax - V2)/V2 for turbines
    # Curtis et al. (1997) Eqn. (2)
    Cp_peak = Cp.min()
    Cp_stag = Cp.max()
    Cp_TE = 0.5 * (Cp[-1] + Cp[0]).item()
    DF = np.sqrt((Cp_TE - Cp_peak) / (Cp_stag - Cp_TE))

    # Peak suction location at minimum pressure coeff
    # Absolute value for if surf coord on SS is -ve
    xpeak = np.abs(zeta_norm[Cp.argmin()].item())

    return xpeak, DF


def separate_waves(F, fs):
    """Perform least-squares wave separation for a microphone set.
    Assumes that time is on the last dimension.
    And only accepts 2D Flowfields

    Parameters
    ----------
    F: FlowField shape (nprobe, nt)
        Unsteady probe data
    fs: float
        Sampling frequency [Hz]

    Returns
    -------
    W: array (2, nf)
        W[0] is upstream-running amplitudes at all frequencies
        W[1] is downstream-running amplitudes at all frequencies
        Normalised by the time-mean pressure
    err: array (nf,)
        Error in the least-squares fit at all frequencies
    f: array (nf,)
        Frequencies at which the wave amplitudes are defined [Hz]

    """

    assert F.ndim == 2

    # Pressure fluctuations wrt mean
    Pav = np.mean(F.P, axis=1, keepdims=True)
    Pprime = F.P - Pav

    # Go to frequency domain
    nt = F.shape[1]
    f = np.fft.rfftfreq(nt, 1.0 / fs)
    Pfft = np.fft.rfft(Pprime, axis=1) / nt * 2.0
    nf = len(f)

    # Wavenumbers
    f1 = f.reshape(1, -1)
    kp = 2.0 * np.pi * f1 / (F.a + F.Vx).mean()
    km = 2.0 * np.pi * f1 / (F.a - F.Vx).mean()

    # Axial coordinates
    x = F.x[:, (0,)]

    # Matrix problem
    A = np.stack((np.exp(-1j * kp * x), np.exp(1j * km * x)), axis=1)

    # Loop over frequencies and solve
    Pwav = np.empty((2, nf), dtype=complex)
    err = np.empty((nf,))
    for n in range(nf):
        b = Pfft[:, (n,)]
        val, resid = np.linalg.lstsq(A[..., n], b, rcond=None)[:2]
        Pwav[:, n] = val.squeeze()
        ref = np.maximum(np.linalg.norm(b), 1e-9)
        err[n] = np.abs(np.linalg.norm(resid) / ref)

    # Put upstream-running first
    W = np.flip(Pwav, axis=0)

    return W, err, f


def cut_blade_sides(grid, offset=0):
    """Nested list of pressure/suction side cuts in each row.

    Parameters
    ----------
    grid : ember.grid.Grid
        Grid object containing the turbomachinery geometry
    offset : int
        Number of cells away from blade surface (default 0)

    Returns
    -------
    list
        Nested list where each element corresponds to a row.
        For each row, returns [Ck0, Cnk] (pressure/suction sides) or None if not found.
    """

    # Assuming a H-mesh
    cuts = []

    for i in range(len(grid.rows)):
        # Check periodics first
        ile = None
        ite = None
        _log_ram(f"Row {i}: checking periodics for blade sides")

        # Iterate over blocks in this row
        for block in grid.rows[i]:
            # Check periodic patches on this block
            for patch in block.patches.periodic:
                # Check if this is a same-block periodic (pitch-wise)
                # For H-mesh, we look for periodics at k boundary that span j but not i
                lim = patch.ijk_lim_abs
                spans_j = np.allclose(lim[1], [0, block.shape[1] - 1])
                spans_i = np.allclose(lim[0], [0, block.shape[0] - 1])
                # Check if at k=0 or k=-1 boundary (single plane)
                at_k_boundary = (lim[2, 0] == lim[2, 1]) and (
                    lim[2, 0] == 0 or lim[2, 0] == block.shape[2] - 1
                )

                if spans_j and at_k_boundary and not spans_i:
                    if lim[0, 0] == 0:
                        ile = lim[0, 1]
                    elif lim[0, 1] == block.shape[0] - 1:
                        ite = lim[0, 0]

            # Now check cusps and inviscid patches on k faces
            for patch in block.patches:
                if (
                    isinstance(
                        patch, (ember.patch.CuspPatch, ember.patch.InviscidPatch)
                    )
                    and patch.const_dim == 2
                ):
                    ite = patch.ijk_lim_abs[0, 0]

        if not ile or not ite:
            cuts.append(None)
            continue

        # Get both sides
        _log_ram(f"Row {i}: found blade sides at i={ile} and i={ite}, cutting")
        Ck0 = grid[i][ile : (ite + 1), :, None, 0 + offset].copy(keep_patches=False)
        Cnk = grid[i][ile : (ite + 1), :, None, -1 - offset].copy(keep_patches=False)
        _log_ram(f"Row {i}: cut blade sides, now clearing patches")

        # Clear patches as they are no longer valid for sliced blocks
        Ck0.patches.clear()
        Cnk.patches.clear()

        C = [Ck0, Cnk]

        # Find the side at highest theta and adjust by pitch
        iu = np.argmax([Ci.t.max() for Ci in C])
        C[iu].set_t(C[iu].t - grid[i].pitch)

        cuts.append(C)

    assert len(cuts) == len(grid.rows)
    assert all(len(c) == 2 for c in cuts)
    return cuts


def cut_blade_surfs(grid, offset=0):
    """O-mesh style cuts for the blades in each row.

    Parameters
    ----------
    grid : ember.grid.Grid
        Grid object containing the turbomachinery geometry
    offset : int
        Number of cells away from blade surface (default 0)

    Returns
    -------
    list
        Nested list where each element corresponds to a row.
        For each row, returns a list of blade surface cuts (FlowField objects).
    """

    surfs = []

    # Check if this is an H-mesh (one block per row)
    is_hmesh = len(grid) == len(grid.rows)

    if is_hmesh:
        _log_ram("Before cutting blade sides")
        row_sides = cut_blade_sides(grid, offset)
        _log_ram(f"After cutting blade sides length {len(row_sides)}")
        for sides in row_sides:
            if sides is None:
                surfs.append(None)
            else:
                cut_now = ember.block_util.concatenate(
                    sides[0].flip(axis=0), sides[1][1:, ...], axis=0
                )
                surfs.append([cut_now])
        _log_ram("After cutting blade surfaces from sides")
    else:
        for row_block in grid.rows:
            # Preallocate list for this row
            surfs.append([])

            # Determine full span nj as the modal nj in this row
            nj_vals, nj_counts = np.unique(
                [b.shape[1] for b in row_block], return_counts=True
            )
            nj = nj_vals[np.argmax(nj_counts)]

            # Loop over blocks and find o-meshes
            for b in row_block:
                if np.allclose(b[0, :, 0].xrt, b[-1, :, 0].xrt) and b.shape[1] == nj:
                    surfs[-1].append(b[:, :, None, offset])

    return surfs


def cut_span(grid, annulus, spf):
    """Cut the grid at a constant span fraction.

    Parameters
    ----------
    grid : ember Grid
        The 3D CFD grid to cut.
    annulus : AnnulusDesigner
        Annulus object used to define the cut surface geometry.
    spf : float
        Span fraction at which to cut, 0 is hub, 1 is casing.

    Returns
    -------
    list of ember.block.Block
        2D structured blocks along the cut surface.
    """
    xr_cut = annulus.get_span_curve(spf)
    cuts = ember.cut.structured_meridional(grid, xr_cut.T)
    return list(cuts)


def incidence_unstructured(grid, machine, ml, irow, spf, plot=False):
    # Pull out 2D cuts of blades and splitters
    _log_ram("Start incidence_unstructured")

    surfs = cut_blade_surfs(grid)[irow]
    _log_ram("Cut blade surfaces")

    nspf = len(spf)

    # Meridional curves for target span fractions
    ist = irow * 2 + 1
    ien = ist + 1
    m = np.linspace(ist, ien, 101)
    xr_spf = machine.ann.evaluate_xr(m.reshape(-1, 1), spf.reshape(1, -1)).reshape(
        2, -1, nspf
    )
    _log_ram("Prepared meridional curves")

    # Meridional velocity vector at inlet to this row
    Vxrt = ml[irow * 2].Vxrt_rel
    _log_ram("Extracted inlet velocity")

    # Loop over main/splitter
    chi = []
    for jbld, surfj in enumerate(surfs):
        _log_ram(f"Processing blade {jbld}")
        surf = surfj.squeeze()
        _log_ram("Squeezed surface")

        # Get the current blade object
        bldnow = machine.bld[irow][jbld]
        _log_ram("Got blade object")

        # Loop over span fractions
        # Unstructure cut through current surface along the
        # target span fraction curves
        xrt_stag = np.zeros((3, nspf))
        xrt_nose = np.zeros((3, nspf))
        xrt_cent = np.zeros((3, nspf))
        for k in range(len(spf)):
            # Cut at this span fraction
            _log_ram(f"Cutting blade {jbld} at spf {spf[k]:.2f}")
            C = ember.cut.structured_meridional(surf[..., None], xr_spf[:, :, k].T)

            # Geometric nose coordinates (used to anchor the stag search)
            xrt_nose[:, k] = bldnow.get_nose(spf[k])
            _log_ram("Got geometric nose coordinates")

            # Stag point coordinates with sub-cell parabolic refinement
            # in arc-length space, so that small flow changes produce a
            # continuous stagnation location instead of jumping between
            # neighbouring grid points.
            istag = get_i_stag(C[0], xrt_LE=xrt_nose[:, k])
            zeta_s = get_zeta_stag(C[0], istag)[0]
            _log_ram(f"Found stagnation point at i={istag[0]}")
            zeta_line = get_zeta(C[0])[:, 0]
            xrt_line = C[0].xrt[:, 0, :]
            xrt_stag[:, k] = [
                np.interp(zeta_s, zeta_line, xrt_line[:, c]) for c in range(3)
            ]

            # Leading edge centre
            xrt_cent[:, k] = bldnow.get_LE_cent(spf[k], 5.0)
            _log_ram("Got leading edge centre coordinates")

        # Calculate the angles
        chi_metal = util.yaw_from_xrt(xrt_nose, xrt_cent, Vxrt)
        chi_flow = util.yaw_from_xrt(xrt_stag, xrt_cent, Vxrt, yaw_ref=chi_metal)

        chi.append(np.stack((chi_metal, chi_flow)))

    return chi
