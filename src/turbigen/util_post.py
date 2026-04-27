"""Functions for post processing, without plotting."""

import numpy as np
import ember.block
import ember.cut
import ember.patch
from turbigen import util


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


def get_i_stag(block):
    """Find i-index of stagnation point for each j-line in a 2D block.

    Locates the stagnation point by finding pressure maxima near the leading
    edge of each spanwise (j) gridline. Uses rotary static pressure to account
    for centrifugal pressure gradients in rotating frames.

    Parameters
    ----------
    block : ember.block.Block
        2D block (shape (ni, nj)) with initialized flow field

    Returns
    -------
    ndarray, shape (nj,)
        i-index of stagnation point for each j-line

    Raises
    ------
    ValueError
        If block is not 2D (ndim != 2)
        If no valid stagnation point found on any j-line

    Notes
    -----
    Algorithm:
    1. Uses rotary static pressure (P_rot) to remove centrifugal effects
    2. Normalizes arc length to [-1, 1] on each j-line
    3. Finds pressure maxima (downward zero crossings of dP/dzeta)
    4. Filters to keep only maxima near LE (|zeta_normalized| < 0.2)
    5. Selects candidate with highest pressure
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

    # Find pressure maxima on each j-line
    _, nj = block.shape[:2]
    i_stag = np.full((nj,), 0, dtype=int)

    for j in range(nj):
        # Calculate pressure gradient
        dP = np.diff(P[:, j])

        # Find indices of downward zero crossings (pressure maxima)
        # Looking for where gradient changes from positive to negative
        izj = np.where(np.diff(np.sign(dP[:-2])) < 0.0)[0] + 1

        # Only keep maxima close to leading edge
        izj = izj[np.abs(z[izj, j]) < 0.2]

        # Select the candidate with maximum pressure
        if len(izj):
            # Take the point with highest pressure among candidates
            i_stag[j] = izj[np.argmax(P[izj, j])]
        else:
            # Take highest pressure anywhere if none near LE
            i_stag[j] = np.argmax(P[:, j])

    return i_stag


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
    print("beans")

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
        Ck0 = grid[i][ile : (ite + 1), :, None, 0 + offset].copy()
        Cnk = grid[i][ile : (ite + 1), :, None, -1 - offset].copy()

        # Clear patches as they are no longer valid for sliced blocks
        Ck0.patches.clear()
        Cnk.patches.clear()

        C = [Ck0, Cnk]

        # Find the side at highest theta and adjust by pitch
        iu = np.argmax([Ci.t.max() for Ci in C])
        C[iu].set_t(C[iu].t - grid[i].pitch)

        cuts.append(C)

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
        row_sides = cut_blade_sides(grid, offset)
        for sides in row_sides:
            if sides is None:
                surfs.append(None)
            else:
                cut_now = ember.block.concatenate(
                    sides[0].flip(axis=0), sides[1][1:, ...], axis=0
                )
                surfs.append([cut_now])
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
    surfs = cut_blade_surfs(grid)[irow]

    nspf = len(spf)

    # Meridional curves for target span fractions
    ist = irow * 2 + 1
    ien = ist + 1
    m = np.linspace(ist, ien, 101)
    xr_spf = machine.ann.evaluate_xr(m.reshape(-1, 1), spf.reshape(1, -1)).reshape(
        2, -1, nspf
    )

    # Meridional velocity vector at inlet to this row
    Vxrt = ml[irow * 2].Vxrt_rel

    # Loop over main/splitter
    chi = []
    for jbld, surfj in enumerate(surfs):
        surf = surfj.squeeze()

        # Get the current blade object
        bldnow = machine.bld[irow][jbld]

        # Loop over span fractions
        # Unstructure cut through current surface along the
        # target span fraction curves
        xrt_stag = np.zeros((3, nspf))
        xrt_nose = np.zeros((3, nspf))
        xrt_cent = np.zeros((3, nspf))
        for k in range(len(spf)):
            # Cut at this span fraction
            C = ember.cut.structured_meridional(surf[..., None], xr_spf[:, :, k].T)

            # Stag point coordinates
            istag = get_i_stag(C[0])[0]
            xrt_stag[:, k] = C[0].xrt[istag, 0, :]

            # Geometric nose coordinates
            xrt_nose[:, k] = bldnow.get_nose(spf[k])

            # Leading edge centre
            xrt_cent[:, k] = bldnow.get_LE_cent(spf[k], 5.0)

        # Calculate the angles
        chi_metal = util.yaw_from_xrt(xrt_nose, xrt_cent, Vxrt)
        chi_flow = util.yaw_from_xrt(xrt_stag, xrt_cent, Vxrt, yaw_ref=chi_metal)

        chi.append(np.stack((chi_metal, chi_flow)))

    return chi
