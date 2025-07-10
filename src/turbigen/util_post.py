"""Functions for post processing, without plotting."""

import numpy as np


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

    # Extract surface distance and normalise to [-1, 1]
    zeta_stag = C.zeta_stag
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
    if use_rot:
        P1, P2 = meanline_row.P_rot
    else:
        P1, P2 = meanline_row.P

    # Get blade surface and slice at span fraction
    surf = grid.cut_blade_surfs(offset)[irow][0]
    xr_spf = machine.ann.get_span_curve(spf)
    surf = surf.meridional_slice(xr_spf)

    # Get surface distance and pressure
    if use_rot:
        P = surf.P_rot
    else:
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
