"""
Core functions for generating geometry from stage aerodynamic parameters.
"""
import scipy.optimize
import scipy.integrate
import compflow
import numpy as np
from collections import namedtuple

# Constants
muref = 1.8e-5
Tref = 288.0

# Define a namedtuple to store all information about a stage design
NonDimStage = namedtuple('NonDimStage', [
    'Yp',
    'Ma',
    'Ma_rel',
    'Al',
    'Al_rel',
    'Lam',
    'Ax_Axin',
    'U_sqrt_cpToin',
    'Pout_Poin',
    'Po_Poin',
    'To_Toin',
])


def nondim_stage_from_Al(
    phi, psi, Al,  # Velocity triangles
    Ma, ga,  # Compressibility
    eta,  # Loss
    Vx_rat=(1., 1.)  # Variation in Vx
):
    """Get N-D geometry, blade speed for set of N-D aerodynamic parameters.

    This compact routine is the only stage design code you will ever need. It
    uses Euler's equation, definitions of the N-D parameters, and standard
    perfect gas compressible flow relations.

    Assumes constant mean radius, angular velocity, hence blade speed.

    From the output of this function, arbitrarily choosing one of Omega or rm
    and providing an inlet stagnation state will completely define the stage in
    dimensional terms.

    TODO - allow for varying the mean radius."""

    # Unpack input
    Ala, Alc = Al

    # First, construct velocity triangles

    # Euler work equation sets tangential velocity upstream of rotor
    # This gives us absolute flow angles everywhere
    tanAlb = np.tan(np.radians(Alc)) * Vx_rat[1] + psi / phi
    Alb = np.degrees(np.arctan(tanAlb))
    Al_all = np.hstack((Ala, Alb, Alc))

    # Get non-dimensional velocities
    Vx_U = np.array([Vx_rat[0], 1., Vx_rat[1]]) * phi
    Vt_U = Vx_U * np.tan(np.radians(Al_all))
    Vtrel_U = Vt_U - 1.
    V_U = np.sqrt(Vx_U**2. + Vt_U**2.)
    Vrel_U = np.sqrt(Vx_U**2. + Vtrel_U**2.)
    Alrel = np.degrees(np.arctan2(Vtrel_U, Vx_U))

    # Use Mach number to get U/cpTo
    V_cpTob = compflow.V_cpTo_from_Ma(Ma, ga)
    U_cpToin = V_cpTob / V_U[1]

    # N-D temperatures from input blade Ma and psi
    cpToin_Usq = 1. / U_cpToin**2
    cpToout_Usq = cpToin_Usq - psi
    cpTo_Usq = np.array([cpToin_Usq, cpToin_Usq, cpToout_Usq])

    # Mach numbers
    Ma_all = compflow.Ma_from_V_cpTo(V_U / np.sqrt(cpTo_Usq), ga)
    Ma_rel = Ma_all * Vrel_U / V_U

    # Capacity
    Q = compflow.mcpTo_APo_from_Ma(Ma_all, ga)
    Q_Qin = Q / Q[0]

    # Second, construct annulus line

    # Use polytropic effy to get entropy change
    To_Toin = cpTo_Usq / cpTo_Usq[0]
    Ds_cp = -(1. - 1. / eta) * np.log(To_Toin[-1])

    # Somewhat arbitrarily, split 50% on stator, 50% on rotor
    s_cp = np.hstack((0., 0.5, 1.)) * Ds_cp

    # Convert to stagnation pressures
    Po_Poin = np.exp((ga / (ga - 1.)) * (np.log(To_Toin) + s_cp))

    # Area ratios = span ratios because rm = const
    cosAl = np.cos(np.radians(Al_all))
    Dr_Drin = np.sqrt(To_Toin) / Po_Poin / Q_Qin * cosAl[0] / cosAl

    # Evaluate some other parameters that we might want to target
    T_Toin = To_Toin / compflow.To_T_from_Ma(Ma_all, ga)
    P_Poin = Po_Poin / compflow.Po_P_from_Ma(Ma_all, ga)
    Porel_Poin = P_Poin * compflow.Po_P_from_Ma(Ma_rel, ga)
    Lam = (T_Toin[2] - T_Toin[1]) / (T_Toin[2] - T_Toin[0])

    Yp_vane = (Po_Poin[0] - Po_Poin[1]) / (Po_Poin[0] - P_Poin[1])
    Yp_blade = (Porel_Poin[1] - Porel_Poin[2]) / (Porel_Poin[1] - P_Poin[2])

    # Assemble all of the data into the output object
    dat = NonDimStage(
        Yp=tuple([Yp_vane, Yp_blade]),
        Al=tuple(Al_all),
        Al_rel=tuple(Alrel),
        Ma=tuple(Ma_all),
        Ma_rel=tuple(Ma_rel),
        Ax_Axin=tuple(Dr_Drin),
        Lam=Lam,
        U_sqrt_cpToin=U_cpToin,
        Pout_Poin=P_Poin[-1],
        Po_Poin=Po_Poin,
        To_Toin=To_Toin,
    )

    return dat


def nondim_stage_from_Lam(
    phi, psi, Lam, Alin,  # Velocity triangles
    Ma, ga,  # Compressibility
    eta,  # Loss
    Vx_rat=(1., 1.)
):
    """This function produces an N-D stage design at fixed reaction.

    It is most convenient to evaluate a design in terms of exit flow angle, and
    not reaction. So iterate on the exit flow angle until the target reaction
    is achieved."""

    # Inner function
    def iter_Al(x):
        stg_now = nondim_stage_from_Al(phi, psi, [Alin, x],
                                       Ma, ga, eta, Vx_rat)
        return stg_now.Lam - Lam

    # Solving for Lam in general is tricky
    # Our strategy is to map out a coarse curve first, pick two points
    # that are close to the desired reaction, then secant iterate
    Al_guess = np.linspace(-89., 89., 9)
    with np.errstate(invalid='ignore'):
        Lam_guess = np.array([iter_Al(Ali) for Ali in Al_guess])
    Al_guess = Al_guess[~np.isnan(Lam_guess)]
    Lam_guess = Lam_guess[~np.isnan(Lam_guess)]
    i1, i2 = np.argmax(Lam_guess), np.argmin(Lam_guess)
    Al_guess, Lam_guess = Al_guess[i1:i2], Lam_guess[i1:i2]
    i0 = np.argmin(np.abs(Lam_guess))

    Al_soln = scipy.optimize.newton(
        iter_Al, x0=Al_guess[i0], x1=Al_guess[i0 - 1])

    stg_out = nondim_stage_from_Al(phi, psi, [Alin, Al_soln],
                                   Ma, ga, eta, Vx_rat)

    return stg_out


def annulus_line(U_sqrt_cpToin, Ax_Axin, htr, cpToin, Omega):
    """Get dimensional annulus line for given stage design and inlet state.

    Parameters
    ----------
    U_sqrt_cpToin : float
        Non-dimensional blade speed.
    Ax_Axin : array, length 3
        Axial area ratio.
    htr : float
        Hub-to-tip radius ratio.
    cpToin : float
        Inlet specific stagnation enthalpy [J/kg].
    Omega : float
        Shaft angular velocity [rad/s].

    Returns
    -------
    rm : array, length 3
        Mean radius [m].
    Dr : array, length 3
        Annulus span [m].
    """

    # Use non-dimensional blade speed to get U, hence mean radius
    U = U_sqrt_cpToin * np.sqrt(cpToin)
    rm = U / Omega

    # Use hub-to-tip ratio to set span (mdot will therefore float)
    Dr_rm = 2. * (1. - htr) / (1. + htr)
    Dr = rm * Dr_rm * np.array(Ax_Axin) / Ax_Axin[1]

    return rm, Dr


def blade_section(chi, a=0.):
    """Makes a simple blade geometry from flow angles."""
    # Copy defaults from MEANGEN (Denton)
    tle = 0.04  # LEADING EDGE THICKNESS/AXIAL CHORD.
    tte = 0.04  # TRAILING EDGE THICKNESS/AXIAL CHORD.
    tmax = 0.25  # MAXIMUM THICKNESS/AXIAL CHORD.
    xtmax = 0.40  # FRACTION OF AXIAL CHORD AT MAXIMUM THICKNESS
    xmodle = 0.02  # FRACTION OF AXIAL CHORD OVER WHICH THE LE IS MODIFIED.
    xmodte = 0.01  # FRACTION OF AXIAL CHORD OVER WHICH THE TE IS MODIFIED.
    tk_typ = 2.0  # FORM OF BLADE THICKNESS DISTRIBUTION.

    # Camber line
    xhat = 0.5 * (1. - np.cos(np.pi * np.linspace(0., 1., 100)))
    tanchi_lim = np.tan(np.radians(chi))
    tanchi = (tanchi_lim[1] - tanchi_lim[0]) * \
        (a * xhat**2. + (1. - a) * xhat) + tanchi_lim[0]
    yhat = np.insert(scipy.integrate.cumtrapz(tanchi, xhat), 0, 0.)

    power = np.log(0.5) / np.log(xtmax)
    xtrans = xhat**power

    # Linear thickness component for leading/trailing edges
    tlin = tle + xhat * (tte - tle)

    # Additional component to make up maximum thickness
    tadd = tmax - (tle + xtmax * (tte - tle))

    # Total thickness
    thick = tlin + tadd * (1.0 - np.abs(xtrans - 0.5)**tk_typ / (0.5**tk_typ))

    # Thin the leading and trailing edges
    xhat1 = 1. - xhat
    fac_thin = np.ones_like(xhat)
    fac_thin[xhat < xmodle] = (xhat[xhat < xmodle] / xmodle)**0.3
    fac_thin[xhat1 < xmodte] = (xhat1[xhat1 < xmodte] / xmodte)**0.3

    # Upper and lower surfaces
    yup = yhat + thick * 0.5 * fac_thin
    ydown = yhat - thick * 0.5 * fac_thin

    # Fillet over the join at thinned LE
    fillet(xhat - xmodle, yup, xmodle / 2.)
    fillet(xhat - xmodle, ydown, xmodle / 2.)

    # # Plot
    # # f,a = plt.subplots()
    # a.plot(xhat,yhat,'k--')
    # a.plot(xhat,yup,'-x')
    # a.plot(xhat,ydown,'-x')
    # a.axis('equal')

    xy = np.vstack((xhat, yup, ydown))

    return xy


def pitch_Zweifel(Z, Al, Ma, Dr, ga, Yp):
    """Calculate pitch-to-chord from Zweifel coefficient."""
    Alr = np.radians(Al)
    Po2_P2 = compflow.Po_P_from_Ma(Ma[1], ga)
    Po2_Po1 = (1. - Yp) / (1. - Yp / Po2_P2)
    Q1 = compflow.mcpTo_APo_from_Ma(Ma[0], ga)
    cosAl = np.cos(Alr)
    V_cpTo_sinAl = compflow.V_cpTo_from_Ma(Ma, ga) * np.sin(Alr)
    s_c = Z * (1. - Po2_Po1 / Po2_P2) / Q1 / \
        cosAl[0] / (V_cpTo_sinAl[1] - V_cpTo_sinAl[0]) / Dr[0] * np.mean(Dr)
    return np.abs(s_c)


def chord_Re(Re, To1, Po1, Ma, ga, rgas, Yp):
    """Set chord length using Reynolds number."""

    # Get exit stagnation pressure
    Po2_P2 = compflow.Po_P_from_Ma(Ma, ga)
    Po2_Po1 = (1. - Yp) / (1. - Yp / Po2_P2)

    # Exit static state
    P2 = Po1 * Po2_Po1 / Po2_P2
    To2_T2 = compflow.To_T_from_Ma(Ma, ga)
    T2 = To1 / To2_T2
    ro2 = P2 / rgas / T2
    cp = rgas * ga / (ga - 1.)
    V2 = compflow.V_cpTo_from_Ma(Ma, ga) * np.sqrt(cp * To1)

    # Get viscosity using 0.62 power approximation
    mu = muref * (T2 / Tref)**0.62

    # Use reynolds number to set chord
    cx = Re * mu / ro2 / V2
    return cx


def fillet(x, r, dx):

    # Get indices for the points at boundary of fillet
    ind = np.array(np.where(np.abs(x) <= dx)[0])

    dr = np.diff(r) / np.diff(x)

    # Assemble matrix problem
    rpts = r[ind[(0, -1), ]]
    xpts = x[ind[(0, -1), ]]
    drpts = dr[ind[(0, -1), ]]
    b = np.atleast_2d(np.concatenate((rpts, drpts))).T
    A = np.array([
        [xpts[0]**3., xpts[0]**2., xpts[0], 1.],
        [xpts[1]**3., xpts[1]**2., xpts[1], 1.],
        [3. * xpts[0]**2., 2. * xpts[0], 1., 0.],
        [3. * xpts[1]**2., 2. * xpts[1], 1., 0.],
    ])
    poly = np.matmul(np.linalg.inv(A), b).squeeze()

    r[ind] = np.polyval(poly, x[ind])


def meridional_mesh(xc, rm, Dr, c, nr):
    """Generate meridional mesh for a blade row."""

    nxb = 101  # Points along blade chord
    nxb2 = nxb // 2  # Points along blade semi-chord

    # Define a cosinusiodal clustering function
    clust = 0.5 * (1. - np.cos(np.pi * np.linspace(0., 1.0, nxb)))
    dclust = np.diff(clust)
    _, dmax = dclust.min(), dclust.max()

    # Numbers of points in inlet/outlet
    nxu = int((xc[1] - xc[0] - 0.5) / dmax)
    nxd = int((xc[3] - xc[2] - 0.5) / dmax)

    # Build up the streamwise grid vector
    x = xc[1] + clust  # Blade
    x = np.insert(x[1:], 0, x[0] + (clust[nxb2:] - 1.0))  # In front of LE
    x = np.append(x[:-1], x[-1] + (clust[:nxb2]))  # Behind TE
    if nxu > 0:
        x = np.insert(x[1:], 0, np.linspace(xc[0], x[0], nxu))  # Inlet
    if nxd > 0:
        x = np.append(x[:-1], np.linspace(x[-1], xc[3], nxd))  # Outlet

    # Trim if needed
    x = x[x > xc[0]]
    x = x[x < xc[-1]]

    # Reset endpoints
    x = np.insert(x, 0, xc[0])
    x = np.append(x, xc[-1])

    # Get indices of edges
    i_edge = [np.where(x == xc[i])[0][0] for i in [1, 2]]
    i_edge[1] = i_edge[1] + 1

    # Number of radial points
    spf = np.linspace(0., 1., nr)
    spf2 = np.atleast_2d(spf).T

    # Now annulus lines
    rh = np.interp(x, xc[1:3], rm - Dr / 2.)
    rc = np.interp(x, xc[1:3], rm + Dr / 2.)

    # smooth the edges
    dxsmth_c = 0.2
    for i in [1, 2]:
        fillet(x - xc[i], rh, dxsmth_c)
        fillet(x - xc[i], rc, dxsmth_c)

    rh2 = np.atleast_2d(rh)
    rc2 = np.atleast_2d(rc)
    r = spf2 * (rc2 - rh2) + rh2
    r = r.T

    # # Scale by chord
    x = x * c

    # f,a = plt.subplots()
    # a.plot(x,r,'k-')
    # a.axis('equal')
    # plt.show()

    return x, r, i_edge


def blade_to_blade_mesh(x, r, ii, chi, nrt, s_c, a=0.):
    """Generate blade section rt, given merid mesh and flow angles."""

    # Define a cosinusiodal clustering function
    clust = 0.5 * (1. - np.cos(np.pi * np.linspace(0., 1.0, nrt)))
    clust3 = (clust[..., None, None]).transpose((2, 1, 0))

    # Chord
    xpass = x[ii[0]:ii[1]] - x[ii[0]]
    c = xpass.ptp()

    # Pitch in terms of theta
    rmid = np.mean(r[ii[0], (0, -1)])
    dt = s_c * c / rmid
    print('dtmid', dt)

    nj = np.shape(chi)[1]
    rt = np.ones(np.shape(r) + (nrt,)) * np.nan
    for j in range(nj):

        rnow = r[ii[0]:ii[1], j][:, None, None]

        # Dimensional section at midspan
        print('Calling blade section, j=%d' % j)
        print('chi = %f %f' % tuple(chi[:, j]))
        sect_now = blade_section(chi[:, j], a) * c
        rt0 = np.interp(xpass, sect_now[0, :], sect_now[1, :])[:, None, None]
        rt1 = (np.interp(xpass, sect_now[0, :], sect_now[2, :])[:, None, None]
               + s_c * c / rmid * rnow)

        # # Scale thickness to be constant along span
        # rt0 = rt0/rmid*rnow
        # rt1 = rt1/rmid*rnow
        # dtnow = (rt0-rt1)/rnow
        # print('dtnow',dtnow.min(),dtnow.max())

        # # Offset by correct circumferential pitch
        # t1 = rt1/rmid
        # rt1 = (t1+dt)*rnow

        rt[ii[0]:ii[1], j, :] = (rt0 + (rt1 - rt0) * clust3).squeeze()

    # Now deal with inlet and exit ducts
    # First just propagate clustering
    rt[:ii[0], :, :] = rt[ii[0], :, :]
    rt[ii[1]:, :, :] = rt[ii[1] - 1, :, :]

    # Check theta range
    dt = rt[ii[0], :, -1] / r[ii[0], :] - rt[ii[0], :, 0] / r[ii[0], :]
    drt = rt[ii[0], :, -1] - rt[ii[0], :, 0]
    print('dt', dt.max(), dt.min())
    print('drt', drt.max(), drt.min())

    # Set endpoints to a uniform distribution
    unif_rt = np.linspace(0., 1., nrt)
    unif_rt3 = (unif_rt[..., None, None]).transpose((2, 1, 0))
    rt[(0, -1), :, :] = rt[(0, -1), :, 0][..., None] + (
        rt[(0, -1), :, -1] - rt[(0, -1), :, 0])[..., None] * unif_rt3

    # We need to map streamwise indices to fractions of cluster relaxation
    # If we have plenty of space, relax linearly over 1 chord, then unif
    # If we have less than 1 chord, relax linearly all the way
    # Relax clustering linearly

    if (x[ii[0]] - x[0]) / c > 1.:
        xnow = x[:ii[0]] - x[ii[0]]
        icl = np.where(xnow / c > -1.)[0]
        lin_x_up = np.zeros((ii[0],))
        lin_x_up[icl] = np.linspace(0., 1., len(icl))
    else:
        lin_x_up = np.linspace(0., 1., ii[0])

    if (x[-1] - x[ii[1] - 1]) / c > 1.:
        icl = np.where(np.abs((x - x[ii[1] - 1]) / c - 0.5) < 0.5)[0]
        lin_x_dn = np.zeros((len(x),))
        lin_x_dn[icl] = np.linspace(1., 0., len(icl))
        lin_x_dn = lin_x_dn[-(len(x) - ii[1]):]
    else:
        lin_x_dn = np.linspace(1., 0., len(x) - ii[1])

    lin_x_up3 = lin_x_up[..., None, None]
    lin_x_dn3 = lin_x_dn[..., None, None]

    rt[:ii[0], :, :] = rt[0, :, :][None, ...] + (
        rt[ii[0], :, :] - rt[0, :, :])[None, ...] * lin_x_up3
    rt[ii[1]:, :, :] = rt[-1, :, :][None, ...] + (
        rt[ii[1], :, :] - rt[-1, :, :])[None, ...] * lin_x_dn3

    return rt


def generate(htr, aft_load=0., dev=(0., 0.),
             lean_type=None, lean_angle=0., G=1.):

    ga = 1.33
    Toin = 1600.
    Poin = 16.0e5
    rgas = 287.14
    cp = rgas * ga / (ga - 1.)
    Omega = 2. * np.pi * 50.
    Z = 0.85
    phi = 0.6
    psi = 1.6
    Lam = 0.5
    Alin = 0.
    Ma = 0.75
    eta = 0.95
    Re = 4.0e6

    rpm_rotor = np.array([Omega / 2. / np.pi * 60.]).astype(np.float32)

    # Get velocity triangles
    stage = nondim_stage_from_Lam(phi, psi, Lam, Alin, Ma, ga, eta)
    print('PR', stage.Pout_Poin)

    # Get mean radius and spans
    rm, Dr = annulus_line(
        stage.U_sqrt_cpToin, stage.Ax_Axin, htr, cp * Toin, Omega,
    )

    print('Dr', Dr)

    # Calculate pitch-to-chord ratio (const Zweifel)
    s_c = (
        pitch_Zweifel(Z, stage.Al[:2], stage.Ma[:2], Dr[:2], ga, stage.Yp[0]),
        pitch_Zweifel(Z,
                      stage.Al_rel[1:],
                      stage.Ma_rel[1:],
                      Dr[1:],
                      ga,
                      stage.Yp[1])
    )

    # Set chord based on Reynolds number
    c = chord_Re(Re, Toin, Poin, stage.Ma[1], ga, rgas, stage.Yp[0])

    # Round pitch-to-chord to nearest whole number of blades
    # Which fit in 8th-annulus sector
    sect = 1. / 8.
    nb = (np.round(
        (2. * np.pi * rm / (np.array(s_c) * c)) * sect) / sect).astype(int)
    print('nb', nb)
    print('old s_c', s_c)
    s_c = 2. * np.pi * rm / nb / c

    print('new s_c', s_c)

    # Number of radial points
    dr_c = 0.04  # Radial grid spacing as fraction of chord
    nr = np.max((int(Dr[1] / (dr_c * c)), 4))
    print('nr', nr)

    print(stage.Al)
    print(stage.Al_rel)

    # Get meridional grids
    duct_len = 2.
    vane_xc = [-duct_len, 0., 1., 1. + G / 2.]
    blade_xc = [1. + G / 2., 1. + G, 2. + G, duct_len + 2. + G]
    vane_x, vane_r, vane_i = meridional_mesh(vane_xc, rm, Dr[:2], c, nr)
    blade_x, blade_r, blade_i = meridional_mesh(blade_xc, rm, Dr[1:], c, nr)

    # Deviation in correct shape
    vane_dev = np.zeros_like(vane_r[vane_i, :])
    vane_dev[1, :] = dev[0]
    blade_dev = np.zeros_like(blade_r[blade_i, :])
    blade_dev[1, :] = dev[1]

    # Radial flow angle variations
    # Free vortex
    vane_Al = np.degrees(np.arctan(
        rm / vane_r[vane_i, :] * np.tan(np.radians(
            np.atleast_2d(stage.Al[:2])).T)))
    blade_Al = np.degrees(np.arctan(
        rm / blade_r[blade_i, :] * np.tan(np.radians(
            np.atleast_2d(stage.Al[1:])).T) - blade_r[blade_i, :] / rm / phi))

    # Add a guess of deviation
    vane_Al += vane_dev
    blade_Al += blade_dev

    # Vxm = phi * Omega * rm
    # Vtm = Vxm * np.tan(np.radians(stage.Al[1]))
    # print(Vxm, Vtm, rm)
    # solve_forced_vortex(Vxm,Vtm,rm,vane_r[blade_i[0],:])

    # f,a=plt.subplots()
    # a.plot((vane_Al.T), vane_r[vane_i[0],:]/rm)
    # a.plot((blade_Al.T), blade_r[blade_i[0],:]/rm)
    # plt.show()

    nrt = 65
    vane_rt = blade_to_blade_mesh(
        vane_x,
        vane_r,
        vane_i,
        vane_Al,
        nrt,
        s_c[0],
        aft_load
    )
    blade_rt = blade_to_blade_mesh(
        blade_x,
        blade_r,
        blade_i,
        blade_Al,
        nrt,
        s_c[1],
        aft_load
    )

    # taper factor
    ARtap = 2.
    taper = np.linspace(1., 1. / np.sqrt(ARtap), nr)

    cxref = np.ptp(vane_x[vane_i])

    xle = np.array([0., 1. + G]) * cxref
    xte = np.array([1., 2. + G]) * cxref

    pside = []
    sside = []
    hub = []
    cas = []
    for xnow, rnow, rtnow, inow, nbi, is_rotor in zip(
            (vane_x, blade_x), (vane_r, blade_r), (vane_rt, blade_rt),
            (vane_i, blade_i), nb, [False, True]):

        pside.append([])
        sside.append([])

        hub.append(np.stack((xnow, rnow[:, 0]), axis=1))
        cas.append(np.stack((xnow, rnow[:, -1]), axis=1))

        dtnow = 2. * np.pi / float(nbi)
        tnow = rtnow / rnow[..., None]
        rt2now = rnow[..., None] * (tnow - dtnow)
        span_now = rnow[inow[0], :].ptp()
        rref_now = rnow[inow[0], 0]
        for ir in range(nr):
            xrrtnow = np.stack((
                xnow[inow[0]:inow[1]],
                rnow[inow[0]:inow[1], ir],
                rtnow[inow[0]:inow[1], ir, 0],
                rt2now[inow[0]:inow[1], ir, -1],), axis=1)

            # Taper the rotor
            if is_rotor:
                xrrtnow[:, 0] = (
                    (xrrtnow[:, 0] - xrrtnow[0, 0]) * taper[ir] + xrrtnow[0, 0]
                    + 0.5 * (xrrtnow[-1, 0] - xrrtnow[0, 0]) * (1. - taper[ir])
                )
                xrrtnow[:, (2, 3)] = (
                    (xrrtnow[:, (2, 3)] - xrrtnow[0, 2]) *
                    taper[ir] + xrrtnow[0, 2]
                )
            # Stack on centroid
            Area = np.trapz(xrrtnow[:, 3] - xrrtnow[:, 2], xrrtnow[:, 0])
            rtc = np.trapz(
                0.5 * (xrrtnow[:, 3] - xrrtnow[:, 2])
                * (xrrtnow[:, 3] + xrrtnow[:, 2]), xrrtnow[:, 0]) / Area

            # Shear the stacking if requested
            if lean_type is not None and not is_rotor:
                tan_nu = np.tan(np.radians(-lean_angle))
                drnow = xrrtnow[0, 1] - rref_now
                if lean_type == 'compound':
                    drtc = tan_nu * (drnow - (drnow**2.) / span_now)
                elif lean_type == 'simple':
                    drtc = tan_nu * drnow
            else:
                drtc = 0.
            # print('j',ir,'drtc',drtc,'drnow',drnow,'rtc',rtc)

            xrrtnow[:, (2, 3)] -= (rtc + drtc)

            pside[-1].append(xrrtnow[:, (0, 1, 2)])
            sside[-1].append(xrrtnow[:, (0, 1, 3)])

            # swap a few points from one side to the other
            if not is_rotor:
                iswp = np.argmax(pside[-1][-1][:, 2])
                sside[-1][-1] = np.append(
                    sside[-1][-1], np.flipud(pside[-1][-1][iswp:-1, :]), axis=0)
                pside[-1][-1] = pside[-1][-1][:(iswp + 1), :]
            else:
                iswp = np.argmin(sside[-1][-1][:, 2])
                pside[-1][-1] = np.append(
                    pside[-1][-1], np.flipud(sside[-1][-1][iswp:-1, :]), axis=0)
                sside[-1][-1] = sside[-1][-1][:(iswp + 1), :]

    hub = np.concatenate((hub[0], hub[1][1:, :]))
    cas = np.concatenate((cas[0], cas[1][1:, :]))

    # f,a = plt.subplots()
    # rtLE = [psi[0,2] for psi in pside[0]]
    # rLE = [psi[0,1] for psi in pside[0]]
    # a.plot(rtLE,rLE,'k-x')
    # a.axis('equal')
    # plt.show()
    # rstrst

    write_geomturbo('mesh.geomTurbo', sside, pside, hub, cas, nb)

    # Make a config for autogrid meshing
    import ConfigParser
    conf = ConfigParser.ConfigParser()
    conf.add_section('mesh')
    conf.set('mesh', 'nrow', 2)
    conf.set('mesh', 'nxu', 41)
    conf.set('mesh', 'nxd', 97)
    conf.set('mesh', 'rpm', [0., rpm_rotor[0]])
    conf.set('mesh', 'xle', [xle[0], xle[1]])
    conf.set('mesh', 'xte', [xte[0], xte[1]])
    conf.set('mesh', 'dy', [1e-4, 1e-4])

    conf.set('mesh', 'unif_span', False)

    nr_htr = np.array([[0.6, 0.9], [121, 65]])
    nr = int(np.interp(htr, nr_htr[0, :], nr_htr[1, :]))
    print(nr)
    conf.set('mesh', 'nr', nr)

    with open('conf.ini', 'w') as f:
        conf.write(f)

    # run_remote( 'mesh.geomTurbo',
        # ['../../script-fix/script_ag.py2', '../../script-fix/script_igg.py2'],
        # '../../script-fix/script_sh', 'conf.ini')

    return


def write_geomturbo(fname, ps, ss, h, c, nb, tips=(None, None), cascade=False):
    """Write blade and annulus geometry to AutoGrid GeomTurbo file.

    Parameters
    ----------

    fname : File name to write
    ps    : Nested list of arrays of pressure-side coordinates,
            ps[row][section][point on section, x/r/rt]
            We allow different sizes for each section and row.
    ss    : Same for suction-side coordinates.
    h     : Array of hub line coordinates, h[axial location, x/r].
    c     : Same for casing line.
    nb    : Iterable of numbers of blades for each row."""

    # Determine numbers of points
    ni_h = np.shape(h)[0]
    ni_c = np.shape(c)[0]

    n_row = len(ps)
    n_sect = [len(psi) for psi in ps]
    ni_ps = [[np.shape(psii)[0] for psii in psi] for psi in ps]
    ni_ss = [[np.shape(ssii)[0] for ssii in ssi] for ssi in ss]

    fid = open(fname, 'w')

    # # Autogrid requires R,X,T coords
    # ps = ps[i][:,[1,0,2]]
    # ss = ss[i][:,[1,0,2]]

    if cascade:
        # Swap the coordinates
        for i in range(n_row):
            for k in range(n_sect[i]):
                ps[i][k] = ps[i][k][:, (1, 2, 0)]
                ss[i][k] = ss[i][k][:, (1, 2, 0)]
    else:
        # Convert RT to T
        for i in range(n_row):
            for k in range(n_sect[i]):
                ps[i][k][:, 2] = ps[i][k][:, 2] / ps[i][k][:, 1]
                ss[i][k][:, 2] = ss[i][k][:, 2] / ss[i][k][:, 1]

    # Write the header
    fid.write('%s\n' % 'GEOMETRY TURBO')
    fid.write('%s\n' % 'VERSION 5.5')
    fid.write('%s\n' % 'bypass no')
    if cascade:
        fid.write('%s\n\n' % 'cascade yes')
    else:
        fid.write('%s\n\n' % 'cascade no')

    # Write hub and casing lines (channel definition)
    fid.write('%s\n' % 'NI_BEGIN CHANNEL')

    # Build the hub and casing line out of basic curves
    # Start the data definition
    fid.write('%s\n' % 'NI_BEGIN basic_curve')
    fid.write('%s\n' % 'NAME thehub')
    fid.write('%s %i\n' % ('DISCRETISATION', 10))
    fid.write('%s %i\n' % ('DATA_REDUCTION', 0))
    fid.write('%s\n' % 'NI_BEGIN zrcurve')
    fid.write('%s\n' % 'ZR')

    # Write the length of hub line
    fid.write('%i\n' % ni_h)

    # Write all the points in x,r
    for i in range(ni_h):
        fid.write('%1.11f\t%1.11f\n' % tuple(h[i, :]))

    fid.write('%s\n' % 'NI_END zrcurve')
    fid.write('%s\n' % 'NI_END basic_curve')

    # Now basic curve for shroud
    fid.write('%s\n' % 'NI_BEGIN basic_curve')
    fid.write('%s\n' % 'NAME theshroud')

    fid.write('%s %i\n' % ('DISCRETISATION', 10))
    fid.write('%s %i\n' % ('DATA_REDUCTION', 0))
    fid.write('%s\n' % 'NI_BEGIN zrcurve')
    fid.write('%s\n' % 'ZR')

    # Write the length of shroud
    fid.write('%i\n' % ni_c)

    # Write all the points in x,r
    for i in range(ni_c):
        fid.write('%1.11f\t%1.11f\n' % tuple(c[i, :]))

    fid.write('%s\n' % 'NI_END zrcurve')
    fid.write('%s\n' % 'NI_END basic_curve')

    # Now lay out the real shroud and hub using the basic curves
    fid.write('%s\n' % 'NI_BEGIN channel_curve hub')
    fid.write('%s\n' % 'NAME hub')
    fid.write('%s\n' % 'VERTEX CURVE_P thehub 0')
    fid.write('%s\n' % 'VERTEX CURVE_P thehub 1')
    fid.write('%s\n' % 'NI_END channel_curve hub')

    fid.write('%s\n' % 'NI_BEGIN channel_curve shroud')
    fid.write('%s\n' % 'NAME shroud')
    fid.write('%s\n' % 'VERTEX CURVE_P theshroud 0')
    fid.write('%s\n' % 'VERTEX CURVE_P theshroud 1')
    fid.write('%s\n' % 'NI_END channel_curve shroud')

    fid.write('%s\n' % 'NI_END CHANNEL')

    # CHANNEL STUFF DONE
    # NOW DEFINE ROWS
    for i in range(n_row):
        fid.write('%s\n' % 'NI_BEGIN nirow')
        fid.write('%s%i\n' % ('NAME r', i + 1))
        fid.write('%s\n' % 'TYPE normal')
        fid.write('%s %i\n' % ('PERIODICITY', nb[i]))
        fid.write('%s %i\n' % ('ROTATION_SPEED', 0))

        hdr = ['NI_BEGIN NINonAxiSurfaces hub',
               'NAME non axisymmetric hub',
               'REPETITION 0',
               'NI_END   NINonAxiSurfaces hub',
               'NI_BEGIN NINonAxiSurfaces shroud',
               'NAME non axisymmetric shroud',
               'REPETITION 0',
               'NI_END   NINonAxiSurfaces shroud',
               'NI_BEGIN NINonAxiSurfaces tip_gap',
               'NAME non axisymmetric tip gap',
               'REPETITION 0',
               'NI_END   NINonAxiSurfaces tip_gap']

        fid.writelines('%s\n' % l for l in hdr)

        fid.write('%s\n' % 'NI_BEGIN NIBlade')
        fid.write('%s\n' % 'NAME Main Blade')

        if tips[i] is not None:
            fid.write('%s\n' % 'NI_BEGIN NITipGap')
            fid.write('%s %f\n' % ('WIDTH_AT_LEADING_EDGE', tips[i][0]))
            fid.write('%s %f\n' % ('WIDTH_AT_TRAILING_EDGE', tips[i][1]))
            fid.write('%s\n' % 'NI_END NITipGap')

        fid.write('%s\n' % 'NI_BEGIN nibladegeometry')
        fid.write('%s\n' % 'TYPE GEOMTURBO')
        fid.write('%s\n' % 'GEOMETRY_MODIFIED 0')
        fid.write('%s\n' % 'GEOMETRY TURBO VERSION 5')
        fid.write('%s %f\n' % ('blade_expansion_factor_hub', 0.1))
        fid.write('%s %f\n' % ('blade_expansion_factor_shroud', 0.1))
        fid.write('%s %i\n' % ('intersection_npts', 10))
        fid.write('%s %i\n' % ('intersection_control', 1))
        fid.write('%s %i\n' % ('data_reduction', 0))
        fid.write('%s %f\n' % ('data_reduction_spacing_tolerance', 1e-006))
        fid.write('%s\n' % ('control_points_distribution '
                            '0 9 77 9 50 0.00622408226922942 0.119480980447523'))
        fid.write('%s %i\n' % ('units', 1))
        fid.write('%s %i\n' % ('number_of_blades', 1))

        fid.write('%s\n' % 'suction')
        fid.write('%s\n' % 'SECTIONAL')
        fid.write('%i\n' % n_sect[i])
        for k in range(n_sect[i]):
            fid.write('%s %i\n' % ('# section', k + 1))
            if cascade:
                fid.write('%s\n' % 'XYZ')
            else:
                fid.write('%s\n' % 'ZRTH')
            fid.write('%i\n' % ni_ss[i][k])
            for j in range(ni_ss[i][k]):
                fid.write('%1.11f\t%1.11f\t%1.11f\n' % tuple(ss[i][k][j, :]))

        fid.write('%s\n' % 'pressure')
        fid.write('%s\n' % 'SECTIONAL')
        fid.write('%i\n' % n_sect[i])
        for k in range(n_sect[i]):
            fid.write('%s %i\n' % ('# section', k + 1))
            if cascade:
                fid.write('%s\n' % 'XYZ')
            else:
                fid.write('%s\n' % 'ZRTH')
            fid.write('%i\n' % ni_ps[i][k])
            for j in range(ni_ps[i][k]):
                fid.write('%1.11f\t%1.11f\t%1.11f\n' % tuple(ps[i][k][j, :]))
        fid.write('%s\n' % 'NI_END nibladegeometry')

        # choose a leading and trailing edge treatment

        #    fid.write('%s\n' % 'BLUNT_AT_LEADING_EDGE')
        fid.write('%s\n' % 'BLENT_AT_LEADING_EDGE')
        #    fid.write('%s\n' % 'BLENT_TREATMENT_AT_TRAILING_EDGE')
        fid.write('%s\n' % 'NI_END NIBlade')

        fid.write('%s\n' % 'NI_END nirow')

    fid.write('%s\n' % 'NI_END GEOMTURBO')

    fid.close()


if __name__ == "__main__":

    pass
