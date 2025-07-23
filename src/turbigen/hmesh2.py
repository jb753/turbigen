import numpy as np
from turbigen import util
import turbigen.grid
import turbigen.geometry

import turbigen.mesh
import dataclasses

import jmesh

logger = util.make_logger()


@dataclasses.dataclass
class H2(turbigen.mesh.Mesher):
    """Improved H-topology mesher."""

    resolution_factor: float = 1.0
    """Multiply the number of points in each direction, keeping relative spacings."""

    skew_max: float = 30.0

    slip_annulus: bool = False

    ER_max: float = 1.2

    dspf_mid_max: float = 0.05

    AR_cusp: float = 0.0
    ni_cusp: int = 0
    nj_tip_min = 9

    yplus: float = np.nan

    plot: bool = False

    def make_grid(self, workdir, mac, dhub, dcas, dsurf, Omega=None):
        """Generate a Grid object for a machine geometry."""

        logger.info("Making an H2-mesh...")

        #
        # Spanwise grid
        #

        # Normalise the spacings
        span = mac.ann.get_span()  # Inlet and exit of each blade row
        span = 0.5 * (span[1:] + span[:-1])[::2]  # Average span of each row

        # Choose a single machine-global spacing at hub/casing
        dspf_hub = np.mean(dhub / span)
        dspf_casing = np.mean(dcas / span)

        # Use smallest tip gap to cluster grid
        tip = np.min(mac.tip)

        # Define spanwise clustering
        # Constant through the machine for all rows
        spf = generate_spanwise_grid_vector(self, dspf_hub, dspf_casing, tip)
        dspan_max = np.diff(spf).max() * span

        #
        # Points on the blades
        #

        # Dimensional grid spacings at blade LE and TE for each row
        chords = mac.ann.chords(0.5)[1::2]  # Meridional chords of each row [m]
        dm_LE = 0.001 * chords
        dm_TE = 0.01 * chords
        AR_chord = 1.0
        dm_max = AR_chord * dspan_max

        nrow = mac.Nrow
        for irow in range(nrow):
            generate_meridional_grid_vectors(self, mac, irow, dm_max[irow])
            pass

            # B


def _theta_limits(
    tq, xrt_u, xrt_l, mlim, Theta=(0.0, 0.0), c=(1.0, 1.0), Theta_max=30.0
):
    """Evaluate pitchwise limits given upper/lower surface section coordinates."""

    # Put geometric leading edge where it should be
    # Must handle axial and radial inlets differently

    # If x varies more than r near LE, is axial, split on min x
    if np.ptp(xrt_u[0][:10]) > np.ptp(xrt_u[1][:10]):
        ind_split = 0
        iule = np.argmin(xrt_u[ind_split])
        ille = np.argmin(xrt_l[ind_split])
    # Otherwise, is radial, split on max r
    elif xrt_u[1][0] < xrt_u[1][-1]:
        ind_split = 1
        iule = np.argmin(xrt_u[ind_split])
        ille = np.argmin(xrt_l[ind_split])
    else:
        ind_split = 1
        iule = np.argmax(xrt_u[ind_split])
        ille = np.argmax(xrt_l[ind_split])

    # If the geometric leading edge is on upper surface
    # we need to move some points from upper to lower
    if xrt_u[ind_split][iule] < xrt_l[ind_split][ille]:
        xrt_l = np.concatenate(
            (np.flip(xrt_u[:, 1 : iule + 1], axis=-1), xrt_l), axis=-1
        )
        xrt_u = xrt_u[:, iule:]
    # If the geometric leading edge is on lower surface
    # we need to move some points from lower to upper
    elif xrt_u[ind_split][iule] > xrt_l[ind_split][ille]:
        xrt_u = np.concatenate(
            (np.flip(xrt_l[:, 1 : ille + 1], axis=-1), xrt_u), axis=-1
        )
        xrt_l = xrt_l[:, ille:]

    # Join the curves together at trailing edge
    if xrt_u[ind_split].max() > xrt_l[ind_split].max():
        xrt_l = np.concatenate((xrt_l, xrt_u[:, (-1,)]), axis=-1)
    else:
        xrt_u = np.concatenate((xrt_u, xrt_l[:, (-1,)]), axis=-1)

    # Evaluate normalised meridional distances for each surface
    m_u = util.cum_arc_length(xrt_u[:2])
    m_l = util.cum_arc_length(xrt_l[:2])
    m_u /= m_u[-1]
    m_l /= m_l[-1]

    m_u = mlim[0] + np.ptp(mlim) * m_u
    m_l = mlim[0] + np.ptp(mlim) * m_l

    # Interpolate the pitchwise limits
    # Values outside unit interval constant at boundary values
    theta_u = np.interp(tq, m_u, xrt_u[2])
    theta_l = np.interp(tq, m_l, xrt_l[2])

    # Look for any turning points in last 5% chord
    # These correspond to TE corner
    dtheta_u = np.diff(theta_u, n=1)
    dtheta_l = np.diff(theta_l, n=1)

    ind_l_up, ind_l_dn = util.zero_crossings(dtheta_l)
    ind_u_up, ind_u_dn = util.zero_crossings(dtheta_u)
    ind_l_te = ind_l_up[tq[ind_l_up] > mlim[1] - 0.2]
    ind_u_te = ind_u_dn[tq[ind_u_dn] > mlim[1] - 0.2]

    # If the process for setting tte does not work, then
    # arbitrarily cluster grid over last 1.0% chord
    # tte = mlim[1] -0.005
    tte = None
    if ind_l_te.size > 0:
        # print(f'TE on lower at {ind_l_te[0]}')
        tte = tq[ind_l_te[-1]]
    elif ind_u_te.size > 0:
        # print(f'TE on upper at {ind_u_te[0]}')
        tte = tq[ind_u_te[-1]]
    else:
        tte = mlim[1] - 0.01

    if np.any(theta_u < theta_l):
        raise Exception("Blade is thicker than calculated pitch!")

    r_u = np.interp(tq, m_u, xrt_u[1])
    r_l = np.interp(tq, m_l, xrt_l[1])
    rref = 0.5 * (r_u + r_l)

    # Skew the mesh upstream of LE and downstream of TE
    dtheta_skew = np.zeros_like(theta_u)
    ind_up = tq < mlim[0]
    ind_dn = tq >= mlim[1]
    Theta_now = np.clip(Theta, -Theta_max, Theta_max)
    tanTheta = np.tan(np.radians(Theta_now))
    if ind_up.any():
        dtheta_skew[ind_up] = (
            tanTheta[0] * c[0] * util.cumtrapz0(1.0 / rref[ind_up], tq[ind_up])
        )
        dtheta_skew[ind_up] -= dtheta_skew[ind_up][-1]
    if ind_dn.any():
        dtheta_skew[ind_dn] = (
            tanTheta[1] * c[1] * util.cumtrapz0(1.0 / rref[ind_dn], tq[ind_dn])
        )
    theta_u += dtheta_skew
    theta_l += dtheta_skew

    return theta_u, theta_l, tte


def generate_spanwise_grid_vector(conf, dspf_hub, dspf_casing, tip):
    """Set the normalised spanwise grid spacing for a machine.

    The spanwise grid is a vector of span fraction from zero to one.

    Parameters
    ----------
    conf : turbigen.mesh.mesher
        The meshing configuration object.
    dspf_hub : float
        Normalised spanwise grid spacing at hub.
    dspf_casing : float
        Normalised spanwise grid spacing at casing.
    tip : float
        Normalised tip gap, zeros if no gap.

    """

    if tip:
        L_main = 1.0 - tip
        L_tip = tip

        # Distribute across tip gap
        dspf_tip = np.minimum(dspf_casing, tip / conf.nj_tip_min)
        clu_tip = (
            jmesh.cluster.double.free(dspf_tip, dspf_tip, 4 * dspf_tip, conf.ER_max)
            * L_tip
            + L_main
        )

        # Distribute across main stream
        clu_main = (
            jmesh.cluster.double.free(dspf_hub, dspf_casing, conf.dspf_mid, conf.ER_max)
            * L_main
        )

        # Resample to the required resolution
        clu_main = util.resample(clu_main, conf.resolution_factor)
        clu_tip = util.resample(clu_tip, conf.resolution_factor)

        # Join together
        spf = np.concatenate((clu_main[:-1], clu_tip))

    else:
        spf = jmesh.cluster.double(
            dspf_hub, dspf_casing, conf.dspf_mid_max, conf.ER_max
        )
        spf = util.resample(spf, conf.resolution_factor)

    assert spf[0] == 0.0
    assert np.isclose(spf[-1], 1.0)
    assert (np.diff(spf) > 0.0).all()

    return spf


def generate_meridional_grid_vectors(conf, mac, irow, dM_max):
    """Distribute points meridionally on hub and casing.

    Parameters
    ----------
    conf : turbigen.mesh.mesher
        The meshing configuration object.
    ann : turbigen.geometry.Annulus
        The annulus geometry to use for meshing.
    irow : int
        The row index of the blade row to mesh.

    """

    nrow = mac.Nrow

    # Get dimensional meridional spacings at blade LE and TE for this row
    # Use a capital M for dimensional lengths
    chord = mac.ann.chords(0.5)[irow]
    dM_LE = 0.001 * chord
    dM_TE = 0.01 * chord

    # Get normalised m boundaries for this row
    m_LE = 1 + irow * 2
    if nrow == 1:
        m_bound = np.array([0.0, 1.0, 2.0, 3.0])
    elif irow == 0:
        m_bound = np.array([-1.0, 0.0, 1.0, 1.5]) + m_LE
    elif irow == nrow - 1:
        m_bound = np.array([-0.5, 0.0, 1.0, 2.0]) + m_LE
    else:
        m_bound = np.array([-0.5, 0.0, 1.0, 1.5]) + m_LE

    # Get query vectors nondimensional meridional coordinates for this row
    # Upstream (inlet boundary to LE)
    # Chord (LE to TE)
    # Downstream (TE to outlet boundary)
    nbrute = 1000
    m_all = np.stack(
        [np.linspace(m_bound[iseg], m_bound[iseg + 1], nbrute) for iseg in range(3)]
    )
    spf_query = np.array([0.0, 1.0]).reshape((2, 1))

    # Now for each of these query meridional segments:
    # 1. Evaluate xr on hub and casing
    # 2. Attempt to cluster using fixed dimensional dm_LE etc
    #    Normalising by the current arc length
    # 3. Take the maximum number of points from hub and casing
    # 4. Regenerate the meridional lines with fixed number of points

    # Arc lengths [inlet/chord/outlet, hub/casing]
    L_ann = np.stack(
        [util.arc_length(mac.ann.evaluate_xr(m, spf_query), axis=-1) for m in m_all]
    )
    print(L_ann.shape)

    # Assemble the dimesional cluster spacings lengths for all segments
    dM_end = [
        [dM_LE, dM_max],
        [dM_LE, dM_TE, dM_max],
        [dM_TE, dM_max],
    ]
    # Normalise the spacings by the arc lengths
    # indexing dm_end[segment][hub/casing, args to cluster]
    dm_end = [dMi / L.reshape(2, 1) for dMi, L in zip(dM_end, L_ann)]

    # First pass of evaluating ni in each segment
    ni_all = np.array(
        [
            [len(jmesh.cluster.single(*dmi, conf.ER_max)) for dmi in dm_end[0]],
            [len(jmesh.cluster.double(*dmi, conf.ER_max)) for dmi in dm_end[1]],
            [len(jmesh.cluster.single(*dmi, conf.ER_max)) for dmi in dm_end[2]],
        ]
    )  # [segment, hub/casing]

    # Now take the max of hub and casing
    ni_max = np.max(ni_all, axis=1)
    ni_tot = np.sum(ni_max) - 2

    # Assemble normalised spacing on hub and casing
    xr_ann = np.full((2, 2, ni_tot), np.nan)
    for iann in range(2):
        clu_upst = np.flip(
            jmesh.cluster.single(*dm_end[0][iann], conf.ER_max, N=ni_max[0])
        )
        clu_chord = jmesh.cluster.double(*dm_end[1][iann], conf.ER_max, N=ni_max[1])
        clu_down = jmesh.cluster.single(*dm_end[2][iann], conf.ER_max, N=ni_max[2])
        m_all = np.concatenate(
            (
                clu_upst + m_bound[0],
                clu_chord[1:] + m_bound[1],
                clu_down[1:] + m_bound[2],
            )
        )

        # TODO need to transform clu to m using m_bound

        xr_ann[iann, :, :] = mac.ann.evaluate_xr(m_all, spf_query[iann])

    import matplotlib.pyplot as plt

    plt.plot(*xr_ann[0], "r-x", label="hub")
    plt.plot(*xr_ann[1], "b-o", label="cas")
    plt.axis("equal")
    plt.show()

    quit()

    # # Cluster on both hub and casing, find max number of points
    # ni_chord = np.max(
    #     [
    #         len(jmesh.cluster.double(dm_LE / L, dm_TE / L, dm_max / L, conf.ER_max))
    #         for L in L_ann
    #     ]
    # )

    # Now regenerate the annulus lines with fixed number of points
    clu_ann = np.stack(
        [
            jmesh.cluster.double(
                dm_LE / L, dm_TE / L, dm_max / L, conf.ER_max, N=ni_chord
            )
            for L in L_ann
        ],
    )
    xr_ann = ii
