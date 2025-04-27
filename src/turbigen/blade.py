from turbigen import util
import numpy as np
import dataclasses
import turbigen.nblade
import turbigen.camber
import turbigen.thickness

logger = util.make_logger()


@dataclasses.dataclass
class BladeDesigner:
    """Store design variables for a blade."""

    spf: np.ndarray
    """Span fractions to define sections on len(spf) = nsect"""

    q_thick: np.ndarray
    """Thickness design variables (nsect, nqthick)"""

    q_camber: np.ndarray
    """Camber design variables (nsect, nqcamber)"""

    camber_type: turbigen.camber.BaseCamber = "quadratic"
    """Which method to generate the camber line."""

    thick_type: turbigen.thickness.BaseThickness = "taylor"
    """Which method to generate the thickness distribution."""

    tip: float = 0.0
    """Tip clearance as fraction of span."""

    vortex_expon: float = -1.0
    """Spanwise swirl distribution, Vt ~ r**vortex_expon."""

    theta_offset: float = 0.0
    """Rotate the blade through this angle."""

    mstack: float = 0.5
    """Stack the blades at this normalised meridional distance."""

    def __post_init__(self):
        # self.number = util.init_subclass_by_signature(
        #     turbigen.nblade.BladeNumberConfig, self.number
        # )

        self.camber_type = util.get_subclass_by_name(
            turbigen.camber.BaseCamber, self.camber_type
        )

        self.thick_type = util.get_subclass_by_name(
            turbigen.thickness.BaseThickness, self.thick_type
        )

        # Check the dimensions of the design variables
        self.spf = np.reshape(self.spf, -1)
        nsect = len(self.spf)
        self.q_thick = np.atleast_2d(self.q_thick)
        self.q_camber = np.atleast_2d(self.q_camber)
        assert (
            self.q_thick.shape[0] == nsect
        ), f"Wrong number of sections for thickness, expected {nsect}, got {self.q_thick.shape[0]}"
        assert (
            self.q_camber.shape[0] == nsect
        ), f"Wrong number of sections for camber, expected {nsect}, got {self.q_camber.shape[0]}"

    def to_dict(self):
        # Built-in dataclasses method gets us most of the way there
        data = dataclasses.asdict(self)

        # Convert the camber and thickness types to strings
        data["camber_type"] = util.camel_to_snake(self.camber_type.__name__)
        data["thick_type"] = util.camel_to_snake(self.thick_type.__name__)

        # Convert ndarray to list
        data["spf"] = data["spf"].tolist()
        data["q_thick"] = data["q_thick"].tolist()
        data["q_camber"] = data["q_camber"].tolist()

        return data

    def apply_recamber(self, mean_line):
        """Convert the stored recamber angles to local tanchi.

        Initially q_camber[:2] are the recamber angles at the LE and TE.
        The flow angles at the mean radius are extracted from the input mean line.
        The variation with radius is given by the stored vortex_expon.
        On exit from this function, q_camber[:2] are the local tanchi values.

        """
        # Calculate the local flow angles
        Alpha_rel = mean_line.Alpha_rel_free_vortex(self.spf, self.vortex_expon)
        # Add the recamber angles to get the local angle
        chi = Alpha_rel + self.q_camber[:, :2]
        if np.any(np.abs(chi) > 90.0):
            raise Exception(f"Cannot set a blade angle over 90 degrees! chi={chi}")
        if np.any(np.abs(chi) > 80.0):
            logger.warning(
                f"WARNING: High blade angles may cause meshing problems: chi={chi}"
            )
        # Take the tangent and store in the class
        self.q_camber[:, :2] = util.tand(chi)

    def get_chi(self, spf):
        """Interpolate metal angles at a given span fraction."""
        cam, _ = self._get_camber_thickness(spf)
        return cam.chi((0.0, 1.0))

    def get_nose(self, spf):
        """Get the nose of the aerofoil leading edge."""

        # Make a meridional grid vector for just the le
        m = util.cluster_cosine(500)
        xrtul = np.stack(self.evaluate_section(spf, m=m), axis=0)
        xrtcam = np.mean(xrtul, axis=0)
        xrtLE = xrtcam[:, 0].squeeze()

        return xrtLE

    def get_LE_cent(self, spf, fac_Rle=1.0):
        """Get the centre of the leading edge."""

        # Make a meridional grid vector for just the le
        cam, thick = self._get_camber_thickness(spf)
        Rle = thick.R_LE / fac_Rle
        m = util.cluster_cosine(500)
        xrtul = np.stack(self.evaluate_section(spf, m=m), axis=0)
        xrtul = xrtul[:, :, m < 2.0 * Rle]
        xrtcam = np.mean(xrtul, axis=0)
        xrtLE = xrtcam[:, np.argmax(m > Rle)]

        return xrtLE

    def undo_recamber(self, mean_line):
        """Convert the stored tanchi back to recamber angles."""
        Alpha_rel = mean_line.Alpha_rel_free_vortex(self.spf, self.vortex_expon)
        # Subtract the local flow angles to get the recamber angles
        # After taking the arctangent
        self.q_camber[:, :2] = util.atand(self.q_camber[:, :2]) - Alpha_rel

    def set_streamsurface(self, streamsurface):
        self.streamsurface = streamsurface

    @property
    def nsect(self):
        return len(self.spf)

    def _get_camber_thickness(self, spf):
        # Create thickness and camber lines
        if len(self.spf) == 1:
            # Constant values
            thick = self.thick_type(self.q_thick[0])
            cam = self.camber_type(self.q_camber[0])
        else:
            # Interpolate the parameters
            qcam = util.interp1d_linear_extrap(self.spf, self.q_camber)
            qthick = util.interp1d_linear_extrap(self.spf, self.q_thick)

            thick = self.thick_type(qthick(spf).reshape(-1))
            cam = self.camber_type(qcam(spf).reshape(-1))

        return cam, thick

    def evaluate_section(self, spf, nchord=10000, m=None):
        """Coordinates of upper and lower surfaces at one span fraction."""

        cam, thick = self._get_camber_thickness(spf)

        # Evaluate midspan meridional chord
        if m is None:
            m = util.cluster_cosine(nchord)

        # Get coordinates of the streamsurface to put this section on
        dydm = cam.dydm(m)
        chi = np.arctan(dydm)
        tau = thick.thick(m)

        # Calculate offsets for perpendicular thickness in b2b plane
        Dm = -tau * np.sin(chi)
        Dy = tau * np.cos(chi)

        # Meridional positions for upper and lower surfaces
        mu = m + Dm
        ml = m - Dm

        # We need to convert the camber line meridional positions into LE/TE
        # meridional positions.
        mcam_LE = np.min((mu.min(), ml.min()))
        mcam_TE = np.max((mu.max(), ml.max()))
        mcam_ptp = mcam_TE - mcam_LE
        mu_LTE = (mu - mcam_LE) / mcam_ptp
        ml_LTE = (ml - mcam_LE) / mcam_ptp
        mcam = (m - mcam_LE) / mcam_ptp
        chord_full = util.arc_length(self.streamsurface(0.5, mcam))

        # now remap to mlim
        mlim = np.array([0.0, 1.0])
        mu_LTE = mlim[0] + np.ptp(mlim) * mu_LTE
        ml_LTE = mlim[0] + np.ptp(mlim) * ml_LTE
        mcam = mlim[0] + np.ptp(mlim) * mcam
        chord = util.arc_length(self.streamsurface(0.5, mcam))

        # Find coordinates on stream surface of upper/lower/camber points
        xru = self.streamsurface(spf, mu_LTE)
        xrl = self.streamsurface(spf, ml_LTE)
        xr = self.streamsurface(spf, mcam)

        # Project camber angle onto streamsurface
        theta = util.cumtrapz0(dydm / xr[1], mcam * chord_full)

        # Stack so that camber theta=0 at the stacking point
        theta -= self.mstack

        # Add on the whole blade angular offset
        theta += self.theta_offset
        # theta += util.interp1d_linear_extrap(self.spf, self.dtheta)(spf)

        # Change in theta if r were constant
        dtu = Dy * chord / xr[1]
        dtl = -Dy * chord / xr[1]

        # Change in rtheta at the average radius between camber and surf
        drtu = dtu * 0.5 * (xr[1] + xru[1])
        drtl = dtl * 0.5 * (xr[1] + xrl[1])

        xrrtu = np.stack((*xru, theta * xru[1] + drtu))
        xrrtl = np.stack((*xrl, theta * xrl[1] + drtl))

        xrtu = xrrtu + 0.0
        xrtu[2] /= xrtu[1]

        xrtl = xrrtl + 0.0
        xrtl[2] /= xrtl[1]

        return xrtu, xrtl

    def surface_length(self, spf):
        """Suction surface length."""
        xrtu, xrtl = self.evaluate_section(spf)
        xrrtu = np.stack((*xrtu[:2],) + (xrtu[1] * xrtu[2],))
        xrrtl = np.stack((*xrtl[:2],) + (xrtl[1] * xrtl[2],))
        Lu = util.arc_length(xrrtu)
        Ll = util.arc_length(xrrtl)
        return np.maximum(Lu, Ll)

    def chord(self, spf):
        """Evaluate meridional length of a line from LE to TE."""

        # Meridional coordinates along camber line
        # are the average of the upper and lower surfaces
        xr = np.stack(self.evaluate_section(spf)).mean(axis=0)[:2]

        # Integrate the arc length
        return util.arc_length(xr)

    def get_coords(self, nspf=20, nchord=100, flip_theta=False):
        """3-D coordinates for this blade row in AutoGrid-style format.

        Parameters
        ----------
        nspf : int
            Number of sections in radial direction.
        nchord : int
            Number of chordwise points along each surface.

        Returns
        -------
        xrt : (2, nspf, nchord, 3) array
            Axial, radial, angular coordinates for this blade. `xrt[0]` is the
            upper surface, with highest theta, `xrt[1]` the lower surface.

        """

        eps = 1e-4
        xrt = np.stack(
            [
                self.evaluate_section(spf, nchord)
                for spf in np.linspace(-eps, 1.0 + eps, nspf)
            ]
        ).transpose(1, 0, 3, 2)

        if flip_theta:
            xrt[:, :, :, 2] *= -1.0

        return xrt
