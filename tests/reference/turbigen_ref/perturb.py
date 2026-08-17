import numpy as np


class Perturbation:
    def __init__(self, state):
        self._state = state

    @property
    def conserved_to_chic(self):
        return self.primitive_to_chic @ self.conserved_to_primitive

    @property
    def primitive_to_conserved(self):
        """Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        conserved variables [rho, rhoVx, rhoVr, rhorVt, rhoe].

        Returns
        -------
        C: (npts, 5, 5) array

        """

        Z = np.zeros(self._state.shape)
        one = np.ones(self._state.shape)
        C = np.stack(
            (
                (
                    one,
                    self._state.Vx,
                    self._state.Vr,
                    self._state.rVt,
                    self._state.drhoe_drho_P,
                ),  # d/drho
                (Z, self._state.rho, Z, Z, self._state.rhoVx),  # d/dVx
                (Z, Z, self._state.rho, Z, self._state.rhoVr),  # d/dVr
                (Z, Z, Z, self._state.r * self._state.rho, self._state.rhoVt),  # d/dVt
                (Z, Z, Z, Z, self._state.drhoe_dP_rho),  # d/dP
            )
        )
        C = np.moveaxis(C, (0, 1), (-1, -2))
        return C

    @property
    def conserved_to_primitive(self):
        """Get a matrix at every node that converts linear pertubations in
        conserved variables [rho, rhoVx, rhoVr, rhorVt, rhoe].
        to perturbations in
        primitive variables [rho, Vx, Vr, Vt, P]

        Returns
        -------
        Cinv: (npts, 5, 5) array

        """
        Z = np.zeros(self._state.shape)
        one = np.ones(self._state.shape)
        Cinv = np.stack(
            (
                (one, Z, Z, Z, Z),
                (-self._state.Vx, one, Z, Z, Z),
                (-self._state.Vr, Z, one, Z, Z),
                (-self._state.Vt, Z, Z, one / self._state.r, Z),
                (
                    (self._state.V**2 - self._state.drhoe_drho_P),
                    -self._state.Vx,
                    -self._state.Vr,
                    -self._state.Vt / self._state.r,
                    one,
                ),
            )
        )
        Cinv[1:4] /= self._state.rho
        Cinv[-1] /= self._state.drhoe_dP_rho
        Cinv = np.moveaxis(Cinv, (0, 1), (-2, -1))
        return Cinv

    @property
    def primitive_to_chic(self):
        """Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        characteristic variables
        [dp-rho*a*dVx, dp+rho*a*dVx, rho*a*dVr, rho*a*dVt, dp - (a^2)*drho].
        [upstream acoustic, downstream acoustic, r-mom, t-mom, entropy wave]

        Returns
        -------
        B: (npts, 5, 5) array

        """

        Z = np.zeros(self._state.shape)
        one = np.ones(self._state.shape)
        rhoa = self._state.rho * self._state.a
        B = np.stack(
            (
                (Z, Z, Z, Z, -(self._state.a**2)),  # d/rho
                (-rhoa, rhoa, Z, Z, Z),  # d/dVx
                (Z, Z, rhoa, Z, Z),  # d/dVr
                (Z, Z, Z, rhoa, Z),  # d/dVt
                (one, one, Z, Z, one),  # d/dP
            )
        )
        B = np.moveaxis(B, (0, 1), (-1, -2))
        return B

    @property
    def chic_to_conserved(self):
        return self.primitive_to_conserved @ self.chic_to_primitive

    @property
    def chic_to_bcond(self):
        return self.primitive_to_bcond @ self.chic_to_primitive

    @property
    def chic_to_primitive(self):
        """Get a matrix at every node that converts linear pertubations in
        characteristic variables
        [dp-rho*a*dVx, dp+rho*a*dVx, rho*a*dVr, rho*a*dVt, dp - (a^2)*drho].
        [upstream acoustic, downstream acoustic, r-mom, t-mom, entropy wave]
        to perturbations in
        primitive variables [rho, Vx, Vr, Vt, P]

        Returns
        -------
        Binv: (npts, 5, 5) array
        """
        zero = np.zeros(self._state.shape)
        one = np.ones(self._state.shape)
        half = one / 2.0
        asq_recip = 1.0 / self._state.a**2
        rhoa_recip = 1.0 / self._state.rho / self._state.a
        Binv = np.stack(
            (
                (
                    asq_recip / 2.0,
                    asq_recip / 2.0,
                    zero,
                    zero,
                    -asq_recip,
                ),
                (-rhoa_recip / 2.0, rhoa_recip / 2, zero, zero, zero),
                (zero, zero, rhoa_recip, zero, zero),
                (zero, zero, zero, rhoa_recip, zero),
                (half, half, zero, zero, zero),
            )
        )
        Binv = np.moveaxis(Binv, (0, 1), (-2, -1))
        return Binv

    @property
    def primitive_to_flux(self):
        """Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        flux variables
        [rhoVx, rhoVx^2+P, rhoVxVr, rhoVxrVt, rhoVx*ho].

        Returns
        -------
        A: (npts, 5, 5) array

        """

        Z = np.zeros(self._state.shape)
        one = np.ones(self._state.shape)
        VxVr = self._state.Vx * self._state.Vr
        VxrVt = self._state.Vx * self._state.rVt
        VxVx = self._state.Vx**2
        dE_drho = (
            self._state.Vx * self._state.ho + self._state.rhoVx * self._state.dhdrho_P
        )
        dE_dVx = self._state.rho * self._state.ho + self._state.rhoVx * self._state.Vx
        A = np.stack(
            (
                (self._state.Vx, VxVx, VxVr, VxrVt, dE_drho),  # d/rho
                (
                    self._state.rho,
                    2.0 * self._state.rhoVx,
                    self._state.rhoVr,
                    self._state.rhorVt,
                    dE_dVx,
                ),  # d/dVx
                (
                    Z,
                    Z,
                    self._state.rhoVx,
                    Z,
                    self._state.rhoVx * self._state.Vr,
                ),  # d/dVr
                (
                    Z,
                    Z,
                    Z,
                    self._state.rhoVx * self._state.r,
                    self._state.rhoVx * self._state.Vt,
                ),  # d/dVt
                (Z, one, Z, Z, self._state.rhoVx * self._state.dhdP_rho),  # d/dP
            )
        )
        A = np.moveaxis(A, (0, 1), (-1, -2))
        return A

    @property
    def flux_to_chic(self):
        return self.primitive_to_chic @ self.flux_to_primitive

    @property
    def bcond_to_cons(self):
        return self.primitive_to_conserved @ self.bcond_to_primitive

    @property
    def flux_to_primitive(self):
        """Get a matrix at every node that converts linear pertubations in
        flux variables
        [rhoVx, rhoVx^2+P, rhoVxVr, rhoVxrVt, rhoVx*ho].
        to perturbations in
        primitive variables [rho, Vx, Vr, Vt, P]

        Returns
        -------
        Ainv: (npts, 5, 5) array
        """
        return np.linalg.inv(self.primitive_to_flux)

    @property
    def primitive_to_bcond(self):
        """Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        boundary condition variables
        [ho, s, tanAlpha, tanBeta, P].

        Returns
        -------
        Y: (npts, 5, 5) array

        """

        Z = np.zeros(self._state.shape)
        one = np.ones(self._state.shape)

        dtanAl_dVx = -self._state.tanAlpha * self._state.Vx / self._state.Vm**2
        dtanAl_dVr = -self._state.tanAlpha * self._state.Vr / self._state.Vm**2
        dtanAl_dVt = 1.0 / self._state.Vm

        dtanBe_dVx = -self._state.Vr / self._state.Vx**2
        dtanBe_dVr = 1.0 / self._state.Vx

        Y = np.stack(
            (
                (self._state.dhdrho_P, self._state.dsdrho_P, Z, Z, Z),  # d/rho
                (self._state.Vx, Z, dtanAl_dVx, dtanBe_dVx, Z),  # d/dVx
                (self._state.Vr, Z, dtanAl_dVr, dtanBe_dVr, Z),  # d/dVr
                (self._state.Vt, Z, dtanAl_dVt, Z, Z),  # d/dVt
                (self._state.dhdP_rho, self._state.dsdP_rho, Z, Z, one),  # d/dP
            )
        )
        Y = np.moveaxis(Y, (0, 1), (-1, -2))
        return Y

    @property
    def bcond_to_primitive(self):
        """Get a matrix at every node that converts linear pertubations in
        boundary condition variables
        [ho, s, tanAlpha, tanBeta, P].
        to perturbations in
        primitive variables [rho, Vx, Vr, Vt, P]

        Returns
        -------
        Yinv: (npts, 5, 5) array

        """
        return np.linalg.inv(self.primitive_to_bcond)

    @property
    def inlet_to_chic(self):
        # Convert downstream-running chics to primitive changes
        # Omit first column corresponding to upstream-running chic
        chic_to_prim = self.chic_to_primitive[..., :, 1:]

        # Convert primitive to inlet bcond changes
        # Omit last row corresponding to static pressure
        prim_to_inlet = self.primitive_to_bcond[..., :-1, :]

        # Reversed transform from inlet to chic
        return np.linalg.inv(prim_to_inlet @ chic_to_prim)
