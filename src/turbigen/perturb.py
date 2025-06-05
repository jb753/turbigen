"""Class to calculate linear perturbations of a state."""

from turbigen import util
import numpy as np


class Perturbator:
    def __init__(self, F, order="C", dtype=np.float64):
        """Calculate linear perturbations of a state.

        Parameters
        ----------
        F : FlowField
            A FlowField to perturb.
        order : {'C', 'F'}, optional
            Memory layout of working vectors and matrices.

        """
        self._F = F
        self.order = order
        self.dtype = dtype

    @property
    def prim(self):
        """Primitive variables."""
        F = self._F
        return util.stack_vector(F.rho, F.Vx, F.Vr, F.Vt, F.P, order=self.order)

    @property
    def cons(self):
        """Conserved variables."""
        F = self._F
        return util.stack_vector(
            F.rho,
            F.rhoVx,
            F.rhoVr,
            F.rho * F.r * F.Vt,
            F.rhoe,
            order=self.order,
        )

    def primitive_to_conserved(self):
        """Matrix to convert primitive to conserved perturbations.

        Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        conserved variables [rho, rhoVx, rhoVr, rhorVt, rhoe].

        This is like matrix C from Holmes (2008).

        Returns
        -------
        C: (npts, 5, 5) array

        """
        F = self._F
        return util.stack_matrix(
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (F.Vx, F.rho, 0.0, 0.0, 0.0),
            (F.Vr, 0.0, F.rho, 0.0, 0.0),
            (F.r * F.Vt, 0.0, 0.0, F.rho * F.r, 0.0),
            (F.drhoe_drho_P, F.rhoVx, F.rhoVr, F.rho * F.Vt, F.drhoe_dP_rho),
            order=self.order,
            shape=F.shape,
            dtype=self.dtype,
        )

    def conserved_to_primitive(self):
        """Get a matrix at every node that converts linear pertubations in
        conserved variables [rho, rhoVx, rhoVr, rhorVt, rhoe].
        to perturbations in
        primitive variables [rho, Vx, Vr, Vt, P]

        This is like matrix Cinv from Holmes (2008).

        Returns
        -------
        Cinv: (npts, 5, 5) array

        """
        F = self._F
        out = util.stack_matrix(
            (1.0, 0, 0, 0, 0),
            (-F.Vx, 1, 0, 0, 0),
            (-F.Vr, 0, 1, 0, 0),
            (-F.Vt, 0, 0, 1 / F.r, 0),
            (
                (F.V**2 - F.drhoe_drho_P),
                -F.Vx,
                -F.Vr,
                -F.Vt / F.r,
                1,
            ),
            order=self.order,
            shape=F.shape,
            dtype=self.dtype,
        )
        out[..., 1:4, :] /= F.rho
        out[..., -1, :] /= F.drhoe_dP_rho
        return out

    def primitive_to_chic(self):
        """Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        characteristic variables
        [dp-rho*a*dVx, dp+rho*a*dVx, rho*a*dVr, rho*a*dVt, dp - (a^2)*drho].
        [upstream acoustic, downstream acoustic, r-mom, t-mom, entropy wave]

        This is like matrix B from Holmes (2008).

        Returns
        -------
        B: (npts, 5, 5) array

        """
        F = self._F
        rhoa = F.rho * F.a
        return util.stack_matrix(
            (0.0, -rhoa, 0.0, 0.0, 1.0),
            (0.0, rhoa, 0.0, 0.0, 1.0),
            (0.0, 0.0, rhoa, 0.0, 0.0),
            (0.0, 0.0, 0.0, rhoa, 0.0),
            (-(F.a**2), 0.0, 0.0, 0.0, 1.0),
            order=self.order,
            shape=F.shape,
            dtype=self.dtype,
        )

    def chic_to_primitive(self):
        """Get a matrix at every node that converts linear pertubations in
        characteristic variables
        [dp-rho*a*dVx, dp+rho*a*dVx, rho*a*dVr, rho*a*dVt, dp - (a^2)*drho].
        [upstream acoustic, downstream acoustic, r-mom, t-mom, entropy wave]
        to perturbations in
        primitive variables [rho, Vx, Vr, Vt, P]

        This is like matrix Binv from Holmes (2008).

        Returns
        -------
        Binv: (npts, 5, 5) array
        """
        F = self._F
        _asq = 1.0 / F.a**2
        _rhoa = 1.0 / F.rho / F.a
        _2asq = _asq * 0.5
        _2rhoa = _rhoa * 0.5
        return util.stack_matrix(
            (_2asq, _2asq, 0.0, 0.0, -_asq),
            (-_2rhoa, _2rhoa, 0.0, 0.0, 0.0),
            (0.0, 0.0, _rhoa, 0.0, 0.0),
            (0.0, 0.0, 0.0, _rhoa, 0.0),
            (0.5, 0.5, 0.0, 0.0, 0.0),
            order=self.order,
            shape=F.shape,
            dtype=self.dtype,
        )

    def primitive_to_flux(self):
        """Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        flux variables
        [rhoVx, rhoVx^2+P, rhoVxVr, rhoVxrVt, rhoVx*ho].

        This is like matrix A from Holmes (2008).
        Does not have an analytical inverse.

        Returns
        -------
        A: (npts, 5, 5) array

        """

        F = self._F
        VxVr = F.Vx * F.Vr
        VxrVt = F.Vx * F.r * F.Vt
        VxVx = F.Vx**2
        dE_drho = F.Vx * F.ho + F.rhoVx * F.dhdrho_P
        dE_dVx = F.rho * F.ho + F.rhoVx * F.Vx

        return util.stack_matrix(
            (F.Vx, F.rho, 0.0, 0.0, 0.0),
            (VxVx, 2.0 * F.rhoVx, 0.0, 0.0, 1.0),
            (VxVr, F.rhoVr, F.rhoVx, 0.0, 0.0),
            (VxrVt, F.rhorVt, 0.0, F.rhoVx * F.r, 0.0),
            (dE_drho, dE_dVx, F.rhoVx * F.Vr, F.rhoVx * F.Vt, F.rhoVx * F.dhdP_rho),
            order=self.order,
            shape=F.shape,
            dtype=self.dtype,
        )

    def primitive_to_bcond(self):
        """Get a matrix at every node that converts linear pertubations in
        primitive variables [rho, Vx, Vr, Vt, P]
        to perturbations in
        boundary condition variables
        [ho, s, tanAlpha, tanBeta, P].

        Does not have an analytical inverse.

        Returns
        -------
        Y: (npts, 5, 5) array

        """

        F = self._F

        dtanAl_dVx = -F.Vt * F.Vx / F.Vm**3
        dtanAl_dVr = -F.Vt * F.Vr / F.Vm**3
        dtanAl_dVt = 1.0 / F.Vm
        dtanBe_dVx = -F.Vr / F.Vx**2
        dtanBe_dVr = 1.0 / F.Vx

        return util.stack_matrix(
            (F.dhdrho_P, F.Vx, F.Vr, F.Vt, F.dhdP_rho),
            (F.dsdrho_P, 0.0, 0.0, 0.0, F.dsdP_rho),
            (0.0, dtanAl_dVx, dtanAl_dVr, dtanAl_dVt, 0.0),
            (0.0, dtanBe_dVx, dtanBe_dVr, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 1.0),
            order=self.order,
            shape=F.shape,
            dtype=self.dtype,
        )
