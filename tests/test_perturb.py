from turbigen import flowfield
from turbigen import perturb
import numpy as np


def test_matrices():
    """Check the primitive to conserved, chic, fluxes, and bcond matrices"""

    np.set_printoptions(precision=3, suppress=True)

    # Define perturbations
    mag = 1e-5
    tol = 1e-2
    perturbations = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0],
        [-0.7, 0.9, -0.7, 0.8, 1.0],
        [-1.2, 3.5, -0.7, 4.1, 2.2],
    ]

    # Initialize a flow field
    F = flowfield.PerfectFlowField(shape=(1,))
    F.cp = 1105.0
    F.gamma = 1.3
    F.mu = 1.8e-5
    F.xrt = 2 * np.ones((3, 1))
    F.Vxrt = [[[100.0], [80.0], [50.0]]]
    F.set_P_T(1.2e5, 295.0)
    F2 = F.copy()

    # Initialise perturbator
    pert = perturb.Perturbator(F)

    for fac_prim in perturbations:
        #
        # Apply primitive perturbation to F2
        dprim = F.primitive * np.array(fac_prim)[..., None] * mag
        F2.set_primitive(F.primitive + dprim)

        # Check conserved
        C = pert.primitive_to_conserved()
        Cinv = pert.conserved_to_primitive()
        assert np.allclose((Cinv @ C).squeeze(), np.eye(5), atol=1e-6), (
            "Primitive-conserved inverse wrong"
        )

        dcons = F2.conserved - F.conserved
        assert np.allclose(dcons, C @ dprim, rtol=tol), (
            "Primitive-conserved matrix wrong"
        )

        # Manually calculate chic vector
        dp = F2.P - F.P
        dVx = F2.Vx - F.Vx
        dVr = F2.Vr - F.Vr
        dVt = F2.Vt - F.Vt
        drho = F2.rho - F.rho
        a = 0.5 * (F2.a + F.a)
        rho = 0.5 * (F2.rho + F.rho)
        dchic = [
            dp - rho * a * dVx,
            dp + rho * a * dVx,
            rho * a * dVr,
            rho * a * dVt,
            dp - (a**2) * drho,
        ]

        # Check prim to chic
        B = pert.primitive_to_chic()
        Binv = pert.chic_to_primitive()
        assert np.allclose((Binv @ B).squeeze(), np.eye(5), atol=1e-6), (
            "Primitive-chic inverse wrong"
        )
        assert np.allclose(dchic, B @ dprim, rtol=tol), "Primitive-chic matrix wrong"

        # Check prim to fluxes
        dflux = F2.fluxx - F.fluxx
        A = pert.primitive_to_flux()
        assert np.allclose(dflux, A @ dprim, rtol=tol)

        # Check prim to bcond
        dbcond = F2.bcond - F.bcond
        Y = pert.primitive_to_bcond()
        assert np.allclose(dbcond, Y @ dprim, rtol=tol)

        # Check inverses if no zeros in dprim
        if not (dprim.squeeze() == 0.0).any():
            assert np.allclose(dprim, Binv @ dchic, rtol=tol)
            assert np.allclose(dprim, Cinv @ dcons, rtol=tol)

    print("ok")


def test_chic_waves():
    """Set up a flow field with travelling waves and check chics recovered."""

    cp = 1005.0
    ga = 1.4
    mu = 1.84e-5
    Vx = 100.0
    L = 1.0
    f = 500.0

    # Set up coordinates
    ni = 50
    nt = 100
    xv = np.linspace(0.0, L, ni)
    xrt = np.stack((xv, np.ones_like(xv), np.zeros_like(xv)))
    omega = 2 * np.pi * f
    t = np.linspace(0.0, 1 / f, nt, endpoint=False)[None, :]
    dt = np.diff(t)[0]
    x = xv[:, None]

    # Mean flow field first
    F = flowfield.PerfectFlowField(shape=(ni, nt))
    F.cp = cp
    F.gamma = ga
    F.mu = mu
    F.xrt = xrt[..., None]
    F.Vx = Vx
    F.Vr = 0.0
    F.Vt = 0.0
    F.set_P_T(1e5, 300.0)

    # Prescribe pressure wave
    Aup = 1e-3
    Adn = 2.2e-3
    a0 = F.a
    rho0 = F.rho
    dPdn = Adn * np.exp(1j * omega * (t - x / a0))
    dPup = Aup * np.exp(1j * omega * (t + x / a0))
    dP = np.real(dPdn + dPup)

    # Momentum for velocity
    # du/dt = -1/rho dp/dx
    dV = np.real(dPdn - dPup) / rho0 / a0

    # Apply to the flowfield
    F.set_P_s(F.P + dP, F.s)
    F.Vx = F.Vx + dV

    # Get changes over a time step
    dU = np.moveaxis(np.diff(F.conserved, axis=-1), 0, -1)[..., None]

    # Convert to chics
    pert = perturb.Perturbator(F)
    primitive_to_chic = pert.primitive_to_chic()
    primitive_to_conserved = pert.primitive_to_conserved()
    conserved_to_chic = primitive_to_chic @ np.linalg.inv(primitive_to_conserved)
    dchic = conserved_to_chic[:, :-1, :, :] @ dU

    # The zero-to-peak amplitudes should match the ratio of up to down waves
    # Neglecting any scaling factors
    Amp_x = np.mean(np.ptp(dchic, axis=0)[:, :2, 0], axis=0)
    Amp_t = np.mean(np.ptp(dchic, axis=1)[:, :2, 0], axis=0)

    assert np.isclose(Amp_x[0] / Amp_x[1], Aup / Adn)
    assert np.isclose(Amp_t[0] / Amp_t[1], Aup / Adn)

    return


if __name__ == "__main__":
    test_matrices()
    test_chic_waves()
    print("All tests passed.")
