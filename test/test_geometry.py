import numpy as np
from turbigen import geometry

def test_section():
    """Verify that blade sections are generated successfully."""
    chi1 = np.arange(-30.0, 30.0, 7)
    chi2 = np.arange(-60.0, 60.0, 7)
    aft = np.arange(-1.0, 1.0, 11)
    for chi1i in chi1:
        for chi2i in chi2:
            for afti in aft:
                xy = geometry.blade_section([chi1i, chi2i], afti)

                # Streamwise coordinate goes in correct direction
                assert np.all(np.diff(xy[0, :], 1) > 0.0)

                # Upper surface is higher than lower surface
                assert np.all(xy[1, :] - xy[2, :] >= 0.0)

                # Surfaces meet at ends
                assert np.all(np.isclose(xy[1, (0, -1)], xy[2, (0, -1)]))

                # Check camber line angles
                yc = np.mean(xy[(1, 2), :], axis=0)
                dyc = np.diff(yc, 1)
                dxc = np.diff(xy[0, :], 1)
                ang = np.degrees(np.arctan2(dyc, dxc))[
                    (0, -1),
                ]
                ang_tol = 0.1
                assert np.all(np.abs(ang - (chi1i, chi2i)) < ang_tol)
