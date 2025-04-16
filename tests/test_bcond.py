"""Verify implementatio of emb solver boundary conditions."""

import numpy as np

import turbigen.compflow_native as cf
import turbigen.fluid
import turbigen.grid
import turbigen.solvers.emb as embsolve
import turbigen.util


def make_grid(use_inlet, use_outlet, use_mixing):
    """Generate a simple two-block grid."""

    # Geometry
    h = 0.1
    L_h = 2.0
    htr = 0.95
    L = h * L_h
    rm = 0.5 * h * (1.0 + htr) / (1.0 - htr)
    rh = rm - 0.5 * h
    rt = rm + 0.5 * h

    # Boundary conditions
    ga = 1.4
    cp = 1005.0
    mu = 1.8e-3
    Ma1 = 0.3
    Po1 = 1e5
    To1 = 300.0

    # Set inlet Ma to get inlet static state
    V = cf.V_cpTo_from_Ma(Ma1, ga) * np.sqrt(cp * To1)
    P1 = Po1 / cf.Po_P_from_Ma(Ma1, ga)
    T1 = To1 / cf.To_T_from_Ma(Ma1, ga)

    # Relative flow angle
    Alpha = 20.0
    Beta = 10.0
    Vt = V * np.sin(np.radians(Alpha))
    Vx = V * np.cos(np.radians(Alpha))

    # Numbers of grid points
    AR_pitch = 1.0
    AR_merid = 2.0
    nj = 33 + 8
    nk = 33
    ni = int(nj * L_h / AR_merid)

    # Use pitchwise aspect ratio to find cell spacing, pitch and Nb
    pitch = h / (nj - 1) * (nk - 1) * AR_pitch
    Nb = int(2.0 * np.pi * rm / pitch)
    dt = 2.0 * np.pi / float(Nb)

    # Make the coordinates
    tv = np.linspace(0.0, dt, nk)
    xv = np.linspace(0.0, L, ni)
    rv = np.linspace(rh, rt, nj)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

    # Split into blocks
    blocks = []
    nblock = 2
    istb = [ni // nblock * iblock for iblock in range(nblock)]
    ienb = [ni // nblock * (iblock + 1) + 1 for iblock in range(nblock)]
    ienb[-1] = ni

    patches = [
        [
            turbigen.grid.PeriodicPatch(k=0),
            turbigen.grid.PeriodicPatch(k=-1),
            turbigen.grid.InviscidPatch(j=0),
            turbigen.grid.InviscidPatch(j=-1),
        ],
        [
            turbigen.grid.PeriodicPatch(k=0),
            turbigen.grid.PeriodicPatch(k=-1),
            turbigen.grid.InviscidPatch(j=0),
            turbigen.grid.InviscidPatch(j=-1),
        ],
    ]

    if use_outlet:
        patches[1].append(turbigen.grid.OutletPatch(i=-1))

    if use_inlet:
        patches[0].append(turbigen.grid.InletPatch(i=0))

    if use_mixing:
        patches[0].append(turbigen.grid.MixingPatch(i=-1))
        patches[1].append(turbigen.grid.MixingPatch(i=0))
    else:
        patches[0].append(turbigen.grid.PeriodicPatch(i=-1))
        patches[1].append(turbigen.grid.PeriodicPatch(i=0))

    for iblock in range(nblock):
        block = turbigen.grid.PerfectBlock.from_coordinates(
            xrt[:, istb[iblock] : ienb[iblock], :, :], Nb, patches[iblock]
        )
        block.label = f"b{iblock}"
        blocks.append(block)

    # Make the grid object
    g = turbigen.grid.Grid(blocks)
    g.check_coordinates()

    # Boundary conditions
    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_P_T(Po1, To1)
    g.apply_inlet(So1, Alpha, 0.0)
    g.calculate_wall_distance()
    g.apply_outlet(P1)

    # Fluid props
    for b in g:
        b.cp = cp
        b.gamma = ga
        b.mu = mu

    # Initial guess with an offset
    mag = 0.01
    for ib, b in enumerate(g):
        dV = (ib + 1) * Vx * mag
        dT = (ib + 1) * To1 * mag
        dP = (ib + 1) * Po1 * mag
        b.Vx = Vx + dV
        b.Vr = -dV
        b.Vt = Vt + dV
        b.set_P_T(P1 + dP, T1 + dT)
        b.Omega = 0.0

    g.match_patches()

    return g


def test_outlet_CFL_0():
    """Without any update from the interior, outlet should force correct values."""

    g = make_grid(use_inlet=False, use_outlet=True, use_mixing=False)
    outlet_patch = g.outlet_patches[0]

    embsolve.Emb(n_step=1000, n_step_avg=1, CFL=0.0).run(g)

    C = outlet_patch.get_cut()
    assert np.isclose(C.P.mean(), outlet_patch.Pout)


def test_inlet_CFL_0():
    """Without any update from the interior, inlet and outlet should force correct values."""

    g = make_grid(use_inlet=True, use_outlet=False, use_mixing=False)

    embsolve.Emb(n_step=1000, n_step_avg=1, CFL=0.0).run(g)

    patch = g.inlet_patches[0]
    C = patch.get_cut()
    htol = C.V.mean() ** 2 * 1e-3
    stol = htol / C.T.mean()
    angtol = 0.01
    assert np.allclose(C.ho, patch.state.h, atol=htol)
    assert np.allclose(C.s, patch.state.s, atol=stol)
    assert np.allclose(C.Alpha, patch.Alpha, atol=angtol)
    assert np.allclose(C.Beta, patch.Beta, atol=angtol)


def test_CFL():
    """Does it nan immediately with non-zero CFL?"""
    g = make_grid(use_inlet=True, use_outlet=True, use_mixing=False)
    embsolve.Emb(
        n_step=2,
        n_step_avg=1,
    ).run(g)


def test_mixer():
    """Does the mixing plane conserve pitchwise-integrated flows?"""

    g = make_grid(use_inlet=True, use_outlet=True, use_mixing=True)

    embsolve.Emb(n_step=5000, n_step_log=100).run(g)

    fluxes = []
    for patch in g.mixing_patches:
        C = patch.get_cut().mix_out()[0]
        fluxes.append(C.fluxes)
    fluxes = np.stack(fluxes)
    flux_avg = np.mean(fluxes, axis=0)
    flux_ref = np.array(
        [flux_avg[0], flux_avg[1], flux_avg[1], C.r * flux_avg[1], flux_avg[-1]]
    )
    err = np.diff(fluxes, axis=0) / flux_ref
    rtol = 2.5e-6
    assert (np.abs(err) < rtol).all()


if __name__ == "__main__":
    # pass
    # test_outlet_CFL_0()
    # test_inlet_CFL_0()
    # test_CFL()
    test_mixer()
