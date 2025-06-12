import numpy as np
import matplotlib.pyplot as plt
import jmesh
import turbigen.fluid
import turbigen.grid
from turbigen.solvers import ember


def get_blade(fname):
    """Read the blade surface data from a CSV file."""

    xy = np.loadtxt(fname, delimiter=",").T / 1000.0  # Convert to meters
    xy1 = xy[:2]
    xy2 = xy[2:]

    # Find last non-zero value in xy2 and cut after that
    ilast = np.where(xy2[1, :] != 0.0)[0][-1]
    xy2 = xy2[:, : ilast + 1]

    # Locate TE and move it to the end
    iTE = np.argmax(xy1[0, :])
    xy2 = np.concatenate((xy2, np.flip(xy1[:, iTE:-1], axis=1)), axis=1)
    xy1 = xy1[:, : (iTE + 1)]

    return xy1, xy2


def get_dimensional_bcond(Ma, Re):
    """Calculate dimensional boundary conditions from non-dimensional values."""

    To1 = 420.0
    cp = 1004.0  # Specific heat at constant pressure [J/(kg*K)]
    ga = 1.4  # Specific heat ratio
    mu = 1.7894e-5  # Dynamic viscosity [kg/(m*s)]
    L = 0.067647  # Reference length [m]

    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)

    # Guess inlet state
    # Will have correct enthalpy but wrong pressure and entropy
    So1.set_P_T(1e5, To1)

    # Guess isentropic velocity
    V2s = Ma * So1.a

    # Iterate to find states
    S2s = So1.copy()
    for _ in range(10):
        # Calculate downstream state
        S2s.set_h_s(So1.h - 0.5 * V2s**2, So1.s)

        # Calculate velocity from Mach
        V2s = Ma * S2s.a

        # Calculate density from Reynolds number
        rho2s = Re * S2s.mu / V2s / L

        # Update exit state
        P2s = rho2s * S2s.rgas * S2s.T
        S2s.set_P_h(P2s, S2s.h)

        # Update inlet state
        So1.set_h_s(So1.h, S2s.s)

    # Verify
    assert np.isclose(So1.T, To1)
    assert np.isclose(So1.h, S2s.h + 0.5 * V2s**2)
    assert np.isclose(V2s / S2s.a, Ma)
    assert np.isclose(Re, S2s.rho * V2s * L / S2s.mu)

    return So1, S2s, V2s


xy1, xy2 = get_blade("scripts/vki_blade.csv")

# Set up axial grid vector
xLE = 0.0
xTE = max(xy1[0].max(), xy2[0].max())
cx = xTE - xLE

# Calculate pitch based on paper numbers
stagger = 55.0
c_ref = 0.067647
g = 0.85 * c_ref

# Offset the blade surface pitchwise
xy2[1] += g

m = jmesh.builder.Builder()

# Cell sizes
dxLE = cx * 0.005
dxTE = cx * 0.005
dxmax = cx * 0.1
dyw = cx * 0.001
dyin = dyw * 10
dyout = dyw * 5

Re = 1e6
Ma = 0.8

So1, S2s, V2s = get_dimensional_bcond(Ma, Re)

# Use flat plate correlations to get viscous length scale
yplus = 30.0
Cf = (2.0 * np.log10(Re) - 0.65) ** -2.3
tauw = Cf * 0.5 * (S2s.rho * V2s**2)
Vtau = np.sqrt(tauw / S2s.rho)
Lvisc = (S2s.mu / S2s.rho) / Vtau
dw = yplus * Lvisc

# Offset vectors for inlet and exit planes
Dxy_in = np.array([-cx, 0.0, 0.0]).reshape(3, 1)
ang = -30.0
Dxy_out = np.array([cx, cx * np.tan(np.radians(ang)), 0.0]).reshape(3, 1)

# Blade vertices
m["B"] = jmesh.primitives.Vertex(*xy1[:, 0])
m["C"] = jmesh.primitives.Vertex(*xy1[:, -1])
m["F"] = jmesh.primitives.Vertex(*xy2[:, 0])
m["G"] = jmesh.primitives.Vertex(*xy2[:, -1])

# Inlet and outlet vertices
m["A"] = m["B"].offset(Dxy_in)
m["E"] = m["F"].offset(Dxy_in)
m["D"] = m["C"].offset(Dxy_out)
m["H"] = m["G"].offset(Dxy_out)

# Blade edges
m["BC"] = jmesh.primitives.Edge(*xy1)
m["FG"] = jmesh.primitives.Edge(*xy2)
ni_chord = m.generate_edge_double_clustered("BC", dxLE, dxTE, dxmax).n
m.generate_edge_double_clustered("FG", dxLE, dxTE, dxmax, N=ni_chord)
nk = m.generate_edge_double_clustered("BF", dw, dw, dxmax).n
m.generate_edge_double_clustered("CG", dw, dw, dxmax, N=nk)

# Inlet and outlet edges
m.generate_edge_double_clustered("AE", dyin, dyin, dxmax, N=nk)
m.generate_edge_double_clustered("DH", dyout, dyout, dxmax, N=nk)
dy_in = m["AE"].ds.max()
ARin = 2.0
ARout = 2.0
dx_in = dy_in * ARin
ni_inlet = m.generate_edge_single_clustered("FE", dxLE, dx_in).n
m.generate_edge_single_clustered("BA", dxLE, dx_in)
dy_out = m["DH"].ds.max()
dx_out = dy_out * ARout
ni_outlet = m.generate_edge_single_clustered("GH", dxTE, dx_out).n
m.generate_edge_single_clustered("CD", dxTE, dx_out)

# All xy edges done.
# Now project spanwise
H = cx * 3.0
edge_projected, edge_spanwise = m.project_edges(list(m.edges.keys()), z=H)
# Coarse clustering in spanwise direction (we will usen inviscid walls)
dzw = H * 0.01
dzmax = H * 0.1
nj = 36
m.generate_edge_double_clustered(edge_spanwise, dzw, dzw, dzmax, N=nj)

m.generate_faces()
print("Number of faces:", len(m.faces))
print(list(m.faces.keys()))
m.generate_blocks()
print("Number of blocks:", len(m.blocks))
print(list(m.blocks.keys()))
# m.plot_xy(z=0.0)
# m.plot_xyz(show_faces=False, show_edges=False)
# plt.show()

# Calculate rmid based on an integer number of blades
Nb = 100
rmid = g * float(Nb) / (2.0 * np.pi)
print(f"g= {g:.5f}, pitch at rmid= {2 * np.pi * rmid / Nb:.5f}, L={m['BF'].S}")

# # Calculate the transformation to polar coords
# circum = 2 * np.pi * geom.roffset
# Nb = np.round(circum / geom.D).astype(int)  # Number of blades
#
# Create the grid object
# Need to sort out z<->y swap and fliping for +ve vols
xyz = [b.xyz for b in m.blocks.values()]
# Sort by min x
xyz = sorted(xyz, key=lambda x: x[0].min())


patches = [
    [
        turbigen.grid.InletPatch(i=0),
        turbigen.grid.PeriodicPatch(i=-1),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
        turbigen.grid.InviscidPatch(j=0),
        turbigen.grid.InviscidPatch(j=-1),
    ],
    [
        turbigen.grid.PeriodicPatch(i=0),
        turbigen.grid.PeriodicPatch(i=-1),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
        turbigen.grid.InviscidPatch(j=0),
        turbigen.grid.InviscidPatch(j=-1),
    ],
    [
        turbigen.grid.OutletPatch(i=-1),
        turbigen.grid.PeriodicPatch(i=0),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
        turbigen.grid.InviscidPatch(j=0),
        turbigen.grid.InviscidPatch(j=-1),
    ],
]

g = turbigen.grid.from_xyz(
    xyz,
    So1,
    Nb,
    rmid,
    ["inlet", "blade", "outlet"],
    sector=True,
    patches=patches,
)
print("Ncells/10^6=", g.ncell / 1e6)
for b in g:
    print(b.label, b.x.min())
    b.set_rho_u(So1.rho, So1.u)
    b.Vx = 0.5 * V2s
    b.Vr = 0.0
    b.Vt = 0.0
    b.Omega = 0.0

g.check_coordinates()

print(2 * np.pi / g[0].Nb)
for k in (0, -1):
    print("Patch k=", k)
    C = g[0][:, :, k]
    print(C.x.min(), C.x.max())
    print(C.r.min(), C.r.max())
    print(C.t.min(), C.t.max(), C.t.mean())


g.match_patches()
g.apply_inlet(So1, 0.0, 0.0)
g.apply_outlet(S2s.P)

ember.Ember().run(g)

# for p in g.periodic_patches:
#     if not p.match:
#         p.block.patches.remove(p)
#
# return g
