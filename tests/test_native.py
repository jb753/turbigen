import turbigen.solvers.native
import turbigen.grid
import numpy as np
import turbigen.compflow_native as cf

def test_cell_to_node():
    x = np.ones((8,3,5,6,7))
    xn = turbigen.solvers.native.cell_to_node(x)
    assert np.allclose(xn,1.)

def test_node_to_face():
    x = np.ones((8,3,5,6,7))
    xf = turbigen.solvers.native.node_to_face(x)
    for xfi in xf:
        assert np.allclose(xfi,1.)

def test_smooth():
    shape = (1,10,12,14)
    x = np.ones(shape) + 0.05*np.random.random_sample(shape)
    for i in range(100):
        x = turbigen.solvers.native.smooth(x)
        if not np.mod(i,10):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            h = ax.contourf(x[0,2,:,:],cmap='RdBu')
            plt.colorbar(h)
            plt.show()
        print(x.ptp())
    assert x.ptp()<1e-3

def test_fluxes():

    g = make_duct()
    b = g[0]
    b.Vx = 5.
    b.Vr = 10.
    b.Vt = 15.
    rho = b.rho

    conservedPhor = np.stack((*b.conserved, b.P, b.ho, b.r))
    Omega = 0.

    Fc, FP = turbigen.solvers.native.get_fluxes(conservedPhor, Omega)




def make_duct():

    # Geometry
    h = 0.1
    L = h * 4.0
    htr = 0.999
    rm = 0.5 * h * (1.0 + htr) / (1.0 - htr)
    rh = rm - 0.5 * h
    rt = rm + 0.5 * h

    # Boundary conditions
    ga = 1.4
    cp = 1005.0
    mu = 1.8e-5
    Alpha = 0.0
    Beta = 0.0
    Po1 = 1e5
    To1 = 300.0

    M = 0.3
    rgas = cp * (ga-1.)/ga
    V = cf.V_cpTo_from_Ma(M,ga)*np.sqrt(cp*To1)
    P1 = Po1/cf.Po_P_from_Ma(M,ga)
    T1 = To1/cf.To_T_from_Ma(M,ga)


    nj = 20
    AR = 1.
    ni = int(nj/h*L)
    nk = 6
    pitch = h/nj*(nk-1)
    Nb = int(2.0 * np.pi * rm / pitch)
    dt = 2.0 * np.pi / float(Nb)
    tv = np.linspace(-dt / 2., dt / 2., nk)
    xv = np.linspace(0., L, ni)
    rv = np.linspace(rh, rt, nj)
    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing='ij'))

    patches = [
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
        turbigen.grid.InletPatch(i=0),
        turbigen.grid.OutletPatch(i=-1),
    ]

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, Nb, patches)
    g = turbigen.grid.Grid([block,])
    g.match_patches()
    g.check_coordinates()

    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_P_T(Po1, To1)
    g.apply_inlet(So1, Alpha, Beta)
    g.calculate_wall_distance()
    g.apply_outlet(P1-2.)

    for b in g:
        b.Vx = V
        b.Vr = 0.
        b.Vt = 0.
        b.cp = cp
        b.gamma = ga
        b.mu = mu
        b.Omega = 0.0
        b.set_P_T(P1, T1)

    return g

test_smooth()
