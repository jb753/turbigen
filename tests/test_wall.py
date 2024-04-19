"""Check cell areas and volumes are correct."""
import turbigen.grid
import numpy as np
import turbigen.compflow_native as cf

def dot(a, b, axis=0):
    return np.sum(a * b, axis=axis)

def make_sector():

    # Geometry
    L = 0.1
    rm = 10.
    dr = 0.1

    r1 = rm-dr/2.
    r2 = rm+dr/2.

    nj = 5
    ni = 30
    nk = 7


    Nb = int(2.*np.pi*rm/dr)
    pitch = 2.*np.pi/Nb

    xv = np.linspace(0, L, ni)
    rv = np.linspace(r1, r2, nj)
    tv = np.linspace(0., pitch, nk)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing='ij'))

    return xrt, Nb


def test_box():

    xrt, Nb = make_sector()
    patches = [ ]
    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, Nb, patches)

    g = turbigen.grid.Grid([block,])
    g.check_coordinates()
    g.match_patches()

    iwall, jwall, kwall, wall = block.get_wall_face()

    assert iwall[0,:,:].all()
    assert iwall[-1,:,:].all()
    assert (~iwall[1:-1,:,:]).all()

    assert jwall[:,0,:].all()
    assert jwall[:,-1,:].all()
    assert (~jwall[:,1:-1,:]).all()

    assert kwall[:,:,0].all()
    assert kwall[:,:,-1].all()
    assert (~kwall[:,:,1:-1]).all()

    assert (~wall[1:-1,1:-1,1:-1]).all()
    assert wall[(0,-1),:,:].all()
    assert wall[:,(0,-1),:].all()
    assert wall[:,:,(0,-1)].all()

def test_stream():

    xrt1, Nb = make_sector()

    xrt2 = xrt1.copy()
    xrt2[0] += xrt1[0].ptp()

    patches1 = [
        turbigen.grid.InletPatch(i=0),
        turbigen.grid.PeriodicPatch(i=-1),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
    ]

    patches2 = [
        turbigen.grid.PeriodicPatch(i=0),
        turbigen.grid.OutletPatch(i=-1),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
    ]

    block1 = turbigen.grid.PerfectBlock.from_coordinates(xrt1, Nb, patches1)
    block2 = turbigen.grid.PerfectBlock.from_coordinates(xrt2, Nb, patches2)

    g = turbigen.grid.Grid([block1, block2])
    g.check_coordinates()
    g.match_patches()

    for b in g:

        iwall, jwall, kwall, wall = b.get_wall_face()

        assert (~iwall).all()
        assert (~kwall).all()

        assert (jwall[:,0,:]).all()
        assert (jwall[:,-1,:]).all()
        assert (~jwall[:,1:-1,:]).all()

        assert wall[:,0,:].all()
        assert wall[:,-1,:].all()
        assert (~wall[:,1:-1,:]).all()


def test_gap():

    xrt, Nb = make_sector()

    ile = 10
    ite = 20

    patches = [
        turbigen.grid.InletPatch(i=0),
        turbigen.grid.OutletPatch(i=-1),
        turbigen.grid.PeriodicPatch(k=0, i=(0,ile)),
        turbigen.grid.PeriodicPatch(k=-1, i=(0,ile)),
        turbigen.grid.PeriodicPatch(k=0, i=(ite,-1)),
        turbigen.grid.PeriodicPatch(k=-1, i=(ite,-1)),
    ]

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, Nb, patches)

    g = turbigen.grid.Grid([block,])
    g.check_coordinates()
    g.match_patches()

    iwall, jwall, kwall, wall = block.get_wall_face()

    assert (~iwall).all()
    assert (jwall[:,0,:]).all()
    assert (jwall[:,-1,:]).all()
    assert (~jwall[:,1:-1,:]).all()
    assert (kwall[ile:ite,:,0]).all()
    assert (kwall[ile:ite,:,-1]).all()

    assert wall[ile:(ite+1),:,0].all()
    assert wall[ile:(ite+1),:,-1].all()
    assert (~wall[:ile,:,0]).all()
    assert (~wall[(ite+1):,:,0]).all()
    assert wall[ile:(ite+1),:,-1].all()
    assert wall[:,0,:].all()
    assert wall[:,-1,:].all()
    assert (~wall[:,1:-1,1:-1]).all()


def test_periodic():

    xrt, Nb = make_sector()

    patches = [
        turbigen.grid.InletPatch(i=0),
        turbigen.grid.OutletPatch(i=-1),
        turbigen.grid.PeriodicPatch(k=0),
        turbigen.grid.PeriodicPatch(k=-1),
    ]

    block = turbigen.grid.PerfectBlock.from_coordinates(xrt, Nb, patches)

    g = turbigen.grid.Grid([block,])
    g.check_coordinates()
    g.match_patches()

    iwall, jwall, kwall, wall = block.get_wall_face()

    assert (~iwall).all()
    assert (~kwall).all()

    assert (jwall[:,0,:]).all()
    assert (jwall[:,-1,:]).all()
    assert (~jwall[:,1:-1,:]).all()

    assert wall[:,0,:].all()
    assert wall[:,-1,:].all()
    assert (~wall[:,1:-1,:]).all()


if __name__=='__main__':

    test_periodic()
    test_gap()
    test_box()
    test_stream()

