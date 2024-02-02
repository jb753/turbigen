import turbigen.solvers.native
import numpy as np

def test_cell_to_node():
    x = np.ones((8,3,5,6,7))
    xn = turbigen.solvers.native.cell_to_node(x)
    assert np.allclose(xn,1.)
