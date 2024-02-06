import turbigen.solvers.native
import numpy as np

def test_cell_to_node():
    x = np.ones((8,3,5,6,7))
    xn = turbigen.solvers.native.cell_to_node(x)
    assert np.allclose(xn,1.)

# def test_smooth():
#     shape = (4,5,6,7)
#     x = np.ones(shape) + 0.1*np.random.random_sample(shape)
#     for _ in range(100):
#         x = turbigen.solvers.native.smooth(x)
#         print(x.ptp())
#     assert x.ptp()<1e-3
