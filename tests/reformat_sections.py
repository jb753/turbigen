import turbigen.util
import numpy as np
from stl import mesh

data = mesh.Mesh.from_file("runs/cascade_stl/post/blade_0.stl")

x = data.x.reshape(-1)
y = data.y.reshape(-1)
z = data.z.reshape(-1)

r = np.sqrt(y**2 + z**2)
t = np.arctan2(z, y)
rt = r * t

rmin = r.min()
rmax = r.max()
spf = (r - rmin) / (rmax - rmin)

spf_fit = np.array([0.3, 0.71])

tol = 5e-2
xrrt = []
for i, spfi in enumerate(spf_fit):
    idx = np.where(np.abs(spf - spfi) < tol)[0]
    xrrt.append(np.stack([x[idx], r[idx], rt[idx]]))

# print(xrrt[0][0].min())
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(*xrrt[1][(0, 2),],'kx')
# ax.axis('equal')
# plt.show()


turbigen.util.write_sections(xrrt, "sections.dat")
