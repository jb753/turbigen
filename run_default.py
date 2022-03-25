from turbigen import design, hmesh, geometry
import json
import numpy as np
import matplotlib.pyplot as plt


with open("turbigen/default_params.json", "r") as f:
    params = json.load(f)

# Mean-line design using the non-dimensionals
stg = design.nondim_stage_from_Lam(**params["mean-line"])

params["3d"]["htr"] = 0.6

# Set geometry using dimensional bcond and 3D design parameter
bcond_and_3d_params = dict(params["bcond"], **params["3d"])
Dstg = design.scale_geometry(stg, **bcond_and_3d_params)

# Make a test section
spf = np.array([0.0, 0.5, 1.0])
chi = Dstg.free_vortex_vane(spf)
xy = geometry.radially_interpolate_section(
    spf, chi, (0.5,), A=geometry.prelim_A() * 0.2
)

# A = np.reshape(params["section"]["Aflat"],params["section"]["shape_A"])
# hmesh.stage_grid( Dstg, [1.,0.5,1.], A)

xyl = geometry.loop_section(xy, repeat_last=False).T
max_circle = geometry.largest_inscribed_circle(xyl)
print(max_circle)

# vor is

l = 1.
x = np.linspace(0.,1.)
ls = np.ones_like(x) * l
zs = np.zeros_like(x)
xf = np.flip(x)
xall = np.concatenate((x,ls,xf,zs))
yall = np.concatenate((zs,x,ls,xf))
xyl = np.stack((xall,yall)).T

fig, ax = plt.subplots()
# ax.plot(xy[0, 0, :], xy[0, 1, :], ".", ms=0.5)
# ax.plot(xy[1, 0, :], xy[1, 1, :], ".", ms=0.5)
ax.plot(xyl[:,0], xyl[:,1],'.')
# ax.scatter(vor[:, 0], vor[:, 1], s=10, c=min_dist, marker="o")
ax.axis("equal")
ax.set_xlim((-0.5, 1.5))
ax.set_ylim((-0.5, 2))

plt.savefig("sects.pdf")


quit()
