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

spf = np.array([0.0, 0.5, 1.0])
chi = Dstg.free_vortex_vane(spf)

hmesh.stage_grid( Dstg, [1.,0.5,1.], **params["section"])

Ap = geometry.prelim_A()
print(Ap)
Apr = np.stack((Ap * 0.1, Ap, Ap * 1.05))

xy = geometry.radially_interpolate_section(
    spf, chi, (0.1,0.5,0.9), A=Apr, spf_A=(0.0, 0.5, 1.0)
)

fig, ax = plt.subplots()
for xyi in xy:
    ax.plot(xyi[0, 0, :], xyi[0, 1, :], "-x")
    ax.plot(xyi[1, 0, :], xyi[1, 1, :], "-+")

ax.axis("equal")
plt.savefig("sects.pdf")


quit()
