import turbigen.yaml
import turbigen.config2
import turbigen.dspace
from pathlib import Path
import numpy as np
from turbigen import util
import matplotlib.pyplot as plt

fname = "runs/0000/config.yaml"
yml = turbigen.yaml.read_yaml(fname)
yml["design_space"]["frac_dof"] = 1.0
conf = turbigen.config2.TurbigenConfig(**yml)
dspace = conf.design_space

#
# for c in dspace.samples:
#     c.mean_line.setup_mean_line(c.inlet.get_inlet())
#
# d = dspace.datum
# d.mean_line.setup_mean_line(d.inlet.get_inlet())


def meshgrid(dspace, datum, N=11, **kwargs):
    # Get datum x
    xd = dspace.independent.get_independent(datum)

    # Assemble grid vectors
    xv = {}
    for ik, k in enumerate(dspace.independent.keys()):
        if k in kwargs:
            # Get the limits from the keyword argument
            if isinstance(kwargs[k], (list, tuple)):
                xv[k] = np.linspace(*kwargs[k], N)
            else:
                xv[k] = kwargs[k]
        else:
            xv[k] = np.array([xd[ik]])

    # Create a meshgrid of the coordinate vectors
    return {k: v for k, v in zip(xv.keys(), np.meshgrid(*xv.values(), indexing="ij"))}


print(dspace.ndof, dspace.nsample)

f = lambda x: x.mean_line_actual["eta_tt"]
eta_tt = dspace.evaluate_samples(f)
print(eta_tt.min(), eta_tt.max())
print(dspace.rmse(f))
f = lambda x: x.mean_line_actual["eta_ts"]
eta_ts = dspace.evaluate_samples(f)
print(eta_ts.min(), eta_ts.max())
print(dspace.rmse(f))
# quit()

# x = dspace.x
# nd, npts = x.shape
# fig, ax = plt.subplots(nd, nd, figsize=(5, 5))
# for i in range(nd):
#     for j in range(nd):
#         ax[i, j].set_xticks(dspace.xlim[(0, -1), j])
#         ax[i, j].set_xlim(dspace.xlim[:, j])
#         if i == j:
#             ax[i, j].hist(x[j, :], bins=10, color="grey")
#         else:
#             ax[i, j].scatter(x[j, :], x[i, :], s=10, alpha=0.5, ec="none")
#             ax[i, j].set_yticks(dspace.xlim[(0, -1), i])
#             ax[i, j].set_ylim(dspace.xlim[:, i])
#         if i == nd - 1:
#             ax[i, j].set_xlabel(dspace.independent.keys()[j], fontsize=10)
#         else:
#             ax[i, j].set_xticklabels([])
#         if j == 0:
#             ax[i, j].set_ylabel(dspace.independent.keys()[i], fontsize=10)
#         else:
#             ax[i, j].set_yticklabels([])
# # plt.tight_layout()
# plt.subplots_adjust(
#     wspace=0.05, hspace=0.05, top=0.98, bottom=0.08, left=0.08, right=0.98
# )
# plt.show()

#
# f = lambda x: x.mean_line.backward(x.mean_line.nominal)["phi2"]
f = lambda x: x.mean_line_actual["eta_tt"]
# print(f(dspace.datum))
# # print(dspace.interpolate(f, [d,d]))
# conf.nblade[0].Co = 0.7
# conf.nblade[1].Co = 0.7
# xg = meshgrid(dspace, conf, psi=(0.8, 2.4), phi2=(0.4, 1.2))
xg = meshgrid(dspace, conf, psi=(0.8, 2.4), phi2=(0.4, 1.2), fac_Ma3_rel=1.05)
print([f"{k} {v.mean()}\n" for k, v in xg.items()])
yg = dspace.evaluate(f, np.stack(list(xg.values())))


plt.figure(layout="constrained")
hc = plt.contour(xg["phi2"].squeeze(), xg["psi"].squeeze(), yg.squeeze())
plt.clabel(hc, inline=True)
plt.xlabel("phi2")
plt.ylabel("psi")
plt.show()
quit()


# confs = dspace.sample(20)
# for i, conf in enumerate(confs):
# conf.save()

# conf.workdir = Path("test_dspace").absolute()
# conf.save()
