import turbigen.yaml
import turbigen.config2
import turbigen.dspace
from pathlib import Path
import numpy as np

fname = "examples/axial_turbine.yaml"
conf = turbigen.config2.TurbigenConfig(**turbigen.yaml.read_yaml(fname))
conf.workdir = Path("test_dspace3").absolute()
conf.workdir.mkdir(exist_ok=True, parents=True)

independent = {
    "mean_line": {
        "phi2": (0.2, 0.8),
        "psi": (0.5, 2.0),
    },
    # "nblade": {0: {"Co": (0.4, 1.0)}}
}

design_space = {
    "datum": conf,
    "independent": independent,
    "order": 2,
}

dspace = turbigen.dspace.DesignSpace(**design_space)

for c in dspace.samples:
    c.mean_line.setup_mean_line(c.inlet.get_inlet())

d = dspace.datum
d.mean_line.setup_mean_line(d.inlet.get_inlet())

print(dspace.ndof, len(dspace.samples))

# f = lambda x: x.mean_line.backward(x.mean_line.nominal)["phi2"]
f = lambda x: x.mean_line.design_vars["psi"]
print(f(dspace.datum))
# print(dspace.interpolate(f, [d,d]))
xg = dspace.meshgrid( psi=(0.5, 1.5), phi2=(0.3, 0.7)).squeeze()
coeff = dspace.fit(f)
yg = dspace.evaluate(coeff, xg)

print(yg.min(), yg.max())

# xs = np.array([dspace.independent.get_independent(c)[0] for c in dspace.samples])

import matplotlib.pyplot as plt
plt.figure(layout='constrained')
hc = plt.contour(xg[0], xg[1], yg)
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
