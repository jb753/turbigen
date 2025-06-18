import turbigen.yaml
from turbigen import util
import turbigen.config2
import turbigen.dspace
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = util.make_logger()
logger.setLevel(level=logging.DEBUG)

fname = "runs/0217/config.yaml"
conf = turbigen.config2.TurbigenConfig(**turbigen.yaml.read_yaml(fname))

dspace = conf.design_space
dspace.setup()


datum = dspace.configs[0]
# datum = turbigen.config2.TurbigenConfig(
#     **turbigen.yaml.read_yaml("runs/0243/config.yaml")
# )
# print(dspace.independent.get_independent_inverse(datum))
# print(datum.blades[0][0].camber)
# quit()

f = lambda x: x.mean_line_actual["Omega"]
xg = dspace.meshgrid(datum, psi=(0.8, 2.4), phi2=(0.4, 1.2)).squeeze()
yg = dspace.converged.evaluate(f, xg)
print(dspace.converged.rmse(f))

print(yg.min(), yg.max())

# xs = np.array([dspace.independent.get_independent(c)[0] for c in dspace.samples])


plt.figure(layout="constrained")
hc = plt.contour(xg[0], xg[1], yg)
plt.clabel(hc, inline=True)
xs = dspace.converged.x
plt.plot(*xs, "ro", markersize=10, label="Samples")
plt.xlabel("phi2")
plt.ylabel("psi")
plt.show()
quit()


# confs = dspace.sample(20)
# for i, conf in enumerate(confs):
# conf.save()

# conf.workdir = Path("test_dspace").absolute()
# conf.save()
