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

fname = "runs/1109/config.yaml"
# fname = "scripts/dspace_test.yaml"
conf = turbigen.config2.TurbigenConfig(**turbigen.yaml.read_yaml(fname))

dspace = conf.design_space
dspace.setup()
quit()

# Plot LE recamber as a function of incidence

datum = dspace.configs[0]
f = lambda x: x.blades[0][0].camber[:, 0]
xg = dspace.meshgrid(datum, phi2=(0.4, 1.2)).squeeze()
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
