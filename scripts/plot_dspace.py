import turbigen.yaml
import turbigen.config2
import turbigen.dspace
from pathlib import Path
import numpy as np
from turbigen import util

fname = "runs/0052/config.yaml"
conf = turbigen.config2.TurbigenConfig(**turbigen.yaml.read_yaml(fname))
dspace = conf.design_space

#
# for c in dspace.samples:
#     c.mean_line.setup_mean_line(c.inlet.get_inlet())
#
# d = dspace.datum
# d.mean_line.setup_mean_line(d.inlet.get_inlet())
#
print(dspace.ndof, len(dspace.samples))
#
# # f = lambda x: x.mean_line.backward(x.mean_line.nominal)["phi2"]
f = lambda x: x.mean_line_actual["htr2"]
# print(f(dspace.datum))
# # print(dspace.interpolate(f, [d,d]))
xg = dspace.meshgrid(conf, psi=(0.8, 2.4), phi2=(0.4, 1.2)).squeeze()
yg = dspace.evaluate(f, xg)

import matplotlib.pyplot as plt

plt.figure(layout="constrained")
hc = plt.contour(*xg, yg)
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
