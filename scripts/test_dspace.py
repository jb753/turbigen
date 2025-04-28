import turbigen.yaml
import turbigen.config2
import turbigen.dspace
from pathlib import Path

fname = "examples/axial_turbine.yaml"
conf = turbigen.config2.TurbigenConfig(**turbigen.yaml.read_yaml(fname))
conf.workdir = Path("test_dspace2").absolute()

independent = {
    "mean_line": {
        "phi2": (0.2, 0.8),
        "psi": (0.5, 2.0),
        "Ma2": (0.5, 0.9),
    },
    "nblade": {0: {"Co": (0.4, 1.0)}, 1: {"Co": (0.3, 0.8)}},
}

design_space = {
    "datum": conf,
    "independent": independent,
}

dspace = turbigen.dspace.DesignSpace(**design_space)

print(dspace.datum.workdir)
print('Bounds:')
print(dspace.get_bounds())
# confs = dspace.sample(10)

# for i, conf in enumerate(confs):
    # conf.save()

# conf.workdir = Path("test_dspace").absolute()
# conf.save()
