import matplotlib.pyplot as plt
import numpy as np
import glob
import json

# Helper function to read a json file and return a dict
def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data


# Get all metadata files and read the json
meta, params = [
    [load_json(fi) for fi in glob.glob("aljaz/**/%s.json" % si)]
    for si in ("meta", "params")
]

# Extract variables to plot
phi = np.array([pi["mean-line"]["phi"] for pi in params])
psi = np.array([pi["mean-line"]["psi"] for pi in params])
eta_isen = np.array([mi["eff_isen"] for mi in meta])
eta_isen_lost = np.array([mi["eff_isen_lost"] for mi in meta])
run_id = ["".join(mi["runid"]) for mi in meta]
err_eta = eta_isen - (1.0 - eta_isen_lost[:, 3])

print("EFFICIENCY ERRORS")
for ri, phii, psii, erri in zip(run_id, phi, psi, err_eta):
    print(
        "Run %s, phi/psi %.2f,%.2f, effy err %.2f percent"
        % (ri, phii, psii, erri * 100.0)
    )


# Make a smith chart
f, a = plt.subplots()
cont = a.tricontourf(phi, psi, eta_isen)
a.set_xlabel("Flow Coefficient")
a.set_ylabel("Stage Loading Coefficient")
cb = f.colorbar(cont)
cb.set_label("Isentropic Efficiency")
plt.savefig("smith.pdf")
