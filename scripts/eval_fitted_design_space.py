"""Read in a previously fitted design space and evaluate it."""

import numpy as np
import turbigen.polynomial
import matplotlib.pyplot as plt

# Load polynomial fit coefficients
FIT_JSON = "scripts/fit_eta_tt.json"
dspace = turbigen.polynomial.FittedDesignSpace(FIT_JSON)

# Define independent vars
DH_rotor = 1.0
Alpha2_rel = -60.0
Ma1_rel = np.linspace(0.3, 0.9)
phi1 = 0.6
PR_tt = 2.0
htr1 = 0.5
Co1 = 0.6
tip = 0.01

# Stack in the correct order
x = np.stack(
    np.broadcast_arrays(
        DH_rotor,
        Alpha2_rel,
        Ma1_rel,
        phi1,
        PR_tt,
        htr1,
        Co1,
        tip,
    ),
)

# Evaluate fit
eta_tt = dspace(x)

# Plot trend with Mach
fig, ax = plt.subplots()
ax.plot(Ma1_rel, eta_tt)
ax.set_xlabel("Inlet Relative Mach, $\mathit{M\kern-0.25ema}_1^\mathrm{rel}$")
ax.set_ylabel("Total-to-total Efficiency, $\eta_\mathrm{tt}$")
plt.tight_layout()
plt.show()
