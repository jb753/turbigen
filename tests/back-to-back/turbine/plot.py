import matplotlib.pyplot as plt
import turbigen.yaml
import numpy as np
import warnings
import matplotlib.gridspec as gridspec

fig = plt.figure()
labs = ["TS3", "EMB"]
lev = np.arange(0.0, 0.45, 0.05)

# Manually define positions for the axes and colorbar
b = 0.3
a = 0.5 * b
d = 0.1 * b
c = d
tot = 2 * a + 2 * b + 4 * d + c
b /= tot
a /= tot
d /= tot
bot = 0.01
top = 0.85
a *= 0.9
ax1_pos = [a + d, bot, b, top]  # [left, bottom, width, height]
ax2_pos = [a + b + 2 * d, bot, b, top]
cbar_pos = [a + 2 * b + 3 * d, 0.1, c, 0.7]  # Narrower and shorter colorbar

# Add axes to the figure
ax1 = fig.add_axes(ax1_pos)
ax2 = fig.add_axes(ax2_pos)
cax = fig.add_axes(cbar_pos)  # Colorbar axis


axs = [ax1, ax2]
for lab, ax in zip(labs, axs):
    d = f"tests/back-to-back/turbine/{lab.lower()}"
    cdat = np.load(d + "/post/contour_Ys_m_2.05.npz")
    pitch = np.ptp(cdat["c1"])
    inv = turbigen.yaml.read_yaml(d + "/inverse.yaml")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = ax.tricontourf(
            cdat["c1"],
            cdat["c2"],
            cdat["v"],
            lev,
            triangles=cdat["triangles"],
            cmap="cubehelix",
            linestyles="none",
            extend="max",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_yticks(())
    ax.set_xticks(())
    ax.set_title(lab + "\n$\overline{Y_s}" + f'={inv["Yh"]:.3f}$')
    ax.axis("off")
    ax.set_xlim(pitch * (np.array([-0.25, 0.25]) - 0.1))

hc = fig.colorbar(cm, cax=cax, label="Entropy Loss, $Y_s$", shrink=0.8)
hc.set_ticks(lev[::2])
# plt.show()
plt.savefig("plots/ts3_emb_turbine.pdf")
