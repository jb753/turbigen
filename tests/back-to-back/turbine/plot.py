import matplotlib.pyplot as plt
import turbigen.yaml
import numpy as np
import warnings

fig = plt.figure()
labs = ["TS3", "EMB"]
lev = np.arange(0.0, 0.45, 0.05)


def add_line_label(ax, line, label, angle=0, position=0.5, offset=0.05, **text_kwargs):
    """
    Adds a label to a matplotlib line with a connecting line of the same color.

    Parameters:
    - ax: The matplotlib Axes object containing the line.
    - line: The Line2D object to label.
    - label: The label text.
    - angle: Angle in degrees for the connecting line (0 degrees is horizontal).
    - position: Normalized position along the data (0=start, 1=end).
    - offset: Distance (in axis coordinates) from the data point for the label.
    - text_kwargs: Additional keyword arguments for the text (e.g., fontsize, fontweight).
    """
    margin = 0.04
    x_data, y_data = line.get_xdata(), line.get_ydata()

    # Ensure position is within [0,1]
    position = np.clip(position, 0, 1)

    # Compute index along the line based on position
    index = int(position * (len(x_data) - 1))
    x_point, y_point = x_data[index], y_data[index]

    # Compute offset based on angle
    radians = np.deg2rad(angle)
    dx = offset * np.cos(radians) * (ax.get_xlim()[1] - ax.get_xlim()[0])
    dy = offset * np.sin(radians) * (ax.get_ylim()[1] - ax.get_ylim()[0])
    dx_margin = margin * np.cos(radians) * (ax.get_xlim()[1] - ax.get_xlim()[0])
    dy_margin = margin * np.sin(radians) * (ax.get_ylim()[1] - ax.get_ylim()[0])

    x_label = x_point + dx
    y_label = y_point + dy

    # Draw connecting line
    ax.plot(
        [x_point, x_label - dx_margin],
        [y_point, y_label - dy_margin],
        color=line.get_color(),
        linewidth=1,
        linestyle="-",
        marker="",
    )

    # Add label
    ax.text(
        x_label,
        y_label,
        label,
        color=line.get_color(),
        ha="center",
        va="center",
        **text_kwargs,
    )


# Relative cost is about x5 lower with EMB. Does not stress the GPU much on  a coarse grid

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
    ax.set_title(lab + "\n$\overline{Y_s}" + f"={inv['Yh']:.3f}$")
    ax.axis("off")
    ax.set_xlim(pitch * (np.array([-0.25, 0.25]) - 0.1))

hc = fig.colorbar(cm, cax=cax, label="Entropy Loss, $Y_s$", shrink=0.8)
hc.set_ticks(lev[::2])
# plt.show()
plt.savefig("plots/ts3_emb_turbine_Ys.pdf")

# Now static pressure distributions
key = "row_0_spf_0.5_blade_0"
dat_Cp = [
    np.load(f"tests/back-to-back/turbine/{solver}/post/pressure_distributions_raw.npz")[
        key
    ]
    for solver in ["ts3", "emb"]
]
fig, ax = plt.subplots(layout="constrained")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-1.4, -0.0)
ax.set_xlabel(r"Normalised Surface Distance, $\zeta/\zeta_\mathrm{TE}$")
ax.set_ylabel(r"Static Pressure, $C_p$")
lines = []

for d in dat_Cp:
    zeta_stag, Cp = d
    zeta_max = zeta_stag.max()
    zeta_min = np.abs(zeta_stag.min())
    zeta_norm = zeta_stag.copy()
    zeta_norm[zeta_norm < 0.0] /= zeta_min
    zeta_norm[zeta_norm > 0.0] /= zeta_max
    lines.append(ax.plot(np.abs(zeta_norm), Cp, marker=""))

# Add labels
poss = (0.86, 0.908)
angs = (235, 235)
for line, label, ang, pos in zip(lines, reversed(labs), angs, poss):
    add_line_label(ax, line[0], label, angle=ang, position=pos, offset=0.08)

plt.savefig("plots/ts3_emb_turbine_Cp.pdf")
