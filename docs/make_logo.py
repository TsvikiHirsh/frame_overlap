"""
The frame_overlap logo (docs/logo.png), a gentle sibling of nbragg's: four
pulse spectra with a Bragg edge each, overlapping in one period (pastel
colours), the counts the detector records (their sum, dark line), the pulse
times (triangles) and the name, on a white round emblem.

    python docs/make_logo.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
FILLS = ["#f7a8c4", "#9adcde", "#c9b8ff", "#ffd09a"]            # nbragg's palette, softened
LINES = ["#e8799f", "#56b9bd", "#9f87ee", "#f0aa62"]
INK = "#46505e"                                                 # the sum, the pulse times and the name
STARTS = [-0.80, -0.38, 0.02, 0.40]                             # pulse times (unequal, as optimised ones are)
FONT = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"


def spectrum(x, x0):
    """A thermal spectrum after a pulse at x0, with a Bragg edge (the transmission jumps up)."""
    u = np.clip(x - x0, 0, None) / 0.09
    flux = u ** 3 * np.exp(-u) / 1.34                            # peak 1 at u = 3
    edge = 0.35 + 0.65 / (1 + np.exp(-(x - x0 - 0.5) / 0.008))
    return 1.1 * flux * edge


x = np.linspace(-1.05, 1.05, 2000)
parts = [spectrum(x, s) + spectrum(x + 2.0, s) for s in STARTS]   # the slow tail wraps around the period
fig = plt.figure(figsize=(4, 4), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-1.08, 1.08)
ax.set_ylim(-1.08, 1.08)
ax.set_aspect("equal")
ax.axis("off")
disc = Circle((0, 0), 1.0, transform=ax.transData)
ax.add_patch(Circle((0, 0), 1.0, color="white", zorder=0))
base = -0.56
for part, fill, line in zip(parts, FILLS, LINES):
    f = ax.fill_between(x, base, base + part, color=fill, alpha=0.75, lw=0, zorder=2)
    f.set_clip_path(disc)
    ln, = ax.plot(x, base + part, color=line, lw=1.1, zorder=3)
    ln.set_clip_path(disc)
total, = ax.plot(x, base + sum(parts), color=INK, lw=2.0, zorder=4, solid_capstyle="round")
total.set_clip_path(disc)
for s in STARTS:
    ax.plot(s + 0.02, 0.33, marker="v", color=INK, ms=6.5, zorder=5, clip_on=False)
ax.plot([-0.95, 0.95], [base, base], color="#c3c8d0", lw=1.0, zorder=4).__getitem__(0).set_clip_path(disc)
font = font_manager.FontProperties(fname=FONT) if os.path.exists(FONT) else font_manager.FontProperties(
    family="serif")
ax.text(0, 0.57, "frame_overlap", color=INK, fontproperties=font, fontsize=28, ha="center", va="center", zorder=6)
ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#cfd4dc", lw=2.5, zorder=7))
fig.savefig(os.path.join(HERE, "logo.png"), transparent=True)
print("written", os.path.join(HERE, "logo.png"))
