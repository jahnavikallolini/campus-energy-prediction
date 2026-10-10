"""Shared plotting style: restrained, colour-blind-safe, publication-oriented."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C

# Okabe-Ito palette, one fixed colour per building across all figures
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#444444"]
BUILDING_COLORS = {b: _PALETTE[i] for i, b in enumerate(C.BUILDINGS)}
LABELS = {b: v["label"] for b, v in C.BUILDINGS.items()}


def apply():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    })


def save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path
