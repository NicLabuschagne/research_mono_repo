"""
theme.py — shared research chart theme for matplotlib and seaborn.

Usage
-----
    from theme import set_theme, diverging_cmap, PALETTE

    set_theme()              # light (default, best for printing)
    set_theme(mode="dark")   # dark, for screens

    # heatmaps centred on zero (blue = positive, orange = negative)
    plt.imshow(data, cmap=diverging_cmap(), vmin=-limit, vmax=limit)

Design rules carried over from the tear sheet
---------------------------------------------
- Categorical colours are assigned in a fixed order and follow the series, never its rank.
- Text uses the ink colours, never a series colour.
- Diverging maps: two hues with a neutral grey at zero; keep the same scale across maps.
- Grid on the y axis only, top and right spines off.
"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ---------------- Light theme (default) ----------------
LIGHT = {
    "surface": "#fcfcfb",      # chart and figure background
    "ink": "#0b0b0b",          # titles and values
    "ink_secondary": "#52514e",  # axis labels, tick labels, notes
    "muted": "#898781",        # axis lines, zero lines, reference lines
    "grid": "#e6e5e1",         # gridlines
    "neutral": "#f0efec",      # diverging midpoint, shaded bands
    "palette": [
        "#2a78d6",  # 1 blue
        "#eb6834",  # 2 orange
        "#1baf7a",  # 3 aqua
        "#eda100",  # 4 yellow
        "#e87ba4",  # 5 magenta
        "#008300",  # 6 green
        "#4a3aa7",  # 7 violet
        "#e34948",  # 8 red
    ],
}

# ---------------- Dark theme (screen use) ----------------
DARK = {
    "surface": "#1a1a19",
    "ink": "#ffffff",
    "ink_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#383835",
    "neutral": "#383835",
    "palette": [
        "#3987e5",  # 1 blue
        "#d95926",  # 2 orange
        "#199e70",  # 3 aqua
        "#c98500",  # 4 yellow
        "#d55181",  # 5 magenta
        "#008300",  # 6 green
        "#9085e9",  # 7 violet
        "#e66767",  # 8 red
    ],
}

# Module-level names for the light theme, so `from theme import PALETTE` works
BG = LIGHT["surface"]
FG = LIGHT["ink_secondary"]
EDGE = LIGHT["muted"]
GRID = LIGHT["grid"]
PALETTE: List[str] = LIGHT["palette"]


def _rc_params(colors: Dict[str, Any]) -> Dict[str, Any]:
    """Build matplotlib rcParams for one colour set."""
    return {
        "figure.facecolor": colors["surface"],
        "axes.facecolor": colors["surface"],
        "savefig.facecolor": colors["surface"],
        "axes.edgecolor": colors["muted"],
        "axes.labelcolor": colors["ink_secondary"],
        "axes.titlecolor": colors["ink"],
        "axes.titlelocation": "left",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": colors["ink"],
        "xtick.color": colors["ink_secondary"],
        "ytick.color": colors["ink_secondary"],
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": colors["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "font.size": 9.5,
        # Treat "$" as text, so dollar amounts in labels are not read as maths
        "text.parse_math": False,
    }


RC_PARAMS: Dict[str, Any] = _rc_params(LIGHT)


def diverging_cmap(mode: str = "light") -> LinearSegmentedColormap:
    """Orange (negative) -> neutral grey (zero) -> blue (positive)."""
    colors = LIGHT if mode == "light" else DARK
    return LinearSegmentedColormap.from_list(
        "research_diverging", [colors["palette"][1], colors["neutral"], colors["palette"][0]]
    )


def set_theme(mode: str = "light") -> Dict[str, Any]:
    """
    Apply the shared research theme to matplotlib and seaborn.

    Parameters
    ----------
    mode : "light" (default, for printing) or "dark" (for screens)

    Returns
    -------
    dict with keys: rc_params (dict), palette (list), colors (dict), diverging_cmap (colormap)
    """
    colors = LIGHT if mode == "light" else DARK
    rc_params = _rc_params(colors)

    sns.set_theme(style="white", rc=rc_params)
    plt.rcParams.update(rc_params)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors["palette"])
    sns.set_palette(colors["palette"])

    return {
        "rc_params": rc_params,
        "palette": colors["palette"],
        "colors": colors,
        "diverging_cmap": diverging_cmap(mode),
    }