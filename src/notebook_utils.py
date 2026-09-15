"""Shared notebook display/formatting helpers.

Import these instead of hand-rolling equivalents (raw `print()` for
commentary, ad hoc `imshow`/`sns.heatmap` calls) so every notebook in this
project renders markdown, tables of contents, and heatmaps the same way.
Enforced via .claude/skills/notebook-practices/SKILL.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display


def md(text: str) -> None:
    """
    Use this whenever a code cell needs to render formatted Markdown (headers,
    bold, bullet lists, links) as its output, instead of printing plain text.
    Typical case: stating an interim finding or a short section takeaway
    inline with the code that produced it.

    Do not use this to display a DataFrame -- return/display the DataFrame
    itself so pandas' own table renderer is used.
    """
    display(Markdown(text))


def table_of_contents(notebook_path: str | Path) -> None:
    """
    Displays the table of contents of a given notebook path -- call this
    once, early (right after the title cell), passing the notebook's own
    path so it reads its current on-disk cells. Built off `##`/`###`/...
    markdown headers, nested by header level.
    """
    toc = []
    with open(notebook_path, "r") as f:
        cells = json.load(f)["cells"]
    for cell in cells:
        if cell["cell_type"] == "markdown":
            for line in cell["source"]:
                match = re.search(r"^#+ \w+", line)
                if match:
                    level = len(line) - len(line.lstrip("#"))
                    link = line.strip(" #\n").replace(" ", "-")
                    toc.append(
                        2 * (level - 2) * " "
                        + (("- [") if level > 1 else "\n[")
                        + line.strip(" #\n")
                        + f"](#{link})"
                    )
    md("### Table of contents")
    md("\n".join(toc))


def plot_heatmap_table(
    table: pd.DataFrame,
    figsize: tuple[float, float] = (14, 8),
    *,
    cmap: str = "Greens",
    vmin: float = 0,
    vmax: float = 100,
    annotate_threshold: float = 50,
    value_format: str = "{:.1f}%",
    title: str | None = None,
    xlabel: str = "Class Name",
    ylabel: str = "Ontology",
    cbar_label: str = "Percentage (%)",
) -> None:
    """
    Use this whenever plotting any matrix-shaped table of numbers as a heatmap
    (per-category scores, coverage tables, confusion-style grids) so every
    heatmap across every notebook shares the same coloring and layout. Set
    `vmax` to whatever value should read as "full" on the color scale -- it
    does not have to be 100 (e.g. use the max count for a count table).

    Cell values are annotated in white above `annotate_threshold` (readable
    against the dark end of `cmap`) and black below it; zero/NaN cells are
    left unannotated.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(table.values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=9)

    for r in range(len(table.index)):
        for c in range(len(table.columns)):
            val = table.iloc[r, c]
            if pd.notna(val) and val > 0:
                color = "white" if val >= annotate_threshold else "black"
                ax.text(c, r, value_format.format(val), ha="center", va="center", fontsize=8, color=color)

    if title is not None:
        ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()
