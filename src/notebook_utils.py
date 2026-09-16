"""Shared notebook display/formatting helpers.

Import these instead of hand-rolling equivalents (raw `print()` for
commentary, ad hoc `imshow`/`sns.heatmap` calls) so every notebook in this
project renders markdown, tables of contents, and heatmaps the same way.
Enforced via .claude/skills/notebook-practices/SKILL.md.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, Markdown, display


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


_NESTED_TABLE_CSS = """
<style>
.nested-table-outer { overflow-x: auto; }
.nested-table-outer table.dataframe { font-size: 12px; border-collapse: collapse; }
.nested-table-outer table.dataframe > thead > tr > th { text-align: left; background: #f0f0f0; position: sticky; top: 0; }
.nested-table-outer table.dataframe > tbody > tr > td { vertical-align: top; text-align: left; padding: 4px 8px; border: 1px solid #ddd; }
.nested-cell { max-height: 220px; overflow: auto; border: 1px solid #ccc; }
.nested-cell table { font-size: 11px; border-collapse: collapse; }
.nested-cell th, .nested-cell td { padding: 2px 5px; border: 1px solid #e0e0e0; }
.nested-empty { color: #999; font-style: italic; }
</style>
"""


_RAWHTML_MARKER = "\x01RAWHTML\x01"


class RawHTML(str):
    """Marks a string as already-safe, pre-built HTML (e.g. text with a substring highlighted
    in `<b>`) -- `_cell_to_html` passes it through unescaped instead of running it through
    `html.escape` like a normal string value. Caller is responsible for escaping any real data
    text before embedding it. Uses a content-marker PREFIX, not just subclassing, because pandas
    can silently downcast `str` subclasses back to plain `str` during DataFrame construction --
    the marker survives that, class identity doesn't (ported from the same
    `previous_work/notebooks/10_gemma_contrast_metadata_collation.ipynb` pattern as
    `display_with_nested_tables`).
    """

    def __new__(cls, content: str) -> Self:
        return super().__new__(cls, _RAWHTML_MARKER + content)


def _cell_to_html(value: object, max_colwidth: int | None = None) -> str:
    """Renders one cell for `display_with_nested_tables`: a nested DataFrame becomes its own
    embedded, scrollable HTML sub-table instead of pandas' opaque `<DataFrame>` repr; a `RawHTML`
    string is passed through unescaped; everything else is HTML-escaped, since the outer table is
    rendered with `escape=False` (required to let the nested `<table>` tags through) -- free text
    can genuinely contain '<'/'>'/'&' and must not be left raw.
    """
    if isinstance(value, str) and value.startswith(_RAWHTML_MARKER):
        return value[len(_RAWHTML_MARKER) :]
    if isinstance(value, pd.DataFrame):
        if len(value) == 0:
            return '<span class="nested-empty">(no rows)</span>'
        # recurse through _cell_to_html for the nested frame's own cells too, so a RawHTML value
        # (e.g. a bolded match) inside a nested table renders correctly instead of being escaped
        # by pandas' own to_html -- otherwise this only worked one level deep.
        inner_view = value.apply(lambda col: col.map(lambda v: _cell_to_html(v, max_colwidth)))
        inner_html = inner_view.to_html(index=False, border=0, escape=False)
        inner_html = re.sub(r">\s+<", "><", inner_html.strip())
        return f'<div class="nested-cell">{inner_html}</div>'
    if isinstance(value, list):
        if len(value) == 0:
            return '<span class="nested-empty">(none)</span>'
        text = ", ".join(str(v).strip() for v in value)
        if max_colwidth is not None and len(text) > max_colwidth:
            text = text[:max_colwidth] + "…"
        return html.escape(text)
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if max_colwidth is not None and len(text) > max_colwidth:
        text = text[:max_colwidth] + "…"
    return html.escape(text)


def _nested_table_html(df: pd.DataFrame, max_colwidth: int | None = None) -> str:
    html_view = df.apply(lambda col: col.map(lambda v: _cell_to_html(v, max_colwidth)))
    table_html = html_view.to_html(escape=False, index=False)
    return _NESTED_TABLE_CSS + f'<div class="nested-table-outer">{table_html}</div>'


def display_with_nested_tables(df: pd.DataFrame, max_colwidth: int | None = None) -> None:
    """Displays a DataFrame as HTML with every DataFrame-valued cell rendered as a real,
    scrollable embedded sub-table instead of pandas' cut-off `<DataFrame>` text repr. Use this
    whenever a row naturally embeds nested per-record data (e.g. a BioSample summary row whose
    `attributes` column holds its own small attribute-name/value table) -- adapted from the
    `display_with_nested_tables` helper in `previous_work/notebooks/10_gemma_contrast_metadata_collation.ipynb`
    (ported and simplified here, not imported, per this project's self-contained-code rule).
    """
    display(HTML(_nested_table_html(df, max_colwidth)))


def json_leaf_values(obj: object) -> list[str]:
    """Every scalar leaf value in a nested JSON-like structure (dict/list of
    dicts/lists/scalars), stringified, `None` excluded. Used by
    `table_covers_json` to check a nested-table view hasn't silently dropped
    a value from the JSON it was built from.
    """
    if isinstance(obj, dict):
        return [v for value in obj.values() for v in json_leaf_values(value)]
    if isinstance(obj, list):
        return [v for item in obj for v in json_leaf_values(item)]
    if obj is None:
        return []
    return [str(obj)]


def table_covers_json(raw: object, df: pd.DataFrame, max_colwidth: int | None = None) -> bool:
    """True iff every scalar leaf value in `raw` appears somewhere in the
    rendered HTML of `df` (as built by `display_with_nested_tables`) -- a
    live completeness check, not an assumption, for whenever a table is
    meant to be trusted as a full substitute for its source JSON.
    """
    table_html = _nested_table_html(df, max_colwidth)
    return all(html.escape(str(v)) in table_html for v in json_leaf_values(raw))
