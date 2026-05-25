"""
src/heatmap.py
──────────────
Interactive heatmap built on log2(TPM + 1) Z-scores.

Layout:
  Y axis  — genes (labels draggable via Plotly pan mode)
  X axis  — individual samples (one column per TSV)
  Below X — coloured group annotation bar matching PCA colours
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_heatmap(
    tpm_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    gene_list: List[str],
    group_colors: Dict[str, str],
    groups: Dict[str, List[str]],
) -> go.Figure:
    """
    Parameters
    ----------
    tpm_df      : genes × samples TPM matrix
    sample_meta : indexed by sample_key, column 'group'
    gene_list   : genes to display (case-insensitive matching against tpm_df.index)
    group_colors: {group_name: hex_color}  — must match PCA colours
    groups      : {group_name: [sample_key, ...]}  — used for column ordering

    Returns
    -------
    Plotly Figure (heatmap rows + thin colour annotation bar)
    """

    # ── Case-insensitive gene matching ────────────────────────────────────────
    idx_upper = {g.upper(): g for g in tpm_df.index}
    matched   = [idx_upper[g.upper()] for g in gene_list if g.upper() in idx_upper]
    missing   = [g for g in gene_list if g.upper() not in idx_upper]

    if not matched:
        raise ValueError(
            f"None of the requested genes were found in the TPM matrix. "
            f"Sample of missing: {missing[:10]}"
        )

    # ── Order samples group-by-group ──────────────────────────────────────────
    ordered_samples: List[str] = []
    for gname, samples in groups.items():
        for s in samples:
            if s in sample_meta.index:
                ordered_samples.append(s)

    sub = tpm_df.loc[matched, ordered_samples].astype(float)

    # ── log2(TPM + 1) then Z-score per gene ──────────────────────────────────
    log_tpm  = np.log2(sub + 1)
    row_mean = log_tpm.mean(axis=1)
    row_std  = log_tpm.std(axis=1).replace(0, 1)
    z        = log_tpm.subtract(row_mean, axis=0).divide(row_std, axis=0)

    n_genes   = len(matched)
    n_samples = len(ordered_samples)

    # ── Subplot: heatmap (row 1) + annotation bar (row 2) ────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.93, 0.07],
        vertical_spacing=0.01,
        shared_xaxes=True,
    )

    # ── Hover text ────────────────────────────────────────────────────────────
    hover = []
    for gene in matched:
        row = []
        for samp in ordered_samples:
            grp   = sample_meta.loc[samp, "group"] if samp in sample_meta.index else "?"
            raw   = round(float(sub.loc[gene, samp]), 2)
            zval  = round(float(z.loc[gene, samp]), 3)
            row.append(
                f"<b>{gene}</b><br>"
                f"Sample: {samp}<br>"
                f"Group: {grp}<br>"
                f"TPM: {raw}<br>"
                f"Z-score: {zval}"
            )
        hover.append(row)

    # ── Heatmap trace ─────────────────────────────────────────────────────────
    fig.add_trace(
        go.Heatmap(
            z=z.values,
            x=ordered_samples,
            y=matched,
            text=hover,
            hoverinfo="text",
            colorscale=[[0, "#d63027"], [0.5, "#f7f7f7"], [1, "#1b984e"]],
            zmid=0,
            colorbar=dict(
                title=dict(text="Z-score<br>(log₂ TPM)", side="right"),
                thickness=12,
                len=0.9,
                y=0.5,
                yanchor="middle",
            ),
        ),
        row=1, col=1,
    )

    # ── Group annotation bar — one rectangle per group ───────────────────────
    # Build a mapping of sample position index so we can calculate x spans
    sample_positions = {samp: i for i, samp in enumerate(ordered_samples)}

    # Group → list of sample indices
    group_sample_indices: Dict[str, list] = {}
    for samp in ordered_samples:
        grp = sample_meta.loc[samp, "group"] if samp in sample_meta.index else "?"
        group_sample_indices.setdefault(grp, []).append(sample_positions[samp])

    # Add one invisible Bar trace per group just for the legend
    for grp, indices in group_sample_indices.items():
        color = group_colors.get(grp, "#aaaaaa")
        fig.add_trace(
            go.Bar(
                x=[ordered_samples[indices[0]]],
                y=[0],                      # height 0 — invisible, legend only
                marker_color=color,
                name=grp,
                showlegend=True,
                legendgroup=grp,
                hoverinfo="skip",
            ),
            row=2, col=1,
        )

    # Draw one filled rectangle per group as a shape on the annotation subplot
    # x values in categorical axes are 0-indexed positions
    for grp, indices in group_sample_indices.items():
        color  = group_colors.get(grp, "#aaaaaa")
        x_start = min(indices) - 0.5
        x_end   = max(indices) + 0.5

        fig.add_shape(
            type="rect",
            xref="x2", yref="y2",          # subplot 2 axes
            x0=x_start, x1=x_end,
            y0=0, y1=1,
            fillcolor=color,
            line=dict(color="white", width=1),
            layer="below",
        )

        # Group name centered inside the rectangle
        fig.add_annotation(
            xref="x2", yref="y2",
            x=(x_start + x_end) / 2,
            y=0.5,
            text=f"<b>{grp}</b>",
            showarrow=False,
            font=dict(size=11, color="white", family="DM Sans, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    gene_px    = max(16, min(34, int(650 / max(n_genes, 1))))
    fig_height = max(400, gene_px * n_genes + 100)

    fig.update_layout(
        height=fig_height,
       width=max(400, n_samples * 60 + 300),
        barmode="stack",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", size=11, color="#111111"),
        legend=dict(
            title=dict(
                text="Group",
                font=dict(color="#111111", size=12),
                side="top",
            ),
            orientation="v",
            x=1.08, y=0.97,
            font=dict(color="#111111"),
        ),
        margin=dict(l=130, r=120, t=40, b=20),
        dragmode="pan",
    )

    # Heatmap axes
    fig.update_xaxes(
        tickangle=-40, tickfont=dict(size=10, color="#111111"),
        showgrid=False, row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Gene",
        tickfont=dict(size=10, color="#111111"),
        title_font=dict(color="#111111"),
        autorange="reversed",
        fixedrange=False,
        showgrid=False, row=1, col=1,
    )

    # Annotation bar axes — hide everything, group labels drawn as annotations
    fig.update_xaxes(
        showticklabels=False, showgrid=False,
        range=[-0.5, n_samples - 0.5],
        row=2, col=1,
    )
    fig.update_yaxes(
        showticklabels=False, showgrid=False,
        range=[0, 1], fixedrange=True, row=2, col=1,
    )

    # ── Warning for unmatched genes ───────────────────────────────────────────
    if missing:
        shown = ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "")
        fig.add_annotation(
            text=f"⚠️ {len(missing)} gene(s) not found in TPM matrix: {shown}",
            xref="paper", yref="paper",
            x=0, y=-0.03,
            showarrow=False,
            font=dict(size=10, color="#cc0033"),
            xanchor="left",
        )

    return fig
