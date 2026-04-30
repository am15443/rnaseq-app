"""
src/pca.py
──────────
PCA on log2(TPM + 1) data — one point per sample.
Interactive 2D and 3D Plotly figures with customisable group colours.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_pca(
    tpm_df: pd.DataFrame,
    n_components: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Parameters
    ----------
    tpm_df : genes × samples TPM matrix

    Returns
    -------
    coords_2d : samples × [PC1, PC2]
    coords_3d : samples × [PC1, PC2, PC3]
    explained : explained variance ratios array
    """
    # samples × genes, log2 transform
    X = np.log2(tpm_df.T.astype(float) + 1)

    # Remove zero-variance genes
    var = X.var(axis=0)
    X   = X.loc[:, var > 0]

    X_scaled = StandardScaler().fit_transform(X)

    n_comp = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca    = PCA(n_components=n_comp)
    coords = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    samples  = tpm_df.columns.tolist()
    coord_df = pd.DataFrame(
        coords,
        index=samples,
        columns=[f"PC{i+1}" for i in range(n_comp)],
    )

    return coord_df[["PC1", "PC2"]], coord_df, explained


def plot_pca_2d(
    coords_2d: pd.DataFrame,
    sample_meta: pd.DataFrame,
    explained: np.ndarray,
    group_colors: Dict[str, str],
    width: int = 800,
    height: int = 550,
) -> go.Figure:
    fig    = go.Figure()
    groups = sample_meta["group"].unique()

    for group in groups:
        samples = sample_meta.index[sample_meta["group"] == group].tolist()
        sub     = coords_2d.loc[[s for s in samples if s in coords_2d.index]]
        color   = group_colors.get(group, "#4361ee")

        fig.add_trace(go.Scatter(
            x=sub["PC1"], y=sub["PC2"],
            mode="markers+text",
            name=group,
            text=sub.index.tolist(),
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=14, color=color,
                        line=dict(width=1.5, color="white"), opacity=0.9),
        ))

    pct = [f"{v*100:.1f}%" for v in explained]
    fig.update_layout(
        xaxis_title=f"PC1 ({pct[0] if pct else ''})",
        yaxis_title=f"PC2 ({pct[1] if len(pct)>1 else ''})",
        legend_title="Group",
        width=width, height=height,
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", size=12),
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#ebebeb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb", zeroline=False)
    return fig


def plot_pca_3d(
    coords_3d: pd.DataFrame,
    sample_meta: pd.DataFrame,
    explained: np.ndarray,
    group_colors: Dict[str, str],
    width: int = 800,
    height: int = 550,
) -> go.Figure:
    fig     = go.Figure()
    has_pc3 = "PC3" in coords_3d.columns
    groups  = sample_meta["group"].unique()

    for group in groups:
        samples = sample_meta.index[sample_meta["group"] == group].tolist()
        sub     = coords_3d.loc[[s for s in samples if s in coords_3d.index]]
        color   = group_colors.get(group, "#4361ee")

        fig.add_trace(go.Scatter3d(
            x=sub["PC1"], y=sub["PC2"],
            z=sub["PC3"] if has_pc3 else [0]*len(sub),
            mode="markers+text",
            name=group,
            text=sub.index.tolist(),
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=8, color=color,
                        line=dict(width=1, color="white"), opacity=0.9),
        ))

    pct = [f"{v*100:.1f}%" for v in explained]
    fig.update_layout(
        scene=dict(
            xaxis_title=f"PC1 ({pct[0] if pct else ''})",
            yaxis_title=f"PC2 ({pct[1] if len(pct)>1 else ''})",
            zaxis_title=f"PC3 ({pct[2] if len(pct)>2 else ''})",
            bgcolor="#fafafa",
        ),
        legend_title="Group",
        width=width, height=height,
        font=dict(family="DM Sans, sans-serif", size=12),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
