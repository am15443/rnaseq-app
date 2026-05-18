"""
src/data_loader.py
──────────────────
Load a single combined TSV file containing all samples.

Only four columns are required regardless of what else is in the file:
    est_counts  |  tpm  |  gene_name  |  srr_id

All other columns are ignored.
"""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

REQUIRED_COLS = {"est_counts", "tpm", "gene_name", "srr_id"}


def load_combined_tsv(uploaded_file) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse a single combined TSV containing all samples.
    Only the four required columns are read; everything else is ignored.

    Returns
    -------
    gene_sample_df : MultiIndex DataFrame — rows=(gene_name, srr_id),
                     columns=[est_counts, tpm]
    srr_ids        : sorted list of all unique srr_ids found in the file
    """
    with st.spinner("Reading TSV file… this may take a moment for large files."):
        raw = uploaded_file.read().decode("utf-8")
        df  = pd.read_csv(io.StringIO(raw), sep="\t", low_memory=False)
        df.columns = df.columns.str.lower().str.strip()

    # ── Column validation ─────────────────────────────────────────────────────
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"TSV is missing required columns: {', '.join(sorted(missing))}.\n"
            f"Required: est_counts, tpm, gene_name, srr_id\n"
            f"Found: {df.columns.tolist()}"
        )

    # ── Keep only the 4 columns we need, drop everything else ────────────────
    df = df[["est_counts", "tpm", "gene_name", "srr_id"]].copy()

    # ── Coerce types ──────────────────────────────────────────────────────────
    df["est_counts"] = pd.to_numeric(df["est_counts"], errors="coerce").fillna(0.0)
    df["tpm"]        = pd.to_numeric(df["tpm"],        errors="coerce").fillna(0.0)
    df["gene_name"]  = df["gene_name"].astype(str).str.strip()
    df["srr_id"]     = df["srr_id"].astype(str).str.strip()

    # ── Drop unmapped genes ───────────────────────────────────────────────────
    bad = {"", "nan", "na", "none", "-", ".", "n/a"}
    df  = df[~df["gene_name"].str.lower().isin(bad)]

    # ── Aggregate transcripts → genes per sample (sum) ────────────────────────
    gene_sample_df = (
        df.groupby(["gene_name", "srr_id"], sort=False)
        .agg(est_counts=("est_counts", "sum"), tpm=("tpm", "sum"))
    )

    srr_ids = sorted(df["srr_id"].unique().tolist())
    return gene_sample_df, srr_ids


def validate_counts(
    gene_sample_df: pd.DataFrame,
    groups: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build aligned gene × sample matrices from the loaded combined data.

    Parameters
    ----------
    gene_sample_df : MultiIndex (gene_name, srr_id) with est_counts & tpm
    groups         : {group_name: [srr_id, ...]}

    Returns
    -------
    counts_df   : (n_genes × n_samples) rounded est_counts as int  — for DGE
    tpm_df      : (n_genes × n_samples) TPM values                 — for PCA/heatmap
    sample_meta : DataFrame indexed by srr_id with column 'group'
    """
    sample_order: List[str] = []
    meta_rows: List[dict]   = []

    for group_name, srr_ids in groups.items():
        for s in srr_ids:
            sample_order.append(s)
            meta_rows.append({"sample": s, "group": group_name})

    if not sample_order:
        raise ValueError("No samples are assigned to any group.")

    sample_meta = pd.DataFrame(meta_rows).set_index("sample")

    # Pivot to gene × sample matrices
    counts_df = (
        gene_sample_df["est_counts"]
        .unstack(level="srr_id")
        .fillna(0)
        .round()
        .astype(int)
    )
    tpm_df = (
        gene_sample_df["tpm"]
        .unstack(level="srr_id")
        .fillna(0.0)
    )

    # Filter to only selected samples in group order
    available = [s for s in sample_order if s in counts_df.columns]
    missing   = [s for s in sample_order if s not in counts_df.columns]
    if missing:
        st.warning(f"⚠️ These srr_ids were not found in the TSV: {missing}")

    counts_df   = counts_df[available]
    tpm_df      = tpm_df[available]

    # Drop all-zero genes
    keep      = counts_df.sum(axis=1) > 0
    counts_df = counts_df.loc[keep]
    tpm_df    = tpm_df.loc[tpm_df.index.isin(counts_df.index)]

    # Trim sample_meta to available samples
    sample_meta = sample_meta.loc[available]

    return counts_df, tpm_df, sample_meta
