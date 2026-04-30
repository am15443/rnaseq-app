"""
src/data_loader.py
──────────────────
Load and validate TSV files with the schema:

    target_id | length | eff_length | est_counts | tpm | gene_name | srr_id

Design decisions:
  - est_counts and tpm are summed per gene_name across isoforms/transcripts.
  - The filename stem is used as the sample key (e.g. "SRR123456" from SRR123456.tsv).
  - DGE is run on rounded est_counts (raw counts).
  - Heatmap visualisation uses tpm.
"""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

REQUIRED_COLS = {"target_id", "length", "eff_length", "est_counts", "tpm", "gene_name", "srr_id"}


def load_tsv_files(uploaded_files) -> Dict[str, pd.DataFrame]:
    """
    Parse a list of Streamlit UploadedFile objects.

    Returns
    -------
    { sample_key : DataFrame indexed by gene_name with columns [est_counts, tpm] }
    """
    result: Dict[str, pd.DataFrame] = {}

    for uf in uploaded_files:
        key = uf.name.rsplit(".", 1)[0]          # strip extension → sample key
        try:
            raw = uf.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(raw), sep="\t")
            df.columns = df.columns.str.lower().str.strip()

            # ── Column validation ─────────────────────────────────────────────
            missing = REQUIRED_COLS - set(df.columns)
            if missing:
                st.warning(
                    f"⚠️ '{uf.name}' is missing required columns: "
                    f"{', '.join(sorted(missing))}. Skipping."
                )
                continue

            # ── Coerce numerics ───────────────────────────────────────────────
            df["est_counts"] = pd.to_numeric(df["est_counts"], errors="coerce").fillna(0.0)
            df["tpm"]        = pd.to_numeric(df["tpm"],        errors="coerce").fillna(0.0)
            df["gene_name"]  = df["gene_name"].astype(str).str.strip()

            # ── Aggregate transcripts → genes (sum) ───────────────────────────
            gene_df = (
                df.groupby("gene_name", sort=False)
                .agg(est_counts=("est_counts", "sum"), tpm=("tpm", "sum"))
            )

            # Remove placeholder / unmapped entries
            bad = {"", "nan", "na", "none", "-", ".", "n/a"}
            gene_df = gene_df[~gene_df.index.str.lower().isin(bad)]

            result[key] = gene_df

        except Exception as e:
            st.warning(f"⚠️ Could not parse '{uf.name}': {e}")

    return result


def validate_counts(
    uploaded_files: Dict[str, pd.DataFrame],
    groups: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build aligned gene × sample matrices from loaded data.

    Returns
    -------
    counts_df   : (n_genes × n_samples) rounded est_counts as int  — for DGE
    tpm_df      : (n_genes × n_samples) TPM values                 — for heatmap
    sample_meta : DataFrame indexed by sample_key with column 'group'
    """
    sample_order: List[str] = []
    meta_rows: List[dict] = []

    for group_name, samples in groups.items():
        for s in samples:
            if s in uploaded_files:
                sample_order.append(s)
                meta_rows.append({"sample": s, "group": group_name})

    if not sample_order:
        raise ValueError("No samples are assigned to any group.")

    sample_meta = pd.DataFrame(meta_rows).set_index("sample")

    counts_frames = {s: uploaded_files[s]["est_counts"].rename(s) for s in sample_order}
    tpm_frames    = {s: uploaded_files[s]["tpm"].rename(s)        for s in sample_order}

    counts_df = pd.concat(counts_frames, axis=1).fillna(0).round().astype(int)
    tpm_df    = pd.concat(tpm_frames,    axis=1).fillna(0.0)

    # Drop all-zero genes
    keep = counts_df.sum(axis=1) > 0
    counts_df = counts_df.loc[keep]
    tpm_df    = tpm_df.loc[tpm_df.index.isin(counts_df.index)]

    return counts_df, tpm_df, sample_meta
