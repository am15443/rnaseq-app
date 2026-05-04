"""
src/dge.py
──────────
Differential Gene Expression for all group pairs using pydeseq2.
Falls back to log-normalised Welch t-test + BH FDR if pydeseq2 unavailable.
"""

from __future__ import annotations

import itertools
import traceback
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from statsmodels.stats.multitest import multipletests

try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    _HAS_PYDESEQ2 = True
except ImportError:
    _HAS_PYDESEQ2 = False


def run_dge_all_pairs(
    counts_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Run DGE for every combination of two groups.

    Parameters
    ----------
    counts_df   : genes × samples, integer est_counts
                  columns = sample ids, index = gene names
    sample_meta : indexed by sample id, column 'group'

    Returns
    -------
    { (groupA, groupB) : results DataFrame }
    Results columns: baseMean, log2FoldChange, pvalue, padj
    Positive log2FoldChange = higher in groupA.
    """
    groups = sample_meta["group"].unique().tolist()
    pairs  = list(itertools.combinations(groups, 2))
    results: Dict[Tuple[str, str], pd.DataFrame] = {}

    for g1, g2 in pairs:
        try:
            if _HAS_PYDESEQ2:
                df = _run_pydeseq2(counts_df, sample_meta, g1, g2)
            else:
                df = _run_fallback(counts_df, sample_meta, g1, g2)
            results[(g1, g2)] = df
        except Exception as e:
            st.error(f"DGE failed for {g1} vs {g2}: {e}")
            st.code(traceback.format_exc())

    return results


def _run_pydeseq2(counts_df, sample_meta, g1, g2):
    # ── Subset to the two groups ──────────────────────────────────────────────
    mask     = sample_meta["group"].isin([g1, g2])
    sub_meta = sample_meta.loc[mask].copy()

    # sub_counts: samples × genes  (pydeseq2 expects samples as rows)
    sub_counts = counts_df.loc[:, sub_meta.index].T.astype(int)

    # ── Both indices must be identical strings ────────────────────────────────
    # Reset both to a clean shared integer-based string index
    shared_index = [f"s{i}" for i in range(len(sub_meta))]

    sub_counts = sub_counts.copy()
    sub_counts.index = shared_index

    meta_df = pd.DataFrame({
        "sample": shared_index,
        "group":  sub_meta["group"].values,
    }).set_index("sample")

    dds = DeseqDataSet(
        counts=sub_counts,
        metadata=meta_df,
        design_factors="group",
        ref_level=["group", g2],
        refit_cooks=True,
        quiet=True,
    )
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=["group", g1, g2], quiet=True)
    stat_res.summary()

    res = stat_res.results_df.copy()
    res.index.name = "gene"
    return res[["baseMean", "log2FoldChange", "pvalue", "padj"]].dropna(subset=["padj"])


def _run_fallback(counts_df, sample_meta, g1, g2):
    """
    Library-size normalised log2 t-test + Benjamini-Hochberg FDR.
    Used when pydeseq2 is not installed.
    """
    mask     = sample_meta["group"].isin([g1, g2])
    sub_meta = sample_meta.loc[mask]
    sub      = counts_df.loc[:, sub_meta.index].astype(float)

    lib_sizes = sub.sum(axis=0)
    scale     = lib_sizes / lib_sizes.median().clip(min=1)
    norm      = sub.divide(scale, axis=1)
    log_norm  = np.log2(norm + 1)

    s1 = sub_meta.index[sub_meta["group"] == g1].tolist()
    s2 = sub_meta.index[sub_meta["group"] == g2].tolist()

    lfcs, pvals, base_means = [], [], []

    for gene in log_norm.index:
        a = log_norm.loc[gene, s1].values
        b = log_norm.loc[gene, s2].values
        lfcs.append(float(np.mean(a) - np.mean(b)))
        base_means.append(float(norm.loc[gene].mean()))
        if len(a) > 1 and len(b) > 1:
            _, p = stats.ttest_ind(a, b, equal_var=False)
        else:
            p = np.nan
        pvals.append(p)

    pvals_arr = np.array(pvals)
    finite    = np.isfinite(pvals_arr)
    padj      = np.full_like(pvals_arr, np.nan)
    if finite.sum() > 0:
        _, padj_fin, _, _ = multipletests(pvals_arr[finite], method="fdr_bh")
        padj[finite] = padj_fin

    res = pd.DataFrame(
        {"baseMean": base_means, "log2FoldChange": lfcs,
         "pvalue": pvals, "padj": padj},
        index=log_norm.index,
    )
    res.index.name = "gene"
    return res.dropna(subset=["padj"])
