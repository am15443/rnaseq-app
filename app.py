"""
app.py
──────
RNAseq Analysis Suite — Streamlit entry point.

Expected input TSV columns (one file per sample):
    target_id | length | eff_length | est_counts | tpm | gene_name | srr_id
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="RNAseq Analysis Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-title {
    font-family: 'DM Mono', monospace;
    font-size: 2.1rem;
    font-weight: 500;
    color: #1a1a2e;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}
.subtitle {
    color: #777;
    font-size: 0.95rem;
    margin-top: 0.1rem;
    margin-bottom: 0;
}
.schema-box {
    background: #f3f4f8;
    border-left: 3px solid #4361ee;
    border-radius: 6px;
    padding: 0.55rem 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #333;
    margin: 0.6rem 0 1rem 0;
    line-height: 1.7;
}
.group-card {
    background: #f8f9ff;
    border: 1.5px solid #e0e4f7;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ── Src imports ───────────────────────────────────────────────────────────────
from src.data_loader import load_tsv_files, validate_counts
from src.dge        import run_dge_all_pairs
from src.pca        import compute_pca, plot_pca_2d, plot_pca_3d
from src.volcano    import plot_volcano
from src.heatmap    import build_heatmap

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("groups",        {}),   # {group_name: [sample_key, ...]}
    ("group_colors",  {}),   # {group_name: hex_color}
    ("uploaded_files",{}),   # {sample_key: DataFrame(gene_name, est_counts, tpm)}
    ("dge_results",   {}),   # {(A,B): results_df}
]:
    if key not in st.session_state:
        st.session_state[key] = default

DEFAULT_COLORS = [
    "#4361ee", "#f72585", "#4cc9f0", "#7209b7",
    "#06d6a0", "#f77f00", "#ef233c", "#3a86ff",
]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🧬 RNAseq Analysis Suite</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload kallisto/STAR count TSVs · Build groups · '
    'Explore differential expression</p>',
    unsafe_allow_html=True,
)
st.markdown("""
<div class="schema-box">
Expected columns per TSV file:<br>
<b>target_id</b> &nbsp;·&nbsp; <b>length</b> &nbsp;·&nbsp; <b>eff_length</b>
&nbsp;·&nbsp; <b>est_counts</b> &nbsp;·&nbsp; <b>tpm</b>
&nbsp;·&nbsp; <b>gene_name</b> &nbsp;·&nbsp; <b>srr_id</b>
</div>
""", unsafe_allow_html=True)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Upload + Group Builder
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Upload Count Files")
    st.caption(
        "One TSV per sample. The filename (minus extension) becomes the sample label."
    )

    uploaded = st.file_uploader(
        "Upload TSV files",
        type=["tsv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        new_data = load_tsv_files(uploaded)
        for k, df in new_data.items():
            st.session_state.uploaded_files[k] = df
        n = len(st.session_state.uploaded_files)
        st.success(f"{n} sample{'s' if n != 1 else ''} loaded")

    st.divider()

    # ── Group builder ─────────────────────────────────────────────────────────
    st.markdown("### 🗂 Sample Groups")
    st.caption("Create groups, name them, pick a colour, then assign samples.")

    all_keys = list(st.session_state.uploaded_files.keys())

    if st.button("➕ Add Group", use_container_width=True):
        idx      = len(st.session_state.groups)
        new_name = f"Group_{idx + 1}"
        st.session_state.groups[new_name]       = []
        st.session_state.group_colors[new_name] = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

    to_delete = []
    for i, gname in enumerate(list(st.session_state.groups.keys())):
        st.markdown('<div class="group-card">', unsafe_allow_html=True)

        col_name, col_color = st.columns([3, 1])
        with col_name:
            new_name = st.text_input(
                "Name", value=gname,
                key=f"gname_{i}", label_visibility="collapsed",
            )
        with col_color:
            color = st.color_picker(
                "Color",
                value=st.session_state.group_colors.get(
                    gname, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                ),
                key=f"gcolor_{i}", label_visibility="collapsed",
            )

        # Rename if changed
        if new_name != gname:
            st.session_state.groups[new_name] = st.session_state.groups.pop(gname)
            st.session_state.group_colors[new_name] = st.session_state.group_colors.pop(gname, color)
            gname = new_name

        st.session_state.group_colors[gname] = color

        selected = st.multiselect(
            "Samples",
            options=all_keys,
            default=[s for s in st.session_state.groups[gname] if s in all_keys],
            key=f"gfiles_{i}",
            placeholder="Assign TSV samples to this group…",
        )
        st.session_state.groups[gname] = selected

        if st.button("🗑 Remove group", key=f"del_{i}", use_container_width=True):
            to_delete.append(gname)

        st.markdown("</div>", unsafe_allow_html=True)

    for g in to_delete:
        st.session_state.groups.pop(g, None)
        st.session_state.group_colors.pop(g, None)
        st.rerun()

    st.divider()

    # ── Run DGE ───────────────────────────────────────────────────────────────
    valid_groups = {k: v for k, v in st.session_state.groups.items() if v}
    can_run = len(valid_groups) >= 2

    if st.button(
        "🚀 Run DGE Analysis", type="primary",
        use_container_width=True, disabled=not can_run,
    ):
        with st.spinner("Running differential gene expression…"):
            try:
                counts_df, tpm_df, sample_meta = validate_counts(
                    st.session_state.uploaded_files,
                    st.session_state.groups,
                )
                results = run_dge_all_pairs(counts_df, sample_meta)
                st.session_state.dge_results = results
                st.session_state["_counts_df"]    = counts_df
                st.session_state["_tpm_df"]       = tpm_df
                st.session_state["_sample_meta"]  = sample_meta
                st.success(f"Done — {len(results)} comparison(s) ready.")
            except Exception as e:
                st.error(f"Analysis error: {e}")

    if not can_run:
        st.caption("⚠️ Define ≥ 2 groups with samples to enable analysis.")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Tabs
# ═════════════════════════════════════════════════════════════════════════════
tab_dge, tab_volcano, tab_pca, tab_heatmap = st.tabs([
    "📊 DGE Results", "🌋 Volcano Plots", "🔵 PCA", "🔥 Heatmap",
])

# ── DGE Results ───────────────────────────────────────────────────────────────
with tab_dge:
    if not st.session_state.dge_results:
        st.info("Run DGE analysis using the sidebar button to see results here.")
    else:
        for (g1, g2), df in st.session_state.dge_results.items():
            with st.expander(f"**{g1}** vs **{g2}**", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Genes tested",          len(df))
                c2.metric("Significant (padj<0.05)", int((df["padj"] < 0.05).sum()))
                c3.metric(f"Up in {g1} (|FC|>2)",
                          int(((df["padj"] < 0.05) & (df["log2FoldChange"] > 1)).sum()))

                st.dataframe(
                    df.sort_values("padj").round(5),
                    use_container_width=True,
                    height=320,
                )
                st.download_button(
                    f"⬇ Download {g1}_vs_{g2}.csv",
                    data=df.to_csv(index=True).encode(),
                    file_name=f"DGE_{g1}_vs_{g2}.csv",
                    mime="text/csv",
                    key=f"dl_{g1}_{g2}",
                )

# ── Volcano Plots ─────────────────────────────────────────────────────────────
with tab_volcano:
    if not st.session_state.dge_results:
        st.info("Run DGE analysis first.")
    else:
        col_fc, col_pv = st.columns(2)
        fc_thresh = col_fc.slider("log₂FC threshold",  0.5, 4.0, 1.0, 0.25)
        pv_thresh = col_pv.slider("-log₁₀(padj) threshold", 1.0, 10.0, 1.301, 0.1,
                                  help="1.301 ≈ padj < 0.05")

        for (g1, g2), df in st.session_state.dge_results.items():
            st.markdown(f"#### {g1} vs {g2}")
            fig = plot_volcano(
                df, g1, g2,
                fc_thresh=fc_thresh,
                neg_log10_padj_thresh=pv_thresh,
                color_up=st.session_state.group_colors.get(g1, "#4361ee"),
                color_down=st.session_state.group_colors.get(g2, "#f72585"),
            )
            st.plotly_chart(fig, use_container_width=True)

# ── PCA ───────────────────────────────────────────────────────────────────────
with tab_pca:
    tpm_ready = "_tpm_df" in st.session_state
    if not tpm_ready:
        # Allow PCA without running DGE if files+groups are set
        valid_for_pca = {k: v for k, v in st.session_state.groups.items() if v}
        if len(valid_for_pca) >= 2 and st.session_state.uploaded_files:
            if st.button("▶ Compute PCA"):
                try:
                    _, tpm_df, sample_meta = validate_counts(
                        st.session_state.uploaded_files,
                        st.session_state.groups,
                    )
                    st.session_state["_tpm_df"]      = tpm_df
                    st.session_state["_sample_meta"] = sample_meta
                    st.rerun()
                except Exception as e:
                    st.error(f"PCA error: {e}")
        else:
            st.info("Upload files and define ≥ 2 groups to compute PCA.")

    if tpm_ready:
        tpm_df      = st.session_state["_tpm_df"]
        sample_meta = st.session_state["_sample_meta"]

        col_w, col_h = st.columns(2)
        pca_w = col_w.slider("Plot width (px)",  400, 1400, 850, 50)
        pca_h = col_h.slider("Plot height (px)", 300, 900,  550, 50)

        st.markdown(
            "💡 **Group colours** are set via the colour pickers in the sidebar "
            "and are shared with the heatmap annotation bar."
        )

        try:
            coords_2d, coords_3d, explained = compute_pca(tpm_df)

            st.markdown("#### 2D PCA")
            fig2d = plot_pca_2d(coords_2d, sample_meta, explained,
                                st.session_state.group_colors,
                                width=pca_w, height=pca_h)
            st.plotly_chart(fig2d, use_container_width=False)

            st.markdown("#### 3D PCA")
            fig3d = plot_pca_3d(coords_3d, sample_meta, explained,
                                st.session_state.group_colors,
                                width=pca_w, height=pca_h)
            st.plotly_chart(fig3d, use_container_width=False)

        except Exception as e:
            st.error(f"PCA failed: {e}")

# ── Heatmap ───────────────────────────────────────────────────────────────────
with tab_heatmap:
    tpm_ready_hm = "_tpm_df" in st.session_state
    if not tpm_ready_hm:
        st.info("Run DGE analysis (or compute PCA) first so TPM data is available.")
    else:
        tpm_df      = st.session_state["_tpm_df"]
        sample_meta = st.session_state["_sample_meta"]

        # ── Gene list CSVs ────────────────────────────────────────────────────
        gene_list_dir = Path("gene_lists")
        gene_list_dir.mkdir(exist_ok=True)
        csv_files = sorted(gene_list_dir.glob("*.csv"))

        selected_genes: set = set()

        if csv_files:
            chosen_csvs = st.multiselect(
                "📋 Select gene list CSV(s)",
                options=[f.stem for f in csv_files],
                help="Place CSV files in the gene_lists/ folder. "
                     "First column should contain gene names.",
            )
            for csv_name in chosen_csvs:
                try:
                    gdf = pd.read_csv(gene_list_dir / f"{csv_name}.csv")
                    col = gdf.columns[0]
                    selected_genes.update(
                        gdf[col].dropna().astype(str).str.strip().str.upper().tolist()
                    )
                except Exception as e:
                    st.warning(f"Could not read {csv_name}.csv: {e}")
        else:
            st.caption(
                "No CSVs found in `gene_lists/`. "
                "Add gene list CSVs there to use the dropdown."
            )

        manual = st.text_input(
            "✏️ Additional genes (comma-separated, case-insensitive)",
            placeholder="e.g.  ACTB, GAPDH, TP53",
        )
        if manual:
            for g in manual.split(","):
                g = g.strip().upper()
                if g:
                    selected_genes.add(g)

        if selected_genes:
            st.caption(f"**{len(selected_genes)}** unique gene(s) selected.")

        if st.button(
            "🔥 Generate Heatmap", type="primary",
            disabled=len(selected_genes) == 0,
        ):
            try:
                fig = build_heatmap(
                    tpm_df,
                    sample_meta,
                    sorted(selected_genes),
                    st.session_state.group_colors,
                    st.session_state.groups,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "💡 Use the pan tool (hand icon) and drag along the Y axis "
                    "to reorder genes visually."
                )
            except Exception as e:
                st.error(f"Heatmap error: {e}")
