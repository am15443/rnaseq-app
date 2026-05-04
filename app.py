"""
app.py
──────
RNAseq Analysis Suite — Streamlit entry point.

Single combined TSV input (all samples in one file):
    target_id | length | eff_length | est_counts | tpm | gene_name | srr_id

Groups and sample assignments are stored entirely in session_state so that
widget interactions (typing, clicking) never reset other widgets.
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

/* ── Base font size boost (Streamlit default is ~14px, we want ~18px) ── */
html { font-size: 18px !important; }

/* ── Dark backgrounds ── */
.stApp { background-color: #0f1117; }
section[data-testid="stSidebar"] { background-color: #1a1d27; }
section[data-testid="stSidebar"] > div { background-color: #1a1d27; }

/* ── White text globally ── */
.stApp, .stApp * {
    color: #ffffff;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar text ── */
section[data-testid="stSidebar"] * { color: #ffffff; }

/* ── Inputs dark ── */
input, textarea, [data-baseweb="input"] input {
    background-color: #2a2d3a !important;
    color: #ffffff !important;
}

/* ── Multiselect dropdown dark ── */
[data-baseweb="select"] > div { background-color: #2a2d3a !important; }
[data-baseweb="menu"] { background-color: #2a2d3a !important; }
[data-baseweb="option"] { background-color: #2a2d3a !important; color: #ffffff !important; }
[data-baseweb="tag"] { background-color: #4361ee !important; color: #ffffff !important; }

/* ── Buttons ── */
.stButton > button {
    background-color: #2a2d3a;
    color: #ffffff;
    border: 1px solid #444;
    font-size: 1rem;
}

/* ── Tab text ── */
button[data-baseweb="tab"] { color: #ffffff !important; }

.main-title {
    font-family: 'DM Mono', monospace;
    font-size: 2.2rem;
    font-weight: 500;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}
.subtitle { color: #aaaaaa; font-size: 1rem; margin-top: 0.1rem; margin-bottom: 0; }
.schema-box {
    background: #1e2130;
    border-left: 3px solid #4361ee;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    color: #dddddd;
    margin: 0.6rem 0 1rem 0;
    line-height: 1.8;
}
.group-card {
    background: #1e2130;
    border: 1.5px solid #333a55;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ── Src imports ───────────────────────────────────────────────────────────────
from src.data_loader import load_combined_tsv, validate_counts
from src.dge         import run_dge_all_pairs
from src.pca         import compute_pca, plot_pca_2d, plot_pca_3d
from src.volcano     import plot_volcano
from src.heatmap     import build_heatmap

# ── Session state defaults ────────────────────────────────────────────────────
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("gene_sample_df", None)
_init("all_srr_ids",    [])
_init("group_ids",      [])          # ordered list of stable group ids (ints)
_init("group_names",    {})          # {gid: str}
_init("group_colors",   {})          # {gid: hex str}
_init("group_samples",  {})          # {gid: [srr_id, ...]}
_init("next_gid",       0)
_init("dge_results",    {})

DEFAULT_COLORS = [
    "#4361ee", "#f72585", "#4cc9f0", "#7209b7",
    "#06d6a0", "#f77f00", "#ef233c", "#3a86ff",
]

# ── Helper: add / remove groups ───────────────────────────────────────────────
def add_group():
    gid = st.session_state.next_gid
    st.session_state.next_gid += 1
    idx = len(st.session_state.group_ids)
    st.session_state.group_ids.append(gid)
    st.session_state.group_names[gid]   = f"Group_{idx + 1}"
    st.session_state.group_colors[gid]  = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
    st.session_state.group_samples[gid] = []

def remove_group(gid):
    st.session_state.group_ids.remove(gid)
    for d in (st.session_state.group_names,
              st.session_state.group_colors,
              st.session_state.group_samples):
        d.pop(gid, None)

# ── Callbacks that write directly into session_state ─────────────────────────
def on_name_change(gid):
    st.session_state.group_names[gid] = st.session_state[f"name_{gid}"]

def on_color_change(gid):
    st.session_state.group_colors[gid] = st.session_state[f"color_{gid}"]

def on_samples_change(gid):
    st.session_state.group_samples[gid] = st.session_state[f"samples_{gid}"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🧬 RNAseq Analysis Suite</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload your combined TSV · Assign srr_ids to groups · '
    'Explore differential expression</p>',
    unsafe_allow_html=True,
)
st.markdown("""
<div class="schema-box">
Expected TSV columns (all samples in one file):<br>
<b>target_id</b> &nbsp;·&nbsp; <b>length</b> &nbsp;·&nbsp; <b>eff_length</b>
&nbsp;·&nbsp; <b>est_counts</b> &nbsp;·&nbsp; <b>tpm</b>
&nbsp;·&nbsp; <b>gene_name</b> &nbsp;·&nbsp; <b>srr_id</b>
</div>
""", unsafe_allow_html=True)
st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Step 1: Upload ────────────────────────────────────────────────────────
    st.markdown("### 📂 Step 1 — Upload Combined TSV")
    st.caption("One file containing all samples. The srr_id column identifies each sample.")

    uploaded = st.file_uploader(
        "Upload combined TSV",
        type=["tsv", "txt"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded is not None and st.session_state.gene_sample_df is None:
        # Only parse once — if user re-uploads a new file they must refresh
        try:
            gene_sample_df, srr_ids = load_combined_tsv(uploaded)
            st.session_state.gene_sample_df = gene_sample_df
            st.session_state.all_srr_ids    = srr_ids
            st.success(f"✅ {len(srr_ids)} sample(s) detected.")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    if st.session_state.gene_sample_df is not None:
        st.caption(f"Samples: {', '.join(st.session_state.all_srr_ids)}")

        # Button to clear and re-upload
        if st.button("↩ Clear & upload new file", use_container_width=True):
            for k in ["gene_sample_df", "all_srr_ids", "dge_results",
                      "_counts_df", "_tpm_df", "_sample_meta"]:
                st.session_state.pop(k, None)
            st.session_state.gene_sample_df = None
            st.session_state.all_srr_ids    = []
            st.session_state.group_ids      = []
            st.session_state.group_names    = {}
            st.session_state.group_colors   = {}
            st.session_state.group_samples  = {}
            st.session_state.next_gid       = 0
            st.session_state.dge_results    = {}
            st.rerun()

    st.divider()

    # ── Step 2: Group Builder ─────────────────────────────────────────────────
    st.markdown("### 🗂 Step 2 — Define Groups")
    st.caption("Create groups, name them, pick a colour, then assign srr_ids.")

    all_srr_ids = st.session_state.all_srr_ids

    if not all_srr_ids:
        st.caption("⬆️ Upload a TSV file first.")
    else:
        st.button("➕ Add Group", on_click=add_group, use_container_width=True)

        for gid in list(st.session_state.group_ids):
            st.markdown('<div class="group-card">', unsafe_allow_html=True)

            col_name, col_color = st.columns([3, 1])
            with col_name:
                st.text_input(
                    "Group name",
                    value=st.session_state.group_names[gid],
                    key=f"name_{gid}",
                    label_visibility="collapsed",
                    on_change=on_name_change,
                    args=(gid,),
                )
            with col_color:
                st.color_picker(
                    "Color",
                    value=st.session_state.group_colors[gid],
                    key=f"color_{gid}",
                    label_visibility="collapsed",
                    on_change=on_color_change,
                    args=(gid,),
                )

            st.multiselect(
                "Assign srr_ids",
                options=all_srr_ids,
                default=st.session_state.group_samples[gid],
                key=f"samples_{gid}",
                label_visibility="collapsed",
                placeholder="Select srr_ids for this group…",
                on_change=on_samples_change,
                args=(gid,),
            )

            st.button(
                "🗑 Remove group",
                key=f"del_{gid}",
                on_click=remove_group,
                args=(gid,),
                use_container_width=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── Step 3: Run Analysis ──────────────────────────────────────────────────
    st.markdown("### 🚀 Step 3 — Run Analysis")

    # Build groups dict {name: [srr_ids]} for downstream modules
    groups_dict = {
        st.session_state.group_names[gid]: st.session_state.group_samples[gid]
        for gid in st.session_state.group_ids
    }
    colors_dict = {
        st.session_state.group_names[gid]: st.session_state.group_colors[gid]
        for gid in st.session_state.group_ids
    }
    valid_groups = {k: v for k, v in groups_dict.items() if v}
    can_run = st.session_state.gene_sample_df is not None and len(valid_groups) >= 2

    if st.button("Run DGE Analysis", type="primary",
                 use_container_width=True, disabled=not can_run):
        with st.spinner("Building matrices and running DGE…"):
            try:
                counts_df, tpm_df, sample_meta = validate_counts(
                    st.session_state.gene_sample_df, groups_dict,
                )
                results = run_dge_all_pairs(counts_df, sample_meta)
                st.session_state.dge_results        = results
                st.session_state["_counts_df"]      = counts_df
                st.session_state["_tpm_df"]         = tpm_df
                st.session_state["_sample_meta"]    = sample_meta
                st.session_state["_groups_dict"]    = groups_dict
                st.session_state["_colors_dict"]    = colors_dict
                st.success(f"Done — {len(results)} comparison(s) ready.")
            except Exception as e:
                st.error(f"Analysis error: {e}")

    if not can_run:
        if st.session_state.gene_sample_df is None:
            st.caption("⬆️ Upload a TSV file to get started.")
        else:
            st.caption("⚠️ Define ≥ 2 groups with srr_ids assigned.")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Tabs
# ═════════════════════════════════════════════════════════════════════════════
tab_samples, tab_dge, tab_volcano, tab_pca, tab_heatmap = st.tabs([
    "🧪 Samples", "📊 DGE Results", "🌋 Volcano Plots", "🔵 PCA", "🔥 Heatmap",
])

# ── Samples overview ──────────────────────────────────────────────────────────
with tab_samples:
    if st.session_state.gene_sample_df is None:
        st.info("Upload a combined TSV file using the sidebar to get started.")
    else:
        st.markdown("#### Samples detected in uploaded file")
        group_map = {
            s: st.session_state.group_names[gid]
            for gid in st.session_state.group_ids
            for s in st.session_state.group_samples[gid]
        }
        summary = pd.DataFrame({
            "srr_id": st.session_state.all_srr_ids,
            "assigned_group": [
                group_map.get(s, "— unassigned —")
                for s in st.session_state.all_srr_ids
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)
        unassigned = [s for s in st.session_state.all_srr_ids if s not in group_map]
        if unassigned:
            st.warning(f"{len(unassigned)} sample(s) not yet assigned: {', '.join(unassigned)}")
        else:
            st.success("All samples are assigned to a group. ✅")

# ── DGE Results ───────────────────────────────────────────────────────────────
with tab_dge:
    if not st.session_state.dge_results:
        st.info("Run DGE analysis using the sidebar to see results here.")
    else:
        for (g1, g2), df in st.session_state.dge_results.items():
            with st.expander(f"**{g1}** vs **{g2}**", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Genes tested", len(df))
                c2.metric("Significant (padj < 0.05)", int((df["padj"] < 0.05).sum()))
                c3.metric(f"Up in {g1} (log2FC > 1)",
                          int(((df["padj"] < 0.05) & (df["log2FoldChange"] > 1)).sum()))
                st.dataframe(df.sort_values("padj").round(5),
                             use_container_width=True, height=320)
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
        colors_dict = st.session_state.get("_colors_dict", {})
        col_fc, col_pv = st.columns(2)
        fc_thresh = col_fc.slider("log₂FC threshold", 0.5, 4.0, 1.0, 0.25)
        pv_thresh = col_pv.slider("-log₁₀(padj) threshold", 1.0, 10.0, 1.301, 0.1,
                                  help="1.301 ≈ padj < 0.05")
        for (g1, g2), df in st.session_state.dge_results.items():
            st.markdown(f"#### {g1} vs {g2}")
            fig = plot_volcano(df, g1, g2,
                               fc_thresh=fc_thresh,
                               neg_log10_padj_thresh=pv_thresh,
                               color_up=colors_dict.get(g1, "#4361ee"),
                               color_down=colors_dict.get(g2, "#f72585"))
            st.plotly_chart(fig, use_container_width=True)

# ── PCA ───────────────────────────────────────────────────────────────────────
with tab_pca:
    tpm_ready = "_tpm_df" in st.session_state
    if not tpm_ready:
        valid_for_pca = {
            st.session_state.group_names[gid]: st.session_state.group_samples[gid]
            for gid in st.session_state.group_ids
            if st.session_state.group_samples[gid]
        }
        if st.session_state.gene_sample_df is not None and len(valid_for_pca) >= 2:
            if st.button("▶ Compute PCA"):
                try:
                    _, tpm_df, sample_meta = validate_counts(
                        st.session_state.gene_sample_df, valid_for_pca,
                    )
                    st.session_state["_tpm_df"]      = tpm_df
                    st.session_state["_sample_meta"] = sample_meta
                    st.session_state["_colors_dict"] = {
                        st.session_state.group_names[gid]: st.session_state.group_colors[gid]
                        for gid in st.session_state.group_ids
                    }
                    st.rerun()
                except Exception as e:
                    st.error(f"PCA error: {e}")
        else:
            st.info("Upload a file and define ≥ 2 groups to compute PCA.")

    if tpm_ready:
        tpm_df      = st.session_state["_tpm_df"]
        sample_meta = st.session_state["_sample_meta"]
        colors_dict = st.session_state.get("_colors_dict", {})

        col_w, col_h = st.columns(2)
        pca_w = col_w.slider("Plot width (px)",  400, 1400, 850, 50)
        pca_h = col_h.slider("Plot height (px)", 300, 900,  550, 50)
        st.caption("💡 Group colours are set via the colour pickers in the sidebar.")

        try:
            coords_2d, coords_3d, explained = compute_pca(tpm_df)
            st.markdown("#### 2D PCA")
            st.plotly_chart(
                plot_pca_2d(coords_2d, sample_meta, explained, colors_dict,
                            width=pca_w, height=pca_h),
                use_container_width=False,
            )
            st.markdown("#### 3D PCA")
            st.plotly_chart(
                plot_pca_3d(coords_3d, sample_meta, explained, colors_dict,
                            width=pca_w, height=pca_h),
                use_container_width=False,
            )
        except Exception as e:
            st.error(f"PCA failed: {e}")

# ── Heatmap ───────────────────────────────────────────────────────────────────
with tab_heatmap:
    if "_tpm_df" not in st.session_state:
        st.info("Run DGE analysis (or compute PCA) first so TPM data is available.")
    else:
        tpm_df      = st.session_state["_tpm_df"]
        sample_meta = st.session_state["_sample_meta"]
        colors_dict = st.session_state.get("_colors_dict", {})
        groups_dict = st.session_state.get("_groups_dict", {})

        gene_list_dir = Path("gene_lists")
        gene_list_dir.mkdir(exist_ok=True)
        csv_files = sorted(gene_list_dir.glob("*.csv"))
        selected_genes: set = set()

        if csv_files:
            chosen_csvs = st.multiselect(
                "📋 Select gene list CSV(s)",
                options=[f.stem for f in csv_files],
                help="Place CSV files in the gene_lists/ folder. First column = gene names.",
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
            st.caption("No CSVs found in gene_lists/. Add gene list CSVs there to use the dropdown.")

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

        if st.button("🔥 Generate Heatmap", type="primary",
                     disabled=len(selected_genes) == 0):
            try:
                fig = build_heatmap(
                    tpm_df, sample_meta, sorted(selected_genes),
                    colors_dict, groups_dict,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 Use the pan tool and drag along the Y axis to reorder genes.")
            except Exception as e:
                st.error(f"Heatmap error: {e}")
