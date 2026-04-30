# RNAseq Analysis & Visualization App

A user-friendly Streamlit application for RNA sequencing analysis and publication-ready plot generation — no coding required.

---

## Overview

This app is designed for bench scientists and bioinformaticians who want to perform differential gene expression analysis and generate high-quality visualizations from STAR-aligned count data without writing a single line of code.

**Key Features:**
- Upload and organize STAR-aligned TSV count files into named sample groups
- Differential Gene Expression (DGE) analysis between all group pairs
- Interactive Volcano Plots for each pairwise comparison
- 2D and 3D PCA plots with customizable group colors
- Gene list-driven Heatmaps with drag-and-drop gene reordering

---

## Input Data Format

The app accepts **TSV files output from STAR alignment** containing raw gene count data. Each TSV file should represent a single sample.

Expected format:
```
gene_id    count
ENSG00000001    142
ENSG00000002    0
...
```

> **Note:** Gene symbol columns (e.g., `gene_name`) are supported alongside Ensembl IDs. The app will attempt to detect the appropriate identifier column automatically.

---

## Getting Started

### 1. Installation

**Prerequisites:** Python 3.9+

```bash
git clone https://github.com/your-org/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage Guide

### Step 1 — Upload Count Files

Use the file uploader to upload one or more STAR-aligned `.tsv` count files. You can upload files from multiple experiments or conditions at once.

### Step 2 — Create Sample Groups

- Click **"Add Group"** to create a new group
- Give each group a descriptive name (e.g., `Control`, `Treatment_24h`)
- Assign each uploaded TSV file to the appropriate group using the dropdown menus
- Create as many groups as needed — there is no limit

### Step 3 — Run Analysis & Generate Plots

Once your groups are defined, navigate to the analysis tabs:

#### Differential Gene Expression (DGE)
- DGE is automatically computed between **every pair of groups**
- For example, if you define groups A, B, and C, the app will run: A vs B, A vs C, and B vs C
- Results include log2 fold change, p-value, and adjusted p-value (FDR)
- Results tables are downloadable as CSV

#### Volcano Plots
- One volcano plot is generated per group pair
- Each plot is labeled at the top left and top right to indicate which group's genes appear on each side
- Significance thresholds (p-value, log2FC) are adjustable

#### PCA
- Each TSV file is represented as a single point
- Points are colored by group
- **Group colors are interactively customizable** using a color picker in the sidebar
- Both **2D and 3D PCA** plots are generated
- Plot dimensions (width/height) can be adjusted

#### Heatmaps
- Place gene list CSV files in the `gene_lists/` folder (see below)
- Select one or more gene lists from the dropdown — overlapping genes are automatically deduplicated
- Manually add additional genes by typing comma-separated gene names (case-insensitive) into the input box
- **Drag and drop genes** to reorder them along the Y axis
- Heatmap layout:
  - **Y axis:** Genes
  - **X axis:** Individual samples (one column per TSV file)
  - A **colored group label bar** sits below the X axis, spanning all samples within each group, colored to match the PCA group colors

---

## Gene List CSVs

To use the heatmap gene list feature, place CSV files in the `gene_lists/` directory at the root of the project. Each CSV should have at least one column containing gene names or symbols.

```
gene_lists/
├── cytokines.csv
├── cell_cycle_genes.csv
└── custom_pathway.csv
```

Expected format (the column header can be anything):
```
gene_name
IL6
TNF
CXCL10
...
```

The app will automatically detect all CSVs in this folder and display them in the dropdown menu by filename.

---

## Project Structure

```
.
├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # Python dependencies
├── gene_lists/             # Place gene list CSVs here
│   └── example_genes.csv
├── src/
│   ├── data_loader.py      # TSV parsing and validation
│   ├── dge.py              # Differential gene expression logic
│   ├── pca.py              # PCA computation and plotting
│   ├── volcano.py          # Volcano plot generation
│   └── heatmap.py          # Heatmap generation and interaction
└── tests/
    └── ...
```

---

## Dependencies

Key packages used (see `requirements.txt` for full list):

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `pydeseq2` | Differential gene expression (DESeq2-based) |
| `plotly` | Interactive volcano, PCA, and heatmap plots |
| `scikit-learn` | PCA computation |
| `scipy` | Statistical testing |

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

[MIT](LICENSE)
