
# scMLEAge

scMLEAge is a single-cell RNA transcriptomic cell-type-specific clock that can help researchers study the biological ages of organisms in the single-cell level. 

![](https://github.com/DaisyCuttie/scMLEAge/blob/main/paper_results/analysis/Figures/Histogram_of_R-squared_by_Celltype.png)

### Individual Model Example
![Limb-muscle cells](https://github.com/DaisyCuttie/scMLEAge/blob/main/paper_results/Model_Figures/Limb-Muscle_skeletal%20muscle%20satellite%20cell_model.png)
![Kidney](https://github.com/DaisyCuttie/scMLEAge/blob/main/paper_results/Model_Figures/Kidney_proximal%20convoluted%20tubule%20epithelial%20cell_model.png)

## Installation

```bash
git clone https://github.com/DaisyCuttie/scMLEAge.git
cd scMLEAge

conda create -n scmleage python=3.10 -y   # or python -m venv .venv
conda activate scmleage

pip install -r requirements.txt
```
---

## Input data requirements

scMLEAge reads a single `.h5ad` file. It must have:

| Requirement | Default | Notes |
|---|---|---|
| Raw (un-normalized) integer counts in `.X` | — | The likelihood model is defined on read counts, so normalized or log-transformed values will not work. |
| A cell type column in `.obs` | `celltype` | Override with `--class-col`. |
| An age column in `.obs` | `age` | Override with `--age-col`. |
| Age values with a unit suffix | `m` (e.g. `3m`, `18m`, `24m`) | Override with `--age-substring`. Predicted ages are written back in the same format. |
| A donor identifier in `.obs` |

---

### Data
The examples here are the applications to the Tabula Muris Senis data https://rdcu.be/eJI2Y that you can download from here https://figshare.com/articles/dataset/Processed_files_to_use_with_scanpy_/8273102/2


## How to run scMLEAge
The example run is shown in the Tutorial.ipynb file.

## Paper link
https://doi.org/10.64898/2025.12.04.692166

## Helper Module Documentation

The `Helper/` directory contains essential utility modules for single-cell aging analysis, providing core data processing, modeling, and visualization functionality. **`Load_Anndata.py`** handles data ingestion and quality filtering, implementing robust preprocessing pipelines that filter out low-count cell types and age-skewed populations while supporting gender-specific filtering. The **`Process_Anndata.py`** module defines the `AnnObject` class, which encapsulates comprehensive single-cell data processing workflows including gene filtering, normalization, PCA analysis, and UMAP visualization with automatic parameter optimization. **`utils.py`** provides specialized computational functions for aging-related analysis, including gene expression normalization, age-correlated gene identification through Pearson correlation analysis. The **`train_and_plot.py`** module implements the core machine learning pipeline and publication-ready violin plots for visualizing prediction accuracy across different age groups. 

---

## Quick start

Run the full pipeline on one organ:

```bash
python run_scmleage_pipeline.py \
  --input-h5ad path/to/Kidney.h5ad \
  --organ Kidney
  --class-col celltype
  --age-col age
  --age-substring age_suffix
```

Only `--input-h5ad` and `--organ` are required. However, class-col, age-col and age-substring is recommended if your age column and cell type column is not named as required above.

<details>
<summary>All options</summary>

| Flag | Default | Purpose |
|---|---|---|
| `--input-h5ad` | *required* | Path to the input `.h5ad`. |
| `--organ` | *required* | Organ name; used in output filenames and correlation-matrix paths. |
| `--class-col` | `celltype` | `.obs` column holding cell type labels. |
| `--age-col` | `age` | `.obs` column holding donor age. |
| `--age-substring` | `m` | Unit suffix stripped from age values. |
| `--processed-folder` | `./Processed_data/` | Where the annotated `.h5ad` is written. |
| `--cor-info-path` | auto | Correlation info matrix directory. Defaults to `HiExpr_correlation_info_matrix/Tabula_Muris_{organ}`. |
| `--r-square-file-path` | auto | Explicit path for the R² output file. |
| `--fig-dir` | `./Model_Figures/` | Per-cell-type model figures. |
| `--r-squareds-dir` | `./R_Squareds` | R² curves per organ. |
| `--n-splits` | `5` | Cross-validation folds. |
| `--random-state` | `42` | Seed for the CV split. |
| `--no-save-processed-h5ad` | off | Skip writing the annotated `.h5ad`. |

</details>


+++ This method facilitate as a cohesive framework for analyzing cellular aging patterns, enabling researchers to identify age-associated gene expression changes, train predictive models, and generate statistical visualizations for aging research applications.
