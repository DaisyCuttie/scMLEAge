from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Helper.r_square_path import ensure_r_square_file_path


warnings.filterwarnings("ignore")


@dataclass
class PipelineConfig:
    input_h5ad: str
    organ: str
    class_col: str = "celltype"
    age_col: str = "age"
    age_substring: str = "m"
    processed_folder: str = "./Processed_data/"
    cor_info_path: str = ""
    r_square_file_path: str = ""
    fig_dir: str = "./Model_Figures/"
    r_squareds_dir: str = "./R_Squareds"
    n_splits: int = 5
    random_state: int = 42
    umap_font_size: int = 16
    save_processed_h5ad: bool = True


def _default_cor_info_path(organ: str) -> str:
    return f"HiExpr_correlation_info_matrix/Tabula_Muris_{organ}"


def _prepare_r_square_file_path(config: PipelineConfig) -> str:
    if config.r_square_file_path:
        file_path = Path(config.r_square_file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return str(file_path)
    return ensure_r_square_file_path(config.organ, config.r_squareds_dir)


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Run the notebook workflow in the same order as Bladder_cells.ipynb."""
    import matplotlib.pyplot as plt
    import scanpy as sc
    from sklearn.model_selection import KFold

    from paper_results.Helper.Load_Anndata import read_and_filter_h5ad
    from paper_results.Helper.Process_Anndata import AnnObject
    import paper_results.Helper.train_and_plot as tp
    import paper_results.Helper.utils as hutil

    raw_adata = read_and_filter_h5ad(
        config.input_h5ad,
        config.class_col,
        config.age_col,
    )

    cor_info_path = config.cor_info_path or _default_cor_info_path(config.organ)
    r_square_file_path = _prepare_r_square_file_path(config)

    font_size_config = {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }

    adataObject = AnnObject(raw_adata, config.class_col, config.age_col, config.age_substring)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    with plt.rc_context(font_size_config):
        sc.pl.umap(
            adataObject.processed_adata,
            color="age",
            ax=axs[0],
            show=False,
            title="UMAP colored by age",
        )
        sc.pl.umap(
            adataObject.processed_adata,
            color=config.class_col,
            ax=axs[1],
            show=False,
            title="UMAP colored by cell type",
        )
    plt.tight_layout()
    plt.show()

    selected_gene_by_frac = hutil.filter_low_genes(
        adataObject.raw_counts, adataObject.celltype_dict
    )
    correlated_index = hutil.get_correlated_genes(
        adataObject, config.organ, selected_gene_by_frac, cor_info_path
    )
    metaCells, metaAges = hutil.get_metaCells(
        adataObject.celltype_dict, adataObject.raw_counts, adataObject.age_dict
    )
    pred_freq_matrices, freq_age_group = hutil.get_gene_frequencies(
        metaCells,
        metaAges,
        adataObject.raw_counts.index,
        selected_gene_by_frac,
    )

    num_features = []
    r_squared_results = {}

    donor_kf = KFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    for celltype in adataObject.celltype_dict:
        print(f"Processing cell type: {celltype}")
        cell_group = adataObject.celltype_dict[celltype]

        res = tp.run_celltype_pipeline(
            celltype=celltype,
            cell_group=cell_group,
            adataObject=adataObject,
            pred_freq_matrices=pred_freq_matrices,
            correlated_index=correlated_index,
            freq_age_group=freq_age_group,
            donor_kf=donor_kf,
            organ=config.organ,
            r_square_file_path=r_square_file_path,
            fig_dir=config.fig_dir,
        )
        adataObject.processed_adata.obs.loc[res["pred_index"], "pred_age"] = res["pred_vals"]
        r_squared_results[celltype] = res["r2_curve"]
        num_features.append(
            {"celltype": celltype, "num_features": res["optimal_gene_count"]}
        )

    tp.write_feature_summary(config.organ, num_features)

    adataObject.processed_adata.obs["pred_age"] = [
        str(int(x)) + "m" for x in adataObject.processed_adata.obs["pred_age"]
    ]

    if config.save_processed_h5ad:
        processed_path = (
            f"{config.processed_folder}{config.organ}_Processed_Data.h5ad"
        )
        Path(config.processed_folder).mkdir(parents=True, exist_ok=True)
        adataObject.processed_adata.write_h5ad(
            filename=processed_path,
            compression="gzip",
        )

    return {
        "adataObject": adataObject,
        "selected_gene_by_frac": selected_gene_by_frac,
        "correlated_index": correlated_index,
        "metaCells": metaCells,
        "metaAges": metaAges,
        "pred_freq_matrices": pred_freq_matrices,
        "freq_age_group": freq_age_group,
        "r_squared_results": r_squared_results,
        "r_square_file_path": r_square_file_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the scMLEAge pipeline in notebook order."
    )
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--organ", required=True)
    parser.add_argument("--class-col", default="celltype")
    parser.add_argument("--age-col", default="age")
    parser.add_argument("--age-substring", default="m")
    parser.add_argument("--processed-folder", default="./Processed_data/")
    parser.add_argument("--cor-info-path", default="")
    parser.add_argument("--r-square-file-path", default="")
    parser.add_argument("--fig-dir", default="./Model_Figures/")
    parser.add_argument("--r-squareds-dir", default="./R_Squareds")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-save-processed-h5ad", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        input_h5ad=args.input_h5ad,
        organ=args.organ,
        class_col=args.class_col,
        age_col=args.age_col,
        age_substring=args.age_substring,
        processed_folder=args.processed_folder,
        cor_info_path=args.cor_info_path,
        r_square_file_path=args.r_square_file_path,
        fig_dir=args.fig_dir,
        r_squareds_dir=args.r_squareds_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
        save_processed_h5ad=not args.no_save_processed_h5ad,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
