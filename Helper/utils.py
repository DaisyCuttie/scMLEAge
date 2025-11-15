import scanpy as sc
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from scipy import stats
from sklearn.linear_model import LinearRegression
import shutil
from scipy.sparse import issparse
from scipy.stats import pearsonr


def get_normalized_matrix(count_matrix):
    anndata = sc.AnnData(count_matrix)
    sc.pp.normalize_total(anndata, target_sum=1e4)
    normalized_expression = anndata.to_df()
    cellnames = []
    for name in normalized_expression.columns:
        cellnames.append(name.replace('.', '-'))
    normalized_expression.columns = cellnames
    normalized_expression = normalized_expression.T
    normalized_expression = normalized_expression.loc[:, (normalized_expression != 0).any(axis=0)]
    return(normalized_expression)


def filter_low_genes(raw_count, celltype_dict):
    selected_gene_by_frac = dict()
    for celltype in tqdm(list(celltype_dict.keys())):
        current_cells = raw_count[celltype_dict[celltype]] 
        norm_curr_cells = get_normalized_matrix(current_cells)
        
        # then select the genes after the filter 
        mean_percent = norm_curr_cells.mean(axis=0)
        argsort_index = np.argsort(mean_percent).values
        counts_top_genes = norm_curr_cells.iloc[:, argsort_index]

        # Adjust selection to get top-expressing genes
        top_genes = list(counts_top_genes.columns)[-int(counts_top_genes.shape[1]):]
        selected_gene_by_frac[celltype] = top_genes
    return selected_gene_by_frac


def get_correlated_genes(adataObject, organ, selected_gene_by_frac, file_path):
    os.makedirs(file_path, exist_ok=True)
    if os.listdir(file_path):
        print("Correlation output directory already exists.")
        print("Deleting previous correlated_info files...")
        shutil.rmtree(file_path)
        os.makedirs(file_path)

    # Transpose normalized counts to genes x cells DataFrame
    norm_counts = pd.DataFrame(
        adataObject.processed_adata.X.toarray(),
        index=adataObject.processed_adata.obs_names,
        columns=adataObject.processed_adata.var_names
    ).T

    correlated_genes = {}
    correlated_index = {}
    correlations = {}

    for celltype, curr_cells in adataObject.celltype_dict.items():
        print(f"Processing cell type {celltype}...")
        curr_norm_cells = norm_counts[curr_cells]
        curr_ages = np.array([adataObject.age_dict[cell] for cell in curr_cells])
        curr_genes = selected_gene_by_frac[celltype]

        corr_vals, p_vals, indices = [], [], []

        for gene in curr_genes:
            expr = curr_norm_cells.loc[gene].values
            r, p = stats.pearsonr(expr, curr_ages)
            corr_vals.append(r)
            p_vals.append(p)
            indices.append(norm_counts.index.get_loc(gene))

        abs_corr = np.abs(corr_vals)
        sort_order = np.argsort(abs_corr)[::-1] ## sorting in descending order

        correlated_genes[celltype] = [curr_genes[i] for i in sort_order]
        correlated_index[celltype] = [indices[i] for i in sort_order]
        correlations[celltype] = [abs_corr[i] for i in sort_order]

        df = pd.DataFrame({
            "Pearson Correlation": corr_vals,
            "Absolute Correlation": abs_corr,
            "P-values": p_vals
        }, index=curr_genes)
        df.to_csv(os.path.join(file_path, f"{celltype}_correlation_info.csv"))

    return correlated_index


def get_metaCells(celltype_dict, raw_counts, age_dict):
    metaCells = {}
    metaAges = {}
    for celltype in celltype_dict:
        curr_cells = celltype_dict[celltype]
        curr_raw_cells = raw_counts[curr_cells].values  # Convert to NumPy array

        curr_ages = np.array([age_dict[cell] for cell in curr_cells])
        age_groups = np.unique(curr_ages)

        metaCells[celltype] = []
        metaAges[celltype] = []
        for age in age_groups:
            age_mask = curr_ages == age
            age_group_cells = curr_raw_cells[:, age_mask]  # Select cells with the current age in NumPy
            # Sum across cells with the same age to create a meta cell
            meta_cell = age_group_cells.sum(axis = 1)
            metaCells[celltype].append(meta_cell)
            metaAges[celltype].append(age)
        metaCells[celltype] = np.array(metaCells[celltype])
    return metaCells, metaAges


def get_gene_frequencies(metaCells, metaAges, gene_names, selected_gene_by_frac, returnTrueFreq = False):
    freq_age_group = {}
    pred_freq_matrices = {}
    freq_matrices = {}

    # Loop through each cell type in adataObject.celltype_dict
    for celltype in metaCells:
        print("processing celltype: ", celltype)
        # Retrieve current cells and ages from adataObject
        curr_cells = metaCells[celltype]
        curr_ages = metaAges[celltype]

        df = pd.DataFrame(curr_cells)
        df.index = curr_ages
        df.columns = gene_names
        
        frequency_matrix = df.div(df.sum(axis = 1), axis = 0).fillna(0)
#         frequency_matrix.to_csv(f"./Frequency_Matrices/{celltype}-frequency_matrix.csv")
        if returnTrueFreq:
            freq_matrices[celltype] = frequency_matrix

        pred_freqs = pd.DataFrame(0, columns=frequency_matrix.columns, index=frequency_matrix.index)

        curr_genes = selected_gene_by_frac[celltype]
        train_ages = np.array(curr_ages).reshape(-1, 1)
        for gene in curr_genes:
            # Only perform regression if the gene is present in the columns
            if gene in frequency_matrix:
                reg = LinearRegression(n_jobs=1).fit(train_ages, frequency_matrix[gene].to_numpy())
                predictions = reg.predict(train_ages)
                pred_freqs[gene] = predictions
        pred_freqs[pred_freqs < 0] = 1e-6
        pred_freqs = pred_freqs.T
        pred_freq_matrices[celltype] = pred_freqs.to_numpy()
        freq_age_group[celltype] = curr_ages
    return pred_freq_matrices, freq_age_group


def powers_of_two_counts(max_len: int) -> list[int]:
    """Return [1,2,4,...] up to max_len (inclusive)."""
    counts, i = [], 0
    while (1 << i) <= max_len:
        counts.append(1 << i)
        i += 1
    return counts or [1]


def r2(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson R² with NaN safety."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    r, _ = pearsonr(x, y)
    if np.isnan(r):
        return 0.0
    return float(r**2)