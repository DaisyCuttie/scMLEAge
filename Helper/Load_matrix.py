import pandas as pd
import numpy as np
import scanpy as sc

def _load_raw_matrix(path):
    raw_count = pd.read_csv(path, header=0, index_col=0)
    return raw_count

def get_normalized_matrix(count_matrix):
    anndata = sc.AnnData(count_matrix)
    sc.pp.normalize_total(anndata, target_sum=1e4)
    sc.pp.log1p(anndata)
    normalized_expression = anndata.to_df()
    cellnames = []
    for name in normalized_expression.columns:
        cellnames.append(name.replace('.', '-'))
    normalized_expression.columns = cellnames
    normalized_expression = normalized_expression.T
    normalized_expression = normalized_expression.loc[:, (normalized_expression != 0).any(axis=0)]
    return(normalized_expression)

def _get_raw_counts(count_matrix):
    raw_count = count_matrix.T
    raw_count = raw_count.loc[:, (raw_count != 0).any(axis=0)].T
    cellnames = []
    for name in raw_count.columns:
        cellnames.append(name.replace('.', '-'))
    raw_count.columns = cellnames
    return raw_count

def main(path):
    raw_matrix = _load_raw_matrix(path)
    raw_count = _get_raw_counts(raw_matrix)
    normalized_expression = get_normalized_matrix(raw_matrix)
    return (raw_count, normalized_expression)