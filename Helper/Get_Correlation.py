import numpy as np
from tqdm import tqdm
from scipy import stats

def get_correlation_info(normalized_expression, celltype_dict, ages_df):
    correlation_matrix = []
    correlated_genes = []
    gene_names = np.array(normalized_expression.columns)
    for celltype in tqdm(list(celltype_dict.keys())):
        cell_group = normalized_expression.T[celltype_dict[celltype]]
        current_ages = ages_df.T[celltype_dict[celltype]]
        current_ages = current_ages.values.flatten()
        pearson_correlation = []
        for expression in np.array(cell_group):
            pearson_correlation.append(abs(stats.pearsonr(expression, current_ages)[0]))
        pearson_correlation = np.nan_to_num(np.array(pearson_correlation))
        order = np.argsort(pearson_correlation)
        correlation_matrix.append(pearson_correlation[order][::-1])
        correlated_genes.append(gene_names[order][::-1])
    return (correlation_matrix, correlated_genes)