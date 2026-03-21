import scanpy as sc
import numpy as np
import pandas as pd
from collections import Counter

class AnnObject:
    def __init__(self, adata, class_col, age_col, substring):
        self.processed_adata = self.anndata_processing(adata)
        self.celltype_dict = self.get_celltype_dict(self.processed_adata, class_col)
        self.raw_counts = self.get_raw_counts(adata)
        self.age_dict = self.get_cleaned_age_dict(self.processed_adata, age_col, substring)
        
    def anndata_processing(self, adata):
        sc.pp.filter_genes(adata, min_cells=5)
        sc.pp.filter_cells(adata, min_genes=500)
        adata.obs['n_counts'] = np.sum(adata.X, axis=1).A1
        adata = adata[adata.obs['n_counts']>=3000]
        sc.pp.normalize_total(adata, target_sum=1e4)
#         print(adata.shape)
#         sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=10, min_disp=0.5)
# #         adata = adata[:, adata.var.highly_variable]
        adata = sc.pp.filter_genes_dispersion(adata, subset = False, min_disp=.5, max_disp=None, 
                                  min_mean=.0125, max_mean=10, n_bins=20, n_top_genes=None, 
                                  log=True, copy=True)
        print("logarithmizing the adata into processed_object")
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10, zero_center=False)
        sc.tl.pca(adata,use_highly_variable=True)
        variance_ratios = adata.uns['pca']['variance_ratio']
        cumulative_variance = variance_ratios.cumsum()
        optimal_pcs = (cumulative_variance < 0.9).sum() + 1
        sc.pp.neighbors(adata, n_neighbors = optimal_pcs)
        sc.tl.louvain(adata, resolution = 1)
        sc.tl.umap(adata)
        return adata
    
    def get_celltype_dict(self, adata, class_col):
        celltype_dict = {}
        celltype_df = pd.DataFrame(adata.obs[class_col])
        for celltype in Counter(celltype_df[class_col]):
            celltype_dict[celltype] = list(adata.obs[adata.obs[class_col] == celltype].index)
        return celltype_dict

    def get_raw_counts(self, adata):
        dense_array = adata.X.T.toarray() if hasattr(adata.X, "toarray") else adata.X.T
        return pd.DataFrame(dense_array, index=adata.var_names, columns=adata.obs_names).astype(int)
    
    def _handle_float(self, x):
        if x - int(x) > 0.5:
            return int(x) + 1 # Round up
        else:
            return int(x) # Round down
        
    def _clean_age_df(self, age_df, age_col, substring=""):
        values = []
        for x in age_df[age_col]:
            if isinstance(x, str):
                try:
                    value = x.replace(substring, "")
                    value = float(value)
                    # Attempt to strip the substring and convert to integer
                    value = self._handle_float(value)
                    values.append(value)
                except ValueError:
                    # integer case
                    if x.isdigit():
                        value = x.replace(substring, "")
                        values.append(int(value))
                    else:
                        # Handle the case where conversion fails
                        warnings.warn(f"Warning: '{x}' could not be converted to an integer.")
                        break
            elif isinstance(x, int):
                values.append(x)
            elif isinstance(x, float):
                value = self._handle_float(x)
                values.append(value)
            else:
                print("Error: x is neither a string, an integer, nor a float:", x)
                break
        age_df[age_col] = values
        return age_df

    def get_cleaned_age_dict(self, adata, age_col, substring):
        age_df = pd.DataFrame(adata.obs[age_col])
        cleaned_age_df = self._clean_age_df(age_df, age_col, substring)
        age_dict = dict(zip(cleaned_age_df.index, cleaned_age_df[age_col]))
        return age_dict