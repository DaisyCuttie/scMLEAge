import pandas as pd
import numpy as np
import scanpy as sc
from collections import Counter


def _filter_low_counts(celltype_df, age_df, celltype_col, threshold):
    print("Checking low count cell types...")
    
    celltype_count = Counter(celltype_df[celltype_col])
    for key in celltype_count:
        if threshold == None:
            unique_ages = np.unique(age_df)
            num_groups = (len(unique_ages) + 1) * 100
            if celltype_count[key] < num_groups:
                print(key, " has too low counts")
                celltype_df = celltype_df[celltype_df[celltype_col] != key]
        else:
            if celltype_count[key] < threshold:
                print(key, " has too low counts")
                celltype_df = celltype_df[celltype_df[celltype_col] != key]
    return celltype_df

def _get_skewed_count_info(adata, class_col, age_col, age_threshold):
    print("Checking skewed count cell types...")
    
    # Compute the fraction of cells for each age group within each cell ontology class
    group_counts = adata.obs.groupby([class_col, age_col]).size()
    total_counts = adata.obs.groupby([class_col]).size()
    
    # Calculate the fraction of each age group within each class
    class_age_fraction = group_counts / total_counts
    
    # Find the cell classes to filter out based on age distribution
    classes_to_filter = class_age_fraction[class_age_fraction > age_threshold].index.get_level_values(0).unique()
    
    return classes_to_filter
    


def read_and_filter_h5ad(filepath, class_col="celltype", age_col="age", filter_gender=True, gender="male", age_threshold=0.8, count_threshold=None):
    """Parameters:
    filepath: path to AnnData object
        The Scanpy AnnData object containing single-cell data.
    class_col: str, optional (default: 'celltype')
        The column name in adata.obs representing the cell ontology class.
    age_col: str, optional (default: 'age')
        The column name in adata.obs representing the age of the cells.
    filter_gender: boolen, optional (default: True)
        Whether you would like to filter a gender out
    gender: str, optional only if the filter_gender is True(default: "male")
        Choose which gender to keep
    age_threshold: float, optional (default: 0.8)
        The threshold fraction for filtering based on age distribution. If one age group has more than this
        fraction of cells in a class, the class will be filtered out.
    count_threshold: int, optional (default: None(100))
        Lower threshold for filtering cell types based on count.
 
    Returns:
    filtered_adata: AnnData object
        The filtered AnnData object with specified cell ontology classes removed based on both criteria."""
    try:
        adata = sc.read_h5ad(filepath)
        
        if filter_gender:
             filtered_adata = adata[adata.obs["sex"] == gender, :].copy()
                
        celltype_df = filtered_adata.obs[[class_col]].copy()
        age_df = filtered_adata.obs[[age_col]].copy()
        
        # Apply the cell count threshold filtering
        celltype_df = _filter_low_counts(celltype_df, age_df, class_col, count_threshold)
    
        # Create a filtered AnnData object based on cell count filtering
        filtered_adata = filtered_adata[celltype_df.index].copy()
        
        # Identify the skewed classes to filter based on age distribution
        classes_to_filter = _get_skewed_count_info(filtered_adata, class_col, age_col, age_threshold)
        
        if len(classes_to_filter):
            print(classes_to_filter[0], " has skewed cell counts")
            
        # Further filter the AnnData object based on age distribution
        final_filtered_adata = filtered_adata[~filtered_adata.obs[class_col].isin(classes_to_filter)].copy()
        
        return final_filtered_adata
    except Exception as e:
        raise(e)