from scipy import sparse
from sklearn.utils import sparsefuncs
import pandas as pd
import numpy as np

def normalization(cell_array):
    df = pd.DataFrame(cell_array)
    X = sparse.csr_matrix(df.values, dtype='float64')
    counts_per_cell = X.sum(axis = 1)  # original counts per cell
    counts_per_cell = np.ravel(counts_per_cell)
    counts_greater_than_zero = counts_per_cell[counts_per_cell > 0]
    after = np.median(counts_greater_than_zero, axis = 0)
    counts_per_cell += counts_per_cell == 0
    counts_per_cell = counts_per_cell / after
    sparsefuncs.inplace_row_scale(X, 1 / counts_per_cell)
    result = pd.DataFrame.sparse.from_spmatrix(X)
    result.columns = df.columns
    return result