from scipy.stats import poisson
import math
from numpy import inf
import numpy as np
from matplotlib.pyplot import figure
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import os
from collections import defaultdict
import seaborn as sns
import pandas as pd
import csv
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import KFold
from . import utils as util

def initialize_output_file(organ, r_squareds_dir="./r_squareds"):
    os.makedirs(r_squareds_dir, exist_ok=True)
    file_path = os.path.join(r_squareds_dir, f"{organ}_r_squared_summary.csv")
    print(file_path)
    if os.path.isfile(file_path):
        print("Deleting previous r_squared file")
        os.remove(file_path)
    return file_path


def write_feature_summary(organ, num_features, output_dir="./Num_Features"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{organ}_num_features_selected.csv")
    with open(filepath, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["celltype", "num_features"])
        writer.writeheader()
        writer.writerows(num_features)
        
        
def _donor_folds(donor_ids: np.ndarray, unique_donors: np.ndarray, donor_kf: KFold):
    for train_idx, test_idx in donor_kf.split(unique_donors):
        train_d = unique_donors[train_idx]
        test_d  = unique_donors[test_idx]
        yield np.isin(donor_ids, train_d), np.isin(donor_ids, test_d)

        
def _test_hyperparameter_by_gene_counts(
    donor_kf: KFold,
    donor_ids: np.ndarray,
    unique_donors: np.ndarray,
    raw_counts: np.ndarray,                  # genes x cells
    ages: np.ndarray,                        # cells
    pred_freq_matrices_celltype: np.ndarray,     # genes x freq_dim
    correlated_index_list: [int],
    age_groups_for_celltype,
    gene_counts: [int],
):
    """Return {gene_count: mean train R² across donor folds}."""
    max_genes = pred_freq_matrices_celltype.shape[0]
    corr_idx = [ix for ix in correlated_index_list if ix < max_genes]

    folds = list(_donor_folds(donor_ids, unique_donors, donor_kf))
    out = {} # Dict[int, float] 

    for g in gene_counts:
        selected_idx = corr_idx[:g]
        if not selected_idx:
            out[g] = 0.0
            continue
        pred_freqs = pred_freq_matrices_celltype[selected_idx, :]
        fold_r2s = []
        for train_mask, _ in folds:
            X_train = raw_counts[selected_idx, :][:, train_mask].T  # cells x genes
            y_train = ages[train_mask]
            yhat = _run_poisson(pred_freqs, X_train, y_train, age_groups_for_celltype)
            fold_r2s.append(util.r2(y_train, yhat))
        out[g] = sum(fold_r2s) / len(fold_r2s)
    return out


def _pick_optimal_gene_count(r2_by_count: dict[int, float]) -> int:
    """Pick gene_count with max R²; tie-break to smaller count."""
    items = sorted(r2_by_count.items(), key=lambda kv: kv[0])  # stable tie-break by smaller count
    return max(items, key=lambda kv: kv[1])[0]


def _run_model_across_fold(donor_kf, donor_ids, unique_donors, raw_cells, ages, optimal_index, optimal_pred_freqs, freq_age_group):
    train_ages = []
    store_train_predict = []
    test_ages = []
    store_test_predict = []
    
    #store the full predicted_ages for plotting 
    full_predicted_ages = [np.nan] * raw_cells.shape[1]
    n_cells = raw_cells.shape[1]
    cell_role = np.array(["none"] * n_cells, dtype=object)
    
    for fold_idx, (train_idx, test_idx) in enumerate(donor_kf.split(unique_donors)):
        train_donors = unique_donors[train_idx]
        test_donors = unique_donors[test_idx]
        train_mask = np.isin(donor_ids, train_donors)
        test_mask = np.isin(donor_ids, test_donors)
        X_train = raw_cells[:, train_mask].T[:, optimal_index]
        X_test = raw_cells[:, test_mask].T[:, optimal_index]
        y_train = ages[train_mask]
        y_test = ages[test_mask]
        # Run the poisson function
        train_predicted_ages = _run_poisson(
            optimal_pred_freqs, X_train, y_train, freq_age_group
        )
        test_predicted_ages = _run_poisson(
            optimal_pred_freqs, X_test, y_test, freq_age_group
        )
        
        np_train_idx = np.where(train_mask)[0]
        np_test_idx  = np.where(test_mask)[0]
        
        # Store for plotting
        train_ages.extend(y_train)
        store_train_predict.extend(train_predicted_ages)
        test_ages.extend(y_test)
        store_test_predict.extend(test_predicted_ages)
        
        # store the full predicted_ages for plotting 
        for idx, pred in zip(np_train_idx, train_predicted_ages):
            full_predicted_ages[idx] = pred
        for idx, pred in zip(np_test_idx, test_predicted_ages):
            full_predicted_ages[idx] = pred
            
    return train_ages, store_train_predict, test_ages, store_test_predict, full_predicted_ages


def _run_poisson(pred_freqs, raw_cells, train_ages, freq_age_group):
    predicted_ages = []
    ### looping through all the age possibilities 
    for index in range(raw_cells.shape[0]):
        cell = raw_cells[index, :]
        cell_sum = np.sum(cell, axis = 0)
        _lambda = pred_freqs * cell_sum # frequency of that gene expression * counts of all genes in that cell
        prob = poisson.pmf(cell.reshape(-1,1), _lambda)
        max_value = np.log(prob)
        max_value[max_value == -inf] = 0
        pred_age = freq_age_group[np.argmax(np.sum(max_value, axis = 0))]
        predicted_ages.append(pred_age)
    return predicted_ages


def _store_predictions(cell_group, indices, predicted_ages, pred_age_plot):
    for idx, pred_age in zip(indices, predicted_ages):
        cell_name = cell_group[idx]
        if cell_name in pred_age_plot:
            print("Already there")
        else:
            pred_age_plot[cell_name] = pred_age
    return pred_age_plot


def run_celltype_pipeline(
    celltype: str,
    cell_group: {str},
    adataObject,
    pred_freq_matrices: {str, np.ndarray},
    correlated_index: {str, list[int]},
    freq_age_group: {str, np.ndarray},
    donor_kf: KFold,
    organ: str,
    r_square_file_path: str,
    fig_dir: str = "./Model_Figures",
    record = True,
    donor_col: str = "mouse.id"
) -> dict:

    raw_cells = np.array(adataObject.raw_counts[cell_group])          # genes x cells
    ages      = np.array([adataObject.age_dict[c] for c in cell_group])
    donor_ids = np.array([adataObject.processed_adata.obs.loc[c, donor_col] for c in cell_group])
    unique_donors = np.unique(donor_ids)

    corr_idx_list = list(correlated_index[celltype])
    gene_counts   = util.powers_of_two_counts(len(corr_idx_list))

    r2_curve = _test_hyperparameter_by_gene_counts(
        donor_kf, donor_ids, unique_donors,
        raw_counts=raw_cells,
        ages=ages,
        pred_freq_matrices_celltype=pred_freq_matrices[celltype],
        correlated_index_list=corr_idx_list,
        age_groups_for_celltype=freq_age_group[celltype],
        gene_counts=gene_counts
    )

    optimal_gene_count = _pick_optimal_gene_count(r2_curve)
    opt_idx = corr_idx_list[:optimal_gene_count]
    optimal_pred_freqs = pred_freq_matrices[celltype][opt_idx, :]

    train_ages, store_train_predict, test_ages, store_test_predict, full_predicted_ages = \
        _run_model_across_fold(donor_kf, donor_ids, unique_donors, raw_cells, ages,
                               opt_idx, optimal_pred_freqs, freq_age_group[celltype])

    # plot 3-panel summary
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6), dpi=600)
    _plot_violin(file_path=r_square_file_path, ax=ax1, x=train_ages, y=store_train_predict,
                 name=celltype, record=False)
    _plot_violin(file_path=r_square_file_path, ax=ax2, x=test_ages, y=store_test_predict,
                 name=celltype, record=record)
    _plot_cell_count(ax3,
                     dict(sorted(Counter(ages).items())))
    fig.text(0.1, 1, f"Model using {optimal_gene_count} genes", fontsize=15)
    fig.suptitle(f"{organ} {celltype} Model", fontsize=18, fontname="helvetica", y=1.03)
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)
    
    fig_path = f"{fig_dir}/{organ}_{celltype}_model.png"
    fig.savefig(fig_path, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close(fig)

    return {
        "celltype": celltype,
        "optimal_gene_count": int(optimal_gene_count),
        "r2_curve": {int(k): float(v) for k, v in r2_curve.items()},
        "figure_path": fig_path,
        "pred_index": cell_group,
        "pred_vals": full_predicted_ages
    }


def _plot_violin(file_path, ax, x, y, name, record = True, text_x = 0.38, text_y = 0.98):
    data = pd.DataFrame({
    'Ages': x,
    'Predicted': y})
    # Calculate MAE
    mae = mean_absolute_error(x, y)
    pearson_rsquared = pearsonr(x, y)[0]**2
    
    if record == True:
        _record_r_squared(file_path, name, pearson_rsquared)
    
    sns.violinplot(x='Ages', y='Predicted', data=data, palette='Set2', ax = ax)
    ax.set_ylim(bottom=0)
    ax.text(text_x, text_y, fr'MAE: {mae:.2f}, $R^2$: {pearson_rsquared:.2f}', transform=ax.transAxes, horizontalalignment='right', verticalalignment='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
    
    ax.set_ylabel("Predicted Age", fontsize = 14)
    ax.set_xlabel("True Age", fontsize = 14)
    if record == True:
        ax.set_title(f"{name} test model", fontsize = 16, fontname = "helvetica")
    else:
        ax.set_title(f"{name} train model", fontsize = 16, fontname = "helvetica")
    
    
def _plot_cell_count(ax, age_dict):
    ages = sorted(list(age_dict.keys()))
    age_counts = [age_dict[a] for a in ages]
    
    # Use different matching colors for each x value
    x = np.arange(len(ages))
    colors = sns.color_palette("Set2", len(ages))

    bars = ax.bar(x, age_counts, width=0.7, color=colors, edgecolor='grey')
    ax.set_xlabel('Age', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title('Num of Cells for Each Age', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in ages])

    # Add text labels above bars for train and test counts
    for i, value in enumerate(age_counts):
        ax.text(i, value, f'{int(value)}', ha='center', va='bottom', fontsize=10, color='black')
     
    
def _record_r_squared(file_path, celltype, r_squared):
    # Create a new DataFrame with celltype as the index and r_squared value
    df = pd.DataFrame({'R_squared': [r_squared]}, index=[celltype])
    if not os.path.isfile(file_path):
        df.to_csv(file_path, mode='w')
    else:
        df.to_csv(file_path, mode='a', header=False)
