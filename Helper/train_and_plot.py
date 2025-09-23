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

def test_model_across_fold(donor_kf, donor_ids, unique_donors, raw_cells, ages, optimal_index, optimal_pred_freqs, freq_age_group):
    train_ages = []
    store_train_predict = []
    test_ages = []
    store_test_predict = []
    
    #store the full predicted_ages for plotting 
    full_predicted_ages = [np.nan] * raw_cells.shape[1]
    
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
        
        # Store for plotting
        train_ages.extend(y_train)
        store_train_predict.extend(train_predicted_ages)
        test_ages.extend(y_test)
        store_test_predict.extend(test_predicted_ages)
        
        # store the full predicted_ages for plotting 
        np_train_idx = np.where(train_mask)[0]
        np_test_idx = np.where(test_mask)[0]
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
    
    
def _plot_cell_count(ax, train_cellCount_dict, test_cellCount_dict):
    ages = sorted(list(train_cellCount_dict.keys()))
    
    # Convert the dictionaries to lists of values for seaborn
    train_counts = list(train_cellCount_dict.values())
    test_counts = list(test_cellCount_dict.values())
    bar_width = 0.4
    # Set position of bars on X axis
    r1 = np.arange(len(ages))
    r2 = [x + bar_width for x in r1]
    # Use different colors for each x value
    unique_colors = sns.color_palette("Set2", len(ages))
    for i in range(len(ages)):
        ax.bar(r1[i], train_counts[i], color=unique_colors[i], width=bar_width, edgecolor='grey', label=f'Train (Age {ages[i]})')
        ax.bar(r2[i], test_counts[i], color=unique_colors[i], alpha=0.7, width=bar_width, edgecolor='grey', label=f'Test (Age {ages[i]})')
    
    ax.set_xlabel('Age', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title('Num of Cells for Each Age (Train vs Test)', fontsize=16)
    ax.set_xticks([r + bar_width / 2 for r in range(len(ages))])
    ax.set_xticklabels(ages)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Add text labels above bars for train and test counts
    for i, value in enumerate(train_counts):
        ax.text(r1[i], value, f'{int(value)}', ha='center', va='bottom', fontsize=10, color='black')
    
    for i, value in enumerate(test_counts):
        ax.text(r2[i], value, f'{int(value)}', ha='center', va='bottom', fontsize=10, color='black')
        
def _record_r_squared(file_path, celltype, r_squared):
    # Create a new DataFrame with celltype as the index and r_squared value
    df = pd.DataFrame({'R_squared': [r_squared]}, index=[celltype])
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        # If the file does not exist, write the new DataFrame to a CSV file
        df.to_csv(file_path, mode='w')
    else:
        # If the file exists, append the new data
        df.to_csv(file_path, mode='a', header=False)
