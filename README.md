
# scBayesAge

scBayesAge is a single-cell RNA transcriptomic cell-type-specific clock that can help researchers study the biological ages of organisms in the single-cell level. 

![](https://github.com/DaisyCuttie/scBayesAge/blob/8cf528407b365c618fe25f4fffe6bb514c5aefad/analysis/Figures/Histogram_of_R-squared_by_Celltype.png)
![Limb-muscle cells]([https://github.com/DaisyCuttie/scBayesAge/blob/main/paper_results/Model_Figures/Limb-Muscle_skeletal muscle satellite cell_model.png](https://github.com/DaisyCuttie/scMLEAge/blob/main/paper_results/Model_Figures/Limb-Muscle_skeletal%20muscle%20satellite%20cell_model.png))



The examples here are the applications to the Tabula Muris Senis data https://rdcu.be/eJI2Y that you can download from here https://figshare.com/articles/dataset/Processed_files_to_use_with_scanpy_/8273102/2




## How to run scMLEAge
The example run is shown in the Tutorial.ipynb file.




## Helper Module Documentation

The `Helper/` directory contains essential utility modules for single-cell aging analysis, providing core data processing, modeling, and visualization functionality. **`Load_Anndata.py`** handles data ingestion and quality filtering, implementing robust preprocessing pipelines that filter out low-count cell types and age-skewed populations while supporting gender-specific filtering. The **`Process_Anndata.py`** module defines the `AnnObject` class, which encapsulates comprehensive single-cell data processing workflows including gene filtering, normalization, PCA analysis, and UMAP visualization with automatic parameter optimization. **`utils.py`** provides specialized computational functions for aging-related analysis, including gene expression normalization, age-correlated gene identification through Pearson correlation analysis. The **`train_and_plot.py`** module implements the core machine learning pipeline and publication-ready violin plots for visualizing prediction accuracy across different age groups. 


+++ This method facilitate as a cohesive framework for analyzing cellular aging patterns, enabling researchers to identify age-associated gene expression changes, train predictive models, and generate statistical visualizations for aging research applications.
