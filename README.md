# Ames Housing Dataset: Comprehensive Machine Learning Evaluation Framework

## Overview

This repository implements a complete end-to-end machine learning pipeline for the Ames Housing Dataset, encompassing exploratory data analysis, feature engineering, and model evaluation across three distinct learning paradigms: regression, unsupervised clustering, and classification. The framework is designed to demonstrate best practices in data science workflows while addressing real-world challenges inherent in housing market data.

## Project Objectives

The primary goal is to conduct a systematic evaluation of machine learning approaches applied to residential property valuation, specifically:

- **Regression Analysis**: Predict continuous sale prices using multiple models and select the best performer
- **Clustering Analysis**: Identify latent market segments and property groupings without using the target variable
- **Classification Task**: Categorize properties into discrete price brackets (Low, Medium, High)

This multi-task approach enables comprehensive assessment of the dataset's characteristics and the comparative effectiveness of different modeling strategies.

## Dataset Description

The Ames Housing Dataset serves as a modern, more comprehensive alternative to the traditional Boston Housing Dataset, offering enhanced complexity and realism for academic and professional machine learning applications.

**Key Characteristics:**
- **Observations**: 2,930 residential property transactions (Ames, Iowa, 2006–2010)
- **Features**: 82 variables (43 categorical, 39 numerical) → 227 after preprocessing
- **Feature Types**: Mixed numerical and categorical attributes with varying cardinality
- **Data Quality**: Contains missing values in 19 features, outliers, and high-cardinality categorical variables
- **Target Variable**: `SalePrice` (continuous) — mean $180,921, median $163,000, skewness 1.88

## Methodology

### 1. Exploratory Data Analysis (EDA)

The initial analysis phase encompasses:

- Univariate and multivariate distribution analysis (histogram, boxplot, KDE, Q-Q plot)
- Target variable distribution characterization before and after log-transformation
- Missing value pattern detection and quantification (top-30 features)
- Correlation structure examination (top-15 features, heatmap, pairplot)
- PCA scree plot on raw numerical features
- Permutation importance analysis (top-15 numerical features)

### 2. Data Preprocessing and Feature Engineering

A robust preprocessing pipeline structured in 10 sequential steps:

**Data Cleaning:**
- Removal of non-predictive identifier columns (`Order`, `PID`)
- Column renaming for Python compatibility (`1st Flr SF` → `FirstFlrSF`, etc.)

**Feature Engineering (6 derived variables):**
- `TotalSF`: sum of first floor + second floor + basement square footage
- `HouseAge`: years since construction (Yr Sold − Year Built)
- `SinceRemod`: years since last remodeling (Yr Sold − Year Remod/Add)
- `HasPool`, `HasFireplace`, `HasGarage`: binary presence flags (0/1)

**Target Transformation:**
- Logarithmic transformation: `SalePrice_log = log1p(SalePrice)` (applied when |skewness| > 0.75)
- Reduces skewness from 1.88 to ~0.12

**Missing Value Imputation:**
- Numerical features: median imputation (robust to outliers)
- Categorical features: explicit `"Missing"` category (preserves absence information)

**Outlier Treatment:**
- IQR clipping: values outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR] are clipped to bounds
- All 2,930 observations preserved (no row removal)

**Encoding Strategies:**
- **Low-cardinality categoricals (≤10 unique values)**: One-Hot Encoding (`drop_first=True` to avoid dummy variable trap)
- **High-cardinality categoricals (>10 unique values)**: Target Encoding Out-of-Fold (mean of `SalePrice` computed on training folds only, preventing data leakage)

**Standardization:**
- Z-score normalization (StandardScaler) on all numerical features
- Required for linear models (Ridge, Lasso); neutral for tree-based models

**Result**: dataset expands from 82 to **227 features**, fully imputed and standardized.

### 3. Feature Selection

Feature selection employs **Mutual Information Regression** to identify the **top-40 features** relative to `SalePrice_log`. This approach:

- Captures non-linear dependencies (unlike Pearson correlation)
- Is model-agnostic (filter-based method)
- Reduces dimensionality from 227 to 40 features

Top-10 features identified: `TotalSF`, `Overall Qual`, `Neighborhood_encoded`, `Gr Liv Area`, `Garage Area`, `Year Built`, `TotalBsmtSF`, `Garage Cars`, `FirstFlrSF`, `HouseAge`.

## Machine Learning Tasks

### Regression

**Objective**: Predict `SalePrice_log` from the top-40 selected features

**Models compared** (5-Fold Cross-Validation):

| Model | R² | RMSE (log) | MAE (log) |
|---|---|---|---|
| Ridge (α=10) | 0.8912 ± 0.022 | 0.1333 | 0.0884 |
| Lasso (α=0.001) | 0.8912 ± 0.023 | 0.1332 | 0.0884 |
| Random Forest (200 trees) | 0.8859 ± 0.024 | 0.1362 | 0.0903 |
| **Gradient Boosting** (200 trees, lr=0.05) | **0.8977 ± 0.020** | **0.1292** | **0.0859** |

**Best model**: Gradient Boosting Regressor (R²=0.8977), corresponding to a typical error of ±13.8% on the real price.

**Evaluation Metrics:**
- Coefficient of Determination (R²)
- Root Mean Squared Error (RMSE, log-scale)
- Mean Absolute Error (MAE, log-scale)
- Feature importance analysis (permutation importance, Random Forest)

### Clustering

**Objective**: Discover latent market segments without using `SalePrice`

**Methodology:**
- Feature selection: top-15 features by intrinsic variance (not MI, to avoid supervised bias)
- Scaling: RobustScaler (median/IQR-based, robust to residual outliers)
- Dimensionality reduction: PCA to 3 principal components (33.4% variance explained)
- Primary algorithm: K-Means (k=3, n_init=100, max_iter=500)
- Alternative: DBSCAN with grid search on eps and min_samples
- Optimal k determined by convergence of three criteria: Elbow Method, Silhouette Score, Davies-Bouldin Index

**Results (k=3):**

| Metric | Value |
|---|---|
| Silhouette Score | 0.8776 (Excellent) |
| Davies-Bouldin Index | 0.2415 (Excellent) |

**Cluster interpretation (external validation via SalePrice mean):**
- Cluster 2 — Entry-level: $139,896 avg price
- Cluster 0 — Standard: $170,757 avg price
- Cluster 1 — Premium: $224,270 avg price

### Classification

**Objective**: Categorize properties into price tiers (Low / Medium / High)

**Configuration:**
- Target: `SalePrice_log` discretized into 3 equipartite classes via quantile binning
- Class balance ratio: 0.988 (near-perfect balance, no resampling needed)
- Algorithm: Random Forest Classifier (200 trees)
- Validation: 5-Fold Cross-Validation

**Results:**

| Metric | Value |
|---|---|
| Accuracy | 84.81% ± 1.86% |
| F1-Score (weighted) | 84.82% ± 1.85% |
| Baseline (majority class) | 33.48% |

**Per-class performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Low (0) | 0.86 | 0.88 | 0.87 |
| Medium (1) | 0.77 | 0.77 | 0.77 |
| High (2) | 0.91 | 0.89 | 0.90 |

## Results Summary

| Task | Primary Metric | Score | Verdict |
|---|---|---|---|
| Regression (GBM) | R² | 0.8977 | Excellent |
| Clustering | Silhouette Score | 0.8776 | Excellent |
| Classification | F1-Score (weighted) | 0.8482 | Good |

Key finding: `TotalSF`, `Overall Qual`, and `Neighborhood` are the dominant features across all three tasks, confirming that size, quality, and location are the fundamental drivers of housing market value.

## Technical Requirements

**Environment:**
- Python 3.8 or higher

**Dependencies:**
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms and tools
- `scipy`: Scientific computing and statistics
- `matplotlib`: Visualization
- `seaborn`: Statistical data visualization

**Installation:**
```bash
pip install -r requirements.txt
```

## Execution Instructions

To execute the complete analytical pipeline:

```bash
python housing_sentinel_v2.py
```

The script automatically locates `AmesHousing.csv` in the same directory or in `../dataset/`. Generated visualizations are saved in the `images/` folder (created automatically).

The script performs the following operations sequentially:
1. Dataset loading and shape reporting
2. Preprocessing and feature engineering (82 → 227 features)
3. Feature selection via Mutual Information (top-40)
4. Regression: training and comparison of 4 models with 5-Fold CV
5. Clustering: K-Means with optimal k selection (Elbow + Silhouette + DBI)
6. Classification: Random Forest with confusion matrix and classification report
7. EDA visualizations (12 images saved to `images/`)
8. Consolidated results summary printed to console

## Output

The script produces 12 visualizations in the `images/` folder:

| File | Content |
|---|---|
| `01_target_distribution.png` | SalePrice distribution (6 plots) |
| `02_missing_values.png` | Missing values by feature |
| `03_correlations.png` | Top-15 correlations + heatmap |
| `04_scatter_top_features.png` | Pairplot top-6 features vs SalePrice |
| `05_pca_scree.png` | PCA scree plot (raw numerical features) |
| `06_kmeans_selection.png` | Elbow + Silhouette (EDA clustering) |
| `07_kmeans_clusters.png` | Cluster scatter PC1 vs PC2 (EDA) |
| `08_feature_importance.png` | Permutation importance top-15 |
| `09_regression_comparison.png` | R² and RMSE comparison across 4 models |
| `10_kmeans_selection.png` | Elbow + Silhouette + DBI (main clustering) |
| `11_kmeans_clusters.png` | Cluster scatter PC1-PC2 and PC1-PC3 |
| `12_confusion_matrix.png` | Classification confusion matrix |

## Future Enhancements

Potential extensions to enhance the framework:

- **Hyperparameter Optimization**: Grid Search or Bayesian Optimization for Ridge α, GBM learning rate and depth
- **Model Interpretability**: SHAP (SHapley Additive exPlanations) values for individual prediction explanation
- **Ensemble Methods**: Stacking Ridge + GBM + Random Forest for potential performance gains
- **Interactive Presentation**: Jupyter Notebook implementation for educational and demonstrative purposes
- **Pipeline Integration**: Fully integrated scikit-learn Pipeline for production deployment with proper train/test leakage prevention
- **External Validation**: Testing on a geographically or temporally distinct housing dataset

## License

This project is intended for academic and educational purposes.
