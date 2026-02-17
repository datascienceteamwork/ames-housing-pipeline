# Ames Housing Dataset: Comprehensive Machine Learning Evaluation Framework

## Overview

This repository implements a complete end-to-end machine learning pipeline for the Ames Housing Dataset, encompassing exploratory data analysis, feature engineering, and model evaluation across three distinct learning paradigms: regression, unsupervised clustering, and classification. The framework is designed to demonstrate best practices in data science workflows while addressing real-world challenges inherent in housing market data.

## Project Objectives

The primary goal is to conduct a systematic evaluation of machine learning approaches applied to residential property valuation, specifically:

- **Regression Analysis**: Predict continuous sale prices of residential properties
- **Clustering Analysis**: Identify latent market segments and property groupings
- **Classification Task**: Categorize properties into discrete price brackets (Low, Medium, High)

This multi-task approach enables comprehensive assessment of the dataset's characteristics and the comparative effectiveness of different modeling strategies.

## Dataset Description

The Ames Housing Dataset serves as a modern, more comprehensive alternative to the traditional Boston Housing Dataset, offering enhanced complexity and realism for academic and professional machine learning applications.

**Key Characteristics:**
- **Features**: 80+ descriptive variables per property
- **Feature Types**: Mixed numerical and categorical attributes
- **Data Quality**: Contains missing values, outliers, and high-cardinality categorical variables
- **Target Variable**: `SalePrice` (continuous)
- **Source**: Located in `data/AmesHousing.csv`

The dataset presents realistic challenges including data quality issues, feature heterogeneity, and complex relationships between predictors and target variables.

**Directory Descriptions:**
- `data/`: Raw dataset files
- `images/`: Generated visualizations organized by analysis type
- `src/`: Core Python implementation
- `report/`: Technical documentation in LaTeX format
- `requirements.txt`: Python dependency specifications

## Methodology

### 1. Exploratory Data Analysis (EDA)

The initial analysis phase encompasses:

- Univariate and multivariate distribution analysis
- Data type identification and cardinality assessment
- Missing value pattern detection and quantification
- Target variable (`SalePrice`) distribution characterization
- Statistical moment analysis (skewness, kurtosis)
- Correlation structure examination

### 2. Data Preprocessing and Feature Engineering

A robust preprocessing pipeline designed to prevent data leakage includes:

**Data Cleaning:**
- Removal of non-predictive identifier columns
- Outlier detection and treatment via Interquartile Range (IQR) clipping

**Feature Engineering:**
- Derived features: `TotalSF` (total square footage), `HouseAge`, `SinceRemod` (time since remodeling)
- Logarithmic transformation of target variable to reduce skewness

**Missing Value Imputation:**
- Numerical features: Median imputation
- Categorical features: Dedicated "Missing" category

**Encoding Strategies:**
- **Low-cardinality categoricals**: One-Hot Encoding
- **High-cardinality categoricals**: Frequency Encoding combined with out-of-fold Target Encoding

**Standardization:**
- Zero-variance feature removal
- Z-score normalization of numerical features

### 3. Feature Selection

Feature selection employs Mutual Information Regression to identify the k most informative features relative to the log-transformed target variable. This approach:

- Reduces dimensionality and computational complexity
- Enhances model generalization capability
- Improves cross-validation stability
- Mitigates multicollinearity effects

## Machine Learning Tasks

### Regression

**Objective**: Predict continuous sale prices

**Model Architecture:**
- Algorithm: Random Forest Regressor
- Validation: K-Fold Cross-Validation

**Evaluation Metrics:**
- Coefficient of Determination (R²)
- Root Mean Squared Error (RMSE, log-scale)
- Mean Absolute Error (MAE, log-scale)
- Feature importance analysis

### Clustering

**Objective**: Discover latent market segments

**Methodology:**
- Dimensionality reduction: Principal Component Analysis (85% variance retention)
- Primary algorithm: K-Means clustering
- Alternative approach: DBSCAN (density-based clustering)

**Evaluation Metrics:**
- Silhouette Score (cluster cohesion and separation)
- Davies-Bouldin Index (cluster compactness)

### Classification

**Objective**: Categorize properties into price tiers

**Configuration:**
- Target classes: Low, Medium, High price brackets
- Algorithm: Random Forest Classifier
- Validation: K-Fold Cross-Validation

**Evaluation Metrics:**
- Classification Accuracy
- Weighted F1-Score
- Precision and Recall
- Baseline comparison (most frequent class predictor)

## Results Summary

Empirical evaluation demonstrates that the Ames Housing Dataset exhibits:

- **Strong Regression Performance**: High predictive accuracy attributable to feature richness and information content
- **Effective Classification**: Clear separability between price tiers enables robust categorization
- **Moderate Clustering Structure**: Latent patterns exist but are less pronounced compared to supervised learning tasks

Detailed quantitative results, including performance metrics and statistical significance tests, are documented in execution logs and the technical report.

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
python src/ames_housing_evaluator.py
```
to install dipendencies:
''' bash 
pip install -r requirements.txt
'''

The script performs the following operations sequentially:
1. Dataset loading and initial inspection
2. Preprocessing and feature engineering
3. Feature selection
4. Regression model training and evaluation
5. Clustering analysis
6. Classification model assessment
7. Consolidated results reporting

## Future Enhancements

Potential extensions to enhance the framework:

- **Hyperparameter Optimization**: Grid Search or Bayesian Optimization for improved model performance
- **Model Comparison**: Evaluation of linear models (Ridge, Lasso) and gradient boosting algorithms (XGBoost, LightGBM)
- **Model Interpretability**: SHAP (SHapley Additive exPlanations) values for feature contribution analysis
- **Interactive Presentation**: Jupyter Notebook implementation for educational and demonstrative purposes
- **Pipeline Integration**: Fully integrated scikit-learn Pipeline for production deployment
- **Ensemble Methods**: Stacking and blending strategies for prediction aggregation

## License

This project is intended for academic and educational purposes.

