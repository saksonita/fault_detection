# Feature Importance Analysis for "파단직전 =1" Condition

This repository contains a comprehensive analysis pipeline for identifying the important features that could cause the "파단직전 =1" (imminent fracture) condition in manufacturing processes.

## Analysis Workflow

The analysis is organized into five sequential Python scripts that form a complete pipeline:

1. **Data Exploration** (`01_data_exploration.py`): Examines the dataset structure, checks for missing values, and analyzes feature distributions.
2. **Correlation Analysis** (`02_correlation_analysis.py`): Identifies correlations between features and the target variable, as well as between different features.
3. **Feature Selection** (`03_feature_selection.py`): Applies various feature selection methods including filter methods (ANOVA F-value, Mutual Information), wrapper methods (RFE), and embedded methods (Random Forest, Gradient Boosting).
4. **Model Building** (`04_model_building.py`): Builds and evaluates multiple classification models with SMOTE to handle class imbalance.
5. **Feature Importance Analysis** (`05_feature_importance_analysis.py`): Analyzes feature importance results from different models and methods to identify the most significant features.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (imblearn)

You can install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## Dataset

The analysis requires the dataset file `엔솔_전처리_파단직전재추출_0401.csv` to be in the same directory as the scripts. This dataset contains various manufacturing process parameters and a binary target variable `파단직전` indicating imminent fracture conditions.

## Running the Analysis

The scripts should be run in sequence:

```bash
# 1. Data Exploration
python 01_data_exploration.py

# 2. Correlation Analysis
python 02_correlation_analysis.py

# 3. Feature Selection
python 03_feature_selection.py

# 4. Model Building
python 04_model_building.py

# 5. Feature Importance Analysis
python 05_feature_importance_analysis.py
```

Alternatively, you can run all scripts in sequence using:

```bash
for script in 01_data_exploration.py 02_correlation_analysis.py 03_feature_selection.py 04_model_building.py 05_feature_importance_analysis.py; do
    python "$script"
done
```

## Output

The analysis generates various outputs in the `analysis` directory:

- **Visualizations**: Feature distributions, boxplots, correlation heatmaps, feature importance plots, ROC curves, and precision-recall curves.
- **Data Files**: Feature rankings, importance scores, and model evaluation results.
- **Summary Files**: Comprehensive summaries of the analysis results, including the most important features identified.

## Key Findings

The analysis identifies several key features that are strongly associated with the "파단직전 =1" condition:

1. GAP-related features ([D0006862] 작업자측 GAP, [D0006866] 작업자측 이전 GAP)
2. Rewinder E.P.C. center ([M005685] 리와인더 E.P.C 중앙)
3. Temperature differences (유도가열온도WS차이)
4. Tension differences (리와인더장력차이)

These features consistently showed high importance across different analytical methods and should be closely monitored to predict and prevent imminent fracture conditions.

## Script Details

### 1. Data Exploration (`01_data_exploration.py`)

This script performs initial exploratory data analysis:
- Loads the dataset and displays basic information
- Checks for missing values
- Analyzes the target variable distribution
- Creates visualizations of feature distributions
- Generates boxplots to check for outliers
- Calculates statistics to understand feature relationships with the target

### 2. Correlation Analysis (`02_correlation_analysis.py`)

This script analyzes correlations:
- Calculates correlations between features and the target variable
- Creates a correlation heatmap for the top features
- Generates scatter plots for the most correlated features
- Identifies highly correlated feature pairs (multicollinearity)

### 3. Feature Selection (`03_feature_selection.py`)

This script applies multiple feature selection methods:
- Filter methods: ANOVA F-value and Mutual Information
- Wrapper methods: Recursive Feature Elimination (RFE)
- Embedded methods: Random Forest and Gradient Boosting importance
- Combines rankings from different methods to identify consistently important features

### 4. Model Building (`04_model_building.py`)

This script builds and evaluates classification models:
- Implements Logistic Regression, Random Forest, Gradient Boosting, and SVM
- Uses SMOTE to handle class imbalance
- Evaluates models using accuracy, ROC-AUC, PR-AUC, and cross-validation
- Extracts feature importance from each model
- Creates visualizations of model performance and feature importance

### 5. Feature Importance Analysis (`05_feature_importance_analysis.py`)

This script performs a comprehensive analysis of feature importance:
- Combines and normalizes feature importance from different models
- Calculates average importance across models
- Identifies features that are consistently important across all methods
- Creates visualizations comparing feature importance
- Generates a final list of the most important features

## Recommendations

Based on the analysis, we recommend:

1. **Implement Real-time Monitoring**: Set up real-time monitoring systems for the top identified features.
2. **Establish Alert Thresholds**: Define threshold values for critical features that, when exceeded, trigger alerts for potential imminent fracture conditions.
3. **Develop Predictive Maintenance System**: Implement a predictive maintenance system using the Random Forest model, which showed the best performance.
4. **Process Optimization**: Review and optimize processes related to GAP control, rewinder edge position control, temperature uniformity, and tension control.
5. **Further Investigation**: Conduct further investigation into the physical mechanisms by which these features influence fracture conditions.
