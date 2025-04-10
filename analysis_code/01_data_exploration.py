
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

import seaborn as sns
import os

# Set up directories
if not os.path.exists('analysis'):
    os.makedirs('analysis')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('../data/엔솔_전처리_파단직전재추출_0401.csv')

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Shape: {df.shape}")
print("\nColumn names:")
print(df.columns.tolist())

# Check for missing values
print("\nMissing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")

# Display basic statistics
print("\nBasic statistics:")
print(df.describe())

# Check target variable distribution
print("\nTarget variable distribution:")
target_counts = df['파단직전'].value_counts()
print(target_counts)
print(f"Percentage of positive cases: {target_counts[1] / len(df) * 100:.4f}%")

# Create a balanced sample for visualization
# (using all positive cases and an equal number of negative cases)
positive_samples = df[df['파단직전'] == 1]
negative_samples = df[df['파단직전'] == 0].sample(n=len(positive_samples), random_state=42)
sample_df = pd.concat([positive_samples, negative_samples], axis=0).reset_index(drop=True)

print(f"\nCreated balanced sample with {len(sample_df)} rows for visualization")

# Identify numeric columns for analysis
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if '파단직전' in numeric_cols:
    numeric_cols.remove('파단직전')  # Remove target from feature list

print(f"\nAnalyzing {len(numeric_cols)} numeric features")

# Create histograms for feature distributions
print("\nCreating feature distribution plots...")
plt.figure(figsize=(20, 15))
for i, col in enumerate(numeric_cols[:25]):  # Limit to first 25 features
    plt.subplot(5, 5, i+1)
    try:
        sns.histplot(data=sample_df, x=col, hue='파단직전', bins=30, kde=True)
    except Exception as e:
        # Fall back to simple histogram if KDE fails
        sns.histplot(data=sample_df, x=col, hue='파단직전', bins=30, kde=False)
    plt.title(col.split(']')[-1] if ']' in col else col)
    plt.tight_layout()
plt.savefig('analysis/feature_distributions.png')
print("Saved feature distributions to analysis/feature_distributions.png")

# Check for outliers using box plots
print("\nCreating boxplots to check for outliers...")
plt.figure(figsize=(20, 15))
for i, col in enumerate(numeric_cols[:25]):  # Limit to first 25 features
    plt.subplot(5, 5, i+1)
    sns.boxplot(data=sample_df, x='파단직전', y=col)
    plt.title(col.split(']')[-1] if ']' in col else col)
    plt.tight_layout()
plt.savefig('analysis/feature_boxplots.png')
print("Saved feature boxplots to analysis/feature_boxplots.png")

# Analyze feature relationships with target
print("\nAnalyzing feature relationships with target variable...")

# Calculate mean values for each feature grouped by target
feature_means = sample_df.groupby('파단직전')[numeric_cols].mean()
print('Feature means by target class:')
print(feature_means)

# Calculate standard deviation for each feature grouped by target
feature_stds = sample_df.groupby('파단직전')[numeric_cols].std()
print('\nFeature standard deviations by target class:')
print(feature_stds)

# Calculate the absolute difference in means between classes
mean_diffs = abs(feature_means.loc[1.0] - feature_means.loc[0.0])
mean_diffs = mean_diffs.sort_values(ascending=False)
print('\nAbsolute differences in means between classes (top 10):')
print(mean_diffs.head(10))

# Calculate the ratio of standard deviations between classes
std_ratios = feature_stds.loc[1.0] / feature_stds.loc[0.0]
std_ratios = std_ratios.sort_values(ascending=False)
print('\nRatios of standard deviations between classes (top 10):')
print(std_ratios.head(10))

# Save the results to a file
with open('analysis/eda_results.txt', 'w', encoding='utf-8') as f:
    f.write('Feature means by target class:\n')
    f.write(str(feature_means))
    f.write('\n\nFeature standard deviations by target class:\n')
    f.write(str(feature_stds))
    f.write('\n\nAbsolute differences in means between classes (top 10):\n')
    f.write(str(mean_diffs.head(10)))
    f.write('\n\nRatios of standard deviations between classes (top 10):\n')
    f.write(str(std_ratios.head(10)))

print("\nSaved EDA results to analysis/eda_results.txt")
print("\nExploratory Data Analysis completed successfully!")
