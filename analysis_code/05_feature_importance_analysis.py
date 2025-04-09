"""
Feature Importance Analysis Script for "파단직전 =1" Condition Analysis

This script analyzes feature importance results from different models and methods
to identify the most significant features for predicting the "파단직전 =1" condition.

Usage:
    python feature_importance_analysis.py

Author: Manus AI
Date: April 9, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up directories
if not os.path.exists('analysis'):
    os.makedirs('analysis')

print("Loading feature importance results from previous steps...")

# Load feature importance results from previous steps
rf_importance = pd.read_csv('analysis/random_forest_importance.csv')
gb_importance = pd.read_csv('analysis/gradient_boosting_importance.csv')
combined_ranks = pd.read_csv('analysis/combined_feature_ranks.csv')

# Read model evaluation results
try:
    with open('analysis/model_evaluation_results.txt', 'r') as f:
        model_results = f.read()
    print("Loaded model evaluation results")
except FileNotFoundError:
    print("Warning: Model evaluation results file not found")
    model_results = ""

print("\nCreating comprehensive feature importance analysis...")

# Combine feature importance from different models
feature_importance_df = pd.DataFrame()

# Add Random Forest importance
rf_importance['Model'] = 'Random Forest'
feature_importance_df = pd.concat([feature_importance_df, rf_importance])

# Add Gradient Boosting importance
gb_importance['Model'] = 'Gradient Boosting'
feature_importance_df = pd.concat([feature_importance_df, gb_importance])

# Normalize importance scores within each model
for model in feature_importance_df['Model'].unique():
    mask = feature_importance_df['Model'] == model
    feature_importance_df.loc[mask, 'Normalized_Importance'] = feature_importance_df.loc[mask, 'Importance'] / feature_importance_df.loc[mask, 'Importance'].max()

# Calculate average normalized importance across models
avg_importance = feature_importance_df.groupby('Feature')['Normalized_Importance'].mean().reset_index()
avg_importance = avg_importance.sort_values('Normalized_Importance', ascending=False)

print("Top 15 features by average normalized importance:")
print(avg_importance.head(15))

# Create a visualization of average feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Normalized_Importance', y='Feature', data=avg_importance.head(15))
plt.title('Average Normalized Feature Importance Across Models')
plt.tight_layout()
plt.savefig('analysis/average_feature_importance.png')
print("Saved average feature importance visualization to analysis/average_feature_importance.png")

# Create a visualization comparing feature importance across models
plt.figure(figsize=(14, 10))
top_features = avg_importance.head(10)['Feature'].tolist()
model_comparison = feature_importance_df[feature_importance_df['Feature'].isin(top_features)]
sns.barplot(x='Normalized_Importance', y='Feature', hue='Model', data=model_comparison)
plt.title('Feature Importance Comparison Across Models')
plt.tight_layout()
plt.savefig('analysis/feature_importance_comparison.png')
print("Saved feature importance comparison visualization to analysis/feature_importance_comparison.png")

# Create a comprehensive feature importance summary
print("\nCreating feature importance summary...")
with open('analysis/feature_importance_summary.txt', 'w') as f:
    f.write('Feature Importance Summary\n')
    f.write('=========================\n\n')
    
    f.write('Top 15 Features by Average Normalized Importance:\n')
    for i, (_, row) in enumerate(avg_importance.head(15).iterrows(), 1):
        f.write(f'{i}. {row["Feature"]} (Avg. Normalized Importance: {row["Normalized_Importance"]:.4f})\n')
    
    f.write('\nTop 15 Features by Combined Ranking from Feature Selection Methods:\n')
    for i, (_, row) in enumerate(combined_ranks.head(15).iterrows(), 1):
        f.write(f'{i}. {row["Feature"]} (Average Rank: {row["Average_Rank"]:.2f})\n')
    
    f.write('\nConsistent Top Features Across All Methods:\n')
    # Identify features that appear in top 10 of both average importance and combined ranking
    top_avg = set(avg_importance.head(10)['Feature'].tolist())
    top_combined = set(combined_ranks.head(10)['Feature'].tolist())
    consistent_features = top_avg.intersection(top_combined)
    
    for i, feature in enumerate(consistent_features, 1):
        avg_rank = combined_ranks[combined_ranks['Feature'] == feature]['Average_Rank'].values[0]
        avg_imp = avg_importance[avg_importance['Feature'] == feature]['Normalized_Importance'].values[0]
        f.write(f'{i}. {feature} (Avg. Rank: {avg_rank:.2f}, Avg. Normalized Importance: {avg_imp:.4f})\n')

print("Saved feature importance summary to analysis/feature_importance_summary.txt")

# Create a final list of most important features
important_features = list(consistent_features)
if len(important_features) < 5:  # Ensure we have at least 5 features
    remaining_from_avg = [f for f in avg_importance.head(10)['Feature'].tolist() if f not in important_features]
    important_features.extend(remaining_from_avg[:5-len(important_features)])

with open('analysis/most_important_features.txt', 'w') as f:
    f.write('Most Important Features for Predicting "파단직전 =1"\n')
    f.write('==============================================\n\n')
    for i, feature in enumerate(important_features, 1):
        f.write(f'{i}. {feature}\n')

print("Saved most important features to analysis/most_important_features.txt")

# Analyze feature relationships
print("\nAnalyzing relationships between top features...")

# Load the original dataset
try:
    df = pd.read_csv('엔솔_전처리_파단직전재추출_0401.csv')
    
    # Create a balanced sample for visualization
    positive_samples = df[df['파단직전'] == 1]
    negative_samples = df[df['파단직전'] == 0].sample(n=len(positive_samples), random_state=42)
    sample_df = pd.concat([positive_samples, negative_samples], axis=0).reset_index(drop=True)
    
    # Create pairplot of top 5 features
    top5_features = important_features[:5] + ['파단직전']
    plt.figure(figsize=(15, 12))
    sns.pairplot(sample_df[top5_features], hue='파단직전', diag_kind='kde')
    plt.suptitle('Pairwise Relationships Between Top 5 Features', y=1.02)
    plt.savefig('analysis/top_features_pairplot.png')
    print("Saved top features pairplot to analysis/top_features_pairplot.png")
except Exception as e:
    print(f"Warning: Could not create pairplot - {str(e)}")

print("\nFeature Importance Analysis completed successfully!")
