"""
Feature Selection Script for "파단직전 =1" Condition Analysis

This script applies various feature selection methods to identify the most important
features for predicting the "파단직전 =1" condition.

Methods used:
1. Filter methods (ANOVA F-value, Mutual Information)
2. Wrapper methods (Recursive Feature Elimination)
3. Embedded methods (Random Forest, Gradient Boosting)

Usage:
    python feature_selection.py

Author: Manus AI
Date: April 9, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Set up directories
if not os.path.exists('analysis'):
    os.makedirs('analysis')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('엔솔_전처리_파단직전재추출_0401.csv')

# Identify features and target
print("\nPreparing features and target...")
X = df.drop('파단직전', axis=1)
y = df['파단직전']

# Get feature names
features = X.columns.tolist()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Filter Methods
print("\n1. Applying filter methods...")

# 1.1 ANOVA F-value
print("\n1.1 ANOVA F-value")
f_selector = SelectKBest(f_classif, k=20)
f_selector.fit(X_train_scaled, y_train)
f_scores = pd.DataFrame({'Feature': features, 'F_Score': f_selector.scores_, 'P_Value': f_selector.pvalues_})
f_scores = f_scores.sort_values('F_Score', ascending=False)

print("Top 10 features by ANOVA F-value:")
print(f_scores.head(10))

# Save results
f_scores.to_csv('analysis/anova_f_scores.csv', index=False)
print("Saved ANOVA F-scores to analysis/anova_f_scores.csv")

# 1.2 Mutual Information
print("\n1.2 Mutual Information")
mi_selector = SelectKBest(mutual_info_classif, k=20)
mi_selector.fit(X_train_scaled, y_train)
mi_scores = pd.DataFrame({'Feature': features, 'MI_Score': mi_selector.scores_})
mi_scores = mi_scores.sort_values('MI_Score', ascending=False)

print("Top 10 features by Mutual Information:")
print(mi_scores.head(10))

# Save results
mi_scores.to_csv('analysis/mutual_information_scores.csv', index=False)
print("Saved Mutual Information scores to analysis/mutual_information_scores.csv")

# 2. Wrapper Methods
print("\n2. Applying wrapper methods...")

# 2.1 Recursive Feature Elimination with Random Forest
print("\n2.1 Recursive Feature Elimination")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=20, step=1)
rfe.fit(X_train_scaled, y_train)
rfe_scores = pd.DataFrame({'Feature': features, 'Selected': rfe.support_, 'Rank': rfe.ranking_})
rfe_scores = rfe_scores.sort_values('Rank')

print("Top 10 features by RFE:")
print(rfe_scores.head(10))

# Save results
rfe_scores.to_csv('analysis/rfe_scores.csv', index=False)
print("Saved RFE scores to analysis/rfe_scores.csv")

# 3. Embedded Methods
print("\n3. Applying embedded methods...")

# 3.1 Random Forest Importance
print("\n3.1 Random Forest Importance")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_importance = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
rf_importance = rf_importance.sort_values('Importance', ascending=False)

print("Top 10 features by Random Forest Importance:")
print(rf_importance.head(10))

# Save results
rf_importance.to_csv('analysis/random_forest_importance.csv', index=False)
print("Saved Random Forest importance to analysis/random_forest_importance.csv")

# 3.2 Gradient Boosting Importance
print("\n3.2 Gradient Boosting Importance")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
gb_importance = pd.DataFrame({'Feature': features, 'Importance': gb.feature_importances_})
gb_importance = gb_importance.sort_values('Importance', ascending=False)

print("Top 10 features by Gradient Boosting Importance:")
print(gb_importance.head(10))

# Save results
gb_importance.to_csv('analysis/gradient_boosting_importance.csv', index=False)
print("Saved Gradient Boosting importance to analysis/gradient_boosting_importance.csv")

# Combine rankings from different methods
print("\nCombining rankings from different methods...")

# Create a dataframe to store all rankings
all_rankings = pd.DataFrame({'Feature': features})

# Add rankings from each method
all_rankings['ANOVA_Rank'] = all_rankings['Feature'].map(
    dict(zip(f_scores['Feature'], range(1, len(features) + 1))))
all_rankings['MI_Rank'] = all_rankings['Feature'].map(
    dict(zip(mi_scores['Feature'], range(1, len(features) + 1))))
all_rankings['RFE_Rank'] = all_rankings['Feature'].map(
    dict(zip(rfe_scores['Feature'], rfe_scores['Rank'])))
all_rankings['RF_Rank'] = all_rankings['Feature'].map(
    dict(zip(rf_importance['Feature'], range(1, len(features) + 1))))
all_rankings['GB_Rank'] = all_rankings['Feature'].map(
    dict(zip(gb_importance['Feature'], range(1, len(features) + 1))))

# Calculate average rank
all_rankings['Average_Rank'] = all_rankings[['ANOVA_Rank', 'MI_Rank', 'RFE_Rank', 'RF_Rank', 'GB_Rank']].mean(axis=1)
all_rankings = all_rankings.sort_values('Average_Rank')

print("Top 10 features by combined ranking:")
print(all_rankings.head(10))

# Save combined rankings
all_rankings.to_csv('analysis/combined_feature_ranks.csv', index=False)
print("Saved combined feature rankings to analysis/combined_feature_ranks.csv")

# Visualize top features from different methods
print("\nCreating visualization of top features from different methods...")
plt.figure(figsize=(15, 10))

# Get top 10 features from each method
top_anova = f_scores.head(10)['Feature'].tolist()
top_mi = mi_scores.head(10)['Feature'].tolist()
top_rfe = rfe_scores.head(10)['Feature'].tolist()
top_rf = rf_importance.head(10)['Feature'].tolist()
top_gb = gb_importance.head(10)['Feature'].tolist()

# Count occurrences of each feature
all_top_features = top_anova + top_mi + top_rfe + top_rf + top_gb
feature_counts = pd.Series(all_top_features).value_counts()

# Plot top features by occurrence count
top_features = feature_counts[feature_counts > 1].index.tolist()
top_feature_counts = feature_counts[feature_counts > 1].values

plt.barh(top_features, top_feature_counts)
plt.xlabel('Number of Methods')
plt.ylabel('Feature')
plt.title('Features Selected by Multiple Methods')
plt.tight_layout()
plt.savefig('analysis/top_features_comparison.png')
print("Saved top features comparison to analysis/top_features_comparison.png")

print("\nFeature Selection completed successfully!")
