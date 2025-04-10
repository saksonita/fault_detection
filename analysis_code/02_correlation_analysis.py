import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up directories
if not os.path.exists('analysis'):
    os.makedirs('analysis')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('../data/엔솔_전처리_파단직전재추출_0401.csv')


# Create a balanced sample for visualization
# (using all positive cases and an equal number of negative cases)
positive_samples = df[df['파단직전'] == 1]
negative_samples = df[df['파단직전'] == 0].sample(n=len(positive_samples), random_state=42)
sample_df = pd.concat([positive_samples, negative_samples], axis=0).reset_index(drop=True)

print(f"Created balanced sample with {len(sample_df)} rows for correlation analysis")

# Identify numeric columns for analysis
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Calculate correlations with target variable
print("\nCalculating correlations with target variable...")
target_correlations = sample_df.corr()['파단직전'].sort_values(ascending=False)

# Display top positive correlations
print("\nCorrelations with target variable (top 15):")
print(target_correlations.head(15))

# Display top negative correlations
print("\nCorrelations with target variable (bottom 15):")
print(target_correlations.tail(15))

# Save correlations to file
target_correlations.to_csv('analysis/target_correlations.csv')
print("Saved target correlations to analysis/target_correlations.csv")

# Create correlation heatmap for top features
print("\nCreating correlation heatmap for top features...")
# Get top 20 features by absolute correlation with target
top_features = target_correlations.drop('파단직전').abs().sort_values(ascending=False).head(20).index.tolist()
top_features.append('파단직전')  # Add target variable
correlation_matrix = sample_df[top_features].corr()

# Create heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Top 20 Features with Target Variable')
plt.tight_layout()
plt.savefig('analysis/correlation_heatmap.png')
print("Saved correlation heatmap to analysis/correlation_heatmap.png")

# Create scatter plots for top correlated features
print("\nCreating scatter plots for top correlated features...")
plt.figure(figsize=(20, 15))
top_pos_corr = target_correlations.drop('파단직전').sort_values(ascending=False).head(5).index.tolist()
top_neg_corr = target_correlations.drop('파단직전').sort_values(ascending=True).head(5).index.tolist()
top_corr_features = top_pos_corr + top_neg_corr

for i, feature in enumerate(top_corr_features):
    plt.subplot(2, 5, i+1)
    sns.scatterplot(data=sample_df, x=feature, y='파단직전', alpha=0.5)
    plt.title(f'{feature}\nCorr: {target_correlations[feature]:.3f}')
    plt.tight_layout()
plt.savefig('analysis/top_correlations_scatter.png')
print("Saved scatter plots to analysis/top_correlations_scatter.png")

# Calculate feature-to-feature correlations to identify multicollinearity
print("\nIdentifying highly correlated feature pairs (multicollinearity)...")
feature_correlations = sample_df[numeric_cols].corr()
high_correlations = []
for i in range(len(feature_correlations.columns)):
    for j in range(i+1, len(feature_correlations.columns)):
        if abs(feature_correlations.iloc[i, j]) > 0.8:  # Threshold for high correlation
            high_correlations.append((feature_correlations.columns[i], 
                                     feature_correlations.columns[j], 
                                     feature_correlations.iloc[i, j]))

# Sort by absolute correlation value
high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

# Save high feature correlations to file
with open('analysis/high_feature_correlations.txt', 'w') as f:
    f.write('Highly correlated feature pairs (|correlation| > 0.8):\n')
    for feat1, feat2, corr in high_correlations:
        f.write(f'{feat1} - {feat2}: {corr:.3f}\n')

print('\nHighly correlated feature pairs (|correlation| > 0.8):')
for feat1, feat2, corr in high_correlations[:10]:  # Show top 10
    print(f'{feat1} - {feat2}: {corr:.3f}')

print("\nSaved high feature correlations to analysis/high_feature_correlations.txt")
print("\nCorrelation Analysis completed successfully!")
