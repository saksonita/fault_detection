
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os

# Set up directories
if not os.path.exists('analysis'):
    os.makedirs('analysis')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('../data/엔솔_전처리_파단직전재추출_0401.csv')

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
print(f"Positive samples in training set: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
print(f"Positive samples in test set: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.2f}%)")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42)
}

# Function to evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    # Create a pipeline with SMOTE to handle class imbalance
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    # Classification report and confusion matrix
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Cross-validation
    cv_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    cv_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        # Get feature importance after SMOTE
        model_fitted = pipeline.named_steps['model']
        importance = model_fitted.feature_importances_
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        # For logistic regression
        model_fitted = pipeline.named_steps['model']
        importance = np.abs(model_fitted.coef_[0])
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': np.zeros(len(features))})
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'report': report,
        'confusion_matrix': cm,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance
    }

# Evaluate all models
print("\nTraining and evaluating models...")
results = {}
for name, model in models.items():
    print(f'Evaluating {name}...')
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    print(f'{name} ROC-AUC: {results[name]["roc_auc"]:.4f}')
    print(f'{name} PR-AUC: {results[name]["pr_auc"]:.4f}')
    print(f'{name} CV ROC-AUC: {results[name]["cv_scores"].mean():.4f} ± {results[name]["cv_scores"].std():.4f}')
    print(f'{name} Top 5 Important Features:')
    print(results[name]['feature_importance'].head(5))
    print('\n')

# Save results to files
print("\nSaving model evaluation results...")
with open('analysis/model_evaluation_results.txt', 'w') as f:
    for name, result in results.items():
        f.write(f'Model: {name}\n')
        f.write(f'Accuracy: {result["accuracy"]:.4f}\n')
        f.write(f'ROC-AUC: {result["roc_auc"]:.4f}\n')
        f.write(f'PR-AUC: {result["pr_auc"]:.4f}\n')
        f.write(f'CV ROC-AUC: {result["cv_scores"].mean():.4f} ± {result["cv_scores"].std():.4f}\n')
        f.write('\nClassification Report:\n')
        f.write(result['report'])
        f.write('\nConfusion Matrix:\n')
        f.write(str(result['confusion_matrix']))
        f.write('\n\nFeature Importance:\n')
        f.write(str(result['feature_importance']))
        f.write('\n\n' + '='*50 + '\n\n')

# Visualize feature importance across models
print("\nCreating feature importance visualizations...")
plt.figure(figsize=(15, 10))
for i, (name, result) in enumerate(results.items()):
    if hasattr(models[name], 'feature_importances_') or hasattr(models[name], 'coef_'):
        plt.subplot(2, 2, i+1)
        top_features = result['feature_importance'].head(10)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'{name} - Top 10 Feature Importance')
        plt.tight_layout()
plt.savefig('analysis/model_feature_importance.png')
print("Saved model feature importance visualization to analysis/model_feature_importance.png")

# Visualize ROC curves
plt.figure(figsize=(10, 8))
for name, result in results.items():
    # Get the fitted pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', models[name])
    ])
    pipeline.fit(X_train, y_train)
    
    # Calculate ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend()
plt.savefig('analysis/roc_curves.png')
print("Saved ROC curves to analysis/roc_curves.png")

# Visualize Precision-Recall curves
plt.figure(figsize=(10, 8))
for name, result in results.items():
    # Get the fitted pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', models[name])
    ])
    pipeline.fit(X_train, y_train)
    
    # Calculate Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
    plt.plot(recall, precision, label=f'{name} (AUC = {result["pr_auc"]:.4f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Different Models')
plt.legend()
plt.savefig('analysis/precision_recall_curves.png')
print("Saved Precision-Recall curves to analysis/precision_recall_curves.png")

print("\nModel Building and Evaluation completed successfully!")
