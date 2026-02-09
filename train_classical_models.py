"""
Train all classical ML models: Random Forest, Logistic Regression, and SVM
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

print("=" * 60)
print("Classical ML Models Training")
print("=" * 60)

# Load dataset
print("\n📂 Loading dataset...")
cleaned_path = os.path.join('datasets', 'cleaned_df.csv')

try:
    df = pd.read_csv(cleaned_path)
    print(f"✅ Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    # Rename columns to match expected format
    column_mapping = {
        'chest_pain_type': 'cp',
        'resting_bp': 'trestbps',
        'cholestrol': 'chol',
        'fasting_blood_suger': 'fbs',
        'max_heart_rate': 'thalach',
        'num_major_vessels': 'ca'
    }
    df = df.rename(columns=column_mapping)
    # Add missing lifestyle columns
    if 'smoking' not in df.columns:
        df['smoking'] = 0
    if 'diabetes' not in df.columns:
        df['diabetes'] = 0
    if 'bmi' not in df.columns:
        df['bmi'] = 25.0
except FileNotFoundError:
    print("⚠️ Cleaned dataset not found. Loading original dataset...")
    df = pd.read_csv(os.path.join('datasets', 'heart.csv'))
    print(f"✅ Loaded original dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Prepare data
print("\n🔧 Preparing data...")
X = df.drop(['target'], axis=1)
if 'Unnamed: 0' in X.columns:
    X = X.drop('Unnamed: 0', axis=1)
y = df['target']

print(f"Features: {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n📊 Train set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Scale features
print("\n⚙️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models directory
os.makedirs('models', exist_ok=True)

# Train Random Forest
print("\n" + "=" * 60)
print("1. Training Random Forest Classifier")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

print("Training...")
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"\n🎯 Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['No Disease', 'Disease']))

print("\n📊 Confusion Matrix:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

# Feature importance
print("\n🔍 Top 10 Feature Importances:")
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)
print(feature_importance_rf.to_string(index=False))

# Save Random Forest
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("\n✅ Random Forest saved: models/random_forest_model.pkl")

# Train Logistic Regression
print("\n" + "=" * 60)
print("2. Training Logistic Regression")
print("=" * 60)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    solver='lbfgs'
)

print("Training...")
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"\n🎯 Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['No Disease', 'Disease']))

print("\n📊 Confusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)

# Coefficients
print("\n🔍 Top 10 Feature Coefficients (absolute values):")
feature_coef_lr = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Coefficient', ascending=False).head(10)
print(feature_coef_lr.to_string(index=False))

# Save Logistic Regression
with open('models/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("\n✅ Logistic Regression saved: models/logistic_regression_model.pkl")

# Train SVM
print("\n" + "=" * 60)
print("3. Training Support Vector Machine (SVM)")
print("=" * 60)

svm_model = SVC(
    probability=True,
    random_state=42,
    class_weight='balanced',
    kernel='rbf',
    gamma='scale'
)

print("Training...")
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"\n🎯 SVM Accuracy: {accuracy_svm * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['No Disease', 'Disease']))

print("\n📊 Confusion Matrix:")
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(cm_svm)

# Save SVM
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
print("\n✅ SVM saved: models/svm_model.pkl")

# Save common scaler and metadata
print("\n" + "=" * 60)
print("Saving Common Resources")
print("=" * 60)

# Save scaler
with open('models/classical_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved: models/classical_scaler.pkl")

# Save feature columns
with open('models/classical_feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("✅ Feature columns saved: models/classical_feature_columns.pkl")

# Save metadata for all models
metadata = {
    'random_forest': {
        'accuracy': accuracy_rf,
        'n_estimators': 100,
        'cm': cm_rf.tolist()
    },
    'logistic_regression': {
        'accuracy': accuracy_lr,
        'max_iter': 1000,
        'cm': cm_lr.tolist()
    },
    'svm': {
        'accuracy': accuracy_svm,
        'kernel': 'rbf',
        'cm': cm_svm.tolist()
    },
    'common': {
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'n_features': X.shape[1],
        'feature_names': X.columns.tolist()
    }
}

with open('models/classical_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✅ Metadata saved: models/classical_metadata.pkl")

# Summary
print("\n" + "=" * 60)
print("🎉 Training Complete - Summary")
print("=" * 60)

print("\n📊 Model Comparison:")
print(f"{'Model':<25} {'Accuracy':<15}")
print("-" * 40)
print(f"{'Random Forest':<25} {accuracy_rf*100:>6.2f}%")
print(f"{'Logistic Regression':<25} {accuracy_lr*100:>6.2f}%")
print(f"{'SVM':<25} {accuracy_svm*100:>6.2f}%")

print("\n✅ All models trained and saved successfully!")
print("\nSaved files:")
print("  - models/random_forest_model.pkl")
print("  - models/logistic_regression_model.pkl")
print("  - models/svm_model.pkl")
print("  - models/classical_scaler.pkl")
print("  - models/classical_feature_columns.pkl")
print("  - models/classical_metadata.pkl")

print("\nYou can now use these models in your Streamlit app.")
