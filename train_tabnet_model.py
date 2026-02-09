import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import pickle
import os

print("=" * 60)
print("TabNet Deep Learning Model Training")
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

# Calculate appropriate batch size based on dataset size
n_samples = len(X_train_scaled)
batch_size = min(32, max(8, n_samples // 8))
virtual_batch_size = max(4, batch_size // 2)
print(f"\n📊 Using batch_size={batch_size}, virtual_batch_size={virtual_batch_size}")

# Initialize TabNet model
print("\n🧠 Initializing TabNet Deep Learning Model...")
tabnet_model = TabNetClassifier(
    n_d=64,                    # Width of the decision prediction layer (INCREASED for better capacity)
    n_a=64,                    # Width of the attention embedding (INCREASED)
    n_steps=5,                 # Number of sequential attention steps (INCREASED)
    gamma=1.5,                 # Feature reuse coefficient (optimized)
    n_independent=2,           # Number of independent GLU layers
    n_shared=2,                # Number of shared GLU layers
    lambda_sparse=1e-3,        # Sparsity loss weight
    momentum=0.02,             # Momentum for batch normalization
    clip_value=1,              # Gradient clipping value
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 50, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',        # Better feature selection
    seed=42,
    verbose=1
)

print("\n🚀 Training TabNet model...")
print("-" * 60)

# Train the model
tabnet_model.fit(
    X_train_scaled, y_train.values,
    eval_set=[(X_test_scaled, y_test.values)],
    eval_name=['test'],
    eval_metric=['accuracy', 'logloss'],
    max_epochs=100,            # Reduced for faster convergence
    patience=20,               # Early stopping patience
    batch_size=batch_size,     # Dynamic batch size based on dataset
    virtual_batch_size=virtual_batch_size,  # Dynamic virtual batch size
    num_workers=0,
    drop_last=False
)

print("\n" + "=" * 60)
print("✅ Training Complete!")
print("=" * 60)

# Evaluate on test set
print("\n📈 Evaluating model performance...")
y_pred = tabnet_model.predict(X_test_scaled)
y_pred_proba = tabnet_model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Positives: {cm[1][1]}")
print(f"True Negatives: {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")

# Feature importance
print("\n🔍 Feature Importance:")
feature_importances = tabnet_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print(feature_importance_df.to_string(index=False))

# Save the model
print("\n💾 Saving trained model...")
os.makedirs('models', exist_ok=True)

# Save TabNet model
tabnet_model.save_model('models/tabnet_model')
print("✅ TabNet model saved: models/tabnet_model.zip")

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved: models/scaler.pkl")

# Save feature columns
with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("✅ Feature columns saved: models/feature_columns.pkl")

# Save model metadata
metadata = {
    'accuracy': accuracy,
    'train_samples': X_train.shape[0],
    'test_samples': X_test.shape[0],
    'n_features': X.shape[1],
    'feature_names': X.columns.tolist()
}

with open('models/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✅ Metadata saved: models/metadata.pkl")

print("\n" + "=" * 60)
print("🎉 Training pipeline completed successfully!")
print("=" * 60)
print("\nYou can now use the trained model in your Streamlit app.")
