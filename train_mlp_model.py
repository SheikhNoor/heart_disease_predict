"""
Train MLP (Multi-Layer Perceptron) Neural Network Model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import pickle
import os

print("=" * 60)
print("MLP Neural Network Model Training")
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

# Scale features (CRITICAL for Neural Networks!)
print("\n⚙️ Scaling features (required for neural networks)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled to zero mean and unit variance")

# Initialize MLP model
print("\n🧠 Initializing MLP Neural Network Model...")
print("Architecture: Input → 128 → 64 → 32 → Output")
print("Activation: ReLU")
print("Optimizer: Adam")

mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers with decreasing neurons
    activation='relu',                  # ReLU activation function
    solver='adam',                      # Adam optimizer (adaptive learning rate)
    alpha=0.0001,                       # L2 regularization parameter
    batch_size='auto',                  # Batch size = min(200, n_samples)
    learning_rate='adaptive',           # Adaptive learning rate
    learning_rate_init=0.001,           # Initial learning rate
    max_iter=500,                       # Maximum iterations
    shuffle=True,                       # Shuffle samples in each iteration
    early_stopping=True,                # Stop when validation score doesn't improve
    validation_fraction=0.1,            # 10% of training data for validation
    n_iter_no_change=20,                # Stop after 20 epochs without improvement
    random_state=42,
    verbose=True                        # Show training progress
)

print("\n🚀 Training MLP model...")
print("-" * 60)

# Train the model
mlp_model.fit(X_train_scaled, y_train)

print("\n" + "=" * 60)
print("✅ Training Complete!")
print("=" * 60)

# Evaluate on test set
print("\n📈 Evaluating model performance...")
y_pred = mlp_model.predict(X_test_scaled)
y_pred_proba = mlp_model.predict_proba(X_test_scaled)

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

# Training statistics
print("\n📈 Training Statistics:")
print(f"Number of iterations: {mlp_model.n_iter_}")
print(f"Number of layers: {mlp_model.n_layers_}")
print(f"Number of outputs: {mlp_model.n_outputs_}")
print(f"Final loss: {mlp_model.loss_:.6f}")

# Feature importance (using input layer weights)
print("\n🔍 Feature Importance (based on input layer weights):")
# Get absolute mean of weights from input to first hidden layer
feature_importances = np.abs(mlp_model.coefs_[0]).mean(axis=1)
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print(feature_importance_df.to_string(index=False))

# Save the model
print("\n💾 Saving trained model...")
os.makedirs('models', exist_ok=True)

# Save MLP model
with open('models/mlp_model.pkl', 'wb') as f:
    pickle.dump(mlp_model, f)
print("✅ MLP model saved: models/mlp_model.pkl")

# Save scaler
with open('models/mlp_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved: models/mlp_scaler.pkl")

# Save feature columns
with open('models/mlp_feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("✅ Feature columns saved: models/mlp_feature_columns.pkl")

# Save model metadata
metadata = {
    'accuracy': accuracy,
    'train_samples': X_train.shape[0],
    'test_samples': X_test.shape[0],
    'n_features': X.shape[1],
    'feature_names': X.columns.tolist(),
    'architecture': '128-64-32',
    'n_iterations': mlp_model.n_iter_,
    'final_loss': mlp_model.loss_
}

with open('models/mlp_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✅ Metadata saved: models/mlp_metadata.pkl")

print("\n" + "=" * 60)
print("🎉 Training pipeline completed successfully!")
print("=" * 60)
print(f"\n✅ MLP Neural Network achieved {accuracy*100:.2f}% accuracy!")
print(f"✅ Model architecture: Input({X.shape[1]}) → 128 → 64 → 32 → Output(2)")
print(f"✅ Trained in {mlp_model.n_iter_} iterations")
print("\nYou can now use the trained model in your Streamlit app.")
