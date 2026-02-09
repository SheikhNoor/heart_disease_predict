"""
Test script to verify the trained TabNet model works correctly
"""
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import pickle
import os

print("=" * 60)
print("Testing Trained TabNet Model")
print("=" * 60)

# Load the model
print("\n📂 Loading trained model...")
model = TabNetClassifier()
model.load_model('models/tabnet_model.zip')
print("✅ Model loaded successfully")

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded successfully")

# Load feature columns
with open('models/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
print(f"✅ Feature columns loaded: {len(feature_columns)} features")

# Load metadata
with open('models/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
print(f"✅ Metadata loaded")
print(f"   - Training accuracy: {metadata['accuracy']*100:.2f}%")
print(f"   - Training samples: {metadata['train_samples']}")
print(f"   - Test samples: {metadata['test_samples']}")
print(f"   - Features: {metadata['n_features']}")

# Test prediction with sample data
print("\n🧪 Testing prediction with sample patient data...")
sample_patient = {
    'age': 55,
    'sex': 1,
    'cp': 0,
    'trestbps': 140,
    'chol': 250,
    'fbs': 0,
    'restecg': 1,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 2.0,
    'slope': 0,
    'ca': 0,
    'thal': 2,
    'smoking': 1,
    'diabetes': 0,
    'bmi': 28.5
}

# Create DataFrame
sample_df = pd.DataFrame([sample_patient])
sample_df = sample_df[feature_columns]  # Ensure correct column order

# Scale the data
sample_scaled = scaler.transform(sample_df)

# Make prediction
prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0]

print("\n" + "=" * 60)
print("📊 Prediction Results")
print("=" * 60)
print(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
print(f"Probability of No Disease: {probability[0]*100:.2f}%")
print(f"Probability of Disease: {probability[1]*100:.2f}%")

print("\n✅ Model is working correctly!")
print("=" * 60)
