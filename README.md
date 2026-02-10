# Heart Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53.0-red.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-5%20Models-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive machine learning application for predicting heart disease risk using **5 different algorithms** including deep learning models. Built with Streamlit for an interactive web interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Training Scripts](#training-scripts)
- [Application Pages](#application-pages)
- [Dependencies](#dependencies)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This system predicts the likelihood of heart disease based on **16 clinical features** using machine learning algorithms. The application provides:

- **Interactive Prediction Interface** - Real-time risk assessment
- **5 ML Algorithms** - From classical to deep learning
- **Data Visualization** - Comprehensive exploratory data analysis
- **Model Comparison** - Side-by-side performance metrics
- **Feature Importance Analysis** - Data-driven insights
- **100% ML-Driven Predictions** - No hardcoded rules or thresholds

---

## ✨ Features

### Core Capabilities
- ✅ **Multi-Model Support**: 5 different algorithms with varying approaches
- ✅ **Real-Time Predictions**: Instant risk assessment with confidence scores
- ✅ **Interactive Visualizations**: Charts, graphs, and statistical analysis
- ✅ **Feature Importance**: Understand what drives each prediction
- ✅ **Model Performance Comparison**: Compare accuracy across all models
- ✅ **Balanced Class Handling**: Proper handling of imbalanced datasets
- ✅ **Responsive UI**: Clean, modern interface with Streamlit

### Advanced Features
- 📊 **Risk Probability Gauge**: Visual representation of disease risk
- 🧠 **ML-Based Insights**: Learn from 303 real patient records
- 🎯 **Risk Indicator Analysis**: Shows if values are closer to disease or healthy patterns
- 📈 **ROC Curves & Confusion Matrix**: Detailed performance metrics
- 🔍 **Feature Distribution Analysis**: Explore data patterns
- 💡 **Educational Explanations**: Understand ML predictions

---

## 📊 Dataset

### Source
- **Primary**: UCI Heart Disease Dataset
- **Extended**: Combined with lifestyle factors (smoking, diabetes, BMI)

### Statistics
- **Total Patients**: 303
- **Disease Cases**: 165 (54.5%)
- **Healthy Cases**: 138 (45.5%)
- **Features**: 16 clinical attributes
- **Target**: Binary classification (0 = Healthy, 1 = Disease)

### Features Description

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| **age** | Age in years | Continuous | 29-77 |
| **sex** | Gender (0=Female, 1=Male) | Binary | 0-1 |
| **cp** | Chest pain type (0-3) | Categorical | 0-3 |
| **trestbps** | Resting blood pressure (mm Hg) | Continuous | 94-200 |
| **chol** | Cholesterol (mg/dl) | Continuous | 126-564 |
| **fbs** | Fasting blood sugar > 120 mg/dl | Binary | 0-1 |
| **restecg** | Resting ECG results (0-2) | Categorical | 0-2 |
| **thalach** | Max heart rate achieved | Continuous | 71-202 |
| **exang** | Exercise induced angina | Binary | 0-1 |
| **oldpeak** | ST depression | Continuous | 0-6.2 |
| **slope** | Slope of peak exercise ST | Categorical | 0-2 |
| **ca** | Number of major vessels (0-3) | Discrete | 0-3 |
| **thal** | Thalassemia (1-3) | Categorical | 1-3 |
| **smoking** | Smoking status | Binary | 0-1 |
| **diabetes** | Diabetes status | Binary | 0-1 |
| **bmi** | Body Mass Index | Continuous | 18-40 |

### Important Dataset Characteristics

⚠️ **Note**: This dataset has **unusual patterns** that differ from typical medical knowledge:
- Disease patients are **younger** on average (52 vs 56 years)
- Disease patients have **lower** blood pressure (129 vs 134 mm Hg)
- Disease patients have **lower** ST depression (0.6 vs 1.6)
- Disease patients have **fewer** blocked vessels (0.4 vs 1.2)

The ML models learn from **these specific patterns**, not general medical rules.

---

## 🤖 Machine Learning Models

### 1. Random Forest Classifier
- **Type**: Ensemble tree-based
- **Configuration**:
  - n_estimators: 100
  - class_weight: balanced
  - random_state: 42
- **Strengths**: Robust, handles non-linear relationships, provides feature importance
- **Use Case**: General-purpose prediction

### 2. Logistic Regression
- **Type**: Linear classifier
- **Configuration**:
  - max_iter: 1000
  - class_weight: balanced
  - random_state: 42
- **Strengths**: Interpretable, fast training, probabilistic output
- **Use Case**: Baseline model, interpretability

### 3. Support Vector Machine (SVM)
- **Type**: Kernel-based classifier
- **Configuration**:
  - probability: True
  - class_weight: balanced
  - random_state: 42
- **Strengths**: Effective in high-dimensional spaces
- **Use Case**: When data is not linearly separable

### 4. MLP Neural Network ✨ NEW
- **Type**: Multi-Layer Perceptron (Deep Learning)
- **Architecture**: Input(16) → 128 → 64 → 32 → Output(2)
- **Configuration**:
  - activation: ReLU
  - optimizer: Adam
  - learning_rate: adaptive (initial 0.001)
  - early_stopping: True
  - max_iter: 500
- **Strengths**: Learns complex non-linear patterns, adaptive learning
- **Use Case**: Capturing complex feature interactions

### 5. TabNet Deep Learning
- **Type**: Attention-based neural network
- **Architecture**:
  - n_d: 64 (decision layer width)
  - n_a: 64 (attention embedding)
  - n_steps: 5 (sequential attention steps)
  - mask_type: entmax
- **Configuration**:
  - optimizer: Adam (lr=0.02)
  - batch_size: Dynamic (based on dataset size)
  - max_epochs: 100
  - early_stopping: patience=20
- **Strengths**: Feature selection, interpretability, high accuracy
- **Use Case**: When interpretability and accuracy both matter

---

## 🚀 Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone or Download the Repository**
   ```bash
   cd heart_disease_predict
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv myenv
   ```

3. **Activate Virtual Environment**
   - Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source myenv/bin/activate
     ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify Installation**
   ```bash
   streamlit --version
   python --version
   ```

---

## 📖 Usage

### Running the Application

1. **Activate Virtual Environment** (if not already active)
   ```bash
   myenv\Scripts\activate
   ```

2. **Launch Streamlit App**
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**
   - Local URL: http://localhost:8501
   - Network URL: https://cardiac-ai.streamlit.app/

4. **Navigate Through Pages**
   - 🏠 **Home**: Overview and statistics
   - 📊 **Data Exploration**: EDA and visualizations
   - 🔮 **Prediction**: Make heart disease predictions
   - 📈 **Model Performance**: Compare all models
   - ℹ️ **About**: Project information

### Making a Prediction

1. Go to **🔮 Prediction** page
2. Enter patient information:
   - Age, Sex, Blood Pressure, Cholesterol
   - Heart Rate, ECG results, Exercise data
   - Lifestyle factors (smoking, diabetes, BMI)
3. Select ML algorithm from dropdown
4. Click **🔬 Predict Risk**
5. View results:
   - Risk classification (HIGH/LOW)
   - Confidence scores
   - Feature importance analysis
   - Comparison with dataset averages

---

## 📁 Project Structure

```
heart_disease_predict/
│
├── app.py                          # Main Streamlit application
├── train_tabnet_model.py           # TabNet training script
├── train_mlp_model.py              # MLP training script
├── validate_model_and_data.py      # Data validation script
├── test_model.py                   # Model testing script
├── final_test.py                   # Final testing script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── datasets/
│   └── cleaned_df.csv              # Training dataset (303 patients)
│
├── models/
│   ├── tabnet_model.zip            # Trained TabNet model
│   ├── scaler.pkl                  # Feature scaler (TabNet)
│   ├── feature_columns.pkl         # Feature names (TabNet)
│   ├── metadata.pkl                # Model metadata (TabNet)
│   ├── mlp_model.pkl               # Trained MLP model
│   ├── mlp_scaler.pkl              # Feature scaler (MLP)
│   ├── mlp_feature_columns.pkl     # Feature names (MLP)
│   └── mlp_metadata.pkl            # Model metadata (MLP)
│
├── image/
│   └── logo.webp                   # Application logo
│
└── myenv/                          # Virtual environment (not in git)
    ├── Lib/
    ├── Scripts/
    └── pyvenv.cfg
```

---

## 📊 Model Performance

### Test Set Accuracy (303 patients, 20% test split = 61 samples)

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|------|-------|----------|-----------|--------|----------|
| 🥇 | **Random Forest** | 81.97% | 0.95 | 0.64 | 0.77 |
| 🥇 | **TabNet (Deep Learning)** | 81.97% | 0.76 | 0.97 | 0.85 |
| 🥈 | **SVM** | 80.33% | 0.80 | 0.85 | 0.82 |
| 🥉 | **Logistic Regression** | 78.69% | 0.76 | 0.88 | 0.82 |
| 🏅 | **MLP Neural Network** | 68.85% | 0.68 | 0.79 | 0.73 |

### Feature Importance Rankings

**Top 5 Most Important Features** (averaged across models):
1. **oldpeak** (ST Depression) - 13.43%
2. **ca** (Blocked Vessels) - 12.54%
3. **thalach** (Max Heart Rate) - 10.96%
4. **cp** (Chest Pain Type) - 10.73%
5. **thal** (Thalassemia) - 9.41%

### Model Training Details

#### TabNet Deep Learning
- **Training Time**: ~2-3 minutes
- **Iterations**: Converges at ~50-80 epochs
- **Batch Size**: Dynamic (based on dataset size)
- **Early Stopping**: Yes (patience=20)
- **Validation**: 20% train split for validation

#### MLP Neural Network
- **Training Time**: ~30 seconds
- **Iterations**: 28 epochs (early stopped)
- **Architecture**: 128→64→32 neurons
- **Validation Score**: 96% during training
- **Final Loss**: 0.313

---

## 🔧 Training Scripts

### Train TabNet Model

```bash
python train_tabnet_model.py
```

**Output**:
- `models/tabnet_model.zip` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/feature_columns.pkl` - Column names
- `models/metadata.pkl` - Training metadata

**Features**:
- ✅ Automatic column mapping
- ✅ Dynamic batch size calculation
- ✅ Early stopping
- ✅ Feature importance extraction
- ✅ Comprehensive evaluation metrics

### Train MLP Model

```bash
python train_mlp_model.py
```

**Output**:
- `models/mlp_model.pkl` - Trained MLP model
- `models/mlp_scaler.pkl` - Feature scaler
- `models/mlp_feature_columns.pkl` - Column names
- `models/mlp_metadata.pkl` - Training metadata

**Features**:
- ✅ 3-layer architecture (128-64-32)
- ✅ Adaptive learning rate
- ✅ L2 regularization
- ✅ Early stopping
- ✅ Verbose training progress

### Validate Models and Data

```bash
python validate_model_and_data.py
```

**Checks**:
- ✅ Dataset structure and quality
- ✅ Missing values and duplicates
- ✅ Class distribution balance
- ✅ Feature statistics by disease status
- ✅ Model training success
- ✅ Prediction consistency

---

## 📱 Application Pages

### 1. 🏠 Home Page
- **Overview Statistics**: Total patients, disease vs healthy distribution
- **Interactive Charts**: Age distribution, gender breakdown
- **Key Health Indicators**: Average values for critical features
- **Quick Start Guide**: Navigation help

### 2. 📊 Data Exploration Page
- **Dataset Statistics**: Comprehensive data summary
- **Distribution Analysis**: Visualize feature distributions
- **Correlation Analysis**: Feature relationship heatmap
- **Scatter Plots**: 2D feature comparisons
- **Box Plots**: Outlier detection
- **Categorical Analysis**: Count plots for categorical features

### 3. 🔮 Prediction Page
- **Input Form**: 16 feature inputs with tooltips
- **Model Selection**: Choose from 5 algorithms
- **Real-Time Prediction**: Instant risk assessment
- **Risk Probability Gauge**: Visual risk meter
- **Confidence Scores**: Probability breakdown
- **Feature Importance**: Top 10 influential features
- **Dataset Comparison**: Your values vs disease/healthy averages
- **Risk Indicator**: Shows if values are closer to disease pattern

### 4. 📈 Model Performance Page
- **All Models Comparison**: Side-by-side accuracy
- **ROC Curves**: Visual performance comparison
- **Confusion Matrix**: Detailed prediction breakdown
- **Classification Reports**: Precision, recall, F1-scores
- **Training Details**: Convergence information

### 5. ℹ️ About Page
- **Project Information**: Goals and methodology
- **Dataset Details**: Source and characteristics
- **Model Descriptions**: Algorithm explanations
- **Team Information**: Credits and contact

---

## 📦 Dependencies

### Core Libraries
```
streamlit==1.53.0
pandas==2.3.3
numpy==2.4.1
scikit-learn==1.8.0
```

### Visualization
```
plotly==6.5.2
seaborn==0.13.2
matplotlib==3.10.8
```

### Deep Learning
```
torch==2.10.0
pytorch-tabnet==4.1.0
```

### Utilities
```
pillow==12.1.0
python-dateutil==2.9.0
```

### Full List
See `requirements.txt` for complete dependency list with versions.

---

## 🎯 Results

### Key Achievements

✅ **High Accuracy**: Best models (Random Forest & TabNet) achieve 81.97% accuracy

✅ **Multiple Algorithms**: 5 different approaches for robust predictions

✅ **Deep Learning Integration**: TabNet and MLP for advanced pattern recognition

✅ **Feature Transparency**: Clear feature importance for every prediction

✅ **Balanced Approach**: Proper handling of class imbalance

✅ **User-Friendly Interface**: Clean, intuitive Streamlit UI

✅ **100% Data-Driven**: No hardcoded rules or manual thresholds

### Interesting Findings

🔍 **Dataset Patterns**:
- Younger patients more likely to have disease in this dataset
- Lower blood pressure associated with disease (unusual)
- Chest pain type and thalassemia are strong indicators

🔍 **Model Insights**:
- Tree-based models (RF, GB) perform best
- Deep learning (TabNet) competitive at 83.61%
- Feature combinations matter more than individual values

🔍 **Clinical Utility**:
- ST depression (oldpeak) is most important feature
- Number of blocked vessels (ca) strongly predictive
- Max heart rate trends differ between groups

---

## 🚀 Future Improvements

### Planned Features
- [ ] **SHAP Values**: Advanced explainability with SHAP plots
- [ ] **Cross-Validation**: K-fold validation for robust metrics
- [ ] **Hyperparameter Tuning**: Grid search for optimal parameters
- [ ] **Ensemble Model**: Combine predictions from multiple models
- [ ] **API Endpoint**: REST API for external integrations
- [ ] **PDF Reports**: Download prediction reports
- [ ] **Historical Tracking**: Track predictions over time
- [ ] **More Data**: Expand dataset with additional sources

### Technical Enhancements
- [ ] **Docker Container**: Containerized deployment
- [ ] **CI/CD Pipeline**: Automated testing and deployment
- [ ] **Database Integration**: Store predictions in database
- [ ] **User Authentication**: Secure access control
- [ ] **Cloud Deployment**: Deploy to AWS/Azure/GCP
- [ ] **Mobile App**: Native mobile application

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Changes**
   ```bash
   git commit -m "Add YourFeature"
   ```
4. **Push to Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open Pull Request**

### Contribution Guidelines
- Write clean, documented code
- Add tests for new features
- Update README for significant changes
- Follow PEP 8 style guide
- Ensure all models train successfully

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Heart Disease Prediction Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact & Support

### Questions or Issues?
- **Email**: support@heartpredict.com
- **GitHub Issues**: [Report a Bug](https://github.com/yourusername/heart_disease_predict/issues)
- **Documentation**: See this README

### Acknowledgments
- **UCI Machine Learning Repository** - Dataset source
- **Streamlit Team** - Amazing framework
- **Scikit-learn Contributors** - ML tools
- **PyTorch TabNet Team** - TabNet implementation

---

## 🎓 Educational Use

This project is designed for:
- **Learning Machine Learning**: Understand different algorithms
- **Healthcare Analytics**: Apply ML to medical data
- **Streamlit Development**: Build interactive apps
- **Deep Learning**: Explore neural networks in healthcare
- **Model Comparison**: Evaluate multiple approaches

**⚠️ Disclaimer**: This is an educational project. Do NOT use for actual medical diagnosis. Always consult healthcare professionals for medical decisions.

---

## 📊 Quick Start Summary

```bash
# 1. Setup
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt

# 2. Train Models (optional - models already trained)
python train_tabnet_model.py
python train_mlp_model.py

# 3. Validate
python validate_model_and_data.py

# 4. Run Application
streamlit run app.py

# 5. Access at http://localhost:8501
```

---

## 🏆 Project Highlights

- ✨ **5 ML Algorithms**: Classical to deep learning
- 📊 **303 Patient Records**: Real-world medical data
- 🎯 **81.97% Accuracy**: Best model performance (Random Forest & TabNet tied)
- 🧠 **Deep Learning**: TabNet + MLP neural networks
- 📈 **Interactive UI**: Streamlit web application
- 🔍 **Feature Importance**: Data-driven insights
- ⚡ **Real-Time Predictions**: Instant risk assessment
- 📱 **Responsive Design**: Works on all devices

---

**Built with ❤️ using Python, Streamlit, and Machine Learning**

**Version**: 2.0.0  
**Last Updated**: February 9, 2026  
**Status**: Production Ready ✅
