# ❤️ Heart Disease Prediction System

<div align="center">

![Heart Disease Prediction](https://img.shields.io/badge/Heart%20Disease-Prediction-red)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53.0-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An advanced machine learning-powered web application for cardiovascular risk assessment**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-machine-learning-models) • [Dataset](#-dataset) • [Credits](#-credits)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Machine Learning Models](#-machine-learning-models)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [Disclaimer](#%EF%B8%8F-disclaimer)
- [Credits](#-credits)
- [License](#-license)

---

## 🎯 Overview

The **Heart Disease Prediction System** is a comprehensive web application that leverages machine learning algorithms to predict the risk of heart disease based on clinical parameters. Built with Streamlit and Python, this application provides an intuitive interface for healthcare professionals and researchers to assess cardiovascular risk quickly and accurately.

The system analyzes **13 key health indicators** including age, blood pressure, cholesterol levels, and ECG results to provide real-time predictions with confidence scores. It features multiple ML models, interactive visualizations, and detailed performance metrics.

---

## ✨ Features

### 🏠 Home Dashboard
- **Real-time Statistics**: Overview of dataset metrics
- **Interactive Visualizations**: Age and gender distribution charts
- **Key Health Indicators**: Average values for critical parameters

### 📊 Data Exploration
- **Dataset Preview**: Browse through patient records
- **Statistical Analysis**: Comprehensive data statistics
- **Feature Distributions**: Analyze individual feature patterns
- **Correlation Analysis**: Discover relationships between variables
- **Advanced Scatter Plots**: Multi-dimensional data visualization

### 🔮 Heart Disease Prediction
- **User-Friendly Input Form**: Enter patient clinical parameters
- **Multiple ML Models**: Choose from 4 different algorithms
- **Real-time Predictions**: Instant risk assessment
- **Probability Gauges**: Visual representation of risk levels
- **Personalized Recommendations**: Health advice based on predictions

### 📈 Model Performance
- **Model Comparison**: Side-by-side accuracy comparison
- **Confusion Matrix**: Detailed classification metrics
- **ROC Curves**: AUC scores and performance visualization
- **Classification Reports**: Precision, recall, and F1-scores

### ℹ️ About
- **Clinical Parameters Guide**: Detailed explanation of all features
- **Data Source Information**: UCI Heart Disease Dataset details
- **Credits & Acknowledgments**: Research contributors

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.12**: Primary programming language
- **Streamlit 1.53.0**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Machine Learning
- **Scikit-learn**: ML algorithms and preprocessing
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
  - Support Vector Machine (SVM)

### Visualization
- **Plotly Express & Graph Objects**: Interactive charts
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical data visualization

### Additional Libraries
- **PIL (Pillow)**: Image processing for logo integration
- **StandardScaler**: Feature normalization

---

## 💻 Installation

### Prerequisites
- Python 3.12 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd heart_disease_predict
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# Linux/Mac
python3 -m venv myenv
source myenv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
Ensure all required packages are installed:
- streamlit
- numpy
- pandas
- seaborn
- scikit-learn
- plotly
- matplotlib
- pillow

---

## 🚀 Usage

### Running the Application

1. **Activate Virtual Environment** (if not already activated)
   ```bash
   # Windows
   myenv\Scripts\activate
   
   # Linux/Mac
   source myenv/bin/activate
   ```

2. **Start the Streamlit Server**
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - The application will automatically open in your default browser

### Using the Application

#### 1. Home Page
- View overall statistics of the dataset
- Explore age and gender distributions
- Check average values for key health indicators

#### 2. Data Exploration
- **Dataset Preview**: Browse patient records
- **Distributions**: Select features to analyze their distribution
- **Correlations**: View correlation heatmap and top correlations
- **Advanced Analysis**: Compare two features with scatter plots

#### 3. Making Predictions
- Navigate to the **Prediction** page
- Enter patient information:
  - **Demographics**: Age, Gender
  - **Clinical Measurements**: Blood Pressure, Cholesterol, Heart Rate
  - **Medical Tests**: ECG Results, Exercise Tests, Thalassemia Type
- Select a machine learning model
- Click **"🔮 Predict Risk"**
- View results:
  - Risk classification (Low/High)
  - Probability gauge
  - Confidence scores
  - Personalized recommendations

#### 4. Model Performance
- Compare accuracy across all models
- View detailed metrics for the best-performing model
- Analyze confusion matrix, ROC curves, and classification reports

---

## 🤖 Machine Learning Models

The application implements **4 different machine learning algorithms**:

### 1. Random Forest Classifier
- **Type**: Ensemble Learning
- **Description**: Combines multiple decision trees
- **Strengths**: High accuracy, handles non-linear relationships
- **Parameters**: 100 estimators, random_state=42

### 2. Gradient Boosting Classifier
- **Type**: Boosting Ensemble
- **Description**: Sequential ensemble of weak learners
- **Strengths**: Excellent performance, feature importance
- **Parameters**: 100 estimators, random_state=42

### 3. Logistic Regression
- **Type**: Linear Model
- **Description**: Probabilistic classification
- **Strengths**: Fast, interpretable, good baseline
- **Parameters**: max_iter=1000, random_state=42

### 4. Support Vector Machine (SVM)
- **Type**: Kernel-based Model
- **Description**: Finds optimal hyperplane
- **Strengths**: Effective in high dimensions
- **Parameters**: probability=True, random_state=42

### Model Training Process
1. **Data Split**: 80% training, 20% testing
2. **Feature Scaling**: StandardScaler normalization
3. **Cross-validation**: Random state for reproducibility
4. **Caching**: Streamlit caching for performance

---

## 📊 Dataset

### UCI Heart Disease Dataset

**Source**: Cleveland Clinic Foundation, Hungarian Institute of Cardiology, University Hospital Zurich, V.A. Medical Center Long Beach

**Statistics**:
- **Total Records**: 303 patients
- **Features**: 14 attributes (13 input features + 1 target)
- **Missing Values**: None (preprocessed)

### Features Description

| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| **age** | Age in years | Numeric | 20-100 |
| **sex** | Gender | Binary | 0 = Female, 1 = Male |
| **cp** | Chest pain type | Categorical | 0-3 (Typical Angina to Asymptomatic) |
| **trestbps** | Resting blood pressure | Numeric | 80-200 mm Hg |
| **chol** | Serum cholesterol | Numeric | 100-400 mg/dl |
| **fbs** | Fasting blood sugar > 120 mg/dl | Binary | 0 = No, 1 = Yes |
| **restecg** | Resting ECG results | Categorical | 0-2 (Normal to LV Hypertrophy) |
| **thalach** | Maximum heart rate achieved | Numeric | 60-220 bpm |
| **exang** | Exercise induced angina | Binary | 0 = No, 1 = Yes |
| **oldpeak** | ST depression induced by exercise | Numeric | 0.0-6.0 |
| **slope** | Slope of peak exercise ST segment | Categorical | 0-2 (Upsloping to Downsloping) |
| **ca** | Number of major vessels | Numeric | 0-3 |
| **thal** | Thalassemia type | Categorical | 1-3 (Normal to Reversible Defect) |
| **target** | Heart disease diagnosis | Binary | 0 = No Disease, 1 = Disease |

### Data Distribution
- **Healthy Patients**: ~54%
- **Heart Disease Patients**: ~46%
- **Age Range**: 29-77 years
- **Gender**: 68% Male, 32% Female

---

## 📁 Project Structure

```
heart_disease_predict/
│
├── app.py                      # Main Streamlit application
├── heart.csv                   # UCI Heart Disease dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── EDA_analysis.ipynb         # Exploratory Data Analysis notebook
│
├── image/
│   └── logo.webp              # Application logo
│
└── myenv/                     # Virtual environment (not in repo)
    ├── Scripts/
    ├── Lib/
    └── Include/
```

### File Descriptions

- **app.py**: Main application file containing all pages, visualizations, and ML logic
- **heart.csv**: Dataset file with 303 patient records
- **requirements.txt**: List of Python package dependencies
- **EDA_analysis.ipynb**: Jupyter notebook for exploratory data analysis
- **image/logo.webp**: Application logo and branding
- **myenv/**: Python virtual environment (excluded from version control)

---

## 📸 Screenshots

### Home Dashboard
- Clean, modern interface with gradient design
- Real-time statistics and key metrics
- Interactive age and gender distribution charts

### Data Exploration
- Full-width dataset preview
- Side-by-side statistics and target distribution
- Correlation heatmaps with color-coded values
- Dynamic feature comparison scatter plots

### Prediction Interface
- User-friendly input form with sliders and dropdowns
- Real-time prediction with animated progress bar
- Risk probability gauge with color-coded zones
- Personalized health recommendations

### Model Performance
- Bar chart comparison of all models
- Confusion matrix with true/false positives
- ROC curves with AUC scores
- Detailed classification metrics

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Areas for Contribution
- Additional ML models (XGBoost, Neural Networks)
- Enhanced visualizations
- Mobile responsiveness improvements
- API integration
- Unit tests and documentation

---

## ⚠️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER**

This application is designed for **educational and research purposes only**. It is **NOT** intended to be used as:
- A substitute for professional medical advice
- A diagnostic tool for clinical use
- A replacement for qualified healthcare providers

### Key Points:
- ❌ **Do NOT** use this application for actual medical diagnosis
- ❌ **Do NOT** make treatment decisions based solely on predictions
- ✅ **Always consult** with qualified healthcare professionals
- ✅ **Seek immediate medical attention** for symptoms of heart disease
- ✅ **Use predictions** as supplementary information only

**Symptoms requiring immediate medical attention:**
- Chest pain or discomfort
- Shortness of breath
- Pain in arms, back, neck, jaw, or stomach
- Cold sweats, nausea, lightheadedness

**Call emergency services (911 or local emergency number) immediately if experiencing these symptoms.**

The developers and contributors assume **no liability** for any medical decisions or outcomes based on the use of this application.

---

## 👨‍💻 Credits

### Developer
**Md Nurullah**
- 💼 Role: Lead Developer & Data Scientist
- 📧 Contact: [mdnurullah.co@gmail.com](mailto:mdnurullah.co@gmail.com)
- 🔗 GitHub: [github.com/SheikhNoor](https://github.com/SheikhNoor)
- 💼 LinkedIn: [linkedin.com/in/md-nurullah-1481b7253](https://www.linkedin.com/in/md-nurullah-1481b7253/)

### Dataset Credits
**Principal Investigators:**
- Dr. Andras Janosi - Hungarian Institute of Cardiology, Budapest
- Dr. William Steinbrunn - University Hospital, Zurich, Switzerland
- Dr. Matthias Pfisterer - University Hospital, Basel, Switzerland
- Dr. Robert Detrano - V.A. Medical Center, Long Beach, and Cleveland Clinic Foundation

**Data Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

### Technologies & Frameworks
- **Streamlit**: Open-source app framework
- **Scikit-learn**: Machine learning library
- **Plotly**: Interactive visualization library
- **Python Community**: For extensive package ecosystem

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Md Nurullah

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

## 📞 Support

For questions, issues, or suggestions:

- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Feature Requests**: Submit a feature request
- 📧 **Email**: [mdnurullah.co@gmail.com](mailto:mdnurullah.co@gmail.com)
- 💬 **GitHub**: [github.com/SheikhNoor](https://github.com/SheikhNoor)
- 💬 **LinkedIn**: [linkedin.com/in/md-nurullah-1481b7253](https://www.linkedin.com/in/md-nurullah-1481b7253/)

---

## 🙏 Acknowledgments

Special thanks to:
- UCI Machine Learning Repository for the dataset
- Streamlit team for the amazing framework
- Scikit-learn developers for ML tools
- Open-source community for continuous support

---

<div align="center">

### 🌟 If you find this project helpful, please consider giving it a star! 🌟

**Made with ❤️ by Md Nurullah**

🏥 Advanced Healthcare Analytics | 🤖 Powered by Machine Learning | 💚 For Better Health Outcomes

</div>

---

**Version**: 1.0.0  
**Last Updated**: January 22, 2026  
**Status**: Active Development
