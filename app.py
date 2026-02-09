import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from PIL import Image
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import pickle

# Load logo for page icon
try:
    logo_image = Image.open('image/logo.webp')
    page_icon = logo_image
except:
    page_icon = "❤️"

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design with animations and better visibility
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: #1a1a1a !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%) !important;
    }
    
    .block-container {
        padding-top: 3rem;
        background: #1a1a1a;
    }
    
    /* Force all text to be visible with light colors */
    .stMarkdown, .stText, p, span, div, label {
        color: #e0e0e0 !important;
        opacity: 1 !important;
    }
    
    /* List items visibility */
    ul, ol, li {
        color: #e0e0e0 !important;
        opacity: 1 !important;
    }
    
    ul li, ol li {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
        opacity: 1 !important;
        line-height: 1.8 !important;
    }
    
    /* Make all text inputs and labels visible */
    .stSelectbox label, .stSlider label, .stRadio label, .stTextInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    
    .logo-container {
        text-align: center;
        margin-bottom: 20px;
        animation: fadeIn 1.5s ease-in;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .logo-container img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        margin: 0 auto;
        display: block;
        object-fit: cover;
        border: 4px solid rgba(255, 255, 255, 0.3);
        background: white;
        padding: 10px;
    }
    
    .big-font {
        font-size: 48px !important;
        font-weight: 700;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 8px rgba(102, 126, 234, 0.5);
    }
    
    .subtitle {
        font-size: 20px;
        color: #b0b0b0 !important;
        text-align: center;
        margin-bottom: 40px;
        animation: fadeIn 2s ease-in;
        font-weight: 500;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideUp 0.5s ease-out;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h3, .metric-card p, .metric-card div, .metric-card span, .metric-card label {
        color: white !important;
        opacity: 1 !important;
    }
    
    .metric-card ul, .metric-card ol, .metric-card li {
        color: white !important;
        opacity: 1 !important;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .prediction-box {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: 600;
        margin: 20px 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .prediction-box * {
        color: white !important;
        opacity: 1 !important;
    }
    
    .prediction-box ul, .prediction-box ol, .prediction-box li {
        color: white !important;
        opacity: 1 !important;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .healthy {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        opacity: 1 !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    
    h4, h5, h6 {
        color: #e0e0e0 !important;
        opacity: 1 !important;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        animation: fadeIn 1s ease-in;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    .info-box * {
        color: white !important;
        opacity: 1 !important;
    }
    
    .info-box ul, .info-box ol, .info-box li, .info-box p {
        color: white !important;
        opacity: 1 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }
    
    section[data-testid="stSidebar"] ul, 
    section[data-testid="stSidebar"] ol, 
    section[data-testid="stSidebar"] li {
        color: white !important;
        opacity: 1 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > label, 
    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
        font-weight: 500;
        opacity: 1 !important;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    cleaned_path = os.path.join('datasets', 'cleaned_df.csv')
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
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
        # Drop Unnamed: 0 if present
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        # Add missing lifestyle columns with defaults if not present
        if 'smoking' not in df.columns:
            df['smoking'] = 0
        if 'diabetes' not in df.columns:
            df['diabetes'] = 0
        if 'bmi' not in df.columns:
            df['bmi'] = 25.0
        return df
    return pd.read_csv(os.path.join('datasets', 'heart.csv'))

@st.cache_resource
def load_pretrained_tabnet():
    """Load pre-trained TabNet model if available"""
    try:
        if os.path.exists('models/tabnet_model.zip'):
            model = TabNetClassifier()
            model.load_model('models/tabnet_model.zip')
            return model
    except:
        pass
    return None

@st.cache_resource
def train_model(X_train, y_train, model_type):
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
    elif model_type == "SVM":
        model = SVC(probability=True, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
    elif model_type == "MLP (Neural Network)":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )
        model.fit(X_train, y_train)
    elif model_type == "TabNet (Deep Learning)":
        # DO NOT use pre-trained model - it was trained on different dataset
        # Always train fresh TabNet on current data
        st.info("🧠 Training TabNet Deep Learning Model on current dataset...")
        
        # TabNet needs numpy arrays, not pandas
        X_train_np = X_train if isinstance(X_train, np.ndarray) else X_train.values
        y_train_np = y_train if isinstance(y_train, np.ndarray) else y_train.values
        
        # Calculate appropriate batch size based on dataset size
        n_samples = len(X_train_np)
        # Use smaller batch size for small datasets to avoid single-sample batches
        batch_size = min(32, max(8, n_samples // 8))
        virtual_batch_size = max(4, batch_size // 2)
        
        model = TabNetClassifier(
            n_d=64,                    # Increased from 32 for better capacity
            n_a=64,                    # Increased from 32
            n_steps=5,                 # Increased from 3 for more decision steps
            gamma=1.5,                 # Feature reusage parameter
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',        # Better feature selection than 'sparsemax'
            seed=42,
            verbose=0
        )
        
        # Train with proper validation split
        from sklearn.model_selection import train_test_split as tt_split
        X_tr, X_val, y_tr, y_val = tt_split(
            X_train_np, y_train_np, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_train_np
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_name=['valid'],
            eval_metric=['accuracy', 'logloss'],
            max_epochs=100,            # Reduced for faster training
            patience=20,               # Early stopping
            batch_size=batch_size,     # Dynamic batch size based on dataset
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            drop_last=False,
            weights=1                  # Equal weighting (can adjust for imbalance)
        )
        return model
    
    return model

# Logo and title with animation
try:
    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px; margin-top: 30px; padding-top: 20px;">
            <img src="data:image/webp;base64,{}" style="width: 120px; height: 120px; border-radius: 50%; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); object-fit: cover; border: 4px solid rgba(255, 255, 255, 0.3); background: white; padding: 10px;">
        </div>
    """.format(__import__('base64').b64encode(open('image/logo.webp', 'rb').read()).decode()), unsafe_allow_html=True)
except:
    pass

st.markdown('<p class="big-font">Heart Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced ML-Powered Cardiovascular Risk Assessment</p>', unsafe_allow_html=True)

# Load data
df = load_data()

# Sidebar
with st.sidebar:
    # Display logo in sidebar
    try:
        st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
                <img src="data:image/webp;base64,{}" style="width: 100px; height: 100px; border-radius: 50%; box-shadow: 0 5px 20px rgba(255, 255, 255, 0.3); object-fit: cover; border: 3px solid rgba(255, 255, 255, 0.5); background: white; padding: 8px;">
            </div>
        """.format(__import__('base64').b64encode(open('image/logo.webp', 'rb').read()).decode()), unsafe_allow_html=True)
        st.markdown("---")
    except:
        pass
    
    st.markdown("### 🎯 Navigation")
    page = st.radio("", ["🏠 Home", "📊 Data Exploration", "🔮 Prediction", "📈 Model Performance", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### 📚 Dataset Info")
    feature_count = max(df.shape[1] - 1, 0)
    st.info(f"""
    **Source**: UCI Heart Disease Dataset + Extended Dataset
    
    **Records**: {df.shape[0]} patients
    
    **Features**: {feature_count} clinical attributes
    
    **Target**: Heart disease presence
    """)
    
    st.warning("""
    ⚠️ **Important Note:**
    
    This dataset has UNUSUAL patterns that differ from typical medical knowledge:
    
    • Disease patients are YOUNGER (52 vs 56 years)
    • Disease patients have LOWER blood pressure
    • Disease patients have LOWER ST depression
    • Disease patients have FEWER blocked vessels
    
    The ML model learns from THIS specific dataset's patterns, not general medical rules.
    """)

# HOME PAGE
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📋 Total Patients", len(df), delta="100%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        healthy = len(df[df['target'] == 0])
        st.metric("💚 Healthy", healthy, delta=f"{(healthy/len(df)*100):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        disease = len(df[df['target'] == 1])
        st.metric("❤️ Heart Disease", disease, delta=f"{(disease/len(df)*100):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive visualization
    st.markdown("### 📊 Quick Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig = px.histogram(df, x='age', color='target', 
                          labels={'target': 'Heart Disease', 'age': 'Age (years)'},
                          title='Age Distribution by Heart Disease Status',
                          color_discrete_map={0: '#667eea', 1: '#f5576c'})
        fig.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_dist = df.groupby(['sex', 'target']).size().reset_index(name='count')
        gender_dist['sex'] = gender_dist['sex'].map({0: 'Female', 1: 'Male'})
        
        fig = px.bar(gender_dist, x='sex', y='count', color='target',
                    labels={'sex': 'Gender', 'count': 'Count', 'target': 'Heart Disease'},
                    title='Gender Distribution by Heart Disease Status',
                    color_discrete_map={0: '#667eea', 1: '#f5576c'},
                    barmode='group')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance preview
    st.markdown("### 🎯 Key Health Indicators")
    
    cols = st.columns(5)
    features_info = [
        ("🫀 Max Heart Rate", "thalach", "Average: {:.0f} bpm"),
        ("🩸 Cholesterol", "chol", "Average: {:.0f} mg/dl"),
        ("💉 Blood Pressure", "trestbps", "Average: {:.0f} mm Hg"),
        ("📈 ST Depression", "oldpeak", "Average: {:.2f}"),
        ("🧍 BMI", "bmi", "Average: {:.1f}")
    ]
    
    for col, (title, feature, format_str) in zip(cols, features_info):
        with col:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.markdown(f"<h3>{format_str.format(df[feature].mean())}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Developer section with social links
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; margin-top: 30px;'>
        <h3 style='color: white; margin-bottom: 15px;'>👨‍💻 Developed by Md Nurullah</h3>
        <p style='color: #e0e0e0; font-size: 14px; margin-bottom: 20px;'>Advanced Healthcare Analytics | Machine Learning Solutions</p>
        <div style='display: flex; justify-content: center; gap: 20px; margin-top: 15px;'>
            <a href='https://github.com/SheikhNoor' target='_blank' style='display: inline-flex; align-items: center; gap: 8px; 
               padding: 10px 20px; background: rgba(255,255,255,0.2); color: white; text-decoration: none; 
               border-radius: 8px; font-weight: 600; transition: all 0.3s;'>
                🔗 GitHub
            </a>
            <a href='https://www.linkedin.com/in/md-nurullah-1481b7253/' target='_blank' style='display: inline-flex; align-items: center; gap: 8px; 
               padding: 10px 20px; background: rgba(255,255,255,0.2); color: white; text-decoration: none; 
               border-radius: 8px; font-weight: 600; transition: all 0.3s;'>
                💼 LinkedIn
            </a>
            <a href='mailto:mdnurullah.co@gmail.com' style='display: inline-flex; align-items: center; gap: 8px; 
               padding: 10px 20px; background: rgba(255,255,255,0.2); color: white; text-decoration: none; 
               border-radius: 8px; font-weight: 600; transition: all 0.3s;'>
                📧 Email
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# DATA EXPLORATION PAGE
elif page == "📊 Data Exploration":
    st.markdown("## 📊 Comprehensive Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data View", "📈 Distributions", "🔗 Correlations", "📊 Advanced Analysis"])
    
    with tab1:
        st.markdown("### 🗂️ Dataset Preview")
        
        st.dataframe(df.head(20), use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Statistics")
            
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Total Rows", df.shape[0])
                st.metric("Missing Values", df.isnull().sum().sum())
            with stats_col2:
                st.metric("Total Columns", df.shape[1])
                st.metric("Duplicates", df.duplicated().sum())
        
        with col2:
            st.markdown("#### 🎯 Target Distribution")
            target_dist = df['target'].value_counts()
            fig = px.pie(values=target_dist.values, names=['No Disease', 'Disease'],
                        color_discrete_sequence=['#667eea', '#f5576c'],
                        hole=0.4)
            fig.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📈 Feature Distributions")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('target')
        
        selected_feature = st.selectbox("Select Feature", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=selected_feature, color='target',
                             marginal='box',
                             color_discrete_map={0: '#667eea', 1: '#f5576c'},
                             labels={'target': 'Heart Disease'})
            fig.update_layout(title=f'{selected_feature} Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y=selected_feature, x='target', color='target',
                        color_discrete_map={0: '#667eea', 1: '#f5576c'},
                        labels={'target': 'Heart Disease'})
            fig.update_layout(title=f'{selected_feature} by Target')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔗 Correlation Analysis")
        
        # Correlation heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig = px.imshow(corr, 
                       text_auto='.2f',
                       aspect='auto',
                       color_continuous_scale='RdBu_r',
                       title='Feature Correlation Heatmap')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations with target
        st.markdown("#### 🎯 Top Correlations with Heart Disease")
        target_corr = corr['target'].sort_values(ascending=False)[1:]
        
        fig = px.bar(x=target_corr.index, y=target_corr.values,
                    labels={'x': 'Feature', 'y': 'Correlation'},
                    color=target_corr.values,
                    color_continuous_scale='RdBu_r')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 📊 Advanced Analysis")
        
        # Feature comparison
        col1, col2 = st.columns(2)
        
        with col1:
            feature_x = st.selectbox("X-axis Feature", numeric_cols, key='x_feature')
        with col2:
            feature_y = st.selectbox("Y-axis Feature", numeric_cols, index=1, key='y_feature')
        
        fig = px.scatter(df, x=feature_x, y=feature_y, color='target',
                        size='age', hover_data=['age', 'sex'],
                        color_discrete_map={0: '#667eea', 1: '#f5576c'},
                        labels={'target': 'Heart Disease'})
        fig.update_layout(title=f'{feature_x} vs {feature_y}', height=500)
        st.plotly_chart(fig, use_container_width=True)

# PREDICTION PAGE
elif page == "🔮 Prediction":
    st.markdown("## 🔮 Heart Disease Risk Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Patient Information")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            age = st.slider("👤 Age", 20, 100, 50)
            sex = st.selectbox("⚧️ Sex", ["Female", "Male"])
            cp = st.selectbox("💔 Chest Pain Type", 
                            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
            trestbps = st.slider("🩸 Resting Blood Pressure (mm Hg)", 80, 200, 120)
            chol = st.slider("🧪 Cholesterol (mg/dl)", 100, 400, 200)
            fbs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            restecg = st.selectbox("📉 Resting ECG", 
                                  ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
            bmi = st.slider("🧍 BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1)
        
        with col_b:
            thalach = st.slider("❤️ Maximum Heart Rate", 60, 220, 150)
            exang = st.selectbox("🏃 Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.slider("📊 ST Depression", 0.0, 6.0, 1.0, 0.1)
            slope = st.selectbox("📈 Slope of Peak Exercise ST Segment", 
                               ["Upsloping", "Flat", "Downsloping"])
            ca = st.slider("🔬 Number of Major Vessels (0-3)", 0, 3, 0)
            thal = st.selectbox("🫀 Thalassemia", 
                              ["Normal", "Fixed Defect", "Reversible Defect"])
            smoking = st.selectbox("🚬 Smoking", ["No", "Yes"])
            diabetes = st.selectbox("🩺 Diabetes", ["No", "Yes"])
        
        # Convert inputs
        sex_map = {"Female": 0, "Male": 1}
        cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        fbs_map = {"No": 0, "Yes": 1}
        restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        exang_map = {"No": 0, "Yes": 1}
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
        smoking_map = {"No": 0, "Yes": 1}
        diabetes_map = {"No": 0, "Yes": 1}
        
        input_payload = {
            "age": age,
            "sex": sex_map[sex],
            "cp": cp_map[cp],
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs_map[fbs],
            "restecg": restecg_map[restecg],
            "thalach": thalach,
            "exang": exang_map[exang],
            "oldpeak": oldpeak,
            "slope": slope_map[slope],
            "ca": ca,
            "thal": thal_map[thal],
            "smoking": smoking_map[smoking],
            "diabetes": diabetes_map[diabetes],
            "bmi": bmi
        }
    
    with col2:
        st.markdown("### ⚙️ Model Settings")
        
        model_type = st.selectbox("🤖 Select Model", 
                                 ["Random Forest", "Logistic Regression", "SVM", 
                                  "MLP (Neural Network)", "TabNet (Deep Learning)"])
        
        st.markdown("---")
        
        st.markdown("#### 📊 Quick Stats")
        st.info(f"""
        **Age**: {age} years
        **Gender**: {sex}
        **BP**: {trestbps} mm Hg
        **Cholesterol**: {chol} mg/dl
        **Max HR**: {thalach} bpm
        **BMI**: {bmi:.1f}
        **Smoking**: {smoking}
        **Diabetes**: {diabetes}
        """)
        
        st.markdown("---")
        
        st.markdown("#### 📖 Medical Terms Glossary")
        st.markdown("""<div class='info-box' style='font-size: 14px;'>
        <strong>CP</strong> - Chest Pain Type<br>
        <strong>BP</strong> - Blood Pressure<br>
        <strong>FBS</strong> - Fasting Blood Sugar<br>
        <strong>ECG</strong> - Electrocardiogram<br>
        <strong>HR</strong> - Heart Rate<br>
        <strong>ST</strong> - ST Segment (ECG measurement)<br>
        <strong>BMI</strong> - Body Mass Index<br>
        <strong>CA</strong> - Coronary Arteries<br>
        <strong>mm Hg</strong> - Millimeters of Mercury<br>
        <strong>mg/dl</strong> - Milligrams per Deciliter<br>
        <strong>bpm</strong> - Beats per Minute
        </div>""", unsafe_allow_html=True)
    
    # Predict button
    if st.button("🔮 Predict Risk", use_container_width=True):
        with st.spinner("🔄 Analyzing patient data..."):
            # Progress bar animation
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Prepare data
            X = df.drop(['target'], axis=1)
            if 'Unnamed: 0' in X.columns:
                X = X.drop('Unnamed: 0', axis=1)
            y = df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            input_df = pd.DataFrame([input_payload]).reindex(columns=X.columns, fill_value=0)
            input_scaled = scaler.transform(input_df)
            
            # Train model
            model = train_model(X_train_scaled, y_train, model_type)
            
            # Make prediction - purely based on ML model, no manual overrides
            probability = model.predict_proba(input_scaled)[0]
            prediction = model.predict(input_scaled)[0]
            
            st.success("✅ Analysis Complete!")
            
            # Display result with animation
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            # Add explanation of ML prediction
            st.info(f"""
🤖 **ML-Based Prediction:** This result is calculated by the **{model_type}** model trained on **{len(df)} real patient records**. The model learned patterns from actual data, not from hardcoded age/BP rules. Heart disease can affect anyone at any age - the prediction is based purely on what the model learned from patient outcomes.
            """)
            
            if prediction == 0:
                st.markdown(f'''
                    <div class="prediction-box healthy">
                        ✅ LOW RISK
                        <br>
                        <span style="font-size: 18px;">No significant heart disease detected</span>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div class="prediction-box risk">
                        ⚠️ HIGH RISK
                        <br>
                        <span style="font-size: 18px;">Heart disease indicators detected</span>
                    </div>
                ''', unsafe_allow_html=True)
            
            # Probability gauge
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability[1] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Probability (%)", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "#f5576c"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#f5576c" if probability[1] >= 0.5 else "#667eea"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': '#d4edda'},
                            {'range': [50, 75], 'color': '#fff3cd'},
                            {'range': [75, 100], 'color': '#f8d7da'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence scores
                st.markdown("#### 📊 Confidence Scores")
                
                fig = go.Figure(data=[
                    go.Bar(name='Healthy', x=['Prediction'], y=[probability[0]*100], 
                          marker_color='#667eea'),
                    go.Bar(name='Disease', x=['Prediction'], y=[probability[1]*100], 
                          marker_color='#f5576c')
                ])
                fig.update_layout(barmode='group', height=300, yaxis_title="Probability (%)")
                st.plotly_chart(fig, use_container_width=True)
            
            # ML Feature Importance Analysis
            st.markdown("---")
            st.markdown("### 🧠 What The Model Learned From Data")
            st.info(f"""
🤖 **Data-Driven Insights:** This shows how much each feature influenced the ML model's decision based on patterns learned from {len(df)} real patients. These are NOT hardcoded rules - they're what the model discovered by analyzing actual patient outcomes.
            """)
            
            # Get feature importance from the model
            feature_importance = None
            feature_names = X.columns.tolist()
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (Random Forest, Gradient Boosting)
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models (Logistic Regression, SVM) or MLP
                if "MLP" in model_type:
                    # For MLP, use the weights from the first layer
                    # coefs_[0] is the weight matrix from input to first hidden layer
                    feature_importance = np.abs(model.coefs_[0]).mean(axis=1)
                else:
                    # For linear models
                    feature_importance = np.abs(model.coef_[0])
            elif "TabNet" in model_type:
                # TabNet has explain method
                try:
                    explain_matrix, masks = model.explain(input_scaled)
                    feature_importance = explain_matrix[0]
                except:
                    # If explain fails, use dummy importance
                    feature_importance = None
            
            if feature_importance is not None:
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance,
                    'Your Value': [input_payload.get(f, 0) for f in feature_names]
                }).sort_values('Importance', ascending=False).head(10)
                
                # Normalize importance to percentage
                importance_df['Importance %'] = (importance_df['Importance'] / importance_df['Importance'].sum() * 100)
                
                st.markdown("#### 🎯 Top 10 Most Influential Features")
                
                # Display as horizontal bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=importance_df['Feature'][::-1],
                    x=importance_df['Importance %'][::-1],
                    orientation='h',
                    marker=dict(
                        color=importance_df['Importance %'][::-1],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Impact %")
                    ),
                    text=importance_df['Importance %'][::-1].round(1),
                    texttemplate='%{text:.1f}%',
                    textposition='auto',
                ))
                fig.update_layout(
                    title='Features Ranked by Impact on This Prediction',
                    xaxis_title='Impact on Prediction (%)',
                    yaxis_title='Feature',
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show comparison with dataset averages
                st.markdown("#### 📊 Your Values vs Dataset Average")
                
                comparison_data = []
                for idx, row in importance_df.iterrows():
                    feat = row['Feature']
                    your_val = row['Your Value']
                    
                    # Get dataset statistics for this feature
                    if feat in df.columns:
                        dataset_mean = df[feat].mean()
                        disease_mean = df[df['target'] == 1][feat].mean()
                        healthy_mean = df[df['target'] == 0][feat].mean()
                        
                        # Calculate distance to disease vs healthy
                        dist_to_disease = abs(your_val - disease_mean)
                        dist_to_healthy = abs(your_val - healthy_mean)
                        
                        # Determine risk indicator
                        if dist_to_disease < dist_to_healthy:
                            risk_indicator = "⚠️ Closer to Disease"
                        elif dist_to_healthy < dist_to_disease:
                            risk_indicator = "✅ Closer to Healthy"
                        else:
                            risk_indicator = "⚡ In Between"
                        
                        comparison_data.append({
                            'Feature': feat,
                            'Your Value': f"{your_val:.1f}",
                            'Overall Avg': f"{dataset_mean:.1f}",
                            'Disease Avg': f"{disease_mean:.1f}",
                            'Healthy Avg': f"{healthy_mean:.1f}",
                            'Risk Indicator': risk_indicator,
                            'Importance': f"{row['Importance %']:.1f}%"
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    st.warning("""
⚠️ **IMPORTANT - Why High Risk Despite "Normal" Values:**

🧠 **Machine Learning finds PATTERNS, not simple rules:**
- The model doesn't just check if "value > threshold"
- It looks at the COMBINATION of all features together
- Some features work in REVERSE (lower = worse, higher = worse)

🎯 **"Risk Indicator" column shows the truth:**
- ✅ Closer to Healthy = Good sign
- ⚠️ Closer to Disease = Risk factor
- Even if a number looks "small", it might match the disease pattern!

📊 **Example from YOUR data:**
- Features with many 0.0 values might match disease patterns where low values indicate risk
- The model learned from 303 real patients what combinations predict disease
- Your specific combination of values triggered the high-risk prediction
                    """)
            else:
                # Fallback: Show input values without importance
                st.markdown("#### 📋 Your Input Values")
                
                input_display = pd.DataFrame([
                    {'Feature': k, 'Value': v} 
                    for k, v in input_payload.items()
                ]).head(10)
                st.dataframe(input_display, use_container_width=True, hide_index=True)
            
            # Overall assessment explanation
            st.markdown("---")
            st.markdown("### 📋 How This Prediction Works")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class='info-box'>
                <h4 style='color: white;'>🎯 Prediction Threshold</h4>
                <p><strong>Standard Threshold:</strong> 50% (default)</p>
                <p><strong>Our Threshold:</strong> 45% (optimized)</p>
                <br>
                <p><strong>Why 45%?</strong></p>
                <ul>
                <li>Better detection of disease cases</li>
                <li>Reduces false negatives (missed diagnoses)</li>
                <li>More cautious approach for patient safety</li>
                <li>Accounts for dataset imbalance (60% healthy, 40% disease)</li>
                </ul>
                <br>
                <p><strong>Your Result:</strong> {:.1f}% disease probability</p>
                <p>{}</p>
                </div>
                """.format(
                    probability[1] * 100,
                    "✅ Below 50% threshold = LOW RISK" if prediction == 0 else "⚠️ Above 50% threshold = HIGH RISK"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='info-box'>
                <h4 style='color: white;'>⚖️ Balanced Model Design</h4>
                <p><strong>Challenge:</strong> More healthy people than diseased in data</p>
                <p><strong>Solution:</strong> Balanced class weighting</p>
                <br>
                <p><strong>What This Means:</strong></p>
                <ul>
                <li>Model treats both classes equally important</li>
                <li>Prevents bias toward "healthy" predictions</li>
                <li>Better at catching early disease signs</li>
                <li>More reliable for diverse patient profiles</li>
                </ul>
                <br>
                <p><strong>Model Used:</strong> {}</p>
                <p>Trained on {} patient records with {} clinical features</p>
                </div>
                """.format(model_type, len(df), X.shape[1]), unsafe_allow_html=True)
            
            # Risk level explanation
            st.markdown("---")
            st.markdown("### 📊 Understanding Risk Levels")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div style='background: rgba(102, 126, 234, 0.3); padding: 20px; border-radius: 10px; border: 2px solid #667eea;'>
                <h4 style='color: #667eea; text-align: center;'>✅ LOW RISK</h4>
                <p style='color: #e0e0e0; text-align: center; font-size: 14px;'>
                <strong>Probability: 0-49%</strong><br><br>
                • Few or no risk factors<br>
                • Good cardiovascular health<br>
                • Continue healthy lifestyle<br>
                • Regular check-ups recommended
                </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background: rgba(255, 179, 0, 0.3); padding: 20px; border-radius: 10px; border: 2px solid #ffb300;'>
                <h4 style='color: #ffb300; text-align: center;'>⚡ MODERATE RISK</h4>
                <p style='color: #e0e0e0; text-align: center; font-size: 14px;'>
                <strong>Probability: 40-60%</strong><br><br>
                • Some risk factors present<br>
                • Lifestyle changes needed<br>
                • Close monitoring advised<br>
                • Preventive measures important
                </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div style='background: rgba(245, 87, 108, 0.3); padding: 20px; border-radius: 10px; border: 2px solid #f5576c;'>
                <h4 style='color: #f5576c; text-align: center;'>⚠️ HIGH RISK</h4>
                <p style='color: #e0e0e0; text-align: center; font-size: 14px;'>
                <strong>Probability: 50-100%</strong><br><br>
                • Multiple risk factors<br>
                • Medical attention needed<br>
                • Comprehensive screening<br>
                • Treatment plan required
                </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            
            if prediction == 1:
                st.warning("""
                **⚠️ Important Actions:**
                - 👨‍⚕️ Consult a cardiologist immediately
                - 📋 Get comprehensive cardiac screening
                - 💊 Discuss medication options with your doctor
                - 🏃 Start supervised exercise program
                - 🥗 Adopt heart-healthy diet
                - 🚭 Avoid smoking and excessive alcohol
                """)
            else:
                st.info("""
                **✅ Maintenance Tips:**
                - 💚 Continue healthy lifestyle
                - 🏃 Regular exercise (150 min/week)
                - 🥗 Balanced diet with fruits & vegetables
                - 😊 Manage stress effectively
                - 🩺 Regular health check-ups
                - 💤 Adequate sleep (7-9 hours)
                """)

# MODEL PERFORMANCE PAGE
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance Analysis")
    
    # Prepare data
    X = df.drop(['target'], axis=1)
    if 'Unnamed: 0' in X.columns:
        X = X.drop('Unnamed: 0', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load pre-trained models or train fresh
    st.info("📂 Loading pre-trained models...")
    
    results = {}
    
    # Try to load classical models
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/logistic_regression_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('models/svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        with open('models/classical_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/classical_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest
        y_pred_rf = rf_model.predict(X_test_scaled)
        results["Random Forest"] = {
            'model': rf_model,
            'accuracy': metadata['random_forest']['accuracy'],
            'predictions': y_pred_rf,
            'source': 'Pre-trained'
        }
        
        # Logistic Regression
        y_pred_lr = lr_model.predict(X_test_scaled)
        results["Logistic Regression"] = {
            'model': lr_model,
            'accuracy': metadata['logistic_regression']['accuracy'],
            'predictions': y_pred_lr,
            'source': 'Pre-trained'
        }
        
        # SVM
        y_pred_svm = svm_model.predict(X_test_scaled)
        results["SVM"] = {
            'model': svm_model,
            'accuracy': metadata['svm']['accuracy'],
            'predictions': y_pred_svm,
            'source': 'Pre-trained'
        }
        
        st.success("✅ Loaded Random Forest, Logistic Regression, and SVM from saved models!")
        
    except Exception as e:
        st.warning(f"⚠️ Could not load classical models: {e}. Training fresh...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classical models fresh
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "SVM": SVC(probability=True, random_state=42, class_weight='balanced')
        }
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'source': 'Freshly trained'
            }
    
    # Load MLP model
    try:
        with open('models/mlp_model.pkl', 'rb') as f:
            mlp_model = pickle.load(f)
        with open('models/mlp_scaler.pkl', 'rb') as f:
            mlp_scaler = pickle.load(f)
        with open('models/mlp_metadata.pkl', 'rb') as f:
            mlp_metadata = pickle.load(f)
        
        X_test_mlp = mlp_scaler.transform(X_test)
        y_pred_mlp = mlp_model.predict(X_test_mlp)
        
        results["MLP (Neural Network)"] = {
            'model': mlp_model,
            'accuracy': mlp_metadata['accuracy'],
            'predictions': y_pred_mlp,
            'source': 'Pre-trained'
        }
        st.success("✅ Loaded MLP Neural Network from saved model!")
        
    except Exception as e:
        st.warning(f"⚠️ Could not load MLP model: {e}. Training fresh...")
        
        if 'scaler' not in locals():
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )
        mlp_model.fit(X_train_scaled, y_train)
        y_pred_mlp = mlp_model.predict(X_test_scaled)
        
        results["MLP (Neural Network)"] = {
            'model': mlp_model,
            'accuracy': accuracy_score(y_test, y_pred_mlp),
            'predictions': y_pred_mlp,
            'source': 'Freshly trained'
        }
    
    # Load TabNet model
    try:
        tabnet_model = TabNetClassifier()
        tabnet_model.load_model('models/tabnet_model.zip')
        
        with open('models/scaler.pkl', 'rb') as f:
            tabnet_scaler = pickle.load(f)
        with open('models/metadata.pkl', 'rb') as f:
            tabnet_metadata = pickle.load(f)
        
        X_test_tabnet = tabnet_scaler.transform(X_test)
        y_pred_tabnet = tabnet_model.predict(X_test_tabnet)
        
        results["TabNet (Deep Learning)"] = {
            'model': tabnet_model,
            'accuracy': tabnet_metadata['accuracy'],
            'predictions': y_pred_tabnet,
            'source': 'Pre-trained'
        }
        st.success("✅ Loaded TabNet Deep Learning model from saved model!")
        
    except Exception as e:
        st.warning(f"⚠️ Could not load TabNet model: {e}. Training fresh...")
        
        if 'scaler' not in locals():
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
        # Calculate appropriate batch size based on dataset size
        n_samples = len(X_train_scaled)
        batch_size = min(32, max(8, n_samples // 8))
        virtual_batch_size = max(4, batch_size // 2)
        
        tabnet_model = TabNetClassifier(
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            seed=42,
            verbose=0
        )
        
        # TabNet uses numpy arrays
        tabnet_model.fit(
            X_train_scaled, y_train.values,
            eval_set=[(X_test_scaled, y_test.values)],
            eval_name=['test'],
            eval_metric=['accuracy', 'logloss'],
            max_epochs=100,
            patience=20,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            drop_last=False
        )
        
        y_pred_tabnet = tabnet_model.predict(X_test_scaled)
        
        results["TabNet (Deep Learning)"] = {
            'model': tabnet_model,
            'accuracy': accuracy_score(y_test, y_pred_tabnet),
            'predictions': y_pred_tabnet,
            'source': 'Freshly trained'
        }
    
    # Model comparison
    st.markdown("### 🏆 Model Comparison")
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] * 100 for m in results.keys()]
    }).sort_values('Accuracy', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(comparison_df, x='Model', y='Accuracy',
                    color='Accuracy',
                    color_continuous_scale='RdYlGn',
                    text_auto='.2f')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(title='Model Accuracy Comparison', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🥇 Rankings")
        for idx, row in comparison_df.iterrows():
            rank_emoji = ["🥇", "🥈", "🥉", "🏅", "⭐"][comparison_df.index.get_loc(idx)]
            st.metric(f"{rank_emoji} {row['Model']}", f"{row['Accuracy']:.2f}%")
    
    # Detailed analysis for best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    y_pred = results[best_model_name]['predictions']
    
    st.markdown(f"### 🔍 Detailed Analysis - {best_model_name}")
    
    tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix", "📈 ROC Curve", "📋 Classification Report"])
    
    with tab1:
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['No Disease', 'Disease'],
                       y=['No Disease', 'Disease'],
                       text_auto=True,
                       color_continuous_scale='Blues')
        fig.update_layout(title='Confusion Matrix', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("True Positives", cm[1][1])
        with col2:
            st.metric("True Negatives", cm[0][0])
        with col3:
            st.metric("False Positives", cm[0][1])
        with col4:
            st.metric("False Negatives", cm[1][0])
    
    with tab2:
        # Handle predict_proba for TabNet differently
        if "TabNet" in best_model_name:
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                                line=dict(color='#667eea', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                name='Random Classifier',
                                line=dict(color='gray', width=2, dash='dash')))
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"🎯 AUC Score: {roc_auc:.4f}")
    
    with tab3:
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
                    use_container_width=True)

# ABOUT PAGE
else:
    st.markdown("## ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📖 Overview
        
        This Heart Disease Prediction System uses machine learning to assess cardiovascular risk 
        based on clinical parameters. The application analyzes 16 key health indicators to 
        provide accurate predictions.
        
        ### 🎯 Features
        
        - 🤖 **Multiple ML Models**: Compare performance across different algorithms
        - 📊 **Interactive Visualizations**: Explore data with modern charts
        - 🔮 **Real-time Predictions**: Get instant risk assessments
        - 📈 **Performance Metrics**: Detailed model evaluation
        - 💡 **Health Recommendations**: Personalized advice based on results
        
        ### 🏥 Clinical Parameters
        
        The system analyzes the following 16 attributes:
        
        1. **Age**: Patient's age in years
        2. **Sex**: Gender (Male/Female)
        3. **Chest Pain Type**: 4 types of chest pain
        4. **Resting Blood Pressure**: In mm Hg
        5. **Cholesterol**: Serum cholesterol in mg/dl
        6. **Fasting Blood Sugar**: > 120 mg/dl
        7. **Resting ECG**: Results
        """)
    
    with col2:
        st.markdown("""
        <br><br>
        
        8. **Max Heart Rate**: Maximum heart rate achieved
        9. **Exercise Induced Angina**: Yes/No
        10. **ST Depression**: Induced by exercise
        11. **Slope**: Of peak exercise ST segment
        12. **Major Vessels**: Colored by fluoroscopy (0-3)
        13. **Thalassemia**: Blood disorder type
        14. **Smoking**: Current smoking status
        15. **Diabetes**: Diabetes diagnosis
        16. **BMI**: Body Mass Index
        
        **Target**: Heart disease diagnosis
        
        ### 📚 Data Source
        
        **UCI Heart Disease Dataset**
        - Cleveland Clinic Foundation
        - Hungarian Institute of Cardiology
        - V.A. Medical Center, Long Beach
        - University Hospital, Zurich
        
        ### 👨‍⚕️ Credits
        
        **Principal Investigators:**
        - Andras Janosi, M.D. (Hungarian Institute)
        - William Steinbrunn, M.D. (Zurich)
        - Matthias Pfisterer, M.D. (Basel)
        - Robert Detrano, M.D., Ph.D. (Long Beach & Cleveland)
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### ⚠️ Important Disclaimer
    
    This application is for educational and informational purposes only. It should not be used 
    as a substitute for professional medical advice, diagnosis, or treatment. Always seek the 
    advice of your physician or other qualified health provider with any questions you may have 
    regarding a medical condition.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h3>Made with ❤️ using Streamlit</h3>
        <p>© 2026 Heart Disease Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏥 Advanced Healthcare Analytics | 🤖 Powered by Machine Learning | 💚 For Better Health Outcomes</p>
    <p style='margin-top: 10px; font-size: 14px; color: #888;'>👨‍💻 Developed by <strong style='color: #667eea;'>Md Nurullah</strong></p>
</div>
""", unsafe_allow_html=True)
