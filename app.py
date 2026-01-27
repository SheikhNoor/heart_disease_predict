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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from PIL import Image

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
        padding-top: 2rem;
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
        margin-bottom: 10px;
        animation: fadeIn 1.5s ease-in;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .logo-container img {
        max-width: 150px;
        height: auto;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
        margin: 0 auto;
        display: block;
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
    df = pd.read_csv('heart.csv')
    return df

@st.cache_resource
def train_model(X_train, y_train, model_type):
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:  # SVM
        model = SVC(probability=True, random_state=42)
    
    model.fit(X_train, y_train)
    return model

# Logo and title with animation
try:
    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
            <img src="data:image/webp;base64,{}" style="max-width: 150px; border-radius: 15px; box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);">
        </div>
    """.format(__import__('base64').b64encode(open('image/logo.webp', 'rb').read()).decode()), unsafe_allow_html=True)
except:
    pass

st.markdown('<p class="big-font">Heart Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced ML-Powered Cardiovascular Risk Assessment</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Display logo in sidebar
    try:
        st.image('image/logo.webp', use_container_width=True)
        st.markdown("---")
    except:
        pass
    
    st.markdown("### 🎯 Navigation")
    page = st.radio("", ["🏠 Home", "📊 Data Exploration", "🔮 Prediction", "📈 Model Performance", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### 📚 Dataset Info")
    st.info("""
    **Source**: UCI Heart Disease Dataset
    
    **Records**: 303 patients
    
    **Features**: 14 clinical attributes
    
    **Target**: Heart disease presence
    """)

# Load data
df = load_data()

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
    
    cols = st.columns(4)
    features_info = [
        ("🫀 Max Heart Rate", "thalach", "Average: {:.0f} bpm"),
        ("🩸 Cholesterol", "chol", "Average: {:.0f} mg/dl"),
        ("💉 Blood Pressure", "trestbps", "Average: {:.0f} mm Hg"),
        ("📈 ST Depression", "oldpeak", "Average: {:.2f}")
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
        
        with col_b:
            thalach = st.slider("❤️ Maximum Heart Rate", 60, 220, 150)
            exang = st.selectbox("🏃 Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.slider("📊 ST Depression", 0.0, 6.0, 1.0, 0.1)
            slope = st.selectbox("📈 Slope of Peak Exercise ST", 
                               ["Upsloping", "Flat", "Downsloping"])
            ca = st.slider("🔬 Number of Major Vessels (0-3)", 0, 3, 0)
            thal = st.selectbox("🫀 Thalassemia", 
                              ["Normal", "Fixed Defect", "Reversible Defect"])
        
        # Convert inputs
        sex_map = {"Female": 0, "Male": 1}
        cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        fbs_map = {"No": 0, "Yes": 1}
        restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        exang_map = {"No": 0, "Yes": 1}
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
        
        input_data = np.array([[
            age, sex_map[sex], cp_map[cp], trestbps, chol, fbs_map[fbs],
            restecg_map[restecg], thalach, exang_map[exang], oldpeak,
            slope_map[slope], ca, thal_map[thal]
        ]])
    
    with col2:
        st.markdown("### ⚙️ Model Settings")
        
        model_type = st.selectbox("🤖 Select Model", 
                                 ["Random Forest", "Gradient Boosting", 
                                  "Logistic Regression", "SVM"])
        
        st.markdown("---")
        
        st.markdown("#### 📊 Quick Stats")
        st.info(f"""
        **Age**: {age} years
        **Gender**: {sex}
        **BP**: {trestbps} mm Hg
        **Cholesterol**: {chol} mg/dl
        **Max HR**: {thalach} bpm
        """)
    
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
            input_scaled = scaler.transform(input_data)
            
            # Train model
            model = train_model(X_train_scaled, y_train, model_type)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            st.success("✅ Analysis Complete!")
            
            # Display result with animation
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
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
                        'bar': {'color': "#f5576c" if probability[1] > 0.5 else "#667eea"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#d4edda'},
                            {'range': [30, 70], 'color': '#fff3cd'},
                            {'range': [70, 100], 'color': '#f8d7da'}
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    with st.spinner("🔄 Training models..."):
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
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
            rank_emoji = ["🥇", "🥈", "🥉", "🏅"][comparison_df.index.get_loc(idx)]
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
        based on clinical parameters. The application analyzes 13 key health indicators to 
        provide accurate predictions.
        
        ### 🎯 Features
        
        - 🤖 **Multiple ML Models**: Compare performance across different algorithms
        - 📊 **Interactive Visualizations**: Explore data with modern charts
        - 🔮 **Real-time Predictions**: Get instant risk assessments
        - 📈 **Performance Metrics**: Detailed model evaluation
        - 💡 **Health Recommendations**: Personalized advice based on results
        
        ### 🏥 Clinical Parameters
        
        The system analyzes the following 14 attributes:
        
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
        14. **Target**: Heart disease diagnosis
        
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
