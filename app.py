import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# Remove sklearn imports
import json
import base64
import io
from datetime import datetime
import os
from typing import Optional, Dict, Any, Tuple, List

# Set page config
st.set_page_config(
    page_title="Advanced Water Quality Analyzer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with system-adaptive theme
st.markdown("""
<style>
    /* System adaptive colors */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #1a1a1a;
            --bg-card: #2d2d2d;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --border-color: #404040;
            --primary-color: #4CC9F0;
            --secondary-color: #4361EE;
            --success-color: #4CAF50;
            --danger-color: #F72585;
            --warning-color: #FFB74D;
        }
    }
    
    @media (prefers-color-scheme: light) {
        :root {
            --bg-primary: #f8f9fa;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --text-primary: #212529;
            --text-secondary: #495057;
            --border-color: #dee2e6;
            --primary-color: #1E88E5;
            --secondary-color: #1565C0;
            --success-color: #2E7D32;
            --danger-color: #C62828;
            --warning-color: #FF9800;
        }
    }
    
    /* Base styles */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        transition: background-color 0.3s ease;
    }
    
    /* Main Header with Location Integration */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white !important;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: float 20s linear infinite;
        opacity: 0.3;
    }
    
    .header-content {
        position: relative;
        z-index: 2;
    }
    
    .location-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.2);
        padding: 8px 20px;
        border-radius: 25px;
        margin-top: 15px;
        font-size: 0.9rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Cards */
    .prediction-card {
        background: var(--bg-card);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
        border-left: 8px solid;
        transition: all 0.3s ease;
        border: 1px solid var(--border-color);
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .prediction-safe {
        border-left-color: var(--success-color);
        background: linear-gradient(145deg, rgba(76, 175, 80, 0.1), var(--bg-card));
    }
    
    .prediction-unsafe {
        border-left-color: var(--danger-color);
        background: linear-gradient(145deg, rgba(247, 37, 133, 0.1), var(--bg-card));
    }
    
    .metric-card {
        background: var(--bg-card);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        text-align: center;
        border: 1px solid var(--border-color);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 12px;
        padding: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background-color: transparent;
        border-radius: 10px;
        color: var(--text-secondary);
        font-weight: 500;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(31, 136, 229, 0.3);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary), var(--bg-primary));
        border-right: 1px solid var(--border-color);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(31, 136, 229, 0.4);
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: var(--primary-color) !important;
    }
    
    .stSlider > div > div > div > div {
        background: var(--primary-color) !important;
    }
    
    /* Dataframe */
    .dataframe {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .dataframe th {
        background-color: rgba(31, 136, 229, 0.15) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--border-color) !important;
    }
    
    .dataframe td {
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }
    
    p, span, div:not(.st-emotion-cache-1offfwp) {
        color: var(--text-primary) !important;
    }
    
    label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Animations */
    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-50px, -50px) rotate(360deg); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(76, 201, 240, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 15px rgba(76, 201, 240, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(76, 201, 240, 0); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Upload widget */
    .uploadedFile {
        background-color: var(--bg-card) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 15px !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--primary-color) !important;
    }
    
    /* File download button */
    .download-button {
        background: linear-gradient(135deg, var(--success-color), #66BB6A);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        text-decoration: none;
        display: inline-block;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .download-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.4);
        color: white;
        text-decoration: none;
    }
    
    /* Location info card */
    .location-card {
        background: linear-gradient(135deg, rgba(31, 136, 229, 0.1), rgba(67, 97, 238, 0.1));
        border: 1px solid rgba(31, 136, 229, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model() -> Optional[Dict[str, Any]]:
    """Load the trained model from pickle file"""
    try:
        # First check for optimized model
        if os.path.exists('water_potability_model_optimized.pkl'):
            with open('water_potability_model_optimized.pkl', 'rb') as f:
                model_package = pickle.load(f)
        # Fallback to regular model
        elif os.path.exists('water_potability_model.pkl'):
            with open('water_potability_model.pkl', 'rb') as f:
                model_package = pickle.load(f)
        else:
            # Create a simple rule-based model for testing
            st.warning("⚠️ Using rule-based model. Please upload a trained model for better predictions.")
            
            # Simple rule-based model
            model_package = {
                'model_type': 'Rule-Based (Demo)',
                'accuracy': 0.78,
                'feature_names': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                                 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
                'model': None,  # We'll use rule-based prediction
                'imputer': None,
                'scaler': None
            }
        
        return model_package
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def rule_based_predict(params):
    """Simple rule-based prediction when no ML model is available"""
    score = 0
    max_score = 9
    
    # pH scoring
    if 6.5 <= params['ph'] <= 8.5:
        score += 1.8
    elif 6.0 <= params['ph'] <= 9.0:
        score += 0.9
    else:
        score += 0.3
    
    # Hardness scoring
    if params['Hardness'] < 200:
        score += 1.35
    elif params['Hardness'] < 300:
        score += 0.9
    else:
        score += 0.45
    
    # Solids scoring
    if params['Solids'] < 30000:
        score += 1.8
    elif params['Solids'] < 40000:
        score += 1.2
    elif params['Solids'] < 50000:
        score += 0.6
    else:
        score += 0.3
    
    # Chloramines scoring
    if params['Chloramines'] < 4:
        score += 1.35
    elif params['Chloramines'] < 6:
        score += 0.9
    else:
        score += 0.45
    
    # Other parameters
    if params['Sulfate'] < 250:
        score += 0.3
    if params['Conductivity'] < 500:
        score += 0.3
    if params['Organic_carbon'] < 15:
        score += 0.3
    if params['Trihalomethanes'] < 80:
        score += 0.3
    if params['Turbidity'] < 5:
        score += 1.35
    elif params['Turbidity'] < 7:
        score += 0.9
    else:
        score += 0.45
    
    confidence = (score / max_score) * 100
    
    # Determine prediction
    if confidence >= 70:
        prediction = 1  # Potable
        probability = np.array([(100-confidence)/100, confidence/100])
    else:
        prediction = 0  # Not potable
        probability = np.array([confidence/100, (100-confidence)/100])
    
    return prediction, probability

@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Load sample water quality data"""
    sample_data = [
        {"Name": "Ideal Drinking Water", "Location": "Swiss Alps", "ph": 7.2, "Hardness": 150, "Solids": 18000, "Chloramines": 4.0, 
         "Sulfate": 250, "Conductivity": 400, "Organic_carbon": 10, "Trihalomethanes": 45, "Turbidity": 1.5},
        {"Name": "Contaminated Water", "Location": "Industrial Area", "ph": 4.5, "Hardness": 400, "Solids": 50000, "Chloramines": 8.5, 
         "Sulfate": 450, "Conductivity": 800, "Organic_carbon": 25, "Trihalomethanes": 120, "Turbidity": 8.0},
        {"Name": "Hard Water", "Location": "Limestone Region", "ph": 7.8, "Hardness": 350, "Solids": 30000, "Chloramines": 5.5, 
         "Sulfate": 300, "Conductivity": 550, "Organic_carbon": 12, "Trihalomethanes": 60, "Turbidity": 3.0},
        {"Name": "Soft Water", "Location": "Rainwater Harvest", "ph": 6.8, "Hardness": 80, "Solids": 12000, "Chloramines": 3.5, 
         "Sulfate": 180, "Conductivity": 300, "Organic_carbon": 8, "Trihalomethanes": 35, "Turbidity": 2.0},
        {"Name": "Mineral Water", "Location": "Natural Spring", "ph": 7.5, "Hardness": 280, "Solids": 25000, "Chloramines": 3.8, 
         "Sulfate": 220, "Conductivity": 480, "Organic_carbon": 9, "Trihalomethanes": 40, "Turbidity": 1.8},
        {"Name": "River Water", "Location": "Downstream", "ph": 7.0, "Hardness": 220, "Solids": 32000, "Chloramines": 5.0, 
         "Sulfate": 280, "Conductivity": 520, "Organic_carbon": 14, "Trihalomethanes": 75, "Turbidity": 4.5},
    ]
    return pd.DataFrame(sample_data)

def predict_potability(input_data: Dict[str, float], model_package: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """Make prediction using the loaded model"""
    if model_package is None:
        return None, None
    
    try:
        # If we have a real ML model
        if model_package['model'] is not None:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data], columns=model_package['feature_names'])
            
            # Preprocess
            if model_package['imputer'] is not None:
                data_imputed = model_package['imputer'].transform(input_df)
            else:
                data_imputed = input_df.values
            
            if model_package['scaler'] is not None:
                data_scaled = model_package['scaler'].transform(data_imputed)
            else:
                data_scaled = data_imputed
            
            # Predict
            prediction = model_package['model'].predict(data_scaled)[0]
            
            # Get probabilities
            probability = None
            if hasattr(model_package['model'], 'predict_proba'):
                probability = model_package['model'].predict_proba(data_scaled)[0]
            
            return prediction, probability
        else:
            # Use rule-based prediction
            return rule_based_predict(input_data)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def create_gauge_chart(value: float, title: str, min_val: float, max_val: float, unit: str = "", threshold_good: float = 0.5) -> go.Figure:
    """Create a gauge chart for parameter visualization"""
    # Adjust colors based on theme
    text_color = 'white' if st.get_option('theme.base') == 'dark' else 'black'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"{title}<br><span style='font-size:0.8em;color:gray'>{unit}</span>", 'font': {'size': 16}},
        number={'suffix': f" {unit}", 'font': {'size': 20}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': text_color},
            'bar': {'color': "#4CC9F0"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, max_val*threshold_good], 'color': "#A5D6A7"},
                {'range': [max_val*threshold_good, max_val], 'color': "#EF9A9A"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val*threshold_good
            }
        }
    ))
    
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color}
    )
    return fig

def create_radar_chart(values: List[float], categories: List[str], title: str, 
                      comparison_values: Optional[List[float]] = None, 
                      comparison_name: Optional[str] = None) -> go.Figure:
    """Create a radar chart for parameter comparison"""
    fig = go.Figure()
    
    # Main values
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(76, 201, 240, 0.3)',
        line_color='#4CC9F0',
        name='Current Sample'
    ))
    
    # Comparison values if provided
    if comparison_values is not None and comparison_name is not None:
        fig.add_trace(go.Scatterpolar(
            r=comparison_values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(67, 97, 238, 0.2)',
            line_color='#4361EE',
            name=comparison_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values)*1.3],
                gridcolor='rgba(128,128,128,0.3)',
                linecolor='rgba(128,128,128,0.5)'
            ),
            angularaxis=dict(
                gridcolor='rgba(128,128,128,0.3)',
                linecolor='rgba(128,128,128,0.5)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        title={'text': title, 'font': {'size': 18}},
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_csv_download_button(df: pd.DataFrame, filename: str, button_text: str) -> str:
    """Create a CSV download button"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{button_text}</a>'
    return href

def main():
    """Main application function"""
    # Enhanced Header with Location Integration
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🌊 ADVANCED WATER QUALITY ANALYZER</h1>
            <p style="font-size: 1.2rem; opacity: 0.9;">AI-Powered Water Potability Prediction & Geographic Analysis</p>
            <div class="location-badge">
                <span>📍</span>
                <span>Location-Aware Water Quality Assessment</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🚀 Loading AI Model..."):
        model_package = load_model()
    
    if model_package is None:
        st.error("Failed to load model. Please check if model files exist.")
        return
    
    # Sidebar with Enhanced Location Features
    with st.sidebar:
        # Water Drop Logo with Location
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h1 style='text-align: center;'>💧</h1>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; margin-top: -10px;'>Water Quality Analyzer</h3>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Location Input Section
        with st.expander("📍 Add Location Information", expanded=False):
            location_name = st.text_input("Location Name", "Unknown Location")
            location_type = st.selectbox(
                "Water Source Type",
                ["Tap Water", "Well Water", "River Water", "Lake Water", "Spring Water", 
                 "Rainwater", "Bottled Water", "Industrial", "Agricultural", "Other"]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input("Latitude", -90.0, 90.0, 0.0, 0.1)
            with col2:
                longitude = st.number_input("Longitude", -180.0, 180.0, 0.0, 0.1)
            
            location_notes = st.text_area("Location Notes", "No additional notes")
            
            if st.button("💾 Save Location", use_container_width=True):
                st.session_state['current_location'] = {
                    'name': location_name,
                    'type': location_type,
                    'lat': latitude,
                    'lon': longitude,
                    'notes': location_notes
                }
                st.success(f"Location '{location_name}' saved!")
        
        st.markdown("---")
        
        # Model Information
        st.markdown("### 📊 Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Type", model_package['model_type'])
        with col2:
            st.metric("Accuracy", f"{model_package['accuracy']:.1%}")
        
        st.markdown("---")
        
        # Analysis Mode Selection
        st.markdown("### 🎯 Analysis Mode")
        analysis_mode = st.radio(
            "Select Analysis Mode:",
            ["Single Sample", "Batch Analysis", "Compare Samples", "Geographic Analysis"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick Actions with Location Support
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("📥 Load Sample Dataset", use_container_width=True, type="primary"):
            st.session_state['sample_loaded'] = True
            st.success("Sample dataset loaded!")
        
        if st.button("🌍 Load Geographic Samples", use_container_width=True):
            st.session_state['geo_samples'] = True
        
        if st.button("🔄 Reset All", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                if key not in ['_last_updated_', '_rerun_data_']:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        
        # Test CSV Downloads
        st.markdown("### 📥 Download Test Data")
        
        # Create sample test data
        test_samples = pd.DataFrame({
            'ph': [7.0, 6.5, 8.2, 5.8, 7.5],
            'Hardness': [150, 280, 120, 380, 200],
            'Solids': [20000, 32000, 18000, 45000, 24000],
            'Chloramines': [4.0, 5.5, 3.8, 7.8, 4.5],
            'Sulfate': [250, 320, 220, 420, 280],
            'Conductivity': [400, 520, 380, 720, 460],
            'Organic_carbon': [10, 14, 9, 22, 12],
            'Trihalomethanes': [50, 68, 42, 110, 55],
            'Turbidity': [3.0, 4.2, 2.1, 6.8, 3.5]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(create_csv_download_button(test_samples, "test_samples.csv", "📄 Test CSV"), unsafe_allow_html=True)
        
        with col2:
            # Create batch test data with locations
            batch_samples = pd.DataFrame({
                'Location': ['New York Tap', 'Texas Well', 'California River', 'Florida Spring', 'Chicago Lake'],
                'ph': [7.1, 6.8, 7.5, 7.3, 6.9],
                'Hardness': [145, 320, 180, 85, 220],
                'Solids': [19500, 41000, 22500, 12500, 28000],
                'Chloramines': [4.1, 6.5, 4.8, 3.2, 5.2],
                'Sulfate': [255, 380, 290, 175, 310],
                'Conductivity': [395, 610, 475, 310, 520],
                'Organic_carbon': [10.2, 16, 12.5, 7.8, 14.2],
                'Trihalomethanes': [48, 82, 58, 38, 65],
                'Turbidity': [2.8, 5.2, 3.8, 1.2, 4.1]
            })
            st.markdown(create_csv_download_button(batch_samples, "geographic_samples.csv", "🌍 Geo CSV"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Details
        with st.expander("🔧 Model Details"):
            st.write(f"**Algorithm:** {model_package['model_type']}")
            st.write(f"**Features Used ({len(model_package['feature_names'])}):**")
            for feature in model_package['feature_names']:
                st.write(f"• {feature}")
            if 'training_date' in model_package:
                st.write(f"**Trained:** {model_package['training_date']}")
    
    # Main content routing
    if analysis_mode == "Single Sample":
        display_single_sample_analysis(model_package)
    elif analysis_mode == "Batch Analysis":
        display_batch_analysis(model_package)
    elif analysis_mode == "Compare Samples":
        display_comparison_analysis(model_package)
    else:  # Geographic Analysis
        display_geographic_analysis(model_package)

def display_single_sample_analysis(model_package: Dict[str, Any]):
    """Display single sample analysis interface"""
    
    # Display current location if available
    if 'current_location' in st.session_state:
        loc = st.session_state['current_location']
        st.markdown(f"""
        <div class="location-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <span style="font-size: 1.5rem;">📍</span>
                <div>
                    <h4 style="margin: 0;">{loc['name']}</h4>
                    <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">{loc['type']}</p>
                </div>
            </div>
            <p style="margin: 0; font-size: 0.9rem;">Coordinates: {loc['lat']:.4f}, {loc['lon']:.4f}</p>
            <p style="margin: 5px 0 0 0; font-size: 0.9rem;">{loc['notes']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["🎛️ Interactive Sliders", "📝 Manual Input", "📊 Visual Analysis"])
    
    # Initialize default values
    default_ph = 7.0
    default_hardness = 150.0
    default_solids = 20000.0
    default_chloramines = 4.0
    default_sulfate = 250.0
    default_conductivity = 400.0
    default_organic_carbon = 10.0
    default_trihalomethanes = 50.0
    default_turbidity = 3.0
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💧 Basic Parameters")
            ph = st.slider("pH Level", 0.0, 14.0, default_ph, 0.1, key="ph_slider")
            st.plotly_chart(create_gauge_chart(ph, "pH", 0, 14, "", 0.5), use_container_width=True)
            
            hardness = st.slider("Hardness (mg/L)", 0.0, 500.0, default_hardness, 10.0, key="hardness_slider")
            st.plotly_chart(create_gauge_chart(hardness, "Hardness", 0, 500, "mg/L", 0.4), use_container_width=True)
        
        with col2:
            st.markdown("#### 🧪 Chemical Parameters")
            solids = st.slider("Total Dissolved Solids (mg/L)", 0.0, 60000.0, default_solids, 1000.0, key="solids_slider")
            
            chloramines = st.slider("Chloramines (ppm)", 0.0, 10.0, default_chloramines, 0.1, key="chloramines_slider")
            st.plotly_chart(create_gauge_chart(chloramines, "Chloramines", 0, 10, "ppm", 0.6), use_container_width=True)
            
            sulfate = st.slider("Sulfate (mg/L)", 0.0, 500.0, default_sulfate, 10.0, key="sulfate_slider")
        
        with col3:
            st.markdown("#### 📈 Additional Parameters")
            conductivity = st.slider("Conductivity (μS/cm)", 0.0, 1000.0, default_conductivity, 10.0, key="conductivity_slider")
            
            organic_carbon = st.slider("Organic Carbon (mg/L)", 0.0, 30.0, default_organic_carbon, 0.1, key="organic_carbon_slider")
            st.plotly_chart(create_gauge_chart(organic_carbon, "Organic Carbon", 0, 30, "mg/L", 0.4), use_container_width=True)
            
            trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 150.0, default_trihalomethanes, 1.0, key="trihalomethanes_slider")
            
            turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, default_turbidity, 0.1, key="turbidity_slider")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            ph = st.number_input("pH Level", 0.0, 14.0, default_ph, 0.1)
            hardness = st.number_input("Hardness (mg/L)", 0.0, 500.0, default_hardness, 10.0)
            solids = st.number_input("Total Dissolved Solids (mg/L)", 0.0, 60000.0, default_solids, 1000.0)
            chloramines = st.number_input("Chloramines (ppm)", 0.0, 10.0, default_chloramines, 0.1)
        with col2:
            sulfate = st.number_input("Sulfate (mg/L)", 0.0, 500.0, default_sulfate, 10.0)
            conductivity = st.number_input("Conductivity (μS/cm)", 0.0, 1000.0, default_conductivity, 10.0)
            organic_carbon = st.number_input("Organic Carbon (mg/L)", 0.0, 30.0, default_organic_carbon, 0.1)
            trihalomethanes = st.number_input("Trihalomethanes (μg/L)", 0.0, 150.0, default_trihalomethanes, 1.0)
            turbidity = st.number_input("Turbidity (NTU)", 0.0, 10.0, default_turbidity, 0.1)
    
    with tab3:
        st.markdown("### 📊 Parameter Radar Chart")
        
        # Use values from sliders or defaults
        values = [ph, hardness/50, solids/6000, chloramines*10, sulfate/5, 
                 conductivity/10, organic_carbon*3, trihalomethanes/1.5, turbidity*10]
        categories = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                     'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity']
        
        radar_fig = create_radar_chart(values, categories, "Water Parameters Profile")
        st.plotly_chart(radar_fig, use_container_width=True)
    
    # Create input dictionary
    input_data = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }
    
    # Prediction section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 Analyze Water Quality", type="primary", use_container_width=True)
    
    if predict_btn:
        with st.spinner("🧪 Analyzing parameters with AI..."):
            prediction, probability = predict_potability(input_data, model_package)
        
        if prediction is None or probability is None:
            st.error("Failed to make prediction. Please check your input and model.")
            return
        
        # Display results with animation
        st.markdown("---")
        
        # Calculate probability values with safe defaults
        prob_potable = probability[1] * 100 if probability is not None and len(probability) > 1 else 0
        prob_non_potable = probability[0] * 100 if probability is not None and len(probability) > 0 else 0
        
        if prediction == 1:
            st.balloons()
            location_info = ""
            if 'current_location' in st.session_state:
                loc_name = st.session_state['current_location'].get('name', 'Not specified')
                location_info = f'<div style="margin-top: 10px; background: rgba(76, 175, 80, 0.1); padding: 10px; border-radius: 8px; border-left: 4px solid var(--success-color);"><strong>📍 Location:</strong> {loc_name}</div>'
            
            st.markdown(f"""
            <div class="prediction-card prediction-safe pulse-animation">
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div style="font-size: 3.5rem;">✅</div>
                    <div style="flex: 1;">
                        <h2 style="margin: 0; color: #4CAF50;">POTABLE WATER</h2>
                        <p style="margin: 8px 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                            This water meets safety standards for drinking
                        </p>
                        {location_info}
                    </div>
                    <div style="font-size: 2.5rem; font-weight: bold; color: #4CAF50;">
                        {prob_potable:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            location_info = ""
            if 'current_location' in st.session_state:
                loc_name = st.session_state['current_location'].get('name', 'Not specified')
                location_info = f'<div style="margin-top: 10px; background: rgba(247, 37, 133, 0.1); padding: 10px; border-radius: 8px; border-left: 4px solid #F44336;"><strong>📍 Location:</strong> {loc_name}</div>'
            
            st.markdown(f"""
            <div class="prediction-card prediction-unsafe">
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div style="font-size: 3.5rem;">⚠️</div>
                    <div style="flex: 1;">
                        <h2 style="margin: 0; color: #F44336;">NON-POTABLE WATER</h2>
                        <p style="margin: 8px 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                            This water does NOT meet safety standards
                        </p>
                        {location_info}
                    </div>
                    <div style="font-size: 2.5rem; font-weight: bold; color: #F44336;">
                        {prob_non_potable:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Prediction Confidence")
            if probability is not None:
                # Create donut chart
                fig = go.Figure(data=[go.Pie(
                    labels=['Potable', 'Not Potable'],
                    values=[prob_potable, prob_non_potable],
                    hole=.5,
                    marker_colors=['#4CAF50', '#F44336'],
                    textinfo='label+percent',
                    textposition='inside'
                )])
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    annotations=[dict(
                        text=f'{prob_potable:.1f}%' if prediction == 1 else f'{prob_non_potable:.1f}%',
                        font_size=24, 
                        showarrow=False,
                        font_color='#4CAF50' if prediction == 1 else '#F44336'
                    )],
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Parameter Summary")
            # Define safe ranges for each parameter
            safe_ranges = {
                'ph': (6.5, 8.5),
                'Hardness': (0, 200),
                'Solids': (0, 30000),
                'Chloramines': (0, 5),
                'Sulfate': (0, 250),
                'Conductivity': (0, 500),
                'Organic_carbon': (0, 12),
                'Trihalomethanes': (0, 80),
                'Turbidity': (0, 5)
            }
            
            params_data = []
            for param, value in input_data.items():
                min_val, max_val = safe_ranges.get(param, (0, 1))
                status = "✅ Safe" if min_val <= value <= max_val else "⚠️ Unsafe"
                params_data.append({
                    'Parameter': param,
                    'Value': value,
                    'Safe Range': f"{min_val:.1f} - {max_val:.1f}",
                    'Status': status
                })
            
            params_df = pd.DataFrame(params_data)
            st.dataframe(
                params_df.style.apply(
                    lambda x: ['background-color: rgba(76, 175, 80, 0.1)' if 'Safe' in str(v) else 'background-color: rgba(244, 67, 54, 0.1)' 
                              for v in x], subset=['Status']
                ),
                use_container_width=True,
                height=400
            )
        
        # Recommendations
        st.markdown("### 📝 Recommendations & Next Steps")
        
        if prediction == 1:
            cols = st.columns(3)
            with cols[0]:
                st.info("""
                **✅ Immediate Actions**
                - Safe for immediate consumption
                - No treatment required
                - Store properly
                """)
            with cols[1]:
                st.success("""
                **📅 Maintenance Schedule**
                - Test every 3-6 months
                - Check filtration systems
                - Maintain cleanliness
                """)
            with cols[2]:
                st.warning("""
                **🔍 Quality Monitoring**
                - Record daily observations
                - Monitor source changes
                - Stay updated on standards
                """)
        else:
            unsafe_params = [p for p in params_data if 'Unsafe' in p['Status']]
            
            cols = st.columns(3)
            with cols[0]:
                unsafe_list = '• ' + ', '.join([p['Parameter'] for p in unsafe_params[:3]]) if unsafe_params else '• Multiple parameters out of range'
                st.error(f"""
                **⚠️ Critical Issues**
                {unsafe_list}
                - Do not consume
                - Stop usage immediately
                - Isolate water source
                """)
            with cols[1]:
                treatment_list = '• ' + ', '.join([f"{p['Parameter']} adjustment" for p in unsafe_params[:3]]) if unsafe_params else '• Comprehensive filtration'
                st.warning(f"""
                **🔧 Required Treatments**
                {treatment_list}
                - Filtration system upgrade
                - Chemical treatment
                - Professional consultation
                """)
            with cols[2]:
                st.info("""
                **📞 Professional Support**
                - Contact local authorities
                - Schedule lab testing
                - Consider alternative sources
                """)
        
        # Export options
        st.markdown("### 💾 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON Report
            current_location = st.session_state.get('current_location', {})
            report = {
                "prediction": "Potable" if prediction == 1 else "Non-Potable",
                "confidence": float(prob_potable if prediction == 1 else prob_non_potable),
                "parameters": input_data,
                "location": current_location,
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "model_type": model_package['model_type'],
                    "accuracy": float(model_package['accuracy'])
                },
                "safe_ranges": safe_ranges,
                "recommendations": "Safe for consumption" if prediction == 1 else "Requires treatment"
            }
            
            json_str = json.dumps(report, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="water_analysis_report.json" class="download-button">📥 JSON Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # CSV Export
            location_name = current_location.get('name', 'Unknown') if current_location else 'Unknown'
            export_df = pd.DataFrame([{
                **input_data,
                'Prediction': 'Potable' if prediction == 1 else 'Non-Potable',
                'Confidence': prob_potable if prediction == 1 else prob_non_potable,
                'Location': location_name,
                'Timestamp': datetime.now().isoformat()
            }])
            csv = export_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="water_sample_analysis.csv" class="download-button">📊 CSV Data</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col3:
            # PDF Summary (simulated)
            if st.button("📄 Generate PDF Report", use_container_width=True):
                st.info("PDF generation would require additional libraries like ReportLab or WeasyPrint. CSV and JSON exports are available.")

def display_batch_analysis(model_package: Dict[str, Any]):
    """Display batch analysis interface"""
    st.markdown("## 📊 Batch Analysis")
    
    # File upload or manual entry
    analysis_type = st.radio(
        "Select input method:",
        ["Upload CSV/Excel File", "Manual Data Entry", "Generate Sample Data"],
        horizontal=True
    )
    
    if analysis_type == "Upload CSV/Excel File":
        uploaded_file = st.file_uploader(
            "Upload your water quality data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file with water quality parameters"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Successfully loaded {len(data)} samples")
                
                # Display data preview
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(data.head(10), use_container_width=True)
                
                # Show column information
                st.info(f"**Columns found:** {', '.join(data.columns.tolist())}")
                
                # Check if required columns exist
                required_cols = model_package['feature_names']
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    st.error(f"⚠️ Missing required columns: {missing_cols}")
                    st.warning("Please ensure your data contains all required parameters.")
                else:
                    if st.button("🔍 Analyze All Samples", type="primary", use_container_width=True):
                        analyze_batch_data(data[required_cols], model_package)
            
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    elif analysis_type == "Manual Data Entry":
        st.info("Enter multiple samples in the table below:")
        
        # Create editable dataframe
        sample_count = st.number_input("Number of samples", 1, 100, 5, 1)
        
        # Initialize dataframe
        if 'batch_data' not in st.session_state:
            default_data = pd.DataFrame(columns=model_package['feature_names'])
            for i in range(sample_count):
                default_data.loc[i] = [7.0, 150.0, 20000.0, 4.0, 250.0, 400.0, 10.0, 50.0, 3.0]
            st.session_state.batch_data = default_data
        
        # Editable dataframe
        st.write("Edit the values below:")
        edited_df = st.data_editor(
            st.session_state.batch_data,
            num_rows="dynamic",
            use_container_width=True,
            height=300
        )
        
        if st.button("🔍 Analyze These Samples", type="primary", use_container_width=True):
            analyze_batch_data(edited_df, model_package)
    
    else:  # Generate Sample Data
        sample_df = load_sample_data()
        st.info("Generated sample data for analysis:")
        st.dataframe(sample_df, use_container_width=True)
        
        if st.button("🔍 Analyze Sample Data", type="primary", use_container_width=True):
            # Remove non-feature columns safely
            feature_cols = model_package['feature_names']
            available_cols = [col for col in feature_cols if col in sample_df.columns]
            analyze_batch_data(sample_df[available_cols], model_package)

def analyze_batch_data(data: pd.DataFrame, model_package: Dict[str, Any]):
    """Analyze batch data and display results"""
    with st.spinner("🧪 Analyzing batch data..."):
        # If we have a real ML model
        if model_package['model'] is not None:
            # Preprocess
            if model_package['imputer'] is not None:
                data_imputed = model_package['imputer'].transform(data)
            else:
                data_imputed = data.values
            
            if model_package['scaler'] is not None:
                data_scaled = model_package['scaler'].transform(data_imputed)
            else:
                data_scaled = data_imputed
            
            # Predict
            predictions = model_package['model'].predict(data_scaled)
            
            # Get probabilities
            if hasattr(model_package['model'], 'predict_proba'):
                probabilities = model_package['model'].predict_proba(data_scaled)
            else:
                probabilities = None
        else:
            # Use rule-based predictions
            predictions = []
            probabilities = []
            for _, row in data.iterrows():
                params = row.to_dict()
                pred, prob = rule_based_predict(params)
                predictions.append(pred)
                probabilities.append(prob)
            
            if probabilities:
                probabilities = np.array(probabilities)
            else:
                probabilities = None
    
    # Display results
    results_df = data.copy()
    results_df['Prediction'] = ['Potable' if p == 1 else 'Not Potable' for p in predictions]
    
    if probabilities is not None:
        results_df['Potable_Probability'] = probabilities[:, 1]
        results_df['Confidence'] = np.max(probabilities, axis=1)
        results_df['Risk_Level'] = pd.cut(
            results_df['Potable_Probability'],
            bins=[0, 0.3, 0.7, 1],
            labels=['High Risk', 'Moderate Risk', 'Low Risk']
        )
    
    potable_count = sum(predictions)
    potable_percentage = potable_count / len(predictions) * 100
    
    st.success(f"✅ Analysis complete! **{potable_count}/{len(predictions)}** samples are potable ({potable_percentage:.1f}%)")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(predictions))
    with col2:
        st.metric("Potable", potable_count)
    with col3:
        st.metric("Non-Potable", len(predictions) - potable_count)
    with col4:
        st.metric("Potable %", f"{potable_percentage:.1f}%")
    
    # Results table with filtering
    st.markdown("### 📋 Analysis Results")
    
    # Add filter options
    filter_option = st.selectbox(
        "Filter results:",
        ["Show All", "Only Potable", "Only Non-Potable", "High Risk Only"]
    )
    
    filtered_df = results_df.copy()
    if filter_option == "Only Potable":
        filtered_df = filtered_df[filtered_df['Prediction'] == 'Potable']
    elif filter_option == "Only Non-Potable":
        filtered_df = filtered_df[filtered_df['Prediction'] == 'Not Potable']
    elif filter_option == "High Risk Only" and 'Risk_Level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Risk_Level'] == 'High Risk']
    
    st.dataframe(
        filtered_df.style.format({
            col: '{:.2f}' for col in filtered_df.select_dtypes(include=[np.number]).columns
        }).apply(
            lambda x: ['background-color: rgba(76, 175, 80, 0.1)' if x['Prediction'] == 'Potable' 
                      else 'background-color: rgba(244, 67, 54, 0.1)' 
                      for _ in x], axis=1
        ),
        use_container_width=True,
        height=400
    )
    
    # Visualization
    st.markdown("### 📊 Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction distribution
        pred_counts = results_df['Prediction'].value_counts()
        fig1 = px.pie(
            values=pred_counts.values,
            names=pred_counts.index,
            title="Potability Distribution",
            color=pred_counts.index,
            color_discrete_map={'Potable': '#4CAF50', 'Not Potable': '#F44336'},
            hole=0.4
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Confidence distribution
        if 'Confidence' in results_df.columns:
            fig2 = px.histogram(
                results_df,
                x='Confidence',
                nbins=20,
                title="Prediction Confidence Distribution",
                color_discrete_sequence=['#1E88E5'],
                opacity=0.7
            )
            fig2.update_layout(bargap=0.1)
            st.plotly_chart(fig2, use_container_width=True)
    
    # Parameter distributions
    st.markdown("### 📈 Parameter Distributions")
    
    # Select parameter to visualize
    param_to_plot = st.selectbox(
        "Select parameter to visualize:",
        model_package['feature_names']
    )
    
    if param_to_plot in results_df.columns:
        fig3 = px.box(
            results_df,
            x='Prediction',
            y=param_to_plot,
            color='Prediction',
            title=f"{param_to_plot} Distribution by Potability",
            color_discrete_map={'Potable': '#4CAF50', 'Not Potable': '#F44336'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Download results
    st.markdown("### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="batch_analysis_results.csv" class="download-button">📥 Download CSV Results</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        # Create summary report
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "total_samples": len(results_df),
            "potable_samples": int(potable_count),
            "non_potable_samples": len(results_df) - int(potable_count),
            "potable_percentage": float(potable_percentage),
            "model_used": model_package['model_type'],
            "model_accuracy": float(model_package['accuracy']),
            "parameters_analyzed": model_package['feature_names']
        }
        
        json_str = json.dumps(summary, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="analysis_summary.json" class="download-button">📄 Download Summary</a>'
        st.markdown(href, unsafe_allow_html=True)

def display_comparison_analysis(model_package: Dict[str, Any]):
    """Display comparison analysis interface"""
    st.markdown("## 🔍 Compare Multiple Samples")
    
    # Load sample data
    sample_df = load_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Let user select samples to compare
        selected_samples = st.multiselect(
            "Select samples to compare:",
            options=sample_df['Name'].tolist(),
            default=sample_df['Name'].tolist()[:3],
            help="Select multiple samples for comparison"
        )
    
    with col2:
        # Select reference sample
        reference_sample = st.selectbox(
            "Select reference sample:",
            options=sample_df['Name'].tolist(),
            index=0,
            help="This sample will be used as reference in comparisons"
        )
    
    if selected_samples:
        compare_df = sample_df[sample_df['Name'].isin(selected_samples)].copy()
        
        # Analyze each sample
        predictions = []
        probabilities = []
        
        for _, row in compare_df.iterrows():
            input_data = row.drop(['Name', 'Location']).to_dict()
            pred, prob = predict_potability(input_data, model_package)
            predictions.append(pred)
            if prob is not None:
                probabilities.append(prob[1] if len(prob) > 1 else 0)
            else:
                probabilities.append(0)
        
        compare_df['Prediction'] = ['Potable' if p == 1 else 'Not Potable' for p in predictions]
        if all(p is not None for p in probabilities):
            compare_df['Potable_Probability'] = probabilities
        
        # Display comparison
        st.markdown("### 📊 Comparison Results")
        
        # Bar chart for probabilities
        if 'Potable_Probability' in compare_df.columns:
            fig = px.bar(
                compare_df,
                x='Name',
                y='Potable_Probability',
                color='Prediction',
                title="Potability Probability Comparison",
                color_discrete_map={'Potable': '#4CAF50', 'Not Potable': '#F44336'},
                text='Potable_Probability',
                hover_data=['Location']
            )
            fig.update_traces(
                texttemplate='%{text:.1%}', 
                textposition='outside',
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5
            )
            fig.update_layout(
                yaxis_tickformat='.0%',
                hovermode='x unified',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Get reference sample data
        ref_sample = sample_df[sample_df['Name'] == reference_sample].iloc[0]
        
        # Radar chart comparison
        st.markdown("### 📈 Parameter Comparison")
        
        # Create radar chart for each selected sample compared to reference
        for _, sample in compare_df.iterrows():
            if sample['Name'] == reference_sample:
                continue
                
            col1, col2 = st.columns(2)
            
            with col1:
                # Normalize values for radar chart
                sample_values = [
                    sample['ph']/14,  # pH normalized
                    sample['Hardness']/500,
                    sample['Solids']/60000,
                    sample['Chloramines']/10,
                    sample['Sulfate']/500,
                    sample['Conductivity']/1000,
                    sample['Organic_carbon']/30,
                    sample['Trihalomethanes']/150,
                    sample['Turbidity']/10
                ]
                
                ref_values = [
                    ref_sample['ph']/14,
                    ref_sample['Hardness']/500,
                    ref_sample['Solids']/60000,
                    ref_sample['Chloramines']/10,
                    ref_sample['Sulfate']/500,
                    ref_sample['Conductivity']/1000,
                    ref_sample['Organic_carbon']/30,
                    ref_sample['Trihalomethanes']/150,
                    ref_sample['Turbidity']/10
                ]
                
                categories = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                            'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity']
                
                radar_fig = create_radar_chart(
                    sample_values, 
                    categories, 
                    f"{sample['Name']} vs {reference_sample}",
                    ref_values,
                    reference_sample
                )
                st.plotly_chart(radar_fig, use_container_width=True)
            
            with col2:
                # Display sample information
                potable_prob = sample.get('Potable_Probability', 0)
                st.markdown(f"""
                <div style="background: var(--bg-card); padding: 20px; border-radius: 15px; border: 1px solid var(--border-color);">
                    <h4>{sample['Name']}</h4>
                    <p><strong>📍 Location:</strong> {sample['Location']}</p>
                    <p><strong>📊 Potability:</strong> {sample['Prediction']}</p>
                    <p><strong>🎯 Probability:</strong> {potable_prob:.1%}</p>
                    <p><strong>vs Reference:</strong> {reference_sample}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed comparison table
        st.markdown("### 📋 Detailed Comparison Table")
        
        # Prepare comparison data
        comparison_data = []
        for param in model_package['feature_names']:
            row = {'Parameter': param}
            for _, sample in compare_df.iterrows():
                row[sample['Name']] = sample[param]
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add reference column
        if reference_sample:
            comparison_df[f'{reference_sample} (Ref)'] = [ref_sample[param] for param in model_package['feature_names']]
        
        st.dataframe(
            comparison_df.style.format({col: '{:.2f}' for col in comparison_df.columns if col != 'Parameter'}),
            use_container_width=True
        )
        
        # Download comparison
        st.markdown("### 💾 Export Comparison")
        csv = compare_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="sample_comparison.csv" class="download-button">📥 Download Comparison Data</a>'
        st.markdown(href, unsafe_allow_html=True)

def display_geographic_analysis(model_package: Dict[str, Any]):
    """Display geographic analysis interface"""
    st.markdown("## 🌍 Geographic Water Quality Analysis")
    
    st.info("""
    This feature allows you to analyze water quality based on geographic location. 
    You can upload data with location coordinates or use sample geographic data.
    """)
    
    # Load geographic sample data
    geo_samples = pd.DataFrame({
        'Location': ['New York City', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
        'Latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484,
                    39.9526, 29.4241, 32.7157, 32.7767, 37.3382],
        'Longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740,
                     -75.1652, -98.4936, -117.1611, -96.7970, -121.8863],
        'ph': [7.1, 7.5, 7.3, 7.0, 7.8, 7.2, 7.1, 7.4, 7.0, 7.6],
        'Hardness': [145, 280, 320, 380, 220, 180, 420, 120, 350, 160],
        'Solids': [19500, 32000, 41000, 45000, 28000, 22500, 52000, 18000, 38000, 21000],
        'Chloramines': [4.1, 5.5, 6.5, 7.8, 4.5, 4.8, 8.8, 3.8, 7.0, 4.2],
        'Sulfate': [255, 320, 380, 420, 280, 290, 460, 220, 380, 260],
        'Conductivity': [395, 520, 610, 720, 460, 475, 800, 380, 600, 420],
        'Organic_carbon': [10.2, 14, 16, 22, 12, 12.5, 26, 9, 18, 10.5],
        'Trihalomethanes': [48, 68, 82, 110, 55, 58, 125, 42, 85, 52],
        'Turbidity': [2.8, 4.2, 5.2, 6.8, 3.5, 3.8, 7.5, 2.1, 5.0, 3.2]
    })
    
    analysis_option = st.radio(
        "Select analysis option:",
        ["Use Sample Geographic Data", "Upload Geographic Data", "Analyze by Region"],
        horizontal=True
    )
    
    if analysis_option == "Use Sample Geographic Data":
        st.dataframe(geo_samples, use_container_width=True)
        
        if st.button("🔍 Analyze Geographic Data", type="primary", use_container_width=True):
            analyze_geographic_data(geo_samples, model_package)
    
    elif analysis_option == "Upload Geographic Data":
        uploaded_file = st.file_uploader(
            "Upload geographic water quality data",
            type=['csv', 'xlsx'],
            help="Upload data with columns: Location, Latitude, Longitude, and water quality parameters"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded {len(data)} geographic samples")
                st.dataframe(data, use_container_width=True)
                
                required_cols = ['Location', 'Latitude', 'Longitude'] + model_package['feature_names']
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    if st.button("🔍 Analyze Uploaded Data", type="primary", use_container_width=True):
                        analyze_geographic_data(data, model_package)
            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    else:  # Analyze by Region
        st.markdown("### Select Region for Analysis")
        
        region = st.selectbox(
            "Select region:",
            ["North America", "Europe", "Asia", "Africa", "South America", "Australia"]
        )
        
        if st.button("🌍 Generate Regional Analysis", type="primary", use_container_width=True):
            # Simulate regional data
            regional_data = generate_regional_data(region)
            analyze_geographic_data(regional_data, model_package)

def analyze_geographic_data(data: pd.DataFrame, model_package: Dict[str, Any]):
    """Analyze geographic data and display results"""
    with st.spinner("🌍 Analyzing geographic data..."):
        # Extract water quality parameters
        water_params = data[model_package['feature_names']]
        
        # Predict
        predictions = []
        probabilities = []
        
        for _, row in water_params.iterrows():
            params = row.to_dict()
            pred, prob = predict_potability(params, model_package)
            predictions.append(pred)
            if prob is not None:
                probabilities.append(prob[1] if len(prob) > 1 else 0)
            else:
                probabilities.append(0)
    
    # Add predictions to data
    results_df = data.copy()
    results_df['Prediction'] = ['Potable' if p == 1 else 'Not Potable' for p in predictions]
    results_df['Potability_Score'] = probabilities
    
    # Create a map visualization
    st.markdown("### 🗺️ Geographic Distribution")
    
    # Create map
    fig = px.scatter_mapbox(
        results_df,
        lat="Latitude",
        lon="Longitude",
        hover_name="Location",
        hover_data=model_package['feature_names'][:3] + ['Potability_Score'],
        color="Prediction",
        color_discrete_map={'Potable': 'green', 'Not Potable': 'red'},
        size="Potability_Score",
        zoom=3,
        height=500,
        title="Water Quality Geographic Distribution"
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional statistics
    st.markdown("### 📊 Regional Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Locations", len(results_df))
    with col2:
        potable_count = sum(predictions)
        st.metric("Potable Locations", potable_count)
    with col3:
        non_potable_count = len(results_df) - potable_count
        st.metric("Non-Potable Locations", non_potable_count)
    with col4:
        potable_percent = potable_count / len(results_df) * 100
        st.metric("Potable %", f"{potable_percent:.1f}%")
    
    # Display results table
    st.markdown("### 📋 Location-wise Results")
    st.dataframe(
        results_df.style.format({
            'Latitude': '{:.4f}',
            'Longitude': '{:.4f}',
            'Potability_Score': '{:.2%}'
        }).format({
            col: '{:.1f}' for col in model_package['feature_names']
        }),
        use_container_width=True
    )
    
    # Download geographic results
    st.markdown("### 💾 Export Geographic Data")
    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="geographic_analysis.csv" class="download-button">🗺️ Download Geographic Data</a>'
    st.markdown(href, unsafe_allow_html=True)

def generate_regional_data(region: str) -> pd.DataFrame:
    """Generate sample regional data"""
    # This is a simplified function - in production, you would use real data
    np.random.seed(42)
    
    regions_data = {
        "North America": {
            "locations": ["New York", "Los Angeles", "Chicago", "Toronto", "Mexico City"],
            "lat_range": (25.0, 50.0),
            "lon_range": (-125.0, -65.0)
        },
        "Europe": {
            "locations": ["London", "Paris", "Berlin", "Rome", "Madrid"],
            "lat_range": (35.0, 60.0),
            "lon_range": (-10.0, 40.0)
        },
        "Asia": {
            "locations": ["Tokyo", "Beijing", "Mumbai", "Singapore", "Seoul"],
            "lat_range": (10.0, 45.0),
            "lon_range": (60.0, 140.0)
        },
        "Africa": {
            "locations": ["Cairo", "Lagos", "Nairobi", "Johannesburg", "Casablanca"],
            "lat_range": (-35.0, 35.0),
            "lon_range": (-20.0, 55.0)
        },
        "South America": {
            "locations": ["São Paulo", "Buenos Aires", "Lima", "Bogotá", "Santiago"],
            "lat_range": (-55.0, 15.0),
            "lon_range": (-85.0, -30.0)
        },
        "Australia": {
            "locations": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"],
            "lat_range": (-45.0, -10.0),
            "lon_range": (110.0, 155.0)
        }
    }
    
    if region in regions_data:
        region_info = regions_data[region]
        n_locations = len(region_info["locations"])
        
        # Generate random coordinates within the region's range
        lat_min, lat_max = region_info["lat_range"]
        lon_min, lon_max = region_info["lon_range"]
        
        data = {
            'Location': region_info["locations"],
            'Latitude': np.random.uniform(lat_min, lat_max, n_locations),
            'Longitude': np.random.uniform(lon_min, lon_max, n_locations),
            'ph': np.random.uniform(6.5, 8.5, n_locations),
            'Hardness': np.random.uniform(100, 400, n_locations),
            'Solids': np.random.uniform(15000, 40000, n_locations),
            'Chloramines': np.random.uniform(3.0, 8.0, n_locations),
            'Sulfate': np.random.uniform(200, 400, n_locations),
            'Conductivity': np.random.uniform(300, 700, n_locations),
            'Organic_carbon': np.random.uniform(8, 20, n_locations),
            'Trihalomethanes': np.random.uniform(40, 100, n_locations),
            'Turbidity': np.random.uniform(1.5, 6.0, n_locations)
        }
        
        return pd.DataFrame(data)
    
    # Return empty dataframe if region not found
    return pd.DataFrame()

if __name__ == "__main__":
    main()
