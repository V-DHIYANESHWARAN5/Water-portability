import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import base64
from datetime import datetime
import io

# Set page config
st.set_page_config(
    page_title="Water Quality Analyzer",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    /* Professional Blue Theme */
    :root {
        --primary-blue: #1E88E5;
        --dark-blue: #1565C0;
        --light-blue: #64B5F6;
        --success-green: #4CAF50;
        --warning-orange: #FF9800;
        --danger-red: #F44336;
        --light-gray: #F5F5F5;
        --dark-gray: #424242;
        --white: #FFFFFF;
    }
    
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Professional Header */
    .main-header {
        background: linear-gradient(135deg, var(--primary-blue), var(--dark-blue));
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin: 15px 0;
        border-left: 5px solid var(--primary-blue);
    }
    
    .result-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin: 20px 0;
        border-top: 5px solid;
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    .result-safe {
        border-top-color: var(--success-green);
        background: linear-gradient(145deg, #f8fff8, white);
    }
    
    .result-unsafe {
        border-top-color: var(--danger-red);
        background: linear-gradient(145deg, #fff8f8, white);
    }
    
    /* Metrics */
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        border: 1px solid #E0E0E0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary-blue), var(--light-blue));
        color: white;
        border: none;
        padding: 14px 28px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(30, 136, 229, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(30, 136, 229, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #F0F2F6;
        padding: 6px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-blue) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(30, 136, 229, 0.3);
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div {
        background: var(--primary-blue) !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = pd.DataFrame({
        'Sample': ['Tap Water', 'Well Water', 'River Water', 'Spring Water', 'Contaminated'],
        'ph': [7.0, 6.8, 7.5, 7.3, 4.5],
        'Hardness': [150, 320, 180, 85, 450],
        'Solids': [200, 420, 250, 125, 1000],
        'Chloramines': [4.0, 6.5, 4.5, 3.2, 9.0],
        'Sulfate': [250, 380, 280, 175, 500],
        'Conductivity': [400, 600, 450, 310, 850],
        'Organic_carbon': [10, 18, 12, 8, 28],
        'Trihalomethanes': [50, 85, 60, 38, 140],
        'Turbidity': [2.0, 5.2, 3.5, 1.2, 9.0],
        'Potability': ['Potable', 'Marginal', 'Potable', 'Potable', 'Not Potable']
    })

# Professional Header
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 2.8rem; margin-bottom: 0.8rem;">💧 Water Quality Analyzer</h1>
    <p style="font-size: 1.3rem; opacity: 0.95;">Professional Water Potability Assessment Tool</p>
    <div style="display: flex; justify-content: center; gap: 15px; margin-top: 20px;">
        <span style="background: rgba(255,255,255,0.2); padding: 6px 18px; border-radius: 20px; font-size: 0.9rem;">
            🔬 Scientific Analysis
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 6px 18px; border-radius: 20px; font-size: 0.9rem;">
            📊 Real-time Results
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 6px 18px; border-radius: 20px; font-size: 0.9rem;">
            💾 Export Ready
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar - Simplified and Professional
with st.sidebar:
    st.markdown("### 💧 Navigation")
    
    analysis_mode = st.selectbox(
        "Select Analysis Mode",
        ["Single Sample Analysis", "Batch Analysis", "Compare Samples"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### 📋 Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Demo Data", use_container_width=True):
            st.success("Demo data loaded!")
            st.rerun()
    
    with col2:
        if st.button("Clear All", use_container_width=True, type="secondary"):
            st.session_state.clear()
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 🎯 Safe Ranges")
    st.markdown("""
    <div class="info-card">
        <p><strong>pH:</strong> 6.5 - 8.5</p>
        <p><strong>Hardness:</strong> < 200 mg/L</p>
        <p><strong>TDS:</strong> < 500 mg/L</p>
        <p><strong>Chloramines:</strong> < 4 ppm</p>
        <p><strong>Turbidity:</strong> < 5 NTU</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("#### 📞 Support")
    st.caption("For technical support, contact: support@waterquality.com")

# Core Analysis Functions
def analyze_water_quality(params):
    """Professional water quality analysis with weighted scoring"""
    score = 0
    issues = []
    max_score = 9
    
    # Weighted scoring system
    # pH (Critical - 20% weight)
    if 6.5 <= params['ph'] <= 8.5:
        score += 1.8
    else:
        issues.append(f"⚠️ pH ({params['ph']}) outside safe range (6.5-8.5)")
    
    # Hardness (Important - 15% weight)
    if params['Hardness'] < 200:
        score += 1.35
    else:
        issues.append(f"⚠️ Hardness ({params['Hardness']} mg/L) > 200 mg/L")
    
    # TDS (Critical - 20% weight)
    if params['Solids'] < 500:
        score += 1.8
    elif params['Solids'] < 1000:
        score += 0.9
        issues.append(f"⚠️ TDS ({params['Solids']} mg/L) approaching limit")
    else:
        issues.append(f"❌ TDS ({params['Solids']} mg/L) > 500 mg/L")
    
    # Chloramines (Important - 15% weight)
    if params['Chloramines'] < 4:
        score += 1.35
    else:
        issues.append(f"⚠️ Chloramines ({params['Chloramines']} ppm) > 4 ppm")
    
    # Turbidity (Important - 15% weight)
    if params['Turbidity'] < 5:
        score += 1.35
    else:
        issues.append(f"⚠️ Turbidity ({params['Turbidity']} NTU) > 5 NTU")
    
    # Other parameters (15% combined weight)
    if params['Sulfate'] < 250:
        score += 0.3
    if params['Conductivity'] < 500:
        score += 0.3
    if params['Organic_carbon'] < 15:
        score += 0.3
    if params['Trihalomethanes'] < 80:
        score += 0.3
    
    confidence = (score / max_score) * 100
    
    # Determine potability
    if confidence >= 80:
        return "SAFE", "✅ Safe for drinking water", issues, confidence
    elif confidence >= 60:
        return "MARGINAL", "⚠️ Requires treatment before consumption", issues, confidence
    else:
        return "UNSAFE", "❌ Unsafe for drinking water", issues, confidence

def create_gauge_chart(value, title, min_val, max_val, good_min, good_max):
    """Create professional gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"<b>{title}</b>", 'font': {'size': 16}},
        number={'font': {'size': 20}, 'suffix': " mg/L" if "mg/L" in title else ""},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [min_val, good_min], 'color': "#EF9A9A"},
                {'range': [good_min, good_max], 'color': "#A5D6A7"},
                {'range': [good_max, max_val], 'color': "#EF9A9A"}
            ],
        }
    ))
    
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "black"}
    )
    return fig

# Main Application Logic
if analysis_mode == "Single Sample Analysis":
    st.markdown("### 🔬 Single Water Sample Analysis")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📋 Parameter Input", "⚡ Quick Test"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Parameters")
            ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1,
                          help="Optimal range: 6.5 - 8.5")
            
            hardness = st.slider("Hardness (mg/L)", 0.0, 500.0, 150.0, 10.0,
                                help="Ideal: < 200 mg/L")
            
            solids = st.slider("Total Dissolved Solids (mg/L)", 0.0, 1000.0, 200.0, 10.0,
                              help="Ideal: < 500 mg/L")
            
            chloramines = st.slider("Chloramines (ppm)", 0.0, 10.0, 4.0, 0.1,
                                   help="Ideal: < 4 ppm")
        
        with col2:
            st.markdown("#### Additional Parameters")
            sulfate = st.slider("Sulfate (mg/L)", 0.0, 500.0, 250.0, 10.0)
            
            conductivity = st.slider("Conductivity (μS/cm)", 0.0, 1000.0, 400.0, 10.0)
            
            organic_carbon = st.slider("Organic Carbon (mg/L)", 0.0, 30.0, 10.0, 0.1)
            
            trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 150.0, 50.0, 1.0)
            
            turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 2.0, 0.1,
                                 help="Ideal: < 5 NTU")
    
    with tab2:
        st.markdown("#### Quick Test Presets")
        
        col1, col2, col3, col4 = st.columns(4)
        
        presets = {
            "Tap Water": {"ph": 7.0, "Hardness": 150, "Solids": 200, "Chloramines": 4.0, 
                         "Sulfate": 250, "Conductivity": 400, "Organic_carbon": 10, 
                         "Trihalomethanes": 50, "Turbidity": 2.0},
            "Well Water": {"ph": 6.8, "Hardness": 320, "Solids": 420, "Chloramines": 6.5,
                          "Sulfate": 380, "Conductivity": 600, "Organic_carbon": 18,
                          "Trihalomethanes": 85, "Turbidity": 5.2},
            "River Water": {"ph": 7.5, "Hardness": 180, "Solids": 250, "Chloramines": 4.5,
                           "Sulfate": 280, "Conductivity": 450, "Organic_carbon": 12,
                           "Trihalomethanes": 60, "Turbidity": 3.5},
            "Spring Water": {"ph": 7.3, "Hardness": 85, "Solids": 125, "Chloramines": 3.2,
                            "Sulfate": 175, "Conductivity": 310, "Organic_carbon": 8,
                            "Trihalomethanes": 38, "Turbidity": 1.2}
        }
        
        for i, (preset_name, preset_values) in enumerate(presets.items()):
            cols = [col1, col2, col3, col4]
            with cols[i]:
                if st.button(preset_name, use_container_width=True):
                    for key, value in preset_values.items():
                        st.session_state[key] = value
                    st.success(f"{preset_name} preset loaded!")
                    st.rerun()
        
        # Load values from session state if available
        if hasattr(st.session_state, 'ph'):
            ph = st.session_state.ph
            hardness = st.session_state.Hardness
            solids = st.session_state.Solids
            chloramines = st.session_state.Chloramines
            sulfate = st.session_state.Sulfate
            conductivity = st.session_state.Conductivity
            organic_carbon = st.session_state.Organic_carbon
            trihalomethanes = st.session_state.Trihalomethanes
            turbidity = st.session_state.Turbidity
    
    # Parameter Visualization
    st.markdown("### 📊 Parameter Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.plotly_chart(create_gauge_chart(ph, "pH Level", 0, 14, 6.5, 8.5), use_container_width=True)
        st.plotly_chart(create_gauge_chart(hardness, "Hardness (mg/L)", 0, 500, 0, 200), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_gauge_chart(solids, "TDS (mg/L)", 0, 1000, 0, 500), use_container_width=True)
        st.plotly_chart(create_gauge_chart(chloramines, "Chloramines (ppm)", 0, 10, 0, 4), use_container_width=True)
    
    with col3:
        st.plotly_chart(create_gauge_chart(turbidity, "Turbidity (NTU)", 0, 10, 0, 5), use_container_width=True)
        st.plotly_chart(create_gauge_chart(conductivity, "Conductivity (μS/cm)", 0, 1000, 0, 500), use_container_width=True)
    
    # Analyze Button
    st.markdown("---")
    if st.button("🚀 Analyze Water Quality", type="primary", use_container_width=True):
        with st.spinner("Analyzing water quality parameters..."):
            # Create parameters dictionary
            params = {
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
            
            # Analyze water quality
            status, message, issues, confidence = analyze_water_quality(params)
            
            # Display Results
            st.markdown("---")
            
            if status == "SAFE":
                st.balloons()
                st.markdown(f"""
                <div class="result-card result-safe">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 20px;">
                            <div style="font-size: 3.5rem;">✅</div>
                            <div>
                                <h2 style="margin: 0; color: #4CAF50;">SAFE FOR DRINKING</h2>
                                <p style="margin: 10px 0 0 0; font-size: 1.1rem; color: #666;">
                                    {message}
                                </p>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 2.5rem; font-weight: bold; color: #4CAF50;">
                                {confidence:.1f}%
                            </div>
                            <div style="color: #666; font-size: 0.9rem;">Confidence Score</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif status == "MARGINAL":
                st.markdown(f"""
                <div class="result-card" style="border-top-color: #FF9800;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 20px;">
                            <div style="font-size: 3.5rem;">⚠️</div>
                            <div>
                                <h2 style="margin: 0; color: #FF9800;">REQUIRES TREATMENT</h2>
                                <p style="margin: 10px 0 0 0; font-size: 1.1rem; color: #666;">
                                    {message}
                                </p>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 2.5rem; font-weight: bold; color: #FF9800;">
                                {confidence:.1f}%
                            </div>
                            <div style="color: #666; font-size: 0.9rem;">Confidence Score</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-unsafe">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 20px;">
                            <div style="font-size: 3.5rem;">❌</div>
                            <div>
                                <h2 style="margin: 0; color: #F44336;">UNSAFE FOR DRINKING</h2>
                                <p style="margin: 10px 0 0 0; font-size: 1.1rem; color: #666;">
                                    {message}
                                </p>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 2.5rem; font-weight: bold; color: #F44336;">
                                {confidence:.1f}%
                            </div>
                            <div style="color: #666; font-size: 0.9rem;">Confidence Score</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display issues if any
            if issues:
                st.markdown("### ⚠️ Issues Identified")
                for issue in issues:
                    st.error(issue)
            
            # Parameter Summary Table
            st.markdown("### 📋 Parameter Summary")
            
            safe_ranges = {
                'ph': (6.5, 8.5),
                'Hardness': (0, 200),
                'Solids': (0, 500),
                'Chloramines': (0, 4),
                'Sulfate': (0, 250),
                'Conductivity': (0, 500),
                'Organic_carbon': (0, 15),
                'Trihalomethanes': (0, 80),
                'Turbidity': (0, 5)
            }
            
            summary_data = []
            for param, value in params.items():
                min_val, max_val = safe_ranges.get(param, (0, 100))
                status = "✅ Within Range" if min_val <= value <= max_val else "⚠️ Out of Range"
                summary_data.append({
                    'Parameter': param,
                    'Value': value,
                    'Safe Range': f"{min_val:.1f} - {max_val:.1f}",
                    'Status': status
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(
                summary_df.style.format({'Value': '{:.2f}'}),
                use_container_width=True,
                height=350
            )
            
            # Recommendations
            st.markdown("### 📝 Recommendations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="info-card">
                    <h4 style="color: #1E88E5;">🔄 Immediate Actions</h4>
                    <p><strong>If Safe:</strong> Continue regular monitoring</p>
                    <p><strong>If Marginal:</strong> Install filtration system</p>
                    <p><strong>If Unsafe:</strong> Stop consumption immediately</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-card">
                    <h4 style="color: #4CAF50;">📅 Testing Schedule</h4>
                    <p>• Monthly: Visual inspection</p>
                    <p>• Quarterly: Basic parameter test</p>
                    <p>• Annually: Comprehensive analysis</p>
                    <p>• Record all test results</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="info-card">
                    <h4 style="color: #FF9800;">🔧 Treatment Options</h4>
                    <p>• Reverse Osmosis</p>
                    <p>• UV Purification</p>
                    <p>• Activated Carbon Filters</p>
                    <p>• Distillation Systems</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Export Results
            st.markdown("### 💾 Export Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON Export
                result_data = {
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "water_quality_status": status,
                    "confidence_score": f"{confidence:.1f}%",
                    "message": message,
                    "parameters": params,
                    "issues_identified": issues,
                    "recommendations": [
                        "Continue monitoring if safe",
                        "Install filtration if marginal",
                        "Seek professional help if unsafe"
                    ]
                }
                
                json_str = json.dumps(result_data, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="water_analysis_{datetime.now().strftime("%Y%m%d")}.json" style="text-decoration: none;"><button style="background: #4CAF50; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; width: 100%; cursor: pointer;">📄 Download JSON Report</button></a>'
                st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                # CSV Export
                export_df = pd.DataFrame([{
                    **params,
                    'Status': status,
                    'Confidence': f"{confidence:.1f}%",
                    'Analysis_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }])
                
                csv = export_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="water_sample_{datetime.now().strftime("%Y%m%d")}.csv" style="text-decoration: none;"><button style="background: #1E88E5; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; width: 100%; cursor: pointer;">📊 Download CSV Data</button></a>'
                st.markdown(href, unsafe_allow_html=True)

elif analysis_mode == "Batch Analysis":
    st.markdown("### 📊 Batch Water Analysis")
    
    tab1, tab2 = st.tabs(["📁 Upload Data", "📋 Use Sample Data"])
    
    with tab1:
        st.markdown("""
        <div class="info-card">
            <h4>📝 Upload Instructions</h4>
            <p><strong>File Format:</strong> CSV or Excel</p>
            <p><strong>Required Columns:</strong> ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity</p>
            <p><strong>Optional:</strong> Sample_ID, Location, Notes</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File loaded successfully! ({len(data)} samples)")
                
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(data, use_container_width=True)
                
                # Check required columns
                required_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                               'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
                
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    if st.button("🔍 Analyze Batch Data", type="primary", use_container_width=True):
                        analyze_batch_data(data)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.markdown("#### Sample Dataset Preview")
        st.dataframe(st.session_state.sample_data, use_container_width=True)
        
        if st.button("🔍 Analyze Sample Data", type="primary", use_container_width=True):
            analyze_batch_data(st.session_state.sample_data)

else:  # Compare Samples
    st.markdown("### 🔍 Compare Water Samples")
    
    st.markdown("#### Select Samples to Compare")
    
    # Let user select samples
    sample_names = st.session_state.sample_data['Sample'].tolist()
    selected_samples = st.multiselect(
        "Choose samples",
        options=sample_names,
        default=sample_names[:3],
        label_visibility="collapsed"
    )
    
    if selected_samples:
        compare_data = st.session_state.sample_data[
            st.session_state.sample_data['Sample'].isin(selected_samples)
        ].copy()
        
        # Display comparison table
        st.markdown("#### 📋 Sample Comparison")
        st.dataframe(compare_data, use_container_width=True)
        
        # Create comparison chart
        st.markdown("#### 📈 Quality Comparison")
        
        # Calculate scores for each sample
        scores = []
        for _, row in compare_data.iterrows():
            params = row[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                         'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']].to_dict()
            status, _, _, confidence = analyze_water_quality(params)
            scores.append({
                'Sample': row['Sample'],
                'Status': status,
                'Score': confidence
            })
        
        scores_df = pd.DataFrame(scores)
        
        # Bar chart
        fig = px.bar(
            scores_df,
            x='Sample',
            y='Score',
            color='Status',
            color_discrete_map={
                'SAFE': '#4CAF50',
                'MARGINAL': '#FF9800',
                'UNSAFE': '#F44336'
            },
            text='Score',
            title="Water Quality Scores by Sample"
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5
        )
        
        fig.update_layout(
            yaxis_title="Quality Score (%)",
            yaxis_range=[0, 100],
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart for parameter comparison
        st.markdown("#### 📊 Parameter Comparison")
        
        if len(selected_samples) <= 5:  # Limit for readability
            fig = go.Figure()
            
            categories = ['pH', 'Hardness', 'TDS', 'Chloramines', 'Sulfate', 
                         'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity']
            
            for _, sample in compare_data.iterrows():
                values = [
                    sample['ph']/14*100,
                    min(sample['Hardness']/5, 100),
                    min(sample['Solids']/10, 100),
                    sample['Chloramines']*10,
                    sample['Sulfate']/5,
                    sample['Conductivity']/10,
                    sample['Organic_carbon']*3.33,
                    min(sample['Trihalomethanes']/1.5, 100),
                    sample['Turbidity']*10
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=sample['Sample']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Download comparison
        st.markdown("### 📥 Export Comparison")
        
        csv = compare_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="sample_comparison_{datetime.now().strftime("%Y%m%d")}.csv" style="text-decoration: none;"><button style="background: #1E88E5; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; width: 100%; cursor: pointer;">📥 Download Comparison Data</button></a>'
        st.markdown(href, unsafe_allow_html=True)

# Helper function for batch analysis
def analyze_batch_data(data):
    """Analyze multiple water samples"""
    with st.spinner("Analyzing batch data..."):
        results = []
        
        for idx, row in data.iterrows():
            params = row[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                         'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']].to_dict()
            
            status, message, issues, confidence = analyze_water_quality(params)
            
            results.append({
                'Sample_ID': f"Sample_{idx+1}",
                'Status': status,
                'Confidence': f"{confidence:.1f}%",
                'Issues_Found': len(issues),
                'Potability': row.get('Potability', 'N/A') if 'Potability' in row.columns else 'N/A'
            })
        
        results_df = pd.DataFrame(results)
        
        # Display results
        st.success(f"✅ Analysis complete! Processed {len(results)} samples")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        safe_count = sum(1 for r in results if r['Status'] == 'SAFE')
        marginal_count = sum(1 for r in results if r['Status'] == 'MARGINAL')
        unsafe_count = sum(1 for r in results if r['Status'] == 'UNSAFE')
        
        with col1:
            st.metric("Total Samples", len(results))
        with col2:
            st.metric("Safe", safe_count)
        with col3:
            st.metric("Marginal", marginal_count)
        with col4:
            st.metric("Unsafe", unsafe_count)
        
        # Results table
        st.markdown("### 📋 Analysis Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Visualization
        st.markdown("### 📊 Results Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            status_counts = results_df['Status'].value_counts()
            fig1 = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                color=status_counts.index,
                color_discrete_map={
                    'SAFE': '#4CAF50',
                    'MARGINAL': '#FF9800',
                    'UNSAFE': '#F44336'
                },
                hole=0.4
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Confidence distribution
            if 'Confidence' in results_df.columns:
                # Extract numeric confidence values
                confidence_values = results_df['Confidence'].str.replace('%', '').astype(float)
                
                fig2 = px.histogram(
                    x=confidence_values,
                    nbins=10,
                    title="Confidence Distribution",
                    color_discrete_sequence=['#1E88E5']
                )
                fig2.update_layout(xaxis_title="Confidence (%)", yaxis_title="Count")
                st.plotly_chart(fig2, use_container_width=True)
        
        # Download results
        st.markdown("### 💾 Export Results")
        
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="batch_analysis_{datetime.now().strftime("%Y%m%d")}.csv" style="text-decoration: none;"><button style="background: #4CAF50; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; width: 100%; cursor: pointer;">📥 Download Batch Results</button></a>'
        st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 20px;">
    <p>💧 <strong>Water Quality Analyzer</strong> | Professional Assessment Tool v2.0</p>
    <p style="font-size: 0.8rem; opacity: 0.7;">For accurate laboratory testing, consult certified water testing facilities.</p>
</div>
""", unsafe_allow_html=True)
