import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import base64
from datetime import datetime

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
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 10px 0;
        border-left: 4px solid var(--primary-blue);
    }
    
    .result-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-top: 4px solid;
    }
    
    .result-safe {
        border-top-color: var(--success-green);
    }
    
    .result-unsafe {
        border-top-color: var(--danger-red);
    }
    
    /* Metrics */
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary-blue), var(--light-blue));
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #F0F0F0;
        padding: 5px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-blue) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💧 Water Quality Analyzer</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">Professional Water Potability Assessment Tool</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3082/3082383.png", width=80)
    st.markdown("### Navigation")
    
    analysis_mode = st.radio(
        "Select Mode:",
        ["Single Analysis", "Batch Analysis", "Sample Comparison"]
    )
    
    st.markdown("---")
    
    st.markdown("### 💡 Quick Actions")
    if st.button("Load Sample Data", use_container_width=True):
        st.session_state.sample_data = pd.DataFrame({
            'Sample': ['Tap Water', 'Well Water', 'River Water', 'Contaminated'],
            'ph': [7.0, 6.8, 7.5, 4.5],
            'Hardness': [150, 320, 180, 450],
            'Solids': [20000, 42000, 25000, 60000],
            'Chloramines': [4.0, 6.5, 4.5, 9.0],
            'Sulfate': [250, 380, 280, 500],
            'Conductivity': [400, 600, 450, 850],
            'Organic_carbon': [10, 18, 12, 28],
            'Trihalomethanes': [50, 85, 60, 140],
            'Turbidity': [2.0, 5.2, 3.5, 9.0]
        })
        st.success("Sample data loaded!")
    
    if st.button("Clear All", use_container_width=True, type="secondary"):
        st.session_state.sample_data = None
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📊 About")
    st.info("""
    This tool analyzes 9 water quality parameters to determine potability.
    
    **Safe Ranges:**
    - pH: 6.5-8.5
    - Hardness: < 200 mg/L
    - TDS: < 500 mg/L
    - Chloramines: < 4 ppm
    - Turbidity: < 5 NTU
    """)

# Main Analysis Functions
def analyze_water_quality(params):
    """Simple rule-based water quality analysis"""
    score = 0
    issues = []
    
    # pH (6.5-8.5 is optimal)
    if 6.5 <= params['ph'] <= 8.5:
        score += 1
    else:
        issues.append(f"pH ({params['ph']}) outside safe range (6.5-8.5)")
    
    # Hardness (less than 200 mg/L is better)
    if params['Hardness'] < 200:
        score += 1
    else:
        issues.append(f"Hardness ({params['Hardness']} mg/L) above 200 mg/L")
    
    # Total Dissolved Solids (TDS) - less than 500 mg/L is ideal
    if params['Solids'] < 500:
        score += 2  # Higher weight for TDS
    elif params['Solids'] < 1000:
        score += 1
    else:
        issues.append(f"TDS ({params['Solids']} mg/L) above 500 mg/L")
    
    # Chloramines (should be less than 4 mg/L)
    if params['Chloramines'] < 4:
        score += 1
    else:
        issues.append(f"Chloramines ({params['Chloramines']} ppm) above 4 ppm")
    
    # Turbidity (should be less than 5 NTU)
    if params['Turbidity'] < 5:
        score += 1
    else:
        issues.append(f"Turbidity ({params['Turbidity']} NTU) above 5 NTU")
    
    # Other parameters (slightly less weight)
    if params['Sulfate'] < 250:
        score += 0.5
    if params['Conductivity'] < 500:
        score += 0.5
    if params['Organic_carbon'] < 15:
        score += 0.5
    if params['Trihalomethanes'] < 80:
        score += 0.5
    
    # Determine potability
    if score >= 7:
        return "POTABLE", "✅ Safe for drinking", issues, score/9
    elif score >= 5:
        return "MARGINAL", "⚠️ Requires treatment", issues, score/9
    else:
        return "NOT POTABLE", "❌ Unsafe for drinking", issues, score/9

def create_gauge(value, title, min_val, max_val, good_range):
    """Create simple gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [min_val, good_range[0]], 'color': "#EF9A9A"},
                {'range': [good_range[0], good_range[1]], 'color': "#A5D6A7"},
                {'range': [good_range[1], max_val], 'color': "#EF9A9A"}
            ],
        }
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# Main App Logic
if analysis_mode == "Single Analysis":
    st.markdown("### 🔬 Single Water Sample Analysis")
    
    tab1, tab2 = st.tabs(["📋 Input Parameters", "⚡ Quick Test"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Parameters")
            ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1,
                          help="Optimal range: 6.5-8.5")
            hardness = st.slider("Hardness (mg/L)", 0.0, 500.0, 150.0, 10.0,
                                help="Less than 200 mg/L is better")
            solids = st.slider("Total Dissolved Solids (mg/L)", 0.0, 1000.0, 200.0, 10.0,
                              help="Less than 500 mg/L is ideal")
            chloramines = st.slider("Chloramines (ppm)", 0.0, 10.0, 4.0, 0.1,
                                   help="Less than 4 ppm is safe")
        
        with col2:
            st.markdown("#### Additional Parameters")
            sulfate = st.slider("Sulfate (mg/L)", 0.0, 500.0, 250.0, 10.0,
                               help="Less than 250 mg/L is ideal")
            conductivity = st.slider("Conductivity (μS/cm)", 0.0, 1000.0, 400.0, 10.0,
                                    help="Less than 500 μS/cm is better")
            organic_carbon = st.slider("Organic Carbon (mg/L)", 0.0, 30.0, 10.0, 0.1,
                                      help="Less than 15 mg/L is better")
            trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 150.0, 50.0, 1.0,
                                       help="Less than 80 μg/L is safe")
            turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 2.0, 0.1,
                                 help="Less than 5 NTU is clear")
    
    with tab2:
        st.markdown("#### Quick Test Presets")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Tap Water", use_container_width=True):
                st.session_state.ph = 7.0
                st.session_state.hardness = 150.0
                st.session_state.solids = 200.0
                st.session_state.chloramines = 4.0
                st.session_state.sulfate = 250.0
                st.session_state.conductivity = 400.0
                st.session_state.organic_carbon = 10.0
                st.session_state.trihalomethanes = 50.0
                st.session_state.turbidity = 2.0
                st.success("Tap water preset loaded!")
        
        with col2:
            if st.button("Well Water", use_container_width=True):
                st.session_state.ph = 6.8
                st.session_state.hardness = 320.0
                st.session_state.solids = 420.0
                st.session_state.chloramines = 6.5
                st.session_state.sulfate = 380.0
                st.session_state.conductivity = 600.0
                st.session_state.organic_carbon = 18.0
                st.session_state.trihalomethanes = 85.0
                st.session_state.turbidity = 5.2
                st.success("Well water preset loaded!")
        
        with col3:
            if st.button("River Water", use_container_width=True):
                st.session_state.ph = 7.5
                st.session_state.hardness = 180.0
                st.session_state.solids = 250.0
                st.session_state.chloramines = 4.5
                st.session_state.sulfate = 280.0
                st.session_state.conductivity = 450.0
                st.session_state.organic_carbon = 12.0
                st.session_state.trihalomethanes = 60.0
                st.session_state.turbidity = 3.5
                st.success("River water preset loaded!")
        
        # Get values from session state or use defaults
        ph = getattr(st.session_state, 'ph', ph)
        hardness = getattr(st.session_state, 'hardness', hardness)
        solids = getattr(st.session_state, 'solids', solids)
        chloramines = getattr(st.session_state, 'chloramines', chloramines)
        sulfate = getattr(st.session_state, 'sulfate', sulfate)
        conductivity = getattr(st.session_state, 'conductivity', conductivity)
        organic_carbon = getattr(st.session_state, 'organic_carbon', organic_carbon)
        trihalomethanes = getattr(st.session_state, 'trihalomethanes', trihalomethanes)
        turbidity = getattr(st.session_state, 'turbidity', turbidity)
    
    # Parameter gauges
    st.markdown("### 📊 Parameter Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.plotly_chart(create_gauge(ph, "pH", 0, 14, (6.5, 8.5)), use_container_width=True)
        st.plotly_chart(create_gauge(hardness, "Hardness", 0, 500, (0, 200)), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_gauge(solids, "TDS", 0, 1000, (0, 500)), use_container_width=True)
        st.plotly_chart(create_gauge(chloramines, "Chloramines", 0, 10, (0, 4)), use_container_width=True)
    
    with col3:
        st.plotly_chart(create_gauge(turbidity, "Turbidity", 0, 10, (0, 5)), use_container_width=True)
        st.plotly_chart(create_gauge(conductivity, "Conductivity", 0, 1000, (0, 500)), use_container_width=True)
    
    # Analyze button
    st.markdown("---")
    if st.button("🚀 Analyze Water Quality", type="primary", use_container_width=True):
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
        
        status, message, issues, confidence = analyze_water_quality(params)
        
        # Display result
        if status == "POTABLE":
            st.balloons()
            st.markdown(f"""
            <div class="result-card result-safe">
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div style="font-size: 3rem;">✅</div>
                    <div>
                        <h2 style="margin: 0; color: #4CAF50;">SAFE FOR DRINKING</h2>
                        <p style="margin: 10px 0; font-size: 1.1rem; color: #333;">
                            {message}
                        </p>
                        <div style="background: rgba(76, 175, 80, 0.1); padding: 15px; border-radius: 8px;">
                            <strong>Confidence Score:</strong> {confidence:.1%}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif status == "MARGINAL":
            st.markdown(f"""
            <div class="result-card" style="border-top-color: #FF9800;">
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div style="font-size: 3rem;">⚠️</div>
                    <div>
                        <h2 style="margin: 0; color: #FF9800;">REQUIRES TREATMENT</h2>
                        <p style="margin: 10px 0; font-size: 1.1rem; color: #333;">
                            {message}
                        </p>
                        <div style="background: rgba(255, 152, 0, 0.1); padding: 15px; border-radius: 8px;">
                            <strong>Confidence Score:</strong> {confidence:.1%}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-unsafe">
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div style="font-size: 3rem;">❌</div>
                    <div>
                        <h2 style="margin: 0; color: #F44336;">UNSAFE FOR DRINKING</h2>
                        <p style="margin: 10px 0; font-size: 1.1rem; color: #333;">
                            {message}
                        </p>
                        <div style="background: rgba(244, 67, 54, 0.1); padding: 15px; border-radius: 8px;">
                            <strong>Confidence Score:</strong> {confidence:.1%}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display issues if any
        if issues:
            st.markdown("### ⚠️ Issues Found")
            for issue in issues:
                st.error(issue)
        
        # Parameter summary
        st.markdown("### 📋 Parameter Summary")
        summary_df = pd.DataFrame({
            'Parameter': list(params.keys()),
            'Value': list(params.values()),
            'Status': ['✅ Safe' if val < limit else '⚠️ Check' 
                      for val, limit in zip(list(params.values()), 
                                          [8.5, 200, 500, 4, 250, 500, 15, 80, 5])]
        })
        st.dataframe(summary_df.style.format({'Value': '{:.2f}'}), use_container_width=True)
        
        # Recommendations
        st.markdown("### 📝 Recommendations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **Immediate Actions:**
            - Follow the safety status above
            - Store water properly
            - Use appropriate containers
            """)
        
        with col2:
            st.success("""
            **Testing Schedule:**
            - Test every 6 months
            - Keep records
            - Monitor changes
            """)
        
        with col3:
            st.warning("""
            **If Unsafe:**
            - Do not consume
            - Seek treatment
            - Contact authorities
            """)
        
        # Export option
        st.markdown("### 📥 Export Results")
        result_data = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "message": message,
            "confidence_score": confidence,
            "parameters": params,
            "issues_found": issues
        }
        
        json_str = json.dumps(result_data, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="water_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">📄 Download Analysis Report</a>'
        st.markdown(href, unsafe_allow_html=True)

elif analysis_mode == "Batch Analysis":
    st.markdown("### 📊 Batch Water Analysis")
    
    tab1, tab2 = st.tabs(["📁 Upload Data", "📝 Sample Data"])
    
    with tab1:
        st.markdown("""
        <div class="info-card">
            <strong>Upload Instructions:</strong>
            <ul>
                <li>Upload a CSV or Excel file</li>
                <li>Include these columns: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity</li>
                <li>You can add additional columns like Sample_ID</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File loaded successfully! ({len(data)} samples)")
                st.dataframe(data, use_container_width=True)
                
                # Check required columns
                required_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                               'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
                
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    if st.button("🔍 Analyze Batch Data", type="primary", use_container_width=True):
                        analyze_batch_data(data[required_cols])
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        if st.session_state.sample_data is not None:
            st.markdown("#### Sample Data Preview")
            st.dataframe(st.session_state.sample_data, use_container_width=True)
            
            if st.button("🔍 Analyze Sample Data", type="primary", use_container_width=True):
                required_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                               'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
                analyze_batch_data(st.session_state.sample_data[required_cols])
        else:
            st.info("No sample data loaded. Click 'Load Sample Data' in the sidebar.")

else:  # Sample Comparison
    st.markdown("### 🔍 Compare Water Samples")
    
    if st.session_state.sample_data is not None:
        # Let user select samples to compare
        sample_names = st.session_state.sample_data['Sample'].tolist()
        selected_samples = st.multiselect(
            "Select samples to compare:",
            options=sample_names,
            default=sample_names[:3]
        )
        
        if selected_samples:
            compare_data = st.session_state.sample_data[st.session_state.sample_data['Sample'].isin(selected_samples)].copy()
            
            # Analyze each sample
            results = []
            for _, row in compare_data.iterrows():
                params = row[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']].to_dict()
                status, message, issues, confidence = analyze_water_quality(params)
                results.append({
                    'Sample': row['Sample'],
                    'Status': status,
                    'Confidence': confidence,
                    'Score': int(confidence * 100)
                })
            
            results_df = pd.DataFrame(results)
            
            # Display comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Comparison Chart")
                fig = px.bar(
                    results_df,
                    x='Sample',
                    y='Score',
                    color='Status',
                    color_discrete_map={
                        'POTABLE': '#4CAF50',
                        'MARGINAL': '#FF9800',
                        'NOT POTABLE': '#F44336'
                    },
                    text='Score',
                    title="Water Quality Scores"
                )
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                fig.update_layout(yaxis_title="Quality Score (%)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Results Table")
                st.dataframe(results_df, use_container_width=True)
            
            # Radar chart for parameters
            st.markdown("#### 📊 Parameter Comparison")
            
            # Prepare data for radar chart
            fig = go.Figure()
            
            for _, row in compare_data.iterrows():
                values = [
                    row['ph']/14 * 100,  # Normalize to 0-100 scale
                    min(row['Hardness']/5, 100),  # Cap at 100
                    min(row['Solids']/10, 100),
                    row['Chloramines'] * 10,
                    row['Sulfate']/5,
                    row['Conductivity']/10,
                    row['Organic_carbon'] * 3.33,
                    min(row['Trihalomethanes']/1.5, 100),
                    row['Turbidity'] * 10
                ]
                
                categories = ['pH', 'Hardness', 'TDS', 'Chloramines', 'Sulfate', 
                            'Conductivity', 'Org Carbon', 'Trihalo', 'Turbidity']
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=row['Sample']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No sample data available. Click 'Load Sample Data' in the sidebar.")

# Helper function for batch analysis
def analyze_batch_data(data):
    """Analyze multiple samples at once"""
    with st.spinner("Analyzing batch data..."):
        results = []
        for idx, row in data.iterrows():
            params = row.to_dict()
            status, message, issues, confidence = analyze_water_quality(params)
            results.append({
                'Sample_ID': f"Sample_{idx+1}",
                'Status': status,
                'Confidence': f"{confidence:.1%}",
                'Issues': len(issues) if issues else 0
            })
        
        results_df = pd.DataFrame(results)
        
        # Summary statistics
        st.success(f"✅ Analysis complete! Processed {len(results)} samples")
        
        col1, col2, col3, col4 = st.columns(4)
        safe_count = sum(1 for r in results if r['Status'] == 'POTABLE')
        
        with col1:
            st.metric("Total Samples", len(results))
        with col2:
            st.metric("Safe", safe_count)
        with col3:
            st.metric("Needs Treatment", sum(1 for r in results if r['Status'] == 'MARGINAL'))
        with col4:
            st.metric("Unsafe", sum(1 for r in results if r['Status'] == 'NOT POTABLE'))
        
        # Results table
        st.markdown("### 📋 Batch Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Pie chart
        st.markdown("### 📊 Distribution")
        status_counts = results_df['Status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            color=status_counts.index,
            color_discrete_map={
                'POTABLE': '#4CAF50',
                'MARGINAL': '#FF9800',
                'NOT POTABLE': '#F44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.markdown("### 📥 Export Results")
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="batch_analysis_{datetime.now().strftime("%Y%m%d")}.csv">📥 Download CSV Results</a>'
        st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>💧 Water Quality Analyzer | Professional Assessment Tool</p>
        <p>For accurate results, consult with certified laboratories</p>
    </div>
    """, unsafe_allow_html=True)
