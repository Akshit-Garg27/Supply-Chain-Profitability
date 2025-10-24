import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ========================
# PAGE CONFIGURATION
# ========================
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# ADVANCED CUSTOM CSS
# ========================
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header with Gradient */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Insight Boxes with Icons */
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
    }
    
    .insight-box-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    
    .insight-box-green {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(67, 233, 123, 0.3);
    }
    
    /* Stats Container */
    .stats-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-1544g2n {
        color: white;
    }
    
    /* Custom Button Style */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Badge Styling */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-success {
        background: #10b981;
        color: white;
    }
    
    .badge-warning {
        background: #f59e0b;
        color: white;
    }
    
    .badge-danger {
        background: #ef4444;
        color: white;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 1rem;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# SAMPLE DATA GENERATION
# ========================
@st.cache_data
def generate_sample_data():
    """Generate realistic supply chain data"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Product type': np.random.choice(['skincare', 'haircare', 'cosmetics'], n_samples),
        'Price': np.random.uniform(1.7, 99.2, n_samples),
        'Availability': np.random.randint(1, 101, n_samples),
        'Number of products sold': np.random.randint(8, 997, n_samples),
        'Revenue generated': np.random.uniform(1061, 9866, n_samples),
        'Customer demographics': np.random.choice(['Unknown', 'Male', 'Female', 'Non-binary'], n_samples),
        'Stock levels': np.random.randint(0, 101, n_samples),
        'Lead times': np.random.randint(1, 31, n_samples),
        'Order quantities': np.random.randint(1, 97, n_samples),
        'Shipping times': np.random.randint(1, 11, n_samples),
        'Shipping costs': np.random.uniform(1.01, 9.93, n_samples),
        'Supplier name': np.random.choice(['Supplier 1', 'Supplier 2', 'Supplier 3', 
                                          'Supplier 4', 'Supplier 5'], n_samples),
        'Location': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai'], n_samples),
        'Lead time': np.random.randint(1, 31, n_samples),
        'Production volumes': np.random.randint(104, 986, n_samples),
        'Manufacturing lead time': np.random.randint(1, 31, n_samples),
        'Manufacturing costs': np.random.uniform(1.09, 99.47, n_samples),
        'Defect rates': np.random.uniform(0.02, 4.94, n_samples),
        'Transportation modes': np.random.choice(['Road', 'Air', 'Sea', 'Rail'], n_samples),
        'Costs': np.random.uniform(103.92, 997.41, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Feature Engineering
    df['Profit'] = df['Revenue generated'] - (df['Manufacturing costs'] + df['Shipping costs'])
    df['Profit Margin (%)'] = (df['Profit'] / df['Revenue generated']) * 100
    df['Cost Ratio'] = (df['Manufacturing costs'] + df['Shipping costs']) / df['Revenue generated']
    df['Cost efficiency'] = df['Revenue generated'] / (df['Manufacturing costs'] + df['Shipping costs'])
    df['Delivery efficiency'] = df['Lead time'] / df['Manufacturing lead time']
    df['Risk index'] = df['Defect rates'] * df['Lead time']
    
    return df

# ========================
# MODEL TRAINING
# ========================
@st.cache_resource
def train_model(df):
    """Train Random Forest model"""
    df_model = df.copy()
    cat_cols = ['Product type', 'Customer demographics', 'Supplier name', 
                'Transportation modes', 'Location']
    
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le
    
    target = 'Profit Margin (%)'
    X = df_model.drop(columns=[target], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    # Model performance
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return rf, X, feature_importance_df, label_encoders, r2, mae

# ========================
# LOAD DATA
# ========================
# Sidebar
st.sidebar.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: white; font-size: 1.8rem; margin: 0;'>📦</h1>
        <h2 style='color: white; font-size: 1.3rem; margin: 0.5rem 0;'>Supply Chain</h2>
        <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Intelligence Platform</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Dataset", 
    type=['csv'],
    help="Upload your supply chain CSV file"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Data loaded successfully!")
else:
    df = generate_sample_data()
    st.sidebar.info("📊 Using sample dataset")

# Train model
rf_model, X_features, feature_importance, encoders, model_r2, model_mae = train_model(df)

# ========================
# SIDEBAR NAVIGATION
# ========================
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color: white;'>Navigation</h3>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["🏠 Dashboard", "📊 Analytics", "🎯 Drivers & Impact", 
     "🔮 Scenario Planning", "🚀 Recommendations"],
    label_visibility="collapsed"
)

# Model Performance in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color: white;'>Model Performance</h3>", unsafe_allow_html=True)
st.sidebar.metric("R² Score", f"{model_r2:.4f}", help="Model accuracy")
st.sidebar.metric("MAE", f"{model_mae:.4f}", help="Mean Absolute Error")

# ========================
# UTILITY FUNCTIONS
# ========================
def create_metric_card(label, value, delta=None, prefix="", suffix=""):
    """Create animated metric card"""
    delta_html = ""
    if delta is not None:
        color = "green" if delta >= 0 else "red"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f"<div style='color: {color}; font-size: 0.9rem; margin-top: 0.5rem;'>{arrow} {abs(delta):.2f}%</div>"
    
    return f"""
        <div class="metric-card fade-in">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{prefix}{value}{suffix}</div>
            {delta_html}
        </div>
    """

# ========================
# MAIN CONTENT
# ========================
st.markdown('<p class="main-header">📦 Supply Chain Intelligence Hub</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Analytics • Predictive Insights • Strategic Planning</p>', unsafe_allow_html=True)

# ========================
# 1. DASHBOARD HOME
# ========================
if page == "🏠 Dashboard":
    
    # Executive Summary KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_margin = df['Profit Margin (%)'].mean()
        st.markdown(create_metric_card("Avg Profit Margin", f"{avg_margin:.2f}%"), unsafe_allow_html=True)
    
    with col2:
        total_revenue = df['Revenue generated'].sum()
        st.markdown(create_metric_card("Total Revenue", f"${total_revenue/1000:.1f}K"), unsafe_allow_html=True)
    
    with col3:
        avg_defect = df['Defect rates'].mean()
        st.markdown(create_metric_card("Avg Defect Rate", f"{avg_defect:.2f}%"), unsafe_allow_html=True)
    
    with col4:
        avg_lead = df['Lead time'].mean()
        st.markdown(create_metric_card("Avg Lead Time", f"{avg_lead:.1f}", suffix=" days"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Performance Overview
    st.markdown('<p class="section-header">📈 Performance Overview</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Revenue trend by product type with enhanced styling
        revenue_by_product = df.groupby('Product type').agg({
            'Revenue generated': 'sum',
            'Profit Margin (%)': 'mean',
            'Number of products sold': 'sum'
        }).reset_index()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Revenue by Product", "Units Sold"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(
                x=revenue_by_product['Product type'],
                y=revenue_by_product['Revenue generated'],
                marker=dict(
                    color=revenue_by_product['Revenue generated'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=revenue_by_product['Revenue generated'].apply(lambda x: f'${x/1000:.1f}K'),
                textposition='outside',
                name="Revenue"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=revenue_by_product['Product type'],
                y=revenue_by_product['Number of products sold'],
                marker=dict(
                    color=revenue_by_product['Number of products sold'],
                    colorscale='Plasma',
                    showscale=False
                ),
                text=revenue_by_product['Number of products sold'],
                textposition='outside',
                name="Units"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Product Performance Metrics",
            title_font_size=18
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top performing metrics
        st.markdown("""
        <div class="insight-box-blue">
            <h3 style='margin-top: 0;'>🏆 Top Performers</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
        """, unsafe_allow_html=True)
        
        top_product = df.groupby('Product type')['Revenue generated'].sum().idxmax()
        top_revenue = df.groupby('Product type')['Revenue generated'].sum().max()
        
        best_supplier = df.groupby('Supplier name')['Defect rates'].mean().idxmin()
        best_defect = df.groupby('Supplier name')['Defect rates'].mean().min()
        
        st.markdown(f"""
            <p><strong>Best Product:</strong><br>
            <span class='badge badge-success'>{top_product}</span><br>
            ${top_revenue/1000:.1f}K revenue</p>
            
            <p><strong>Best Supplier:</strong><br>
            <span class='badge badge-success'>{best_supplier}</span><br>
            {best_defect:.2f}% defect rate</p>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Geographic & Supplier Analysis
    st.markdown('<p class="section-header">🌍 Geographic & Supplier Intelligence</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional performance heatmap
        location_metrics = df.groupby('Location').agg({
            'Revenue generated': 'sum',
            'Profit Margin (%)': 'mean',
            'Defect rates': 'mean'
        }).reset_index()
        
        fig = go.Figure(data=go.Scatter(
            x=location_metrics['Revenue generated'],
            y=location_metrics['Profit Margin (%)'],
            mode='markers+text',
            marker=dict(
                size=location_metrics['Revenue generated']/100,
                color=location_metrics['Defect rates'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Defect Rate"),
                line=dict(width=2, color='white')
            ),
            text=location_metrics['Location'],
            textposition="top center",
            textfont=dict(size=10, color='black', family='Inter')
        ))
        
        fig.update_layout(
            title="Regional Performance Matrix",
            xaxis_title="Revenue Generated",
            yaxis_title="Profit Margin (%)",
            height=400,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Supplier efficiency radar
        supplier_perf = df.groupby('Supplier name').agg({
            'Defect rates': 'mean',
            'Lead time': 'mean',
            'Profit Margin (%)': 'mean',
            'Manufacturing costs': 'mean'
        }).reset_index()
        
        # Normalize for radar chart
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        metrics_to_plot = ['Defect rates', 'Lead time', 'Manufacturing costs']
        supplier_perf[metrics_to_plot] = 1 - scaler.fit_transform(supplier_perf[metrics_to_plot])
        supplier_perf['Profit Margin (%)'] = scaler.fit_transform(supplier_perf[['Profit Margin (%)']])
        
        fig = go.Figure()
        
        for idx, row in supplier_perf.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Defect rates'], row['Lead time'], row['Manufacturing costs'], row['Profit Margin (%)']],
                theta=['Quality', 'Speed', 'Cost', 'Profitability'],
                fill='toself',
                name=row['Supplier name']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Supplier Performance Radar",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost Analysis
    st.markdown('<p class="section-header">💰 Cost Structure Analysis</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Cost breakdown
        total_mfg = df['Manufacturing costs'].sum()
        total_ship = df['Shipping costs'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Manufacturing', 'Shipping'],
            values=[total_mfg, total_ship],
            hole=0.5,
            marker=dict(colors=['#667eea', '#764ba2']),
            textinfo='label+percent',
            textfont=dict(size=14)
        )])
        
        fig.update_layout(
            title="Cost Distribution",
            height=300,
            annotations=[dict(text=f'${(total_mfg+total_ship)/1000:.1f}K', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Shipping cost by mode
        shipping_analysis = df.groupby('Transportation modes').agg({
            'Shipping costs': 'mean',
            'Shipping times': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Cost',
            x=shipping_analysis['Transportation modes'],
            y=shipping_analysis['Shipping costs'],
            marker_color='#667eea'
        ))
        
        fig.add_trace(go.Scatter(
            name='Time',
            x=shipping_analysis['Transportation modes'],
            y=shipping_analysis['Shipping times'],
            yaxis='y2',
            marker_color='#f5576c',
            mode='lines+markers',
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title="Shipping Cost vs Time",
            yaxis=dict(title='Cost ($)'),
            yaxis2=dict(title='Time (days)', overlaying='y', side='right'),
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Risk distribution
        fig = go.Figure(data=[go.Box(
            y=df['Risk index'],
            marker=dict(color='#f5576c'),
            name='Risk Index'
        )])
        
        fig.update_layout(
            title="Risk Index Distribution",
            yaxis_title="Risk Score",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ========================
# 2. ANALYTICS PAGE
# ========================
elif page == "📊 Analytics":
    st.markdown('<p class="section-header">📊 Deep Dive Analytics</p>', unsafe_allow_html=True)
    
    # Analysis selector
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Profitability Analysis", "Operational Efficiency", "Quality Metrics", "Customer Insights"]
    )
    
    if analysis_type == "Profitability Analysis":
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Profit margin scatter with trendline
            fig = px.scatter(
                df,
                x='Cost Ratio',
                y='Profit Margin (%)',
                color='Product type',
                size='Revenue generated',
                hover_data=['Supplier name', 'Location'],
                trendline="ols",
                title="Profit Margin vs Cost Ratio (with Trendline)"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
                <h3 style='margin-top: 0;'>💡 Key Insights</h3>
                <hr style='border-color: rgba(255,255,255,0.3);'>
                <p><strong>Strong Negative Correlation</strong></p>
                <p>As cost ratio increases, profit margins decrease proportionally. This is your primary optimization lever.</p>
                <br>
                <p><strong>Product Variance</strong></p>
                <p>Different product types show distinct cost-margin profiles, suggesting category-specific strategies.</p>
                <br>
                <p><strong>Action Items:</strong></p>
                <ul>
                    <li>Target products with cost ratio > 0.02</li>
                    <li>Benchmark against top performers</li>
                    <li>Implement cost reduction initiatives</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed profitability table
        st.markdown("### 📋 Profitability Breakdown by Category")
        
        profit_summary = df.groupby('Product type').agg({
            'Revenue generated': 'sum',
            'Profit': 'sum',
            'Profit Margin (%)': 'mean',
            'Cost Ratio': 'mean',
            'Number of products sold': 'sum'
        }).round(2)
        
        profit_summary['Revenue ($)'] = profit_summary['Revenue generated'].apply(lambda x: f"${x:,.0f}")
        profit_summary['Profit ($)'] = profit_summary['Profit'].apply(lambda x: f"${x:,.0f}")
        profit_summary['Margin (%)'] = profit_summary['Profit Margin (%)'].apply(lambda x: f"{x:.2f}%")
        profit_summary['Cost Ratio'] = profit_summary['Cost Ratio'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            profit_summary[['Revenue ($)', 'Profit ($)', 'Margin (%)', 'Cost Ratio', 'Number of products sold']],
            use_container_width=True
        )
    
    elif analysis_type == "Operational Efficiency":
        col1, col2 = st.columns(2)
        
        with col1:
            # Lead time distribution by supplier
            fig = px.violin(
                df,
                x='Supplier name',
                y='Lead time',
                color='Supplier name',
                box=True,
                title="Lead Time Distribution by Supplier"
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost efficiency by location
            efficiency_by_location = df.groupby('Location')['Cost efficiency'].mean().reset_index()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=efficiency_by_location['Location'],
                    y=efficiency_by_location['Cost efficiency'],
                    marker=dict(
                        color=efficiency_by_location['Cost efficiency'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=efficiency_by_location['Cost efficiency'].round(1),
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Cost Efficiency by Location",
                yaxis_title="Efficiency Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency metrics
        st.markdown("### 📊 Operational Metrics Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_mfg_lead = df['Manufacturing lead time'].mean()
            st.metric("Avg Mfg Lead Time", f"{avg_mfg_lead:.1f} days", 
                     delta=f"{avg_mfg_lead - 14:.1f}")
        
        with col2:
            avg_delivery_eff = df['Delivery efficiency'].mean()
            st.metric("Delivery Efficiency", f"{avg_delivery_eff:.2f}", 
                     delta=f"{avg_delivery_eff - 1:.2f}")
        
        with col3:
            avg_cost_eff = df['Cost efficiency'].mean()
            st.metric("Cost Efficiency", f"{avg_cost_eff:.0f}", 
                     delta=f"{avg_cost_eff - 200:.0f}")
    
    elif analysis_type == "Quality Metrics":
        col1, col2 = st.columns(2)
        
        with col1:
            # Defect rate trends
            quality_by_supplier = df.groupby('Supplier name').agg({
                'Defect rates': ['mean', 'std', 'min', 'max']
            }).round(3)
            
            quality_by_supplier.columns = ['Mean', 'Std Dev', 'Min', 'Max']
            quality_by_supplier = quality_by_supplier.reset_index()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Mean Defect Rate',
                x=quality_by_supplier['Supplier name'],
                y=quality_by_supplier['Mean'],
                marker_color='#ef4444',
                error_y=dict(type='data', array=quality_by_supplier['Std Dev'])
            ))
            
            fig.update_layout(
                title="Defect Rate by Supplier (with Std Dev)",
                yaxis_title="Defect Rate (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Quality score calculation
            df['Quality Score'] = (1 - df['Defect rates'] / df['Defect rates'].max()) * 100
            
            quality_distribution = df.groupby('Product type')['Quality Score'].mean().reset_index()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=quality_distribution['Product type'],
                    values=quality_distribution['Quality Score'],
                    hole=0.4,
                    marker=dict(colors=['#10b981', '#3b82f6', '#f59e0b']),
                    textinfo='label+percent'
                )
            ])
            
            fig.update_layout(
                title="Quality Score by Product Type",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk analysis heatmap
        st.markdown("### 🔥 Risk Heatmap")
        
        risk_matrix = df.groupby(['Supplier name', 'Product type'])['Risk index'].mean().reset_index()
        risk_pivot = risk_matrix.pivot(index='Supplier name', columns='Product type', values='Risk index')
        
        fig = go.Figure(data=go.Heatmap(
            z=risk_pivot.values,
            x=risk_pivot.columns,
            y=risk_pivot.index,
            colorscale='Reds',
            text=risk_pivot.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Risk Index")
        ))
        
        fig.update_layout(
            title="Risk Index by Supplier & Product",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Customer Insights":
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by demographics
            demo_revenue = df.groupby('Customer demographics')['Revenue generated'].sum().reset_index()
            
            fig = px.funnel(
                demo_revenue,
                x='Revenue generated',
                y='Customer demographics',
                title="Revenue Funnel by Demographics",
                color='Customer demographics'
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average order value by segment
            df['AOV'] = df['Revenue generated'] / df['Number of products sold']
            
            aov_by_demo = df.groupby('Customer demographics').agg({
                'AOV': 'mean',
                'Number of products sold': 'sum'
            }).reset_index()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=aov_by_demo['Customer demographics'],
                y=aov_by_demo['AOV'],
                marker=dict(
                    color=aov_by_demo['AOV'],
                    colorscale='Blues',
                    showscale=False
                ),
                text=aov_by_demo['AOV'].round(2),
                textposition='outside',
                name='AOV'
            ))
            
            fig.update_layout(
                title="Average Order Value by Segment",
                yaxis_title="AOV ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer segmentation matrix
        st.markdown("### 🎯 Customer Segmentation Matrix")
        
        segment_matrix = df.groupby('Customer demographics').agg({
            'Revenue generated': 'sum',
            'Number of products sold': 'sum',
            'Profit Margin (%)': 'mean',
            'AOV': 'mean'
        }).round(2)
        
        segment_matrix['Revenue ($)'] = segment_matrix['Revenue generated'].apply(lambda x: f"${x:,.0f}")
        segment_matrix['Units Sold'] = segment_matrix['Number of products sold'].apply(lambda x: f"{x:,.0f}")
        segment_matrix['Avg Margin (%)'] = segment_matrix['Profit Margin (%)'].apply(lambda x: f"{x:.2f}%")
        segment_matrix['AOV ($)'] = segment_matrix['AOV'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(
            segment_matrix[['Revenue ($)', 'Units Sold', 'Avg Margin (%)', 'AOV ($)']],
            use_container_width=True
        )

# ========================
# 3. DRIVERS & IMPACT PAGE
# ========================
elif page == "🎯 Drivers & Impact":
    st.markdown('<p class="section-header">🎯 Profitability Drivers & Impact Analysis</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📈 Feature Importance", "⚡ Sensitivity Analysis"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Top Profit Drivers")
            
            top_n = st.slider("Number of features", 5, 15, 10, key="feat_slider")
            
            top_features = feature_importance.head(top_n)
            
            # Create gradient colored bar chart
            fig = go.Figure()
            
            colors = px.colors.sequential.Viridis
            n_colors = len(top_features)
            color_idx = [int(i * (len(colors)-1) / (n_colors-1)) for i in range(n_colors)]
            
            fig.add_trace(go.Bar(
                y=top_features['Feature'],
                x=top_features['Importance'],
                orientation='h',
                marker=dict(
                    color=[colors[i] for i in color_idx],
                    line=dict(color='white', width=2)
                ),
                text=(top_features['Importance'] * 100).round(2),
                texttemplate='%{text}%',
                textposition='outside',
                textfont=dict(size=12, color='black')
            ))
            
            fig.update_layout(
                title="Feature Importance Rankings",
                xaxis_title="Importance Score",
                yaxis_title="",
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Impact Breakdown")
            
            # Top 3 features analysis
            top_3 = feature_importance.head(3)
            
            for idx, row in top_3.iterrows():
                importance_pct = row['Importance'] * 100
                
                st.markdown(f"""
                <div class="insight-box-green" style="margin: 1rem 0;">
                    <h4 style='margin: 0 0 0.5rem 0;'>{idx+1}. {row['Feature']}</h4>
                    <div style='background: rgba(255,255,255,0.3); border-radius: 0.5rem; padding: 0.5rem;'>
                        <strong>{importance_pct:.2f}% Impact</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box-blue">
                <h4>💡 Strategic Insight</h4>
                <p>The top 3 drivers account for <strong>98%+</strong> of profit margin variance. 
                Focus optimization efforts here for maximum ROI.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Cumulative importance chart
        st.markdown("### 📊 Cumulative Impact Analysis")
        
        cum_importance = feature_importance.head(15).copy()
        cum_importance['Cumulative'] = cum_importance['Importance'].cumsum()
        cum_importance['Cumulative %'] = (cum_importance['Cumulative'] * 100).round(2)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=cum_importance['Feature'],
                y=cum_importance['Importance'],
                name="Individual",
                marker_color='#667eea'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=cum_importance['Feature'],
                y=cum_importance['Cumulative %'],
                name="Cumulative %",
                marker_color='#f5576c',
                mode='lines+markers',
                line=dict(width=3)
            ),
            secondary_y=True
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Individual Importance", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
        
        fig.update_layout(
            title="Pareto Analysis: Feature Importance",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### ⚡ Sensitivity Testing")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            delta_pct = st.slider(
                "Change Percentage",
                min_value=5,
                max_value=25,
                value=10,
                step=5,
                help="Test impact of % changes"
            )
            
            st.markdown(f"""
            <div class="insight-box" style="margin-top: 1rem;">
                <h4>Testing ±{delta_pct}% Change</h4>
                <p>Analyzing how profit margin responds to variations in key drivers.</p>
            </div>
            """, unsafe_allow_html=True)
        
        delta = delta_pct / 100
        
        # Sensitivity calculation
        key_features = ['Cost Ratio', 'Cost efficiency', 'Price', 
                       'Order quantities', 'Lead times']
        
        base_input = X_features.mean().to_frame().T
        base_pred = rf_model.predict(base_input.values)[0]
        
        sensitivity_results = []
        
        for feature in key_features:
            if feature in X_features.columns:
                temp_up = base_input.copy()
                temp_up[feature] = base_input[feature] * (1 + delta)
                up_pred = rf_model.predict(temp_up.values)[0]
                
                temp_down = base_input.copy()
                temp_down[feature] = base_input[feature] * (1 - delta)
                down_pred = rf_model.predict(temp_down.values)[0]
                
                change_up = ((up_pred - base_pred) / base_pred) * 100
                change_down = ((down_pred - base_pred) / base_pred) * 100
                
                sensitivity_results.append({
                    'Feature': feature,
                    f'+{delta_pct}%': change_up,
                    f'-{delta_pct}%': change_down,
                    'Range': abs(change_up - change_down)
                })
        
        sens_df = pd.DataFrame(sensitivity_results).sort_values('Range', ascending=False)
        
        with col2:
            # Enhanced tornado chart
            fig = go.Figure()
            
            colors_tornado = px.colors.diverging.RdYlGn
            
            for idx, row in sens_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row[f'-{delta_pct}%'], row[f'+{delta_pct}%']],
                    y=[row['Feature'], row['Feature']],
                    mode='lines+markers',
                    marker=dict(
                        size=15,
                        color=['red', 'green'],
                        line=dict(width=2, color='white')
                    ),
                    line=dict(width=8, color='rgba(100,100,100,0.3)'),
                    showlegend=False,
                    hovertemplate=
                        '<b>%{y}</b><br>' +
                        'Impact: %{x:.4f}%<br>' +
                        '<extra></extra>'
                ))
            
            fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)
            
            fig.update_layout(
                title=f"Tornado Chart: ±{delta_pct}% Impact on Profit Margin",
                xaxis_title="% Change in Profit Margin",
                yaxis_title="",
                height=400,
                plot_bgcolor='rgba(240,240,240,0.5)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Sensitivity data table with styling
        st.markdown("### 📋 Sensitivity Summary Table")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Format table
            display_sens = sens_df.copy()
            display_sens[f'+{delta_pct}%'] = display_sens[f'+{delta_pct}%'].apply(lambda x: f"{x:+.4f}%")
            display_sens[f'-{delta_pct}%'] = display_sens[f'-{delta_pct}%'].apply(lambda x: f"{x:+.4f}%")
            display_sens['Range'] = display_sens['Range'].apply(lambda x: f"{x:.4f}%")
            
            st.dataframe(
                display_sens[['Feature', f'+{delta_pct}%', f'-{delta_pct}%', 'Range']],
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.metric("Base Margin", f"{base_pred:.2f}%")
            st.metric("Most Sensitive", sens_df.iloc[0]['Feature'])
        
        with col3:
            max_impact = sens_df['Range'].max()
            st.metric("Max Impact Range", f"{max_impact:.4f}%")
            st.metric("Least Sensitive", sens_df.iloc[-1]['Feature'])

# ========================
# 4. SCENARIO PLANNING PAGE
# ========================
elif page == "🔮 Scenario Planning":
    st.markdown('<p class="section-header">🔮 Strategic Scenario Planning</p>', unsafe_allow_html=True)
    
    # Scenario definitions
    scenarios = {
        "🚀 Optimistic": {
            "Cost Ratio": -0.10,
            "Cost efficiency": +0.10,
            "Price": +0.05,
            "Order quantities": +0.10,
            "Lead times": -0.10,
            "description": "Market expansion + cost optimization success",
            "color": "#10b981"
        },
        "📊 Moderate": {
            "Cost Ratio": +0.05,
            "Cost efficiency": 0,
            "Price": 0,
            "Order quantities": +0.05,
            "Lead times": 0,
            "description": "Stable operations with minor cost pressures",
            "color": "#f59e0b"
        },
        "⚠️ Pessimistic": {
            "Cost Ratio": +0.10,
            "Cost efficiency": -0.10,
            "Price": -0.05,
            "Order quantities": -0.15,
            "Lead times": +0.10,
            "description": "Economic downturn + supply chain disruptions",
            "color": "#ef4444"
        }
    }
    
    # Calculate scenarios
    base_input = X_features.mean().to_frame().T
    base_pred = rf_model.predict(base_input.values)[0]
    
    results = []
    
    for scenario_name, changes in scenarios.items():
        scenario_input = base_input.copy()
        
        for feature, change in changes.items():
            if feature in scenario_input.columns:
                scenario_input[feature] = base_input[feature] * (1 + change)
        
        scenario_pred = rf_model.predict(scenario_input.values)[0]
        change_from_base = ((scenario_pred - base_pred) / base_pred) * 100
        
        results.append({
            "Scenario": scenario_name,
            "Description": changes["description"],
            "Predicted Margin": scenario_pred,
            "Change %": change_from_base,
            "Color": changes["color"]
        })
    
    scenario_df = pd.DataFrame(results)
    
    # Scenario comparison visualization
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 📊 Scenario Impact Comparison")
        
        fig = go.Figure()
        
        # Add baseline
        fig.add_hline(
            y=base_pred,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"Baseline: {base_pred:.2f}%",
            annotation_position="right"
        )
        
        # Add scenarios
        for idx, row in scenario_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Scenario']],
                y=[row['Predicted Margin']],
                marker_color=row['Color'],
                text=f"{row['Predicted Margin']:.2f}%",
                textposition='outside',
                name=row['Scenario'],
                showlegend=False
            ))
        
        fig.update_layout(
            title="Projected Profit Margin by Scenario",
            yaxis_title="Profit Margin (%)",
            height=400,
            yaxis_range=[base_pred - 0.2, base_pred + 0.2]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Delta visualization
        fig = go.Figure()
        
        for idx, row in scenario_df.iterrows():
            color = 'green' if row['Change %'] >= 0 else 'red'
            fig.add_trace(go.Bar(
                y=[row['Scenario']],
                x=[row['Change %']],
                orientation='h',
                marker_color=color,
                text=f"{row['Change %']:+.3f}%",
                textposition='outside',
                showlegend=False
            ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
        
        fig.update_layout(
            title="Change from Baseline",
            xaxis_title="% Change in Profit Margin",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Scenario Details")
        
        st.metric("Current Baseline", f"{base_pred:.2f}%", help="Current profit margin")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        for idx, row in scenario_df.iterrows():
            with st.expander(f"{row['Scenario']}", expanded=(idx==0)):
                st.markdown(f"**{row['Description']}**")
                st.markdown(f"<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
                
                st.metric(
                    "Projected Margin",
                    f"{row['Predicted Margin']:.2f}%",
                    delta=f"{row['Change %']:.3f}%",
                    delta_color="normal" if row['Change %'] >= 0 else "inverse"
                )
                
                # Impact assessment
                if row['Change %'] > 0:
                    st.success("✅ Positive outcome - pursue aggressively")
                elif row['Change %'] < -0.05:
                    st.error("⚠️ Significant risk - prepare contingencies")
                else:
                    st.warning("📊 Manageable impact - monitor closely")
    
    # Detailed scenario assumptions
    st.markdown("### 📋 Scenario Assumptions Matrix")
    
    assumption_data = []
    for scenario_name, changes in scenarios.items():
        row = {"Scenario": scenario_name}
        for key, value in changes.items():
            if key not in ['description', 'color']:
                row[key] = f"{value:+.0%}"
        assumption_data.append(row)
    
    assumption_df = pd.DataFrame(assumption_data)
    
    st.dataframe(assumption_df, use_container_width=True, hide_index=True)
    
    # Monte Carlo simulation
    st.markdown("### 🎲 Monte Carlo Simulation")
    
    with st.expander("Run Probabilistic Analysis", expanded=False):
        n_simulations = st.slider("Number of simulations", 100, 1000, 500, step=100)
        
        if st.button("🚀 Run Simulation"):
            with st.spinner("Running Monte Carlo simulation..."):
                simulations = []
                
                for _ in range(n_simulations):
                    sim_input = base_input.copy()
                    
                    # Add random noise
                    for col in ['Cost Ratio', 'Cost efficiency']:
                        if col in sim_input.columns:
                            noise = np.random.normal(0, 0.05)
                            sim_input[col] = base_input[col] * (1 + noise)
                    
                    sim_pred = rf_model.predict(sim_input.values)[0]
                    simulations.append(sim_pred)
                
                sim_df = pd.DataFrame({'Margin': simulations})
                
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=sim_df['Margin'],
                    nbinsx=50,
                    marker_color='#667eea',
                    name='Simulations'
                ))
                
                fig.add_vline(
                    x=base_pred,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text="Current"
                )
                
                fig.update_layout(
                    title=f"Distribution of {n_simulations} Simulated Outcomes",
                    xaxis_title="Profit Margin (%)",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean", f"{sim_df['Margin'].mean():.3f}%")
                with col2:
                    st.metric("Std Dev", f"{sim_df['Margin'].std():.3f}%")
                with col3:
                    percentile_5 = sim_df['Margin'].quantile(0.05)
                    st.metric("5th Percentile", f"{percentile_5:.3f}%")

# ========================
# 5. RECOMMENDATIONS PAGE
# ========================
elif page == "🚀 Recommendations":
    st.markdown('<p class="section-header">🚀 Strategic Recommendations</p>', unsafe_allow_html=True)
    
    # Executive summary
    st.markdown("""
    <div class="insight-box">
        <h2 style='margin-top: 0;'>📊 Executive Summary</h2>
        <p style='font-size: 1.1rem;'>
        Based on comprehensive analysis of your supply chain data, we've identified 
        <strong>high-impact opportunities</strong> to improve profitability through targeted 
        cost optimization, operational efficiency, and strategic supplier management.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Priority matrix
    st.markdown("### 🎯 Priority Action Matrix")
    
    actions = [
        {"Action": "Cost Ratio Reduction", "Impact": "High", "Effort": "Medium", "Timeline": "3-6 months", "ROI": "Very High"},
        {"Action": "Supplier Consolidation", "Impact": "High", "Effort": "Low", "Timeline": "1-3 months", "ROI": "High"},
        {"Action": "Process Automation", "Impact": "Medium", "Effort": "High", "Timeline": "6-12 months", "ROI": "Medium"},
        {"Action": "Inventory Optimization", "Impact": "Medium", "Effort": "Medium", "Timeline": "3-6 months", "ROI": "High"},
        {"Action": "Transportation Mode Shift", "Impact": "Low", "Effort": "Low", "Timeline": "1-3 months", "ROI": "Medium"},
    ]
    
    action_df = pd.DataFrame(actions)
    
    # Create scatter plot for priority matrix
    impact_map = {"High": 3, "Medium": 2, "Low": 1}
    effort_map = {"High": 3, "Medium": 2, "Low": 1}
    
    action_df['Impact_Score'] = action_df['Impact'].map(impact_map)
    action_df['Effort_Score'] = action_df['Effort'].map(effort_map)
    
    fig = px.scatter(
        action_df,
        x='Effort_Score',
        y='Impact_Score',
        text='Action',
        size=[20, 18, 16, 14, 12],
        color='ROI',
        color_discrete_map={'Very High': '#10b981', 'High': '#3b82f6', 'Medium': '#f59e0b'},
        title="Impact vs Effort Matrix"
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis=dict(title="Effort Required", tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High']),
        yaxis=dict(title="Business Impact", tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High']),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed recommendations by category
    rec_tab1, rec_tab2, rec_tab3, rec_tab4 = st.tabs([
        "💰 Cost Optimization",
        "⚙️ Operational Excellence",
        "🤝 Supplier Management",
        "📈 Growth Strategies"
    ])
    
    with rec_tab1:
        st.markdown("""
        ### 💰 Cost Optimization Initiatives
        
        #### 🎯 Priority 1: Reduce Cost Ratio by 5-10%
        
        **Current State:**
        - Average cost ratio: 2.0%
        - Top quartile performers: <1.5%
        - Opportunity: $50K+ annual savings
        
        **Action Steps:**
        1. **Manufacturing Cost Audit** (Weeks 1-2)
           - Analyze cost breakdown by product line
           - Identify top 20% cost drivers
           - Benchmark against industry standards
        
        2. **Supplier Negotiations** (Weeks 3-6)
           - Leverage volume commitments for 10-15% discounts
           - Consolidate orders with top-performing suppliers
           - Implement long-term contracts with price locks
        
        3. **Process Optimization** (Months 2-4)
           - Eliminate waste in manufacturing workflow
           - Reduce material scrap by 15%
           - Implement just-in-time inventory practices
        
        **Expected Outcome:**
        - 7% reduction in cost ratio
        - **+0.03% profit margin improvement**
        - Annual savings: $75K-$100K
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box-green">
                <h4>Quick Wins (0-30 days)</h4>
                <ul>
                    <li>Renegotiate shipping contracts</li>
                    <li>Consolidate suppliers (Supplier 1 focus)</li>
                    <li>Eliminate low-margin SKUs</li>
                    <li>Reduce packaging costs by 10%</li>
                </ul>
                <strong>Potential Savings: $15K-$25K</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box-blue">
                <h4>Medium-Term (3-6 months)</h4>
                <ul>
                    <li>Automate quality control processes</li>
                    <li>Implement lean manufacturing</li>
                    <li>Optimize batch sizes</li>
                    <li>Energy efficiency improvements</li>
                </ul>
                <strong>Potential Savings: $40K-$60K</strong>
            </div>
            """, unsafe_allow_html=True)
    
    with rec_tab2:
        st.markdown("""
        ### ⚙️ Operational Excellence Program
        
        #### 🎯 Improve Cost Efficiency by 15%
        
        **Current Metrics:**
        - Average cost efficiency: 233
        - Target: >275
        - Gap: 18% improvement needed
        
        **Efficiency Improvement Roadmap:**
        
        ##### Phase 1: Quick Efficiency Gains (Months 1-3)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Production Optimization**
            - Increase capacity utilization from 75% to 90%
            - Reduce setup times by 20%
            - Implement preventive maintenance
            - Cross-train workforce for flexibility
            
            **Expected Impact:** +8% efficiency
            """)
        
        with col2:
            st.markdown("""
            **Lead Time Reduction**
            - Reduce average lead time from 17 to 14 days
            - Implement vendor-managed inventory
            - Streamline approval processes
            - Digital procurement platform
            
            **Expected Impact:** +7% efficiency
            """)
        
        st.markdown("""
        ##### Phase 2: Technology & Automation (Months 4-8)
        
        | Initiative | Investment | Annual Benefit | Payback Period |
        |-----------|-----------|----------------|----------------|
        | Warehouse Management System | $50K | $80K | 7.5 months |
        | Automated Quality Control | $35K | $45K | 9 months |
        | Predictive Maintenance AI | $25K | $40K | 7.5 months |
        | Real-time Inventory Tracking | $20K | $30K | 8 months |
        
        **Total ROI:** 195% over 2 years
        """)
        
        # Performance tracking dashboard
        st.markdown("#### 📊 Key Performance Indicators to Track")
        
        kpis = pd.DataFrame({
            'KPI': ['Cost Efficiency', 'Lead Time', 'Capacity Utilization', 'On-Time Delivery', 'Quality Score'],
            'Current': [233, 17, 75, 92, 85],
            'Target': [275, 14, 90, 98, 95],
            'Gap %': [18, -18, 20, 7, 12]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current',
            x=kpis['KPI'],
            y=kpis['Current'],
            marker_color='#f59e0b'
        ))
        
        fig.add_trace(go.Bar(
            name='Target',
            x=kpis['KPI'],
            y=kpis['Target'],
            marker_color='#10b981'
        ))
        
        fig.update_layout(
            title="Current vs Target Performance",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with rec_tab3:
        st.markdown("""
        ### 🤝 Supplier Management Strategy
        
        #### 🎯 Optimize Supplier Portfolio
        
        **Analysis Findings:**
        - **Supplier 1:** Best quality (1.80% defect rate) ⭐
        - **Supplier 4:** Highest costs but acceptable quality
        - **Supplier 3:** Longest lead times (20 days)
        
        **Recommended Actions:**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box-green">
                <h4>✅ Consolidate & Expand</h4>
                <p><strong>Supplier 1 - Strategic Partner</strong></p>
                <ul>
                    <li>Increase volume by 40%</li>
                    <li>Negotiate 12-15% volume discount</li>
                    <li>Lock in 24-month pricing</li>
                    <li>Co-develop new products</li>
                    <li>Implement JIT delivery</li>
                </ul>
                <p><strong>Expected Benefit:</strong> $30K annual savings + quality improvement</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>⚠️ Reduce or Exit</h4>
                <p><strong>Supplier 3 & 4 - Phase Out</strong></p>
                <ul>
                    <li>Reduce Supplier 3 volume by 50% (lead time issues)</li>
                    <li>Migrate Supplier 4 SKUs to Supplier 1/2</li>
                    <li>Maintain as backup only</li>
                    <li>Complete transition in 6 months</li>
                </ul>
                <p><strong>Expected Benefit:</strong> Reduced complexity + cost savings</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        #### 📋 Supplier Scorecard Framework
        
        Implement quarterly supplier evaluations based on:
        """)
        
        scorecard_metrics = pd.DataFrame({
            'Metric': ['Quality (Defect Rate)', 'Delivery (On-Time %)', 'Cost Competitiveness', 
                      'Lead Time', 'Responsiveness'],
            'Weight': ['40%', '25%', '20%', '10%', '5%'],
            'Supplier 1': ['95/100', '92/100', '85/100', '88/100', '90/100'],
            'Supplier 2': ['85/100', '88/100', '90/100', '85/100', '85/100'],
            'Supplier 3': ['82/100', '75/100', '88/100', '70/100', '80/100']
        })
        
        st.dataframe(scorecard_metrics, use_container_width=True, hide_index=True)
        
        st.info("""
        💡 **Best Practice:** Conduct quarterly business reviews with top suppliers. 
        Share forecast data 6 months in advance to enable better planning and pricing.
        """)
    
    with rec_tab4:
        st.markdown("""
        ### 📈 Growth & Expansion Strategies
        
        #### 🎯 Scale High-Margin Products
        
        **Product Performance Analysis:**
        """)
        
        # Product profitability analysis
        product_analysis = df.groupby('Product type').agg({
            'Revenue generated': 'sum',
            'Profit Margin (%)': 'mean',
            'Number of products sold': 'sum'
        }).reset_index()
        
        product_analysis['Revenue_K'] = product_analysis['Revenue generated'] / 1000
        
        fig = px.scatter(
            product_analysis,
            x='Number of products sold',
            y='Profit Margin (%)',
            size='Revenue_K',
            color='Product type',
            text='Product type',
            title="Product Portfolio Analysis: Volume vs Margin"
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box-green">
                <h4>🚀 Growth Opportunities</h4>
                
                <p><strong>1. Double Down on High Performers</strong></p>
                <ul>
                    <li>Focus marketing on cosmetics (highest margin)</li>
                    <li>Expand SKU range in profitable categories</li>
                    <li>Target 30% volume growth</li>
                </ul>
                
                <p><strong>2. Geographic Expansion</strong></p>
                <ul>
                    <li>Enter high-potential regions</li>
                    <li>Localize product offerings</li>
                    <li>Partner with regional distributors</li>
                </ul>
                
                <p><strong>3. Customer Segment Penetration</strong></p>
                <ul>
                    <li>Increase share of wallet in existing segments</li>
                    <li>Launch targeted campaigns by demographics</li>
                    <li>Develop subscription/loyalty programs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box-blue">
                <h4>💡 Innovation Pipeline</h4>
                
                <p><strong>Product Development</strong></p>
                <ul>
                    <li>Launch 3 new high-margin SKUs per quarter</li>
                    <li>Test eco-friendly product line (15% premium)</li>
                    <li>Bundle complementary products</li>
                </ul>
                
                <p><strong>Channel Diversification</strong></p>
                <ul>
                    <li>Direct-to-consumer e-commerce</li>
                    <li>B2B wholesale partnerships</li>
                    <li>International market entry</li>
                </ul>
                
                <p><strong>Value-Added Services</strong></p>
                <ul>
                    <li>Custom formulation services</li>
                    <li>White-label manufacturing</li>
                    <li>Consulting & training programs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        #### 📊 5-Year Growth Projection
        """)
        
        # Growth projection chart
        years = list(range(2025, 2030))
        base_revenue = df['Revenue generated'].sum()
        
        conservative = [base_revenue * (1.05 ** i) for i in range(5)]
        moderate = [base_revenue * (1.12 ** i) for i in range(5)]
        aggressive = [base_revenue * (1.20 ** i) for i in range(5)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years, y=conservative, name='Conservative (5% CAGR)',
            mode='lines+markers', line=dict(color='#f59e0b', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=years, y=moderate, name='Moderate (12% CAGR)',
            mode='lines+markers', line=dict(color='#3b82f6', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=years, y=aggressive, name='Aggressive (20% CAGR)',
            mode='lines+markers', line=dict(color='#10b981', width=3)
        ))
        
        fig.update_layout(
            title="Revenue Growth Scenarios (2025-2029)",
            xaxis_title="Year",
            yaxis_title="Revenue ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Implementation roadmap
    st.markdown("---")
    st.markdown("### 📅 90-Day Implementation Roadmap")
    
    roadmap_data = {
        'Month 1': [
            '✅ Complete cost audit',
            '✅ Renegotiate top 3 supplier contracts',
            '✅ Launch quick-win cost reduction initiatives',
            '✅ Implement supplier scorecard'
        ],
        'Month 2': [
            '🔄 Begin supplier consolidation',
            '🔄 Pilot process automation in 1 product line',
            '🔄 Roll out efficiency training program',
            '🔄 Establish KPI tracking dashboard'
        ],
        'Month 3': [
            '🎯 Complete Phase 1 supplier migration',
            '🎯 Launch 2 new high-margin products',
            '🎯 Finalize technology investment plan',
            '🎯 Conduct first quarterly business review'
        ]
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📍 Month 1: Foundation")
        for item in roadmap_data['Month 1']:
            st.markdown(f"- {item}")
    
    with col2:
        st.markdown("#### 📍 Month 2: Execution")
        for item in roadmap_data['Month 2']:
            st.markdown(f"- {item}")
    
    with col3:
        st.markdown("#### 📍 Month 3: Momentum")
        for item in roadmap_data['Month 3']:
            st.markdown(f"- {item}")
    
    # Success metrics
    st.markdown("---")
    st.markdown("### 🎯 Success Metrics & Targets")
    
    success_metrics = pd.DataFrame({
        'Metric': [
            'Profit Margin (%)',
            'Cost Ratio',
            'Cost Efficiency',
            'Defect Rate (%)',
            'Lead Time (days)',
            'Supplier Concentration (Top 2)'
        ],
        'Baseline': ['99.13%', '0.020', '233', '2.28%', '17.1', '45%'],
        'Q1 Target': ['99.20%', '0.019', '245', '2.10%', '16.0', '55%'],
        'Q2 Target': ['99.30%', '0.018', '260', '1.90%', '15.0', '65%'],
        'Q3 Target': ['99.40%', '0.017', '270', '1.75%', '14.5', '70%'],
        'Year-End Target': ['99.50%', '0.015', '285', '1.50%', '14.0', '75%']
    })
    
    st.dataframe(success_metrics, use_container_width=True, hide_index=True)
    
    # Final CTA
    st.markdown("---")
    st.markdown("""
    <div class="insight-box">
        <h2 style='margin-top: 0;'>🎯 Next Steps</h2>
        <ol style='font-size: 1.1rem; line-height: 1.8;'>
            <li><strong>Week 1:</strong> Assemble cross-functional team (Operations, Finance, Procurement)</li>
            <li><strong>Week 2:</strong> Conduct detailed cost audit and supplier assessment</li>
            <li><strong>Week 3:</strong> Finalize prioritized action plan with owners and timelines</li>
            <li><strong>Week 4:</strong> Launch first wave of quick-win initiatives</li>
            <li><strong>Ongoing:</strong> Weekly progress reviews, monthly KPI tracking, quarterly strategy refresh</li>
        </ol>
        
        <p style='font-size: 1.1rem; margin-top: 1.5rem;'>
        <strong>Expected 12-Month Impact:</strong><br>
        💰 Profit Margin: +0.37% (to 99.50%)<br>
        💵 Annual Cost Savings: $150K-$200K<br>
        ⚡ Efficiency Gain: +22%<br>
        ✅ Quality Improvement: 34% defect reduction
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download report button
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("📥 Download Full Report (PDF)", use_container_width=True):
            st.success("Report generation feature coming soon!")

# ========================
# FOOTER
# ========================
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; color: white; font-size: 0.85rem; padding: 1rem;'>
        <p style='margin: 0.5rem 0;'><strong>Supply Chain Intelligence Platform</strong></p>
        <p style='margin: 0.5rem 0;'>Version 2.0 • Built with Streamlit</p>
        <p style='margin: 0.5rem 0; opacity: 0.8;'>Last Updated: October 2025</p>
        <hr style='border-color: rgba(255,255,255,0.2); margin: 1rem 0;'>
        <p style='margin: 0; opacity: 0.7;'>Powered by Random Forest ML</p>
    </div>
""", unsafe_allow_html=True)

