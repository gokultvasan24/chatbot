import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Analysis & Visualization Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Title and description
st.title("📊 Comprehensive Data Analysis & Visualization Tool")
st.markdown("""
This tool provides automated data analysis and visualization for any uploaded dataset.
It will automatically detect data types and generate appropriate visualizations and statistics.
""")

# Sidebar for file upload and controls
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.data = df
            st.success(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    st.divider()
    
    # Analysis options
    st.header("⚙️ Analysis Options")
    show_missing = st.checkbox("Show missing data analysis", value=True)
    show_numeric = st.checkbox("Show numeric variable analysis", value=True)
    show_categorical = st.checkbox("Show categorical variable analysis", value=True)
    show_correlation = st.checkbox("Show correlation analysis", value=True)
    show_distributions = st.checkbox("Show distribution plots", value=True)
    
    # Export options
    st.divider()
    st.header("💾 Export Options")
    if st.button("Generate Report"):
        if st.session_state.data is not None:
            st.session_state.generate_report = True
        else:
            st.warning("Please upload data first")

# Main content area
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Overview", 
        "📊 Descriptive Statistics", 
        "📈 Visualizations",
        "🔗 Correlations",
        "📑 Report"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.subheader("First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Data Types")
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values,
            'Null %': (df.isna().sum().values / len(df) * 100).round(2),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtypes_df, use_container_width=True)
        
        # Missing data visualization
        if show_missing and df.isna().any().any():
            st.subheader("Missing Data Visualization")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Missing Values Count', 'Missing Values Percentage')
            )
            
            missing_counts = df.isna().sum()
            missing_percent = (missing_counts / len(df) * 100)
            
            # Filter columns with missing values
            missing_cols = missing_counts[missing_counts > 0]
            
            if len(missing_cols) > 0:
                # Bar chart for missing counts
                fig.add_trace(
                    go.Bar(x=missing_cols.index, y=missing_cols.values,
                          name='Missing Count', marker_color='red'),
                    row=1, col=1
                )
                
                # Bar chart for missing percentages
                fig.add_trace(
                    go.Bar(x=missing_cols.index, y=missing_percent[missing_cols.index],
                          name='Missing %', marker_color='orange'),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No missing values found in the dataset")
    
    # Tab 2: Descriptive Statistics
    with tab2:
        st.header("Descriptive Statistics")
        
        # Numeric variables
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and show_numeric:
            st.subheader("Numeric Variables Summary")
            
            # Calculate statistics
            stats_df = pd.DataFrame()
            for col in numeric_cols:
                stats_df[col] = [
                    df[col].count(),
                    df[col].mean(),
                    df[col].std(),
                    df[col].var(),
                    df[col].min(),
                    df[col].quantile(0.25),
                    df[col].median(),
                    df[col].quantile(0.75),
                    df[col].max(),
                    df[col].skew(),
                    df[col].kurtosis(),
                    df[col].isna().sum()
                ]
            
            stats_df.index = ['Count', 'Mean', 'Std Dev', 'Variance', 'Min', 
                             '25%', 'Median', '75%', 'Max', 'Skewness', 
                             'Kurtosis', 'Missing']
            
            st.dataframe(stats_df.round(3), use_container_width=True)
            
            # Interpretation
            with st.expander("📖 Interpretation Guide"):
                st.markdown("""
                - **Skewness**: 
                    - Near 0: Symmetric distribution
                    - Positive: Right-skewed (tail to the right)
                    - Negative: Left-skewed (tail to the left)
                - **Kurtosis**:
                    - Near 0: Normal distribution
                    - Positive: Heavy tails, sharp peak
                    - Negative: Light tails, flat peak
                """)
        
        # Categorical variables
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0 and show_categorical:
            st.subheader("Categorical Variables Summary")
            
            selected_cat = st.selectbox("Select categorical variable", cat_cols)
            
            # Frequency table
            freq_df = df[selected_cat].value_counts().reset_index()
            freq_df.columns = ['Category', 'Frequency']
            freq_df['Percentage'] = (freq_df['Frequency'] / len(df) * 100).round(2)
            freq_df['Cumulative %'] = freq_df['Percentage'].cumsum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(freq_df, use_container_width=True)
            
            with col2:
                # Bar chart
                fig = px.bar(freq_df.head(15), x='Category', y='Frequency',
                           title=f'Top 15 Categories - {selected_cat}',
                           color='Frequency', color_continuous_scale='Viridis')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Visualizations
    with tab3:
        st.header("Data Visualizations")
        
        if show_distributions:
            # Numeric distributions
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.subheader("Numeric Variable Distributions")
                
                # Select variables for visualization
                selected_numeric = st.multiselect(
                    "Select numeric variables to visualize",
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))]
                )
                
                if selected_numeric:
                    # Create subplot grid
                    n_cols = min(2, len(selected_numeric))
                    n_rows = (len(selected_numeric) + 1) // 2
                    
                    fig = make_subplots(
                        rows=n_rows, cols=n_cols,
                        subplot_titles=selected_numeric
                    )
                    
                    for i, col in enumerate(selected_numeric):
                        row = i // n_cols + 1
                        col_num = i % n_cols + 1
                        
                        # Add histogram
                        fig.add_trace(
                            go.Histogram(x=df[col].dropna(), name=col,
                                       marker_color='lightblue', opacity=0.7),
                            row=row, col=col_num
                        )
                        
                        # Add density line
                        hist_data = df[col].dropna()
                        if len(hist_data) > 1:
                            kernel_density = stats.gaussian_kde(hist_data)
                            x_range = np.linspace(hist_data.min(), hist_data.max(), 100)
                            fig.add_trace(
                                go.Scatter(x=x_range, y=kernel_density(x_range),
                                         mode='lines', name=f'{col} density',
                                         line=dict(color='red', width=2)),
                                row=row, col=col_num
                            )
                    
                    fig.update_layout(height=300 * n_rows, showlegend=False)
                    fig.update_xaxes(title_text="Value")
                    fig.update_yaxes(title_text="Frequency")
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Box plots
            if len(selected_numeric) > 0:
                st.subheader("Box Plots")
                fig = go.Figure()
                for col in selected_numeric:
                    fig.add_trace(go.Box(y=df[col].dropna(), name=col))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Categorical visualizations
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                st.subheader("Categorical Variable Visualizations")
                
                selected_cat_viz = st.selectbox(
                    "Select categorical variable for visualization",
                    cat_cols,
                    key='cat_viz'
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    value_counts = df[selected_cat_viz].value_counts().head(15)
                    fig_bar = px.bar(
                        x=value_counts.index, y=value_counts.values,
                        title=f'Bar Chart - {selected_cat_viz}',
                        labels={'x': 'Category', 'y': 'Count'}
                    )
                    fig_bar.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Pie chart (if <= 10 categories)
                    if len(value_counts) <= 10:
                        fig_pie = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f'Pie Chart - {selected_cat_viz}'
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Tab 4: Correlations
    with tab4:
        st.header("Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2 and show_correlation:
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Correlation Heatmap")
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Pearson Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Correlation Statistics")
                
                # Get upper triangle of correlation matrix
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                # Find significant correlations (|r| > 0.5)
                significant = upper_tri.stack()
                significant = significant[abs(significant) > 0.5].sort_values(ascending=False)
                
                if len(significant) > 0:
                    st.write("**Strong Correlations (|r| > 0.5):**")
                    for (var1, var2), corr in significant.items():
                        strength = "Positive" if corr > 0 else "Negative"
                        st.write(f"• {var1} & {var2}: {corr:.3f} ({strength})")
                else:
                    st.write("No strong correlations found")
            
            # Scatter plot matrix
            if len(numeric_cols) <= 5:
                st.subheader("Scatter Plot Matrix")
                fig = px.scatter_matrix(
                    df[numeric_cols],
                    dimensions=numeric_cols,
                    title="Pairwise Relationships"
                )
                fig.update_traces(diagonal_visible=False)
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation tests
            with st.expander("Detailed Correlation Tests"):
                st.subheader("Correlation Test Results")
                
                results = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        var1 = numeric_cols[i]
                        var2 = numeric_cols[j]
                        
                        # Remove missing values
                        clean_data = df[[var1, var2]].dropna()
                        if len(clean_data) > 3:
                            corr, p_value = stats.pearsonr(clean_data[var1], clean_data[var2])
                            
                            sig_level = ""
                            if p_value < 0.001:
                                sig_level = "***"
                            elif p_value < 0.01:
                                sig_level = "**"
                            elif p_value < 0.05:
                                sig_level = "*"
                            
                            results.append({
                                'Variable 1': var1,
                                'Variable 2': var2,
                                'Correlation': round(corr, 3),
                                'P-value': round(p_value, 4),
                                'Significance': sig_level,
                                'Sample Size': len(clean_data)
                            })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric variables for correlation analysis")
    
    # Tab 5: Report
    with tab5:
        st.header("Analysis Report")
        
        if st.button("Generate Complete Report"):
            with st.spinner("Generating report..."):
                # Create report content
                report_lines = []
                report_lines.append("=" * 60)
                report_lines.append("COMPREHENSIVE DATA ANALYSIS REPORT")
                report_lines.append("=" * 60)
                report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"Dataset: {uploaded_file.name if uploaded_file else 'Unknown'}")
                report_lines.append(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Dataset structure
                report_lines.append("\n" + "=" * 40)
                report_lines.append("DATASET STRUCTURE")
                report_lines.append("=" * 40)
                
                for col in df.columns:
                    report_lines.append(f"\nColumn: {col}")
                    report_lines.append(f"  Type: {df[col].dtype}")
                    report_lines.append(f"  Non-null: {df[col].count()}/{len(df)}")
                    report_lines.append(f"  Unique: {df[col].nunique()}")
                
                # Numeric statistics
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    report_lines.append("\n" + "=" * 40)
                    report_lines.append("NUMERIC VARIABLE STATISTICS")
                    report_lines.append("=" * 40)
                    
                    for col in numeric_cols:
                        report_lines.append(f"\n{col}:")
                        report_lines.append(f"  Mean: {df[col].mean():.3f}")
                        report_lines.append(f"  Std Dev: {df[col].std():.3f}")
                        report_lines.append(f"  Median: {df[col].median():.3f}")
                        report_lines.append(f"  Min: {df[col].min():.3f}")
                        report_lines.append(f"  Max: {df[col].max():.3f}")
                        report_lines.append(f"  Skewness: {df[col].skew():.3f}")
                
                # Join report
                report_text = "\n".join(report_lines)
                
                # Display report
                st.text_area("Generated Report", report_text, height=400)
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report_text,
                    file_name=f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

else:
    # Welcome message when no data is loaded
    st.info("👆 Please upload a CSV or Excel file to begin analysis")
    
    # Example of what the app can do
    st.markdown("""
    ### Features:
    - **📊 Automated Analysis**: Automatically detects data types and generates appropriate statistics
    - **📈 Interactive Visualizations**: Dynamic plots with Plotly for better exploration
    - **🔍 Missing Data Analysis**: Identifies and visualizes missing values
    - **📐 Descriptive Statistics**: Comprehensive statistical summaries
    - **🔄 Correlation Analysis**: Examines relationships between variables
    - **📑 Report Generation**: Export complete analysis reports
    
    ### Supported File Formats:
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    """)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit • Comprehensive Data Analysis Tool v1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
