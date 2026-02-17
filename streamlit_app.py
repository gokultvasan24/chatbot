import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    shapiro, normaltest, anderson, kstest,
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon,
    kruskal, f_oneway, friedmanchisquare,
    pearsonr, spearmanr, kendalltau,
    chi2_contingency, fisher_exact, barnard_exact
)
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import DescrStatsW
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Statistical Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'test_suggestions' not in st.session_state:
    st.session_state.test_suggestions = {}

# Title and description
st.title("📊 Advanced Statistical Analysis & Visualization Tool")
st.markdown("""
This tool automatically analyzes your data, suggests appropriate statistical tests 
(both parametric and non-parametric), and performs comprehensive statistical analysis 
based on data characteristics.
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
    
    # Analysis settings
    st.header("⚙️ Analysis Settings")
    significance_level = st.slider(
        "Significance Level (α)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01
    )
    
    auto_suggest = st.checkbox("Auto-suggest statistical tests", value=True)
    perform_tests = st.checkbox("Automatically perform suggested tests", value=True)
    
    st.divider()
    
    # Visualization settings
    st.header("📈 Visualization Settings")
    plot_theme = st.selectbox(
        "Plot Theme",
        ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"]
    )
    
    color_palette = st.selectbox(
        "Color Palette",
        ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo"]
    )

# Main content area
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Data Overview", 
        "📊 Normality Tests",
        "📈 Hypothesis Testing",
        "🔗 Correlation Analysis",
        "📉 Regression Analysis",
        "📑 Complete Report"
    ])
    
    # Tab 1: Data Overview with Automatic Test Suggestions
    with tab1:
        st.header("Dataset Overview & Test Suggestions")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())
        with col4:
            st.metric("Complete Cases", df.dropna().shape[0])
        
        # Automatic data type detection and test suggestions
        st.subheader("🔍 Automatic Test Suggestions Based on Data")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create suggestion dataframe
        suggestions = []
        
        for col in numeric_cols:
            data = df[col].dropna()
            n = len(data)
            
            # Check normality
            if n >= 3:
                if n <= 5000:
                    stat, p_value = shapiro(data)
                    is_normal = p_value > significance_level
                else:
                    stat, p_value = normaltest(data)
                    is_normal = p_value > significance_level
                
                # Suggest tests based on normality
                if is_normal:
                    suggestions.append({
                        'Variable': col,
                        'Type': 'Numeric (Normal)',
                        'Parametric Tests': 't-test, ANOVA, Pearson Correlation',
                        'Non-Parametric Alternatives': 'Mann-Whitney U, Kruskal-Wallis, Spearman',
                        'Normality Test': f"Shapiro-Wilk p={p_value:.4f}",
                        'Distribution': 'Normal'
                    })
                else:
                    suggestions.append({
                        'Variable': col,
                        'Type': 'Numeric (Non-normal)',
                        'Parametric Tests': '⚠️ Not recommended',
                        'Non-Parametric Tests': 'Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Spearman',
                        'Normality Test': f"Shapiro-Wilk p={p_value:.4f}",
                        'Distribution': 'Non-normal'
                    })
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            suggestions.append({
                'Variable': col,
                'Type': 'Categorical',
                'Parametric Tests': 'N/A',
                'Non-Parametric Tests': f'Chi-square, Fisher\'s Exact (if 2x2)',
                'Categories': f'{unique_count} unique values',
                'Distribution': 'Categorical'
            })
        
        suggestions_df = pd.DataFrame(suggestions)
        st.dataframe(suggestions_df, use_container_width=True)
        
        # Summary statistics by variable type
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Numeric Variables Summary")
            if numeric_cols:
                numeric_summary = df[numeric_cols].describe().T
                numeric_summary['skew'] = df[numeric_cols].skew()
                numeric_summary['kurtosis'] = df[numeric_cols].kurtosis()
                st.dataframe(numeric_summary.round(3), use_container_width=True)
        
        with col2:
            st.subheader("📋 Categorical Variables Summary")
            if categorical_cols:
                cat_summary = []
                for col in categorical_cols[:5]:  # Limit to first 5
                    cat_summary.append({
                        'Variable': col,
                        'Categories': df[col].nunique(),
                        'Mode': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                        'Mode Freq': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0,
                        'Missing': df[col].isna().sum()
                    })
                st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)
    
    # Tab 2: Normality Tests
    with tab2:
        st.header("Normality Assessment")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            selected_var = st.selectbox("Select variable for normality tests", numeric_cols)
            
            if selected_var:
                data = df[selected_var].dropna()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Statistical Tests")
                    
                    # Shapiro-Wilk Test
                    if len(data) <= 5000:
                        stat, p_value = shapiro(data)
                        st.write(f"**Shapiro-Wilk Test:**")
                        st.write(f"- Statistic: {stat:.4f}")
                        st.write(f"- P-value: {p_value:.4f}")
                        st.write(f"- {'✓ Normal' if p_value > significance_level else '✗ Non-normal'} (α={significance_level})")
                    
                    # D'Agostino's K^2 Test
                    stat, p_value = normaltest(data)
                    st.write(f"\n**D'Agostino's K² Test:**")
                    st.write(f"- Statistic: {stat:.4f}")
                    st.write(f"- P-value: {p_value:.4f}")
                    st.write(f"- {'✓ Normal' if p_value > significance_level else '✗ Non-normal'}")
                    
                    # Anderson-Darling Test
                    result = anderson(data)
                    st.write(f"\n**Anderson-Darling Test:**")
                    st.write(f"- Statistic: {result.statistic:.4f}")
                    st.write(f"- Critical values: {result.critical_values}")
                    st.write(f"- Significance level: {result.significance_level}")
                    
                    # Kolmogorov-Smirnov Test
                    stat, p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
                    st.write(f"\n**Kolmogorov-Smirnov Test:**")
                    st.write(f"- Statistic: {stat:.4f}")
                    st.write(f"- P-value: {p_value:.4f}")
                
                with col2:
                    st.subheader("Visual Normality Checks")
                    
                    # Q-Q Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=np.random.normal(0, 1, len(data)),
                        y=np.sort(data),
                        mode='markers',
                        name='Q-Q Plot',
                        marker=dict(color='blue', size=5)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[-3, 3],
                        y=[data.mean() - 3*data.std(), data.mean() + 3*data.std()],
                        mode='lines',
                        name='Reference Line',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title="Q-Q Plot",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Distribution plot
                    fig = px.histogram(
                        data, nbins=30,
                        title=f"Distribution of {selected_var}",
                        labels={'value': selected_var, 'count': 'Frequency'}
                    )
                    fig.add_vline(x=data.mean(), line_dash="dash", line_color="red",
                                 annotation_text=f"Mean: {data.mean():.2f}")
                    fig.add_vline(x=data.median(), line_dash="dash", line_color="green",
                                 annotation_text=f"Median: {data.median():.2f}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric variables found for normality testing")
    
    # Tab 3: Hypothesis Testing
    with tab3:
        st.header("Statistical Hypothesis Testing")
        
        test_type = st.selectbox(
            "Select test type",
            ["Two-group comparison (Independent)", 
             "Two-group comparison (Paired)",
             "Multiple groups comparison",
             "Proportion test",
             "Goodness of fit"]
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if test_type == "Two-group comparison (Independent)":
            st.subheader("Independent Two-Group Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                numeric_var = st.selectbox("Select numeric variable", numeric_cols, key='ind_num')
            
            with col2:
                group_var = st.selectbox("Select grouping variable", categorical_cols, key='ind_cat')
            
            if numeric_var and group_var:
                groups = df[group_var].dropna().unique()
                if len(groups) == 2:
                    group1_data = df[df[group_var] == groups[0]][numeric_var].dropna()
                    group2_data = df[df[group_var] == groups[1]][numeric_var].dropna()
                    
                    # Check normality
                    _, p1 = shapiro(group1_data) if len(group1_data) <= 5000 else (None, 0)
                    _, p2 = shapiro(group2_data) if len(group2_data) <= 5000 else (None, 0)
                    
                    st.write(f"**Group 1 ({groups[0]}):** n={len(group1_data)}")
                    st.write(f"**Group 2 ({groups[1]}):** n={len(group2_data)}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Parametric Test (t-test)")
                        
                        # Check variance equality
                        _, p_var = stats.levene(group1_data, group2_data)
                        
                        if p_var > significance_level:
                            t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=True)
                            st.write("**Equal variance assumed**")
                        else:
                            t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=False)
                            st.write("**Equal variance not assumed (Welch's t-test)**")
                        
                        st.write(f"t-statistic: {t_stat:.4f}")
                        st.write(f"p-value: {p_value:.4f}")
                        
                        if p_value < significance_level:
                            st.success(f"✓ Significant difference (p < {significance_level})")
                        else:
                            st.info(f"✗ No significant difference (p > {significance_level})")
                    
                    with col2:
                        st.subheader("Non-parametric Test (Mann-Whitney U)")
                        
                        u_stat, p_value = mannwhitneyu(group1_data, group2_data)
                        st.write(f"U-statistic: {u_stat:.4f}")
                        st.write(f"p-value: {p_value:.4f}")
                        
                        if p_value < significance_level:
                            st.success(f"✓ Significant difference (p < {significance_level})")
                        else:
                            st.info(f"✗ No significant difference (p > {significance_level})")
                    
                    # Visualization
                    fig = go.Figure()
                    fig.add_trace(go.Box(y=group1_data, name=str(groups[0])))
                    fig.add_trace(go.Box(y=group2_data, name=str(groups[1])))
                    fig.update_layout(title="Group Comparison Box Plot")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Grouping variable should have exactly 2 categories (found {len(groups)})")
        
        elif test_type == "Multiple groups comparison":
            st.subheader("Multiple Groups Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                numeric_var = st.selectbox("Select numeric variable", numeric_cols, key='multi_num')
            
            with col2:
                group_var = st.selectbox("Select grouping variable", categorical_cols, key='multi_cat')
            
            if numeric_var and group_var:
                groups = df[group_var].dropna().unique()
                group_data = [df[df[group_var] == g][numeric_var].dropna() for g in groups]
                
                st.write(f"**Number of groups:** {len(groups)}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Parametric Test (ANOVA)")
                    
                    f_stat, p_value = f_oneway(*group_data)
                    st.write(f"F-statistic: {f_stat:.4f}")
                    st.write(f"p-value: {p_value:.4f}")
                    
                    if p_value < significance_level:
                        st.success(f"✓ Significant differences among groups (p < {significance_level})")
                        
                        # Post-hoc test
                        if len(groups) > 2:
                            st.subheader("Post-hoc Analysis (Tukey HSD)")
                            
                            # Prepare data for Tukey
                            tukey_data = df[[numeric_var, group_var]].dropna()
                            tukey_result = pairwise_tukeyhsd(tukey_data[numeric_var], 
                                                             tukey_data[group_var])
                            
                            result_df = pd.DataFrame(data=tukey_result.summary().data[1:],
                                                    columns=tukey_result.summary().data[0])
                            st.dataframe(result_df)
                    else:
                        st.info(f"✗ No significant differences among groups (p > {significance_level})")
                
                with col2:
                    st.subheader("Non-parametric Test (Kruskal-Wallis)")
                    
                    h_stat, p_value = kruskal(*group_data)
                    st.write(f"H-statistic: {h_stat:.4f}")
                    st.write(f"p-value: {p_value:.4f}")
                    
                    if p_value < significance_level:
                        st.success(f"✓ Significant differences among groups (p < {significance_level})")
                    else:
                        st.info(f"✗ No significant differences among groups (p > {significance_level})")
                
                # Visualization
                fig = go.Figure()
                for i, g in enumerate(groups):
                    fig.add_trace(go.Violin(y=group_data[i], name=str(g), box_visible=True))
                fig.update_layout(title="Group Comparison (Violin Plot)")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Correlation Analysis
    with tab4:
        st.header("Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Select variables for correlation
            selected_vars = st.multiselect(
                "Select variables for correlation analysis",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if len(selected_vars) >= 2:
                # Check normality for each variable
                normality_results = {}
                for var in selected_vars:
                    data = df[var].dropna()
                    if len(data) <= 5000:
                        _, p_val = shapiro(data)
                        normality_results[var] = p_val > significance_level
                
                st.subheader("Correlation Matrix")
                
                # Calculate different correlation coefficients
                pearson_corr = df[selected_vars].corr(method='pearson')
                spearman_corr = df[selected_vars].corr(method='spearman')
                kendall_corr = df[selected_vars].corr(method='kendall')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Pearson Correlation**")
                    fig = px.imshow(
                        pearson_corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Pearson (Parametric)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Spearman Correlation**")
                    fig = px.imshow(
                        spearman_corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Spearman (Non-parametric)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    st.write("**Kendall Correlation**")
                    fig = px.imshow(
                        kendall_corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Kendall Tau (Non-parametric)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed correlation tests
                st.subheader("Detailed Correlation Tests")
                
                # Create pairwise correlation tests
                results = []
                for i in range(len(selected_vars)):
                    for j in range(i+1, len(selected_vars)):
                        var1 = selected_vars[i]
                        var2 = selected_vars[j]
                        
                        # Remove missing values
                        clean_data = df[[var1, var2]].dropna()
                        
                        if len(clean_data) > 3:
                            # Pearson
                            r_pearson, p_pearson = pearsonr(clean_data[var1], clean_data[var2])
                            
                            # Spearman
                            r_spearman, p_spearman = spearmanr(clean_data[var1], clean_data[var2])
                            
                            # Kendall
                            r_kendall, p_kendall = kendalltau(clean_data[var1], clean_data[var2])
                            
                            # Recommendation based on normality
                            both_normal = normality_results.get(var1, False) and normality_results.get(var2, False)
                            
                            results.append({
                                'Variable 1': var1,
                                'Variable 2': var2,
                                'Pearson r': round(r_pearson, 3),
                                'Pearson p': round(p_pearson, 4),
                                'Spearman ρ': round(r_spearman, 3),
                                'Spearman p': round(p_spearman, 4),
                                'Kendall τ': round(r_kendall, 3),
                                'Kendall p': round(p_kendall, 4),
                                'Recommended': 'Pearson' if both_normal else 'Spearman/Kendall',
                                'N': len(clean_data)
                            })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Scatter plot matrix
                st.subheader("Scatter Plot Matrix")
                fig = px.scatter_matrix(
                    df[selected_vars],
                    dimensions=selected_vars,
                    title="Pairwise Relationships"
                )
                fig.update_traces(diagonal_visible=False)
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric variables for correlation analysis")
    
    # Tab 5: Regression Analysis
    with tab5:
        st.header("Regression Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                dependent_var = st.selectbox("Select dependent variable (Y)", numeric_cols)
            
            with col2:
                independent_vars = st.multiselect(
                    "Select independent variables (X)",
                    [col for col in numeric_cols if col != dependent_var]
                )
            
            if dependent_var and len(independent_vars) >= 1:
                # Prepare data
                X = df[independent_vars].copy()
                y = df[dependent_var].copy()
                
                # Remove missing values
                mask = ~(X.isna().any(axis=1) | y.isna())
                X_clean = X[mask]
                y_clean = y[mask]
                
                # Add constant for intercept
                X_with_const = sm.add_constant(X_clean)
                
                # Fit model
                model = sm.OLS(y_clean, X_with_const).fit()
                
                # Model summary
                st.subheader("Regression Model Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R-squared", f"{model.rsquared:.4f}")
                with col2:
                    st.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
                with col3:
                    st.metric("F-statistic", f"{model.fvalue:.2f}")
                
                st.write(f"**F-test p-value:** {model.f_pvalue:.4f}")
                
                # Coefficients table
                coef_df = pd.DataFrame({
                    'Variable': ['Intercept'] + independent_vars,
                    'Coefficient': model.params.values,
                    'Std Error': model.bse.values,
                    't-value': model.tvalues.values,
                    'p-value': model.pvalues.values,
                    'Conf. Interval Low': model.conf_int()[0],
                    'Conf. Interval High': model.conf_int()[1]
                })
                
                st.dataframe(coef_df.round(4), use_container_width=True)
                
                # Diagnostic plots
                st.subheader("Regression Diagnostics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Residuals vs Fitted
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=model.fittedvalues,
                        y=model.resid,
                        mode='markers',
                        marker=dict(color='blue', size=5)
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    fig.update_layout(
                        title="Residuals vs Fitted Values",
                        xaxis_title="Fitted Values",
                        yaxis_title="Residuals"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Q-Q plot of residuals
                    from scipy import stats
                    
                    theoretical_quantiles = stats.norm.ppf(
                        np.linspace(0.01, 0.99, len(model.resid))
                    )
                    sorted_residuals = np.sort(model.resid)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=sorted_residuals,
                        mode='markers',
                        marker=dict(color='blue', size=5)
                    ))
                    
                    # Add reference line
                    z = np.polyfit(theoretical_quantiles, sorted_residuals, 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=p(theoretical_quantiles),
                        mode='lines',
                        name='Reference Line',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Q-Q Plot of Residuals",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Check assumptions
                st.subheader("Assumption Checks")
                
                # Normality of residuals
                _, p_norm = stats.shapiro(model.resid) if len(model.resid) <= 5000 else (None, 1)
                st.write(f"**Normality of residuals (Shapiro-Wilk):** p-value = {p_norm:.4f}")
                if p_norm > significance_level:
                    st.success("✓ Residuals appear normally distributed")
                else:
                    st.warning("⚠️ Residuals may not be normally distributed")
                
                # Homoscedasticity
                _, p_het = stats.bartlett(model.fittedvalues, model.resid)
                st.write(f"**Homoscedasticity (Bartlett's test):** p-value = {p_het:.4f}")
                if p_het > significance_level:
                    st.success("✓ Variances appear homogeneous")
                else:
                    st.warning("⚠️ Heteroscedasticity may be present")
    
    # Tab 6: Complete Report
    with tab6:
        st.header("Complete Statistical Analysis Report")
        
        if st.button("Generate Complete Statistical Report"):
            with st.spinner("Generating comprehensive report..."):
                report_lines = []
                report_lines.append("=" * 70)
                report_lines.append("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
                report_lines.append("=" * 70)
                report_lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"Dataset: {uploaded_file.name if uploaded_file else 'Unknown'}")
                report_lines.append(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
                report_lines.append(f"Significance Level: α = {significance_level}")
                
                # Section 1: Data Overview
                report_lines.append("\n" + "=" * 50)
                report_lines.append("1. DATA OVERVIEW")
                report_lines.append("=" * 50)
                
                report_lines.append(f"\nTotal Observations: {df.shape[0]}")
                report_lines.append(f"Total Variables: {df.shape[1]}")
                report_lines.append(f"Complete Cases: {df.dropna().shape[0]}")
                report_lines.append(f"Missing Values: {df.isna().sum().sum()}")
                
                # Section 2: Variable Types
                report_lines.append("\n" + "=" * 50)
                report_lines.append("2. VARIABLE CLASSIFICATION")
                report_lines.append("=" * 50)
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                
                report_lines.append(f"\nNumeric Variables ({len(numeric_cols)}):")
                for col in numeric_cols:
                    report_lines.append(f"  • {col}")
                
                report_lines.append(f"\nCategorical Variables ({len(categorical_cols)}):")
                for col in categorical_cols:
                    report_lines.append(f"  • {col}")
                
                # Section 3: Normality Assessment
                report_lines.append("\n" + "=" * 50)
                report_lines.append("3. NORMALITY ASSESSMENT")
                report_lines.append("=" * 50)
                
                for col in numeric_cols[:10]:  # Limit to first 10
                    data = df[col].dropna()
                    if len(data) >= 3:
                        if len(data) <= 5000:
                            stat, p_val = shapiro(data)
                            report_lines.append(f"\n{col}:")
                            report_lines.append(f"  • Shapiro-Wilk p-value: {p_val:.4f}")
                            report_lines.append(f"  • Distribution: {'Normal' if p_val > significance_level else 'Non-normal'}")
                            report_lines.append(f"  • Skewness: {data.skew():.3f}")
                            report_lines.append(f"  • Kurtosis: {data.kurtosis():.3f}")
                
                # Section 4: Recommended Statistical Tests
                report_lines.append("\n" + "=" * 50)
                report_lines.append("4. RECOMMENDED STATISTICAL TESTS")
                report_lines.append("=" * 50)
                
                for col in numeric_cols:
                    data = df[col].dropna()
                    if len(data) >= 3:
                        if len(data) <= 5000:
                            _, p_val = shapiro(data)
                            is_normal = p_val > significance_level
                            
                            report_lines.append(f"\n{col}:")
                            if is_normal:
                                report_lines.append("  • Parametric tests recommended:")
                                report_lines.append("    - t-test (2 groups)")
                                report_lines.append("    - ANOVA (multiple groups)")
                                report_lines.append("    - Pearson correlation")
                            else:
                                report_lines.append("  • Non-parametric tests recommended:")
                                report_lines.append("    - Mann-Whitney U test (2 groups)")
                                report_lines.append("    - Kruskal-Wallis test (multiple groups)")
                                report_lines.append("    - Spearman correlation")
                
                # Section 5: Key Findings
                report_lines.append("\n" + "=" * 50)
                report_lines.append("5. KEY STATISTICAL FINDINGS")
                report_lines.append("=" * 50)
                
                # Check for significant correlations
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    significant_corrs = []
                    for i in range(len(numeric_cols)):
                        for j in range(i+1, len(numeric_cols)):
                            if abs(corr_matrix.iloc[i, j]) > 0.5:
                                significant_corrs.append(
                                    f"{numeric_cols[i]} & {numeric_cols[j]}: {corr_matrix.iloc[i, j]:.3f}"
                                )
                    
                    if significant_corrs:
                        report_lines.append("\nStrong Correlations (|r| > 0.5):")
                        for corr in significant_corrs[:10]:
                            report_lines.append(f"  • {corr}")
                
                # Create report text
                report_text = "\n".join(report_lines)
                
                # Display report
                st.text_area("Statistical Analysis Report", report_text, height=500)
                
                # Download button
                st.download_button(
                    label="📥 Download Statistical Report",
                    data=report_text,
                    file_name=f"statistical_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

else:
    # Welcome message when no data is loaded
    st.info("👆 Please upload a CSV or Excel file to begin statistical analysis")
    
    st.markdown("""
    ### 🎯 Features:
    
    #### 1. **Automatic Test Suggestions**
    - Normality testing (Shapiro-Wilk, D'Agostino's K², Anderson-Darling)
    - Automatic recommendation of parametric vs non-parametric tests
    - Based on data distribution and sample size
    
    #### 2. **Comprehensive Statistical Tests**
    - **Group Comparisons**: t-test, Mann-Whitney U, ANOVA, Kruskal-Wallis
    - **Paired Tests**: Paired t-test, Wilcoxon signed-rank
    - **Correlation**: Pearson, Spearman, Kendall Tau
    - **Categorical**: Chi-square, Fisher's exact test
    - **Post-hoc**: Tukey HSD for multiple comparisons
    
    #### 3. **Regression Analysis**
    - Multiple linear regression
    - Diagnostic plots and assumption checks
    - Coefficient interpretation
    
    #### 4. **Visualization**
    - Distribution plots with normality curves
    - Q-Q plots for normality assessment
    - Correlation heatmaps (parametric and non-parametric)
    - Box plots and violin plots for group comparisons
    - Residual diagnostic plots
    
    #### 5. **Complete Statistical Report**
    - Automated report generation
    - Key findings highlighted
    - Recommendations for further analysis
    """)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center'>
        <p>Advanced Statistical Analysis Tool • Automatically suggests and performs appropriate tests based on your data</p>
    </div>
    """,
    unsafe_allow_html=True
)
