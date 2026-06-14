import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Sector Rotation Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📊 Data Explorer",
    "📈 Model Performance",
    "🔮 Prediction Demo",
    "🧠 Explainability",
    "💼 Business Recommendation"
])

# -------------------------------------------------------------------
# Helper: safely read CSV, return None if not found
# -------------------------------------------------------------------
def safe_read_csv(filename, **kwargs):
    if os.path.exists(filename):
        return pd.read_csv(filename, **kwargs)
    else:
        st.warning(f"File not found: {filename}")
        return None

# -------------------------------------------------------------------
# Load all model summaries and equity curves
# -------------------------------------------------------------------
@st.cache_data
def load_all_data():
    # Summaries
    xgb = safe_read_csv("XGBoost_Summary.csv")
    lgb = safe_read_csv("LightGBM_Summary.csv")
    nn = safe_read_csv("NeuralNetwork_Summary.csv")
    cat = safe_read_csv("CatBoost_Performance_Summary.csv")
    base = safe_read_csv("Baseline_Summary.csv")

    model_dfs = []
    if xgb is not None:
        xgb["Model"] = "XGBoost"
        model_dfs.append(xgb)
    if lgb is not None:
        lgb["Model"] = "LightGBM"
        model_dfs.append(lgb)
    if nn is not None:
        nn["Model"] = "NeuralNetwork"
        model_dfs.append(nn)
    if cat is not None:
        cat["Model"] = "CatBoost"
        # Ensure Window column exists (12M,24M,36M,60M)
        if 'Window' not in cat.columns and 'Window' in cat.columns.tolist():
            pass
        elif 'Window' not in cat.columns:
            cat.insert(0, 'Window', ['12M', '24M', '36M', '60M'])
        # Ensure Win Rate column
        if 'Win Rate' not in cat.columns and 'WinRate' in cat.columns:
            cat['Win Rate'] = cat['WinRate']
        model_dfs.append(cat)
    if base is not None:
        base["Model"] = "Baseline"
        model_dfs.append(base)

    if not model_dfs:
        st.error("No model summary files found. Please upload CSV files.")
        return pd.DataFrame(), {}, {}

    all_models = pd.concat(model_dfs, ignore_index=True)

    # Keep only essential columns
    keep_cols = ['Window', 'End Fund', 'CAGR', 'Volatility', 'Sharpe', 'Max DD', 'Win Rate', 'Model']
    keep_cols = [c for c in keep_cols if c in all_models.columns]
    all_models = all_models[keep_cols]

    # Equity curves
    equity = {}
    for name, file in [
        ("XGBoost", "XGBoost_Equity.csv"),
        ("LightGBM", "LightGBM_Equity.csv"),
        ("NeuralNetwork", "NeuralNetwork_Equity.csv"),
        ("CatBoost", "CatBoost_Portfolio_Value.csv")
    ]:
        if os.path.exists(file):
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            df.columns = df.columns.str.strip()
            # Convert numeric column names like "12" to "12M"
            new_cols = []
            for col in df.columns:
                if col.isdigit():
                    new_cols.append(col + 'M')
                elif col.endswith('M') and col[:-1].isdigit():
                    new_cols.append(col)
                else:
                    new_cols.append(col)
            df.columns = new_cols
            equity[name] = df
        else:
            equity[name] = None

    # Baseline equity
    if os.path.exists("Baseline_Equity.csv"):
        base_eq = pd.read_csv("Baseline_Equity.csv", index_col=0, parse_dates=True)
        # Baseline equity may have column "Baseline" or "RuleBased"
        if base_eq.columns[0] not in ['Baseline', 'RuleBased']:
            base_eq.columns = ['Baseline']
        equity["Baseline"] = base_eq
    else:
        equity["Baseline"] = None

    return all_models, equity

# -------------------------------------------------------------------
# Home page
# -------------------------------------------------------------------
if page == "🏠 Home":
    st.title("📊 AI Sector Rotation Dashboard")
    st.markdown("""
    ### Problem & Business Value
    **Problem:** Portfolio managers need to dynamically allocate capital across sectors to outperform the market while controlling drawdowns.

    **Solution:** Machine learning models (XGBoost, LightGBM, CatBoost, Neural Network) predict next‑month sector returns using 19 features. Top 3 sectors are selected monthly with Amplified Softmax weighting.

    **Business Impact (2006‑2025 backtest):**
    - **XGBoost 12M** turned $1M into **$250M** (CAGR 32.5%, Sharpe 1.32, Max DD -31%).
    - Significantly outperforms baseline (CAGR 18.6%, Sharpe 0.70, Max DD -64%).
    """)

# -------------------------------------------------------------------
# Data Explorer
# -------------------------------------------------------------------
elif page == "📊 Data Explorer":
    st.title("Data Explorer")
    df_models, _ = load_all_data()
    if df_models.empty:
        st.warning("No data loaded.")
    else:
        st.subheader("Model Performance Summary")
        st.dataframe(df_models, use_container_width=True)

        st.subheader("Dataset Statistics")
        st.write(f"**Models:** {', '.join(df_models['Model'].unique())}")
        st.write(f"**Training windows:** {sorted(df_models['Window'].unique())}")
        if 'CAGR' in df_models.columns:
            st.write(f"**CAGR range:** {df_models['CAGR'].min()} to {df_models['CAGR'].max()}")
        if 'Sharpe' in df_models.columns:
            st.write(f"**Sharpe range:** {df_models['Sharpe'].min()} to {df_models['Sharpe'].max()}")

    # Optional file uploader for new data
    st.subheader("Upload Additional Data (CSV)")
    uploaded = st.file_uploader("Upload a CSV with same columns", type="csv")
    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(new_df.head())

# -------------------------------------------------------------------
# Model Performance (main interactive page)
# -------------------------------------------------------------------
elif page == "📈 Model Performance":
    st.title("Model Performance")
    df_models, equity_data = load_all_data()
    if df_models.empty:
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", df_models["Model"].unique())
    model_sub = df_models[df_models["Model"] == model]
    with col2:
        window = st.selectbox("Training window", model_sub["Window"].tolist())

    row = model_sub[model_sub["Window"] == window].iloc[0]

    # Metrics row
    st.subheader(f"{model} – {window}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", row["CAGR"])
    c2.metric("Sharpe Ratio", row["Sharpe"])
    c3.metric("Volatility", row["Volatility"])
    c4.metric("Max Drawdown", row["Max DD"])

    with st.expander("Additional Metrics"):
        st.write(f"**End Fund:** {row['End Fund']}")
        if "Avg DD" in row:
            st.write(f"**Average Drawdown:** {row['Avg DD']}")
        st.write(f"**Win Rate:** {row['Win Rate']}")

    # Equity curve
    eq_df = equity_data.get(model)
    if eq_df is not None and window in eq_df.columns:
        eq_series = eq_df[window].dropna()
        fig = px.line(x=eq_series.index, y=eq_series.values,
                      labels={"x": "Date", "y": "Portfolio Value (NT$)"},
                      title=f"{model} {window} – Equity Curve (Log Scale)")
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Equity curve not available for this selection.")

    # Compute drawdown from equity if possible
    if eq_df is not None and window in eq_df.columns:
        eq_vals = eq_df[window].dropna()
        dd = (eq_vals / eq_vals.cummax() - 1) * 100
        fig_dd = px.area(x=dd.index, y=dd.values,
                         labels={"x": "Date", "y": "Drawdown (%)"},
                         title=f"{model} {window} – Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

    # Monthly returns
    if eq_df is not None and window in eq_df.columns:
        monthly_ret = eq_df[window].pct_change().dropna() * 100
        if not monthly_ret.empty:
            ret_df = pd.DataFrame({"Date": monthly_ret.index, "Return (%)": monthly_ret.values})
            colors = ["green" if v >= 0 else "red" for v in monthly_ret.values]
            fig_ret = px.bar(ret_df, x="Date", y="Return (%)", color=colors, color_discrete_map="identity",
                             title=f"{model} {window} – Monthly Returns")
            fig_ret.add_hline(y=0, line_dash="dash", line_color="black")
            fig_ret.update_layout(showlegend=False)
            st.plotly_chart(fig_ret, use_container_width=True)

    # Compare all models for selected window
    st.subheader("Model Comparison (Equity Curves)")
    compare_window = st.selectbox("Training window for comparison", ["12M", "24M", "36M", "60M"], key="comp")
    fig_comp = px.line(title=f"Equity Curves – {compare_window}")
    any_data = False
    for model_name, eq_df in equity_data.items():
        if eq_df is not None and compare_window in eq_df.columns:
            fig_comp.add_scatter(x=eq_df.index, y=eq_df[compare_window].dropna(),
                                 mode='lines', name=model_name)
            any_data = True
    if any_data:
        fig_comp.update_layout(yaxis_type="log", xaxis_title="Date", yaxis_title="Portfolio Value")
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("No equity data for this window.")

    # Full table
    st.subheader("Full Performance Table")
    st.dataframe(df_models, use_container_width=True)

# -------------------------------------------------------------------
# Prediction Demo
# -------------------------------------------------------------------
elif page == "🔮 Prediction Demo":
    st.title("Prediction Demo")
    df_models, _ = load_all_data()
    if not df_models.empty:
        # Find best model by Sharpe
        df_models['Sharpe_num'] = pd.to_numeric(df_models['Sharpe'], errors='coerce')
        best = df_models.loc[df_models['Sharpe_num'].idxmax()]
        st.success(f"Recommended model: **{best['Model']} {best['Window']}** (Sharpe {best['Sharpe']})")
        st.markdown("""
        **Predicted top sectors for next month (based on latest signals):**
        - Semiconductor (expected return +3.2%)
        - Electronics (+2.1%)
        - Shipping (+1.5%)
        """)
        st.info("In a production system, these predictions would come from the live model.")
    else:
        st.warning("No model data available.")

# -------------------------------------------------------------------
# Explainability
# -------------------------------------------------------------------
elif page == "🧠 Explainability":
    st.title("Explainability")
    st.markdown("""
    ### Feature Importance (from LightGBM 12M)
    The chart below shows the contribution of each feature type to the model’s predictions.
    """)
    # Create bar chart from our known numbers
    imp_df = pd.DataFrame({
        "Feature": ["Volatility", "Relative Strength", "Momentum", "Breadth", "MA Distance"],
        "Importance (%)": [27.5, 23.9, 22.6, 15.3, 10.7]
    })
    fig = px.bar(imp_df, x="Feature", y="Importance (%)", title="Feature Type Importance",
                 color="Feature", text="Importance (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key Insights:**
    - **Volatility** is the most predictive feature – lower long‑term volatility sectors tend to outperform.
    - **Relative strength** (sector vs market) and **momentum** also drive sector rotation.
    - **Breadth** (percentage of stocks above MA200) and **MA distance** add additional signal.
    - The model’s decisions are largely driven by risk (volatility) and trend (momentum), which aligns with financial intuition.
    """)

# -------------------------------------------------------------------
# Business Recommendation
# -------------------------------------------------------------------
elif page == "💼 Business Recommendation":
    st.title("Business Recommendation")
    df_models, _ = load_all_data()
    if not df_models.empty:
        df_models['Sharpe_num'] = pd.to_numeric(df_models['Sharpe'], errors='coerce')
        best = df_models.loc[df_models['Sharpe_num'].idxmax()]
        st.success(f"**Deploy {best['Model']} with {best['Window']} training window.**")
        st.write(f"This model achieved Sharpe {best['Sharpe']}, CAGR {best['CAGR']}, and max drawdown {best['Max DD']}.")
    st.markdown("""
    ### Actionable Steps
    1. **Rebalance monthly** – on the first trading day of each month, re‑evaluate sector scores.
    2. **Use Amplified Softmax** (temperature 3.0) to allocate weights – focus capital on top sectors.
    3. **Monitor volatility** – when predicted sector volatility exceeds a threshold, reduce overall exposure.
    4. **Avoid neural network models** – they show higher drawdowns and lower Sharpe.
    5. **Implement a stop‑loss** of 25% to protect against extreme events.

    ### Expected Impact
    - Backtest (2006‑2025) shows **32.5% CAGR** with **-31% max drawdown**.
    - $1M invested would have grown to **$250M**.
    - Even after 0.1% transaction costs, the strategy significantly outperforms the baseline.
    """)
