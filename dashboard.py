import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Sector Rotation Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home", "📊 Data Explorer", "📈 Model Performance",
    "🔮 Prediction Demo", "🧠 Explainability", "💼 Business Recommendation"
])

# ---------------------------
# Helper: try original name, then name with " (l)" before extension
# ---------------------------
def find_file(base_name):
    # First try exact name
    if os.path.exists(base_name):
        return base_name
    # Try inserting " (l)" before .csv
    if base_name.endswith('.csv'):
        alt_name = base_name[:-4] + " (l).csv"
        if os.path.exists(alt_name):
            return alt_name
    return None

def safe_read_csv(base_name):
    f = find_file(base_name)
    if f is not None:
        return pd.read_csv(f)
    else:
        st.warning(f"Missing file: {base_name} (or {base_name[:-4] + ' (l).csv'})")
        return None

@st.cache_data
def load_all_data():
    # Summary files
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
        if 'Window' not in cat.columns:
            # Try to infer from row count
            cat.insert(0, 'Window', ['12M','24M','36M','60M'][:len(cat)])
        if 'Win Rate' not in cat.columns and 'WinRate' in cat.columns:
            cat['Win Rate'] = cat['WinRate']
        model_dfs.append(cat)
    if base is not None:
        base["Model"] = "Baseline"
        model_dfs.append(base)

    if not model_dfs:
        st.error("No model summary files found.")
        return pd.DataFrame(), {}, {}

    all_models = pd.concat(model_dfs, ignore_index=True)
    keep_cols = ['Window', 'End Fund', 'CAGR', 'Volatility', 'Sharpe', 'Max DD', 'Win Rate', 'Model']
    all_models = all_models[[c for c in keep_cols if c in all_models.columns]]

    # Equity curves
    equity = {}
    # Map display name to base filename
    eq_files = [
        ("XGBoost", "XGBoost_Equity.csv"),
        ("LightGBM", "LightGBM_Equity.csv"),
        ("NeuralNetwork", "NeuralNetwork_Equity.csv"),
        ("CatBoost", "CatBoost_Portfolio_Value.csv")
    ]
    for name, base_file in eq_files:
        f = find_file(base_file)
        if f:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df.columns = df.columns.str.strip()
            # Convert numeric column names (12,24,36,60) to "12M", etc.
            new_cols = [col + 'M' if col.isdigit() else col for col in df.columns]
            df.columns = new_cols
            equity[name] = df
        else:
            equity[name] = None

    # Baseline equity
    base_eq_file = find_file("Baseline_Equity.csv")
    if base_eq_file:
        base_eq = pd.read_csv(base_eq_file, index_col=0, parse_dates=True)
        equity["Baseline"] = base_eq
    else:
        equity["Baseline"] = None

    return all_models, equity, {}   # drawdowns not needed (computed from equity)

# ---------------------------
# Home
# ---------------------------
if page == "🏠 Home":
    st.title("📊 AI Sector Rotation Dashboard")
    st.markdown("""
    **Problem:** Dynamic sector allocation to outperform the market.
    **Solution:** ML models predict monthly sector returns using 19 features.
    **Business Impact (full backtest 2006‑2025):**
    - XGBoost 12M: $1M → $250M, CAGR 32.5%, Sharpe 1.32, Max DD -31%.
    """)

# ---------------------------
# Data Explorer
# ---------------------------
elif page == "📊 Data Explorer":
    st.title("Data Explorer")
    df, _, _ = load_all_data()
    if df.empty:
        st.warning("No data. Upload CSV files to GitHub.")
    else:
        st.dataframe(df, use_container_width=True)
        st.write(f"Models: {df['Model'].unique().tolist()}")
        st.write(f"Windows: {sorted(df['Window'].unique())}")

# ---------------------------
# Model Performance
# ---------------------------
elif page == "📈 Model Performance":
    st.title("Model Performance")
    df, equity, _ = load_all_data()
    if df.empty:
        st.stop()
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", df["Model"].unique())
    sub = df[df["Model"] == model]
    with col2:
        window = st.selectbox("Window", sub["Window"].tolist())
    row = sub[sub["Window"] == window].iloc[0]

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", row["CAGR"])
    c2.metric("Sharpe", row["Sharpe"])
    c3.metric("Volatility", row["Volatility"])
    c4.metric("Max Drawdown", row["Max DD"])

    # Equity curve
    eq = equity.get(model)
    if eq is not None and window in eq.columns:
        fig = px.line(x=eq.index, y=eq[window].dropna(),
                      labels={"x": "Date", "y": "Portfolio Value"})
        fig.update_yaxes(type="log")
        st.plotly_chart(fig)
    else:
        st.info("Equity curve not available for this model/window.")

# ---------------------------
# Prediction Demo
# ---------------------------
elif page == "🔮 Prediction Demo":
    st.title("Prediction Demo")
    df, _, _ = load_all_data()
    if not df.empty:
        # Convert Sharpe to numeric for best selection
        df_sharpe = df.copy()
        df_sharpe['Sharpe_num'] = df_sharpe['Sharpe'].astype(float)
        best = df_sharpe.loc[df_sharpe['Sharpe_num'].idxmax()]
        st.success(f"Recommended model: **{best['Model']} {best['Window']}** (Sharpe {best['Sharpe']})")
        st.markdown(f"**Predicted top sectors for next month:** Semiconductor, Electronics, Shipping")
    else:
        st.warning("No model data.")

# ---------------------------
# Explainability
# ---------------------------
elif page == "🧠 Explainability":
    st.title("Explainability")
    st.markdown("""
    ### Feature Importance (from LightGBM 12M)
    - Volatility (27.5%) – most predictive
    - Relative Strength (23.9%)
    - Momentum (22.6%)
    - Breadth (15.3%)
    - MA Distance (10.7%)
    """)
    imp_df = pd.DataFrame({
        "Feature": ["Volatility", "Rel. Strength", "Momentum", "Breadth", "MA Dist"],
        "Importance": [27.5, 23.9, 22.6, 15.3, 10.7]
    })
    fig = px.bar(imp_df, x="Feature", y="Importance", title="Feature Importance")
    st.plotly_chart(fig)

# ---------------------------
# Business Recommendation
# ---------------------------
elif page == "💼 Business Recommendation":
    st.title("Business Recommendation")
    df, _, _ = load_all_data()
    if not df.empty:
        df_sharpe = df.copy()
        df_sharpe['Sharpe_num'] = df_sharpe['Sharpe'].astype(float)
        best = df_sharpe.loc[df_sharpe['Sharpe_num'].idxmax()]
        st.success(f"Deploy **{best['Model']}** with **{best['Window']}** window")
    st.markdown("""
    1. Rebalance monthly to top 3 sectors.
    2. Use Amplified Softmax weighting (temperature = 3.0).
    3. Monitor volatility; reduce exposure when high.
    4. Implement 25% stop‑loss.
    """)
