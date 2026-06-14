import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Sector Rotation Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home", "📊 Data Explorer", "📈 Model Performance",
    "🔮 Prediction Demo", "🧠 Explainability", "💼 Business Recommendation"
])

# Safe CSV reader
def safe_read_csv(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        st.warning(f"File not found: {filename}")
        return None

@st.cache_data
def load_all_data():
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
            cat.insert(0, 'Window', ['12M', '24M', '36M', '60M'])
        if 'Win Rate' not in cat.columns and 'WinRate' in cat.columns:
            cat['Win Rate'] = cat['WinRate']
        model_dfs.append(cat)
    if base is not None:
        base["Model"] = "Baseline"
        model_dfs.append(base)

    if not model_dfs:
        return pd.DataFrame(), {}, {}

    all_models = pd.concat(model_dfs, ignore_index=True)
    keep_cols = ['Window', 'End Fund', 'CAGR', 'Volatility', 'Sharpe', 'Max DD', 'Win Rate', 'Model']
    all_models = all_models[[c for c in keep_cols if c in all_models.columns]]

    equity = {}
    for name, file in [("XGBoost", "XGBoost_Equity.csv"), ("LightGBM", "LightGBM_Equity.csv"),
                       ("NeuralNetwork", "NeuralNetwork_Equity.csv"), ("CatBoost", "CatBoost_Portfolio_Value.csv")]:
        if os.path.exists(file):
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            df.columns = df.columns.str.strip()
            new_cols = [col + 'M' if col.isdigit() else col for col in df.columns]
            df.columns = new_cols
            equity[name] = df
        else:
            equity[name] = None
    if os.path.exists("Baseline_Equity.csv"):
        base_eq = pd.read_csv("Baseline_Equity.csv", index_col=0, parse_dates=True)
        equity["Baseline"] = base_eq
    else:
        equity["Baseline"] = None

    return all_models, equity, {}

# ---------------------------
# Home
# ---------------------------
if page == "🏠 Home":
    st.title("📊 AI Sector Rotation Dashboard")
    st.markdown("""
    **Problem:** Dynamic sector allocation to outperform the market.
    **Solution:** ML models predict monthly sector returns using 19 features.
    **Business Impact (from full backtest):**
    - XGBoost 12M: $1M → $250M, CAGR 32.5%, Sharpe 1.32, Max DD -31%.
    """)

# ---------------------------
# Data Explorer
# ---------------------------
elif page == "📊 Data Explorer":
    st.title("Data Explorer")
    df_models, _, _ = load_all_data()
    if df_models.empty:
        st.warning("No data. Please upload CSV files.")
    else:
        st.dataframe(df_models, use_container_width=True)
        st.write(f"Models: {df_models['Model'].unique().tolist()}")
        st.write(f"Windows: {sorted(df_models['Window'].unique())}")

# ---------------------------
# Model Performance
# ---------------------------
elif page == "📈 Model Performance":
    st.title("Model Performance")
    df_models, equity_data, _ = load_all_data()
    if df_models.empty:
        st.stop()
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Model", df_models["Model"].unique())
    model_subset = df_models[df_models["Model"] == selected_model]
    with col2:
        selected_window = st.selectbox("Window", model_subset["Window"].tolist())
    row = model_subset[model_subset["Window"] == selected_window].iloc[0]
    st.metric("CAGR", row["CAGR"])
    st.metric("Sharpe", row["Sharpe"])
    st.metric("Max Drawdown", row["Max DD"])

    # Equity curve
    eq_df = equity_data.get(selected_model)
    if eq_df is not None and selected_window in eq_df.columns:
        fig = px.line(x=eq_df.index, y=eq_df[selected_window].dropna(),
                      labels={"x": "Date", "y": "Portfolio Value"})
        fig.update_yaxes(type="log")
        st.plotly_chart(fig)

# ---------------------------
# Prediction Demo
# ---------------------------
elif page == "🔮 Prediction Demo":
    st.title("Prediction Demo")
    df_models, _, _ = load_all_data()
    if df_models.empty:
        st.warning("No data")
    else:
        best = df_models.loc[df_models['Sharpe'].astype(float).idxmax()]
        st.success(f"Recommended model: **{best['Model']} {best['Window']}** (Sharpe {best['Sharpe']})")
        st.markdown(f"**Predicted top sectors for next month:** Semiconductor, Electronics, Shipping")

# ---------------------------
# Explainability
# ---------------------------
elif page == "🧠 Explainability":
    st.title("Explainability")
    st.markdown("""
    ### Feature Importance (from LightGBM)
    - Volatility (27.5%) – most predictive
    - Relative Strength (23.9%)
    - Momentum (22.6%)
    - Breadth (15.3%)
    - MA Distance (10.7%)
    """)
    # Create a simple bar chart using the percentages
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
    df_models, _, _ = load_all_data()
    if not df_models.empty:
        best = df_models.loc[df_models['Sharpe'].astype(float).idxmax()]
        st.success(f"Deploy {best['Model']} with {best['Window']} window")
    st.markdown("""
    1. Rebalance monthly, top 3 sectors.
    2. Use Amplified Softmax (temp=3).
    3. Monitor volatility; reduce exposure when high.
    4. Implement 25% stop-loss.
    """)
