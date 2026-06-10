import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")
st.title("📊 AI Sector Rotation – Model Comparison (2006‑2025)")
st.markdown("Compare XGBoost, LightGBM, Neural Network, CatBoost, and Baseline – all on the same period (trading starts 2006, data 2000‑2025).")

@st.cache_data
def load_data():
    # Summaries
    xgb = pd.read_csv("XGBoost_Summary.csv")
    lgb = pd.read_csv("LightGBM_Summary.csv")
    nn = pd.read_csv("NeuralNetwork_Summary.csv")
    cat = pd.read_csv("CatBoost_Performance_Summary.csv")
    base = pd.read_csv("Baseline_Summary.csv")

    # Add model column
    xgb["Model"] = "XGBoost"
    lgb["Model"] = "LightGBM"
    nn["Model"] = "NeuralNetwork"
    cat["Model"] = "CatBoost"
    base["Model"] = "Baseline"

    all_models = pd.concat([xgb, lgb, nn, cat, base], ignore_index=True)

    # Keep only the columns we want to display
    keep_cols = ['Window', 'End Fund', 'CAGR', 'Volatility', 'Sharpe', 'Max DD', 'Win Rate', 'Model']
    if 'Win Rate' not in all_models.columns and 'WinRate' in all_models.columns:
        all_models['Win Rate'] = all_models['WinRate']
    all_models = all_models[keep_cols]

    # Equity curves
    equity = {}
    # Standard models
    for name, file in [("XGBoost", "XGBoost_Equity.csv"),
                       ("LightGBM", "LightGBM_Equity.csv"),
                       ("NeuralNetwork", "NeuralNetwork_Equity.csv"),
                       ("CatBoost", "CatBoost_Portfolio_Value.csv")]:
        if os.path.exists(file):
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            df.columns = df.columns.str.strip()
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

    # Baseline equity (special handling)
    if os.path.exists("Baseline_Equity.csv"):
        df_base = pd.read_csv("Baseline_Equity.csv", index_col=0, parse_dates=True)
        window_name = base["Window"].iloc[0]  # "RuleBased"
        df_base.columns = [window_name]
        equity["Baseline"] = df_base
    else:
        equity["Baseline"] = None

    # Drawdowns (optional)
    drawdown = {}
    for name, file in [("XGBoost", "XGBoost_Drawdowns.csv"),
                       ("LightGBM", "LightGBM_Drawdowns.csv"),
                       ("NeuralNetwork", "NeuralNetwork_Drawdowns.csv"),
                       ("CatBoost", None)]:
        if file and os.path.exists(file):
            dd = pd.read_csv(file, index_col=0, parse_dates=True)
            dd.columns = dd.columns.str.strip()
            new_cols = []
            for col in dd.columns:
                if col.isdigit():
                    new_cols.append(col + 'M')
                elif col.endswith('M') and col[:-1].isdigit():
                    new_cols.append(col)
                else:
                    new_cols.append(col)
            dd.columns = new_cols
            drawdown[name] = dd
        else:
            drawdown[name] = None

    return all_models, equity, drawdown

df_models, equity_data, drawdown_data = load_data()

# Sidebar
st.sidebar.header("Model Selection")
model_list = df_models["Model"].unique().tolist()
selected_model = st.sidebar.selectbox("Choose a model:", model_list, key="model_selector")

model_subset = df_models[df_models["Model"] == selected_model]
window_options = model_subset["Window"].tolist()
selected_window = st.sidebar.selectbox("Training window (months):", window_options, key="window_selector")

row = model_subset[model_subset["Window"] == selected_window].iloc[0]

st.subheader(f"📈 {selected_model} – {selected_window} Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("CAGR", row["CAGR"])
col2.metric("Sharpe Ratio", row["Sharpe"])
col3.metric("Volatility", row["Volatility"])
col4.metric("Max Drawdown", row["Max DD"])

with st.expander("More metrics"):
    st.write(f"**End Fund:** {row['End Fund']}")
    if "Avg DD" in row:
        st.write(f"**Average Drawdown:** {row['Avg DD']}")
    st.write(f"**Win Rate:** {row['Win Rate']}")

# Equity curve
st.subheader(f"📈 Equity Curve – {selected_model} {selected_window}")
eq_df = equity_data.get(selected_model)
if eq_df is not None and selected_window in eq_df.columns:
    eq_series = eq_df[selected_window].dropna()
    fig_eq = px.line(x=eq_series.index, y=eq_series.values,
                     labels={"x": "Date", "y": "Portfolio Value (NT$)"},
                     title=f"{selected_model} {selected_window} – Portfolio Growth")
    fig_eq.update_yaxes(type="log")
    st.plotly_chart(fig_eq, use_container_width=True)
else:
    st.info("Equity curve data not available for this model/window.")

# Drawdown
st.subheader(f"📉 Drawdown – {selected_model} {selected_window}")
dd_df = drawdown_data.get(selected_model)
if dd_df is not None and selected_window in dd_df.columns:
    dd_series = dd_df[selected_window].dropna() * 100
    fig_dd = px.area(x=dd_series.index, y=dd_series.values,
                     labels={"x": "Date", "y": "Drawdown (%)"},
                     title=f"{selected_model} {selected_window} – Drawdown")
    st.plotly_chart(fig_dd, use_container_width=True)
else:
    # Compute from equity if possible
    if eq_df is not None and selected_window in eq_df.columns:
        eq_vals = eq_df[selected_window].dropna()
        cummax = eq_vals.cummax()
        dd_computed = (eq_vals - cummax) / cummax * 100
        fig_dd = px.area(x=dd_computed.index, y=dd_computed.values,
                         labels={"x": "Date", "y": "Drawdown (%)"},
                         title=f"{selected_model} {selected_window} – Drawdown (computed)")
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.warning("No drawdown or equity data for this model/window.")

# Monthly returns bar chart
st.subheader(f"📆 Monthly Returns – {selected_model} {selected_window}")
if eq_df is not None and selected_window in eq_df.columns:
    eq_vals = eq_df[selected_window].dropna()
    monthly_ret = eq_vals.pct_change().dropna() * 100
    if not monthly_ret.empty:
        fig_ret = px.bar(x=monthly_ret.index, y=monthly_ret.values,
                         labels={"x": "Date", "y": "Return (%)"},
                         title=f"{selected_model} {selected_window} – Monthly Returns")
        fig_ret.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_ret, use_container_width=True)
    else:
        st.info("Not enough data to compute monthly returns.")
else:
    st.info("Equity data missing – cannot compute monthly returns.")

# ============================================================
# Comparison graph across models (same training window)
# INCLUDES BASELINE AS A FIXED REFERENCE
# ============================================================
st.subheader("📊 Model Comparison per Training Window (with Baseline)")
window_for_compare = st.selectbox("Select training window to compare:", 
                                   ["12M", "24M", "36M", "60M"], 
                                   key="comp_window")

fig_comp = px.line(title=f"Equity Curves – {window_for_compare} Training Window")
any_data = False

# Add standard models for the selected window
for model_name, eq_df in equity_data.items():
    if model_name == "Baseline":
        continue  # handle baseline separately below
    if eq_df is not None and window_for_compare in eq_df.columns:
        eq_series = eq_df[window_for_compare].dropna()
        fig_comp.add_scatter(x=eq_series.index, y=eq_series.values,
                             mode='lines', name=model_name)
        any_data = True

# Add Baseline (fixed strategy, independent of training window)
baseline_eq = equity_data.get("Baseline")
if baseline_eq is not None and not baseline_eq.empty:
    # Baseline has a single column (e.g., "RuleBased")
    baseline_col = baseline_eq.columns[0]
    eq_base = baseline_eq[baseline_col].dropna()
    fig_comp.add_scatter(x=eq_base.index, y=eq_base.values,
                         mode='lines', name="Baseline",
                         line=dict(dash="dash", color="purple"))
    any_data = True

if not any_data:
    st.info("No equity data available for the selected window.")
else:
    fig_comp.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value (NT$)",
                           yaxis_type="log")
    st.plotly_chart(fig_comp, use_container_width=True)

# Full comparison table
st.subheader("📋 All Models – Performance Table")
st.dataframe(df_models, use_container_width=True)

st.caption("Data from XGBoost, LightGBM, NeuralNetwork, CatBoost, and Baseline (2020‑2026).")
