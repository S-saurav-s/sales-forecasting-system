import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📈 Sales Forecasting Dashboard")
st.markdown("Forecasting beverage sales using SARIMA, Prophet, XGBoost and LSTM")

# =========================================================
# LOAD STATES
# =========================================================

@st.cache_data
def load_states():
    response = requests.get(f"{API_URL}/states")
    return response.json()

states_data = load_states()

states = [s["state"] for s in states_data]

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Forecast Settings")

selected_state = st.sidebar.selectbox(
    "Select State",
    states
)

forecast_weeks = st.sidebar.slider(
    "Forecast Horizon (Weeks)",
    min_value=1,
    max_value=12,
    value=8
)

# =========================================================
# GET FORECAST
# =========================================================

payload = {
    "states": [selected_state],
    "horizon_weeks": forecast_weeks
}

response = requests.post(
    f"{API_URL}/forecast",
    json=payload
)

forecast_data = response.json()[0]

# =========================================================
# METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Best Model",
    forecast_data["best_model"].upper()
)

col2.metric(
    "MAPE",
    f'{forecast_data["model_metrics"]["mape"]:.2f}%'
)

col3.metric(
    "RMSE",
    f'{forecast_data["model_metrics"]["rmse"]:,.0f}'
)

# =========================================================
# FORECAST TABLE
# =========================================================

st.subheader(f"Forecast for {selected_state}")

forecast_df = pd.DataFrame(forecast_data["forecast"])

st.dataframe(
    forecast_df,
    use_container_width=True
)

# =========================================================
# CHART
# =========================================================

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    forecast_df["date"],
    forecast_df["predicted"],
    marker="o"
)

ax.fill_between(
    forecast_df["date"],
    forecast_df["lower_95"],
    forecast_df["upper_95"],
    alpha=0.3
)

ax.set_title(f"{selected_state} Sales Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Sales")

plt.xticks(rotation=45)

st.pyplot(fig)

# =========================================================
# STATE MODEL SUMMARY
# =========================================================

st.subheader("All States Performance")

summary_df = pd.DataFrame(states_data)

summary_df = summary_df.sort_values("mape")

st.dataframe(
    summary_df,
    width="stretch"
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown("Built with FastAPI + Streamlit")