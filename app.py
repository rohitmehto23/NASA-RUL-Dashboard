import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="NASA RUL Dashboard")

st.title("🚀 AI Powered Predictive Maintenance")
st.markdown("### NASA Turbofan Engine RUL Prediction - Model Comparison")

# Load data
@st.cache_data
def load_results():
    return pd.read_csv('results.csv')

Results = load_results()

# Metrics
col1, col2, col3 = st.columns(3)
best = Results.loc[Results['R2-Test'].idxmax()]
col1.metric("🏆 Best Model", best['Model'])
col2.metric("✅ Best R²", f"{best['R2-Test']:.3f}")
col3.metric("📉 Lowest RMSE", f"{best['RMSE-Test']:.3f}")

# Results table (SIMPLE - no styling error)
st.subheader("📊 Model Performance")
st.dataframe(Results)

# Charts
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.bar(Results, x='Model', y='R2-Test', 
                          title="R²-Test Comparison"), use_container_width=True)
with col2:
    st.plotly_chart(px.bar(Results, x='Model', y='RMSE-Test', 
                          title="RMSE-Test Comparison"), use_container_width=True)

st.markdown("---")
st.caption("NASA CMAPSS Dataset - Powered by Streamlit")

