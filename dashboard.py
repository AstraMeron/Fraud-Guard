import streamlit as st
import requests
import pandas as pd
import plotly.express as px # Great for interactive charts

# Page Config
st.set_page_config(page_title="Fraud-Guard Admin", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Fraud-Guard: Interactive Fraud Analytics")

# --- 1. BUSINESS IMPACT METRICS ---
# These simulate business value for your Week 12 report
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Total Transactions", "151,112", "+12%")
col_m2.metric("Detected Fraud", "14,151", "9.3%", delta_color="inverse")
col_m3.metric("Fraud Cases Prevented", "8,402")
col_m4.metric("Estimated Savings", "$420,100", "+$15k")

st.divider()

# --- 2. INTERACTIVE PREDICTION SECTION ---
st.header("🔍 Real-Time Fraud Probe")
col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("Transaction Details")
    with st.form("prediction_form"):
        p_val = st.number_input("Purchase Value ($)", min_value=1, value=50)
        u_age = st.slider("User Age", 18, 90, 30)
        u_browser = st.selectbox("Browser", options=[0, 1, 2, 3], format_func=lambda x: ["Chrome", "Firefox", "Safari", "Edge"][x])
        u_source = st.selectbox("Source", options=[0, 1, 2], format_func=lambda x: ["Ads", "SEO", "Direct"][x])
        u_sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female", horizontal=True)
        u_time = st.number_input("Seconds since Signup", value=5000)
        
        submit = st.form_submit_button("Run Fraud Analysis")

with col_result:
    st.subheader("Model Decision")
    if submit:
        with st.spinner("Analyzing patterns..."):
            payload = {
                "purchase_value": p_val,
                "age": u_age,
                "browser": u_browser,
                "sex": u_sex,
                "source": u_source,
                "time_diff": u_time,
                "user_transaction_count": 1
            }
            try:
                res = requests.post("http://127.0.0.1:5000/predict", json=payload)
                data = res.json()
                
                if data['prediction'] == 1:
                    st.error(f"### 🚩 HIGH RISK: {data['class_label']}")
                    st.progress(data['fraud_probability'])
                    st.write(f"Confidence: **{data['fraud_probability']*100:.2f}%**")
                    st.warning("Recommendation: Flag for manual review / Block transaction.")
                else:
                    st.success(f"### ✅ LOW RISK: {data['class_label']}")
                    st.progress(data['fraud_probability'])
                    st.write(f"Fraud Probability: **{data['fraud_probability']*100:.2f}%**")
                    st.info("Recommendation: Proceed with transaction.")
            except Exception as e:
                st.error("Connection Error: Is the Flask server (serve_model.py) running?")

st.divider()

# --- 3. DATA EXPLORATION & VISUALIZATIONS ---
st.header("📊 Global Fraud Insights")
tab1, tab2 = st.tabs(["Explainability (SHAP)", "Geography & Trends"])

with tab1:
    st.image("reports/figures/shap_summary_plot.png", use_container_width=True)
    st.write("**Insight:** Features on the right (red/pink) push the model toward a Fraud prediction.")

with tab2:
    # Creating a dummy dataframe to show how interactivity works
    # In your real case, you'd load your 'Fraud_Data.csv' here
    chart_data = pd.DataFrame({
        'Country': ['USA', 'China', 'Japan', 'Ethiopia', 'UK'],
        'Fraud_Count': [450, 320, 150, 80, 210]
    })
    fig = px.bar(chart_data, x='Country', y='Fraud_Count', title="Top Countries by Fraud Volume", color='Fraud_Count')
    st.plotly_chart(fig, use_container_width=True)

st.caption("Fraud-Guard v1.0 | Data Updated: 2026-02-19")