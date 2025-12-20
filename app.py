import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import time
from github import Github  # PyGithub
from streamlit_star_rating import st_star_rating

# -----------------------------
# Page Config & Dark Theme
# -----------------------------
st.set_page_config(
    page_title="Return Risk Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Supabase-inspired dark glassmorphism
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0d1117, #161b22);
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        text-shadow: 0 2px 8px rgba(88, 166, 255, 0.3);
    }
    .glass-card {
        background: rgba(13, 17, 23, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        border: 1px solid rgba(88, 166, 255, 0.2);
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        margin: 20px 0;
    }
    .stSelectbox > div > div, .stNumberInput > div, .stTextInput > div, .stTextArea > div {
        background: rgba(22, 27, 34, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(88, 166, 255, 0.3);
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f6feb, #58a6ff);
        border: none;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.4);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(31, 111, 235, 0.6);
    }
    .stMetric {
        background: rgba(13, 17, 23, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(88, 166, 255, 0.2);
    }
    div[data-testid="stDataFrame"] {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 16px;
    }
    .footer {
        text-align: center;
        color: #8b949e;
        font-size: 14px;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model & Preprocessors
# -----------------------------
@st.cache_resource
def load_artifacts():
    base_path = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base_path, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    le_category = joblib.load(os.path.join(base_path, "le_category.pkl"))
    le_country = joblib.load(os.path.join(base_path, "le_country.pkl"))
    return model, scaler, le_category, le_country

model, scaler, le_category, le_country = load_artifacts()

# -----------------------------
# Title
# -----------------------------
st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 40px;">
        <h1 style="font-size: 48px; margin: 0;">🛒 E-Commerce Return Risk Predictor</h1>
        <h3 style="margin: 10px 0 0; color: #8b949e;">Predict if a customer is likely to return their order</h3>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# Model Metrics
# -----------------------------
MODEL_METRICS = {
    "Accuracy": "0.9245",
    "Precision (Return Class)": "0.68",
    "Recall (Return Class)": "0.62",
    "AUC Score": "0.91"
}

# -----------------------------
# Prediction Section
# -----------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.header("📦 Enter Product & Order Details")

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Product Category", options=le_category.classes_, index=0)
    unit_price = st.number_input("Unit Price (£)", min_value=0.0, max_value=500.0, value=5.0, step=0.5)
    total_price = st.number_input("Approximate Total Order Value (£)", min_value=0.0, max_value=1000.0, value=50.0, step=5.0)
    customer_return_rate = st.slider("Customer's Past Return Rate (%)", 0.0, 20.0, 1.7,
                                     help="Historical return behavior of this customer")

with col2:
    country = st.selectbox("Customer Country", options=le_country.classes_, index=0)
    month = st.selectbox("Month of Purchase", options=list(range(1, 13)),
                         format_func=lambda x: datetime(2025, x, 1).strftime('%B'))
    is_holiday = st.checkbox("Holiday Season (Nov-Dec)", value=False)
    is_weekend = st.checkbox("Weekend Order", value=False)

if st.button("🔍 Predict Return Chance", type="primary", use_container_width=True):
    with st.spinner("Analyzing order details..."):
        time.sleep(1.5)
        input_data = pd.DataFrame({
            'UnitPrice': [unit_price],
            'TotalPrice': [total_price],
            'Month': [month],
            'Hour': [12],
            'IsWeekend': [int(is_weekend)],
            'IsHolidaySeason': [int(is_holiday)],
            'CustomerReturnRate': [customer_return_rate / 100],
            'Category': [le_category.transform([category])[0]],
            'Country': [le_country.transform([country])[0]]
        })
        scaled_input = scaler.transform(input_data)
        return_probability = model.predict_proba(scaled_input)[0][1]
        prediction = "Return Likely" if return_probability > 0.05 else "Return Unlikely"

    st.markdown("<br>", unsafe_allow_html=True)
    risk_color = "#FF5252" if return_probability > 0.05 else "#4CAF50"
    highlight_color = "#FFEB3B" if return_probability > 0.05 else "#FFFFFF"

    st.markdown(f"""
        <div style='text-align: center; padding: 40px; border-radius: 20px;
                    background: linear-gradient(135deg, {risk_color}aa, #0d1117);
                    backdrop-filter: blur(12px); box-shadow: 0 12px 40px rgba(0,0,0,0.6);
                    border: 1px solid {risk_color}44;'>
            <h1 style='margin:0; font-size: 80px; color: white;'>{return_probability:.1%}</h1>
            <h3 style='margin:10px 0; color: #c9d1d9;'>Probability of Return</h3>
            <h2 style='margin:10px 0; color: {highlight_color};'>{prediction}</h2>
        </div>
    """, unsafe_allow_html=True)

    if return_probability <= 0.05:
        st.success("🎉 Low Risk! This order is likely to be kept.")
        st.balloons()
    else:
        st.error("⚠️ High Risk! This order may be returned.")
        st.snow()

    if return_probability > 0.05:
        st.warning("**Recommendation**: Enhance product photos, detailed sizing charts, or clear material description.")
    else:
        st.success("**Great choice!** High customer satisfaction expected.")

st.markdown("</div>", unsafe_allow_html=True)

# Metrics Card
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>🔬 Model Performance Metrics (Test Set)</h4>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", MODEL_METRICS["Accuracy"])
c2.metric("Precision (Returns)", MODEL_METRICS["Precision (Return Class)"])
c3.metric("Recall (Returns)", MODEL_METRICS["Recall (Return Class)"])
c4.metric("AUC Score", MODEL_METRICS["AUC Score"])
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Feedback Section
# -----------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>📝 Give Your Feedback</h2>", unsafe_allow_html=True)
st.write("Your feedback helps improve the model and user experience!")

with st.form(key="feedback_form", clear_on_submit=True):
    name = st.text_input("Your Name *", placeholder="e.g., Umar Farooq")

    colf1, colf2 = st.columns(2)
    with colf1:
        st.write("**Usability of the App**")
        usability_rating = st_star_rating("Rate usability", maxValue=5, defaultValue=4, size=30)
    with colf2:
        st.write("**Accuracy & Relevance of Prediction**")
        accuracy_relevance = st_star_rating("Rate accuracy", maxValue=5, defaultValue=4, size=30)

    suggestions = st.text_area("Suggestions for Improvement",
                               placeholder="e.g., Add product images, allow bulk upload, show explanations, etc.")

    submitted = st.form_submit_button("🚀 Submit Feedback", type="primary", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("⚠️ Please enter your name.")
        else:
            feedback_entry = {
                "Name": name,
                "Usability_Rating": usability_rating,
                "Accuracy_Relevance_Rating": accuracy_relevance,
                "Suggestions": suggestions,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            try:
                g = Github(st.secrets["GITHUB_TOKEN"])
                repo = g.get_repo(f"{st.secrets['GITHUB_USERNAME']}/{st.secrets['REPO_NAME']}")
                file_path = "feedback.csv"
                branch = st.secrets.get("BRANCH", "main")

                try:
                    contents = repo.get_contents(file_path, ref=branch)
                    raw_url = f"https://raw.githubusercontent.com/{st.secrets['GITHUB_USERNAME']}/{st.secrets['REPO_NAME']}/{branch}/{file_path}"
                    df_existing = pd.read_csv(raw_url)
                    df_updated = pd.concat([df_existing, pd.DataFrame([feedback_entry])], ignore_index=True)
                except:
                    df_updated = pd.DataFrame([feedback_entry])

                csv_content = df_updated.to_csv(index=False)

                if 'contents' in locals():
                    repo.update_file(path=file_path, message=f"Feedback from {name}", content=csv_content, sha=contents.sha, branch=branch)
                else:
                    repo.create_file(path=file_path, message=f"Initial feedback from {name}", content=csv_content, branch=branch)

                st.success(f"✅ Thank you, **{name}**! Your feedback has been saved.")
                st.balloons()
            except Exception as e:
                st.error(f"Error saving feedback: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Feedback Table with Auto-Refresh
# -----------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>📊 All Submitted Feedbacks</h2>", unsafe_allow_html=True)

# Session state for refresh control
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# Auto-refresh every 15 seconds
time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
if time_since_refresh > 15:
    st.rerun()

try:
    branch = st.secrets.get("BRANCH", "main")
    feedback_url = f"https://raw.githubusercontent.com/{st.secrets['GITHUB_USERNAME']}/{st.secrets['REPO_NAME']}/{branch}/feedback.csv"
    df_feedback = pd.read_csv(feedback_url)
    
    st.dataframe(df_feedback, use_container_width=True, hide_index=True)
    
    csv_data = df_feedback.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Feedbacks as CSV",
        data=csv_data,
        file_name=f"feedbacks_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Update last refresh time only on successful load
    st.session_state.last_refresh = datetime.now()

except:
    st.info("No feedback submitted yet or loading...")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
    <div class="footer glass-card">
        <strong>Data Science Assignment 4</strong> | BSCS-F22 | Instructor: Ghulam Ali<br>
        Model: XGBoost Classifier | Dataset: Online Retail (UCI/Kaggle)<br>
        Deployment: Streamlit Cloud | Version 1.0 — December 2025
    </div>
    <p style="text-align: center; color: #58a6ff; margin-top: 20px;">
        ⭐ Feedback saved to GitHub • Real-time updates • Glassmorphic Dark Theme
    </p>
""", unsafe_allow_html=True)
