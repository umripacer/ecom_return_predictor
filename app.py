import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import time
from github import Github  # PyGithub
from streamlit_star_rating import st_star_rating  # NEW: Animated star rating component

# -----------------------------
# Page Config & Dark Theme Setup
# -----------------------------
st.set_page_config(
    page_title="Return Risk Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Force dark theme + Supabase-inspired color palette (dark emerald green + glassmorphism)
st.markdown("""
    <style>
        /* Main background & Supabase-like dark theme */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
        }
        
        /* Glassmorphism cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.65);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 20px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
            margin-bottom: 2rem;
        }
        
        /* Header styling */
        h1, h2, h3, h4 {
            color: #34d399 !important; /* Supabase emerald green */
            text-align: center;
        }
        
        /* Metrics & result card */
        .stMetric {
            background: rgba(52, 211, 153, 0.15);
            border-radius: 12px;
            padding: 10px;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }
        
        /* Dataframe glass effect + fade edges */
        .stDataFrame {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        
        /* Fade top/bottom of table for 3D depth */
        div[data-testid="stVerticalBlock"] > div.element-container > div.stDataFrame {
            mask-image: linear-gradient(to bottom, transparent 0%, black 10%, black 90%, transparent 100%);
            -webkit-mask-image: linear-gradient(to bottom, transparent 0%, black 10%, black 90%, transparent 100%);
        }
        
        /* Buttons - emerald green */
        .stButton > button {
            background: #34d399;
            color: #0f172a;
            border-radius: 12px;
            font-weight: bold;
            border: none;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            background: #10b981;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(52, 211, 153, 0.3);
        }
        
        hr {
            border-color: #334155;
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
# Title with 3D glass effect
# -----------------------------
st.markdown("""
    <div class="glass-card">
        <h1 style='text-align: center; color: #34d399; text-shadow: 0 4px 15px rgba(52,211,153,0.4);'>
            🛒 E-Commerce Product Return Risk Predictor
        </h1>
        <h3 style='text-align: center; color: #94a3b8;'>
            Predict if a customer is likely to return their order
        </h3>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# Model Performance Metrics
# -----------------------------
MODEL_METRICS = {
    "Accuracy": "0.9245",
    "Precision (Return Class)": "0.68",
    "Recall (Return Class)": "0.62",
    "AUC Score": "0.91"
}

# -----------------------------
# Main Prediction Section (Glass Card)
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
    with st.spinner("Analyzing order details and predicting risk..."):
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
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(f"""
            <div style='text-align: center; padding: 40px; border-radius: 24px;
                        background: linear-gradient(135deg, {'#f87171' if return_probability > 0.05 else '#34d399'}, #1e293b);
                        backdrop-filter: blur(12px); box-shadow: 0 12px 30px rgba(0,0,0,0.5);
                        border: 1px solid rgba(255,255,255,0.1);'>
                <h1 style='margin:0; font-size: 72px; color: white; text-shadow: 0 4px 20px rgba(0,0,0,0.6);'>
                    {return_probability:.1%}
                </h1>
                <h3 style='margin:10px 0; color: #e2e8f0;'>Probability of Return</h3>
                <h2 style='margin:15px 0; color: {'#fbbf24' if return_probability > 0.05 else '#ffffff'};'>
                    {prediction}
                </h2>
            </div>
        """, unsafe_allow_html=True)

    if return_probability <= 0.05:
        st.success("🎉 Low Risk! This order is likely to be kept.")
        st.balloons()
    else:
        st.error("⚠️ High Risk! This order may be returned.")
        st.snow()

    if return_probability > 0.05:
        st.warning("**Recommendation**: Enhance product photos, detailed sizing charts, or clear material description for this category.")
    else:
        st.success("**Great choice!** High customer satisfaction expected.")

    st.markdown("<h4 style='text-align: center;'>🔬 Model Performance Metrics (Test Set)</h4>", unsafe_allow_html=True)
    colm1, colm2, colm3, colm4 = st.columns(4)
    colm1.metric("Accuracy", MODEL_METRICS["Accuracy"])
    colm2.metric("Precision (Returns)", MODEL_METRICS["Precision (Return Class)"])
    colm3.metric("Recall (Returns)", MODEL_METRICS["Recall (Return Class)"])
    colm4.metric("AUC Score", MODEL_METRICS["AUC Score"])

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Feedback Section (Glass Card + Star Ratings)
# -----------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #34d399;'>📝 Give Your Feedback</h2>", unsafe_allow_html=True)
st.write("Your feedback helps improve the model and user experience!")

with st.form(key="feedback_form", clear_on_submit=True):
    name = st.text_input("Your Name *", placeholder="e.g., Umar Farooq")
    
    colf1, colf2 = st.columns(2)
    with colf1:
        st.markdown("<p style='text-align:center; color:#94a3b8;'>Usability of the App</p>", unsafe_allow_html=True)
        usability_rating = st_star_rating("", maxValue=5, defaultValue=4, size=30, read_only=False)
    with colf2:
        st.markdown("<p style='text-align:center; color:#94a3b8;'>Accuracy & Relevance of Prediction</p>", unsafe_allow_html=True)
        accuracy_relevance = st_star_rating("", maxValue=5, defaultValue=4, size=30, read_only=False)
    
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
                    repo.update_file(
                        path=file_path,
                        message=f"New feedback from {name}",
                        content=csv_content,
                        sha=contents.sha,
                        branch=branch
                    )
                else:
                    repo.create_file(
                        path=file_path,
                        message=f"Initial feedback from {name}",
                        content=csv_content,
                        branch=branch
                    )
                st.success(f"✅ Thank you, **{name}**! Your feedback has been recorded and saved to GitHub.")
                st.balloons()
                # Force refresh feedback table
                st.rerun()
            except Exception as e:
                st.error(f"Error saving feedback to GitHub: {str(e)}")
                st.info("Feedback could not be saved permanently.")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Feedback Table (Realtime + Glass Effect)
# -----------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #34d399;'>📊 All Submitted Feedbacks</h2>", unsafe_allow_html=True)

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
except Exception as e:
    st.info("No feedback submitted yet or unable to load from GitHub.")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
    <div class='glass-card' style='text-align: center; padding: 1.5rem; margin-top: 3rem;'>
        <p style='color: #94a3b8; font-size: 14px; margin:5px;'>
            <strong>Data Science Assignment 4</strong> | BSCS-F22 | Instructor: Ghulam Ali<br>
            Model: XGBoost Classifier | Dataset: Online Retail (UCI/Kaggle)<br>
            Deployment: Streamlit Cloud | Version 1.0 — December 2025
        </p>
        <p style='color: #64748b; font-size: 13px; margin-top: 15px;'>
            **Feedback Persistence**: Feedbacks are now directly saved to your GitHub repository using the GitHub API. 
            New entries will appear in <code>feedback.csv</code> on GitHub and in the app after refresh.
        </p>
    </div>
""", unsafe_allow_html=True)
