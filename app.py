import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import time

# -----------------------------
# Load Model & Preprocessors
# -----------------------------
@st.cache_resource
def load_artifacts():
    base_path = os.path.dirname(__file__)  # folder of app.py
    model = joblib.load(os.path.join(base_path, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    le_category = joblib.load(os.path.join(base_path, "le_category.pkl"))
    le_country = joblib.load(os.path.join(base_path, "le_country.pkl"))
    return model, scaler, le_category, le_country

model, scaler, le_category, le_country = load_artifacts()

# -----------------------------
# Page Config & Title
# -----------------------------
st.set_page_config(page_title="Return Risk Predictor", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <h1 style='text-align: center; color: #1E88E5; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
        🛒 E-Commerce Product Return Risk Predictor
    </h1>
    <h3 style='text-align: center; color: #424242;'>
        Predict if a customer is likely to return their order
    </h3>
    <hr style='border: 2px solid #1E88E5; border-radius: 5px;'>
""", unsafe_allow_html=True)

# -----------------------------
# Model Performance Metrics (From Training Results)
# -----------------------------
MODEL_METRICS = {
    "Accuracy": "0.9245",
    "Precision (Return Class)": "0.68",
    "Recall (Return Class)": "0.62",
    "AUC Score": "0.91"
}

# -----------------------------
# Main Prediction Section
# -----------------------------
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

# Predict Button
if st.button("🔍 Predict Return Chance", type="primary", use_container_width=True):
    with st.spinner("Analyzing order details and predicting risk..."):
        time.sleep(1.5)  # Small delay for effect

        # Prepare input
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

    # Animated Result Reveal
    st.markdown("<br>", unsafe_allow_html=True)
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(f"""
            <div style='text-align: center; padding: 30px; border-radius: 20px; 
                        background: linear-gradient(135deg, {'#FF5252' if return_probability > 0.05 else '#4CAF50'}, #ffffff);
                        box-shadow: 0 8px 20px rgba(0,0,0,0.15); color: white;'>
                <h1 style='margin:0; font-size: 60px;'>{return_probability:.1%}</h1>
                <h3 style='margin:5px 0;'>Probability of Return</h3>
                <h2 style='margin:10px 0; color: {'#FFEB3B' if return_probability > 0.05 else '#FFFFFF'};'>{prediction}</h2>
            </div>
        """, unsafe_allow_html=True)

    # Celebration Animation
    if return_probability <= 0.05:
        st.success("🎉 Low Risk! This order is likely to be kept.")
        st.balloons()
    else:
        st.error("⚠️ High Risk! This order may be returned.")
        st.snow()

    # Recommendations
    if return_probability > 0.05:
        st.warning("**Recommendation**: Enhance product photos, detailed sizing charts, or clear material description for this category.")
    else:
        st.success("**Great choice!** High customer satisfaction expected.")

    # Show Model Performance
    st.markdown("<br><h4 style='text-align: center;'>🔬 Model Performance Metrics (Test Set)</h4>", unsafe_allow_html=True)
    colm1, colm2, colm3, colm4 = st.columns(4)
    colm1.metric("Accuracy", MODEL_METRICS["Accuracy"])
    colm2.metric("Precision (Returns)", MODEL_METRICS["Precision (Return Class)"])
    colm3.metric("Recall (Returns)", MODEL_METRICS["Recall (Return Class)"])
    colm4.metric("AUC Score", MODEL_METRICS["AUC Score"])

# -----------------------------
# Feedback Form
# -----------------------------
st.markdown("<br><hr style='border-top: 3px dashed #1E88E5;'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #1E88E5;'>📝 Give Your Feedback</h2>", unsafe_allow_html=True)
st.write("Your feedback helps improve the model and user experience!")

with st.form(key="feedback_form", clear_on_submit=True):
    name = st.text_input("Your Name *", placeholder="e.g., Umar Farooq")
    colf1, colf2 = st.columns(2)
    with colf1:
        usability_rating = st.slider("Usability of the App (1 = Poor, 5 = Excellent)", 1, 5, 4)
    with colf2:
        accuracy_relevance = st.slider("Accuracy & Relevance of Prediction (1-5)", 1, 5, 4)

    suggestions = st.text_area("Suggestions for Improvement", 
                               placeholder="e.g., Add product images, allow bulk upload, show explanations, etc.")
    submitted = st.form_submit_button("🚀 Submit Feedback", type="primary", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("⚠️ Please enter your name.")
        else:
            feedback_entry = pd.DataFrame([{
                "Name": name,
                "Usability_Rating": usability_rating,
                "Accuracy_Relevance_Rating": accuracy_relevance,
                "Suggestions": suggestions,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            
            # Use relative path for feedback.csv
            csv_path = os.path.join(os.path.dirname(__file__), "feedback.csv")
            if os.path.exists(csv_path):
                df_existing = pd.read_csv(csv_path)
                df_updated = pd.concat([df_existing, feedback_entry], ignore_index=True)
            else:
                df_updated = feedback_entry

            df_updated.to_csv(csv_path, index=False)
            st.success(f"✅ Thank you, **{name}**! Your feedback has been recorded.")
            st.balloons()

# -----------------------------
# Footer
# -----------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; color: #666; font-size: 14px;'>
        <strong>Data Science Assignment 4</strong> | BSCS-F22 | Instructor: Ghulam Ali<br>
        Model: XGBoost Classifier | Dataset: Online Retail (UCI/Kaggle)<br>
        Deployment: Streamlit Cloud | Version 1.0 — December 2025
    </p>
""", unsafe_allow_html=True)
