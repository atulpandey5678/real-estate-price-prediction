import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.predict import predict_price


st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1.5rem;
    }
    .prediction-label {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .prediction-value {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🏠 Real Estate Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict property prices across Indian cities using Machine Learning</div>', unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    property_area = st.number_input("Property Area (sq ft)", min_value=100, max_value=10000, value=1200, step=50)
    bedrooms = st.selectbox("Number of Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("Number of Bathrooms", options=[1, 2, 3, 4, 5, 6], index=1)

with col2:
    year_built = st.number_input("Year Built", min_value=1950, max_value=2025, value=2015, step=1)
    garage_size = st.selectbox("Garage Size (Cars)", options=[0, 1, 2, 3], index=1)
    lot_area = st.number_input("Lot Area (sq ft)", min_value=200, max_value=20000, value=2500, step=100)

st.markdown("---")

location = st.selectbox("Location Tier", options=["Metro", "Tier1", "Tier2", "Tier3"], index=0)
property_type = st.selectbox("Property Type", options=["Apartment", "Villa", "Independent House"], index=0)
furnishing_status = st.selectbox("Furnishing Status", options=["Furnished", "Semi-Furnished", "Unfurnished"], index=1)

st.markdown("")

if st.button("🔍 Predict Price"):
    input_data = {
        "property_area": property_area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "garage_size": garage_size,
        "lot_area": lot_area,
        "location": location,
        "property_type": property_type,
        "furnishing_status": furnishing_status,
    }

    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

    try:
        predicted_price = predict_price(input_data, model_dir)
        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">Estimated Property Price</div>
                <div class="prediction-value">₹ {predicted_price:,.2f} Lakhs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.error("Model not found. Please run the training pipeline first: `python -m src.train_model`")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #adb5bd; font-size: 0.85rem;">'
    "Built with Streamlit • Model: XGBoost / Random Forest / Linear Regression"
    "</div>",
    unsafe_allow_html=True,
)
