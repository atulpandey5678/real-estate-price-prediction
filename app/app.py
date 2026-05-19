import os
import sys
import time
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.predict import predict_price

st.set_page_config(
    page_title="Real Estate Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Advanced Custom CSS: Professional White & Red Corporate Theme
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global App Background */
    .stApp {
        background-color: #f8f9fa; /* Very light gray/white */
        color: #1e293b;
    }

    /* Top bar adjustment */
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid #e2e8f0;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1050px;
    }

    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Headers */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #0f172a;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out forwards;
        opacity: 0;
    }
    
    .hero-title span {
        color: #d90429; /* Professional Crimson Red */
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        font-weight: 400;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInUp 0.8s ease-out 0.2s forwards;
        opacity: 0;
    }

    /* Input Card Styling */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Text inside inputs */
    div[data-baseweb="base-input"],
    div[data-baseweb="base-input"] > input {
        background: transparent !important;
        color: #0f172a !important;
        font-weight: 500 !important;
    }
    
    /* Red Hover/Focus effects */
    div[data-baseweb="select"] > div:hover, 
    div[data-baseweb="input"]:hover,
    div[data-baseweb="select"] > div:focus-within, 
    div[data-baseweb="input"]:focus-within {
        border-color: #ef233c !important;
        box-shadow: 0 8px 16px rgba(239, 35, 60, 0.12) !important;
        transform: translateY(-2px);
    }

    /* Labels */
    .stSelectbox label, .stNumberInput label {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Main Action Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #d90429 0%, #b71c1c 100%);
        color: #ffffff !important;
        border: none !important;
        padding: 0.9rem 2rem !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        box-shadow: 0 4px 14px 0 rgba(217, 4, 41, 0.39) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-top: 1.5rem;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 10px 25px 0 rgba(217, 4, 41, 0.45) !important;
    }
    
    .stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Prediction Result Box */
    .prediction-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #d90429;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.08);
        animation: slideInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        opacity: 0;
    }

    .prediction-label {
        color: #64748b;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }

    .prediction-value {
        color: #0f172a;
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1.1;
    }
    
    .insight-box {
        margin-top: 1rem;
        display: flex;
        justify-content: center;
        gap: 2rem;
    }
    
    .insight-item {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .insight-title {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
    }
    
    .insight-value {
        font-size: 1.25rem;
        color: #d90429;
        font-weight: 700;
    }

    /* Staggered form animations */
    .stNumberInput, .stSelectbox {
        animation: fadeIn 0.6s ease-out forwards;
        opacity: 0;
    }
    div[data-testid="column"]:nth-child(1) .stNumberInput { animation-delay: 0.3s; }
    div[data-testid="column"]:nth-child(1) .stSelectbox { animation-delay: 0.4s; }
    div[data-testid="column"]:nth-child(2) .stNumberInput { animation-delay: 0.5s; }
    div[data-testid="column"]:nth-child(2) .stSelectbox { animation-delay: 0.6s; }
    div[data-testid="column"]:nth-child(3) .stSelectbox { animation-delay: 0.7s; }

    /* Keyframes */
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInUp {
        0% { opacity: 0; transform: translateY(40px) scale(0.98); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# App Layout
# ---------------------------------------------------------
st.markdown('<div class="hero-title">NexEstate <span>Valuation</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Enterprise-Grade AI Property Appraisal</div>', unsafe_allow_html=True)

# Using st.container to group the fields visually
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        property_area = st.number_input("Property Area (sq ft)", min_value=100, max_value=10000, value=1200, step=50)
        bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5, 6], index=1)
        
    with col2:
        year_built = st.number_input("Year Built", min_value=1950, max_value=2025, value=2015, step=1)
        garage_size = st.selectbox("Garage Size (Cars)", options=[0, 1, 2, 3], index=1)
        lot_area = st.number_input("Lot Area (sq ft)", min_value=200, max_value=20000, value=2500, step=100)
        
    with col3:
        location = st.selectbox("Location Tier", options=["Metro", "Tier1", "Tier2", "Tier3"], index=0)
        property_type = st.selectbox("Property Type", options=["Apartment", "Villa", "Independent House"], index=0)
        furnishing_status = st.selectbox("Furnishing Status", options=["Furnished", "Semi-Furnished", "Unfurnished"], index=1)

st.markdown("<br/>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Prediction Logic & Animations
# ---------------------------------------------------------
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("Calculate Market Value")

if predict_clicked:
    # Adding a visual progress bar and calculation delay for UX
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.markdown("<div style='text-align: center; color: #64748b; font-weight: 500;'>Analyzing market trends...</div>", unsafe_allow_html=True)
    for percent_complete in range(0, 100, 20):
        time.sleep(0.1)
        progress_bar.progress(percent_complete + 20)
    
    status_text.empty()
    progress_bar.empty()

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
        
        # Calculate derived metrics for added functionality
        price_per_sqft = (predicted_price * 100000) / property_area
        age_of_property = 2026 - year_built
        
        st.markdown(
            f"""
            <div class="prediction-container">
                <div class="prediction-label">Estimated Market Value</div>
                <div class="prediction-value">Rs. {predicted_price:,.2f} L</div>
                
                <div class="insight-box">
                    <div class="insight-item">
                        <div class="insight-title">Price per Sq.Ft</div>
                        <div class="insight-value">Rs. {price_per_sqft:,.0f}</div>
                    </div>
                    <div class="insight-item">
                        <div class="insight-title">Property Age</div>
                        <div class="insight-value">{age_of_property} Years</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Additional Functionality: Streamlit Metric
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("View Advanced Analytics", expanded=False):
            st.markdown("### Market Indicators")
            metric1, metric2, metric3 = st.columns(3)
            
            # Simple logic to determine demand based on location
            demand_indicator = "High Demand" if location in ["Metro", "Tier1"] else "Stable Demand"
            delta_val = "12% 📈" if location == "Metro" else "5% 📈"
            
            metric1.metric("Location Rating", location, delta=demand_indicator)
            metric2.metric("Market Trend (YoY)", "Positive", delta=delta_val)
            metric3.metric("Liquidity Score", "Very High" if furnishing_status == "Furnished" else "Moderate")
            
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first by running `python -m src.train_model`.")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; font-weight: 500;">'
    "Powered by Advanced Machine Learning • NexEstate Systems"
    "</div>",
    unsafe_allow_html=True,
)
