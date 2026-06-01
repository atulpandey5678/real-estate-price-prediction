live link : https://real-estate-price-prediction-bvyxm63jntpb8t3zjlu8cc.streamlit.app/


# Real Estate Price Prediction

A machine learning application to predict property prices in Indian cities. The application features a web interface built with Streamlit and uses models like XGBoost, Random Forest, and Linear Regression to estimate property values based on various features such as area, location, and amenities.

## Problem Statement

The Indian real estate market is vast, diverse, and often lacks pricing transparency. This project addresses the challenge of **estimating property prices** based on key structural and locational features, enabling buyers, sellers, and agents to make data-driven pricing decisions.

## Dataset Description

A synthetic India-focused housing dataset with **5,000 records** and the following features:

| Feature | Description |
|---|---|
| `property_area` | Built-up area in square feet |
| `bedrooms` | Number of bedrooms (1–6) |
| `bathrooms` | Number of bathrooms (1–6) |
| `year_built` | Year the property was constructed |
| `garage_size` | Garage capacity (0–3 cars) |
| `lot_area` | Total lot/plot area in square feet |
| `location` | City tier — Metro, Tier1, Tier2, Tier3 |
| `property_type` | Apartment, Villa, or Independent House |
| `furnishing_status` | Furnished, Semi-Furnished, or Unfurnished |
| `price` | **Target** — Price in INR Lakhs |

## Features

- **Predictive Modeling**: Estimate property prices using trained ML models.
- **Interactive UI**: A Streamlit web application for easy interaction.
- **Data Preprocessing**: Handling missing values, categorical encoding, etc.
- **Model Training**: Easily retrain models with new data.


