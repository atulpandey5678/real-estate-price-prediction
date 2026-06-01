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

## Project Structure

```text
real_estate_project/
├── app/
│   └── app.py                  # Streamlit application
├── data/
│   └── train.csv               # Dataset
├── models/
│   ├── feature_columns.json    # Saved feature names for prediction
│   └── house_price_model.pkl   # Serialized ML model
├── src/
│   ├── data_preprocessing.py   # Data cleaning and handling missing values
│   ├── evaluate_model.py       # Metrics for model evaluation
│   ├── feature_engineering.py  # Feature transformations and encoding
│   ├── predict.py              # Prediction pipeline
│   └── train_model.py          # Model training pipeline
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/atulpandey5678/real-estate-price-prediction.git
   cd real-estate-price-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**1. Train the model**

If you want to retrain the model, run the training pipeline:
```bash
python -m src.train_model
```
This will evaluate multiple models (Linear Regression, Random Forest, XGBoost) and save the best performing model.

**2. Make a quick prediction from CLI**

Test the model prediction from the command line:
```bash
python -m src.predict
```

**3. Run the Web Application**

Launch the Streamlit app to interact with the model via a web browser:
```bash
python -m streamlit run app/app.py
```
The app will automatically open in your default browser at `http://localhost:8502`.


