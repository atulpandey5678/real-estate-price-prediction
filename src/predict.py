"""
Module for loading a trained model and making real estate price predictions.
"""

import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, List


def load_model(path: str) -> Any:
    """Loads the serialized machine learning model."""
    return joblib.load(path)


def load_feature_columns(path: str) -> List[str]:
    """Loads the required feature columns for the model."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_input(input_dict: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    """Formats the input dictionary into a dataframe compatible with the model."""
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df, dtype=int)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def make_prediction(model: Any, input_data: pd.DataFrame) -> float:
    """Uses the model to predict the price and returns a rounded float."""
    prediction = model.predict(input_data)
    return round(float(prediction[0]), 2)


def predict_price(input_dict: Dict[str, Any], model_dir: str) -> float:
    """End-to-end prediction function for a single input instance."""
    model_path = os.path.join(model_dir, "house_price_model.pkl")
    columns_path = os.path.join(model_dir, "feature_columns.json")

    model = load_model(model_path)
    feature_columns = load_feature_columns(columns_path)
    input_data = prepare_input(input_dict, feature_columns)
    
    return make_prediction(model, input_data)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")

    sample_input = {
        "property_area": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 2015,
        "garage_size": 1,
        "lot_area": 3000,
        "location": "Metro",
        "property_type": "Apartment",
        "furnishing_status": "Semi-Furnished",
    }

    predicted_price = predict_price(sample_input, model_dir)
    print(f"Predicted Price: Rs. {predicted_price} Lakhs")
