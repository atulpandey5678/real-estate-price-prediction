import os
import json
import joblib
import pandas as pd
import numpy as np


def load_model(path):
    return joblib.load(path)


def load_feature_columns(path):
    with open(path, "r") as f:
        return json.load(f)


def prepare_input(input_dict, feature_columns):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df, dtype=int)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    return round(float(prediction[0]), 2)


def predict_price(input_dict, model_dir):
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
    print(f"Predicted Price: ₹{predicted_price} Lakhs")
