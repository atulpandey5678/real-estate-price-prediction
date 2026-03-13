# 🏠 Real Estate Price Prediction (India Focus)

A production-quality Machine Learning project that predicts residential property prices across Indian cities using structured housing data. Built with clean ML engineering practices and designed for professional portfolios.

---

## 📌 Problem Statement

The Indian real estate market is vast, diverse, and often lacks pricing transparency. This project addresses the challenge of **estimating property prices** based on key structural and locational features, enabling buyers, sellers, and agents to make data-driven pricing decisions.

---

## 📊 Dataset Description

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

---

## 🛠 Technologies Used

- **Python 3.10+**
- **pandas** / **NumPy** — Data manipulation
- **scikit-learn** — ML models & evaluation
- **XGBoost** — Gradient boosting regressor
- **Matplotlib** / **Seaborn** — Visualization
- **joblib** — Model serialization
- **Streamlit** — Interactive web application

---

## 🏗 Project Structure

```
real_estate_price_prediction/
├── data/
│   └── train.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── models/
│   ├── house_price_model.pkl
│   └── feature_columns.json
├── app/
│   └── app.py
├── scripts/
│   └── generate_dataset.py
├── requirements.txt
└── README.md
```

---

## 🤖 Model Training Approach

### Pipeline

1. **Data Preprocessing** — Handle missing values (median for numerical, mode for categorical), remove columns with >40% missing data
2. **Feature Engineering** — One-hot encode categorical variables, split into 80/20 train/test sets
3. **Model Training** — Train three regression models:
   - Linear Regression
   - Random Forest Regressor (200 trees, max_depth=15)
   - XGBoost Regressor (300 estimators, learning_rate=0.05)
4. **Model Selection** — Automatically select the best model based on R² score
5. **Serialization** — Save the best model and feature columns using joblib/JSON

---

## 📈 Evaluation Metrics

All models are evaluated on the test set using:

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average absolute difference |
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **R² Score** | Coefficient of Determination — variance explained |

---

## 🚀 How to Run

### 1. Clone & Install

```bash
git clone https://github.com/your-username/real-estate-price-prediction.git
cd real-estate-price-prediction
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python scripts/generate_dataset.py
```

### 3. Train the Model

```bash
python -m src.train_model
```

This will train all three models, evaluate them, select the best one, and save it to `models/`.

### 4. Run Predictions (CLI)

```bash
python -m src.predict
```

### 5. Launch the Web App

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥 Streamlit App

The web interface allows users to input property details and receive instant price predictions:

- **Input Fields**: Property area, bedrooms, bathrooms, year built, garage size, lot area, location tier, property type, furnishing status
- **Output**: Estimated price in INR Lakhs
- **Design**: Clean, modern gradient-styled interface

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
