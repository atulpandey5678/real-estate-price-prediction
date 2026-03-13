import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import get_preprocessed_data


def encode_categorical_features(df):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    return df


def split_data(df, target_column="price", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_engineered_data(data_path, target_column="price", test_size=0.2, random_state=42):
    df = get_preprocessed_data(data_path)
    df = encode_categorical_features(df)
    X_train, X_test, y_train, y_test = split_data(
        df, target_column=target_column, test_size=test_size, random_state=random_state
    )
    feature_columns = X_train.columns.tolist()
    return X_train, X_test, y_train, y_test, feature_columns
