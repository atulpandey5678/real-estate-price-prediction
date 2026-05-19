"""
Module for feature engineering, including categorical encoding and splitting the data.
"""

import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from src.data_preprocessing import get_preprocessed_data


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes categorical columns, dropping the first category to avoid multicollinearity."""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    return df


def split_data(df: pd.DataFrame, target_column: str = "price", test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the dataframe into training and testing subsets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_engineered_data(data_path: str, target_column: str = "price", test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Runs the feature engineering pipeline and returns the train/test splits along with feature names."""
    df = get_preprocessed_data(data_path)
    df = encode_categorical_features(df)
    
    X_train, X_test, y_train, y_test = split_data(
        df, target_column=target_column, test_size=test_size, random_state=random_state
    )
    feature_columns = X_train.columns.tolist()
    
    return X_train, X_test, y_train, y_test, feature_columns
