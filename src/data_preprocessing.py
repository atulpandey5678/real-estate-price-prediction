"""
Module for handling data preprocessing tasks such as loading data,
handling missing values, and dropping columns with excessive missing data.
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Loads dataset from the specified CSV path."""
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values in numerical and categorical columns."""
    df = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def remove_high_missing_columns(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """Removes columns that have missing values above the given threshold."""
    df = df.copy()
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    return df


def get_preprocessed_data(path: str, missing_threshold: float = 0.4) -> pd.DataFrame:
    """Runs the full data preprocessing pipeline."""
    df = load_data(path)
    df = remove_high_missing_columns(df, threshold=missing_threshold)
    df = handle_missing_values(df)
    return df
