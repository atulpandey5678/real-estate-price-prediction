import pandas as pd
import numpy as np


def load_data(path):
    return pd.read_csv(path)


def handle_missing_values(df):
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


def remove_high_missing_columns(df, threshold=0.4):
    df = df.copy()
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def get_preprocessed_data(path, missing_threshold=0.4):
    df = load_data(path)
    df = remove_high_missing_columns(df, threshold=missing_threshold)
    df = handle_missing_values(df)
    return df
