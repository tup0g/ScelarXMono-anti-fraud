from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_time_features(df: pd.DataFrame, column: str = "transaction_time") -> pd.DataFrame:
    data = df.copy()
    if column in data.columns:
        ts = pd.to_datetime(data[column], errors="coerce")
        data[f"{column}_hour"] = ts.dt.hour
        data[f"{column}_dayofweek"] = ts.dt.dayofweek
        data[f"{column}_is_weekend"] = (ts.dt.dayofweek >= 5).astype("Int64")
    return data


def build_preprocessor(df: pd.DataFrame, target_col: str = "is_fraud") -> ColumnTransformer:
    feature_df = df.drop(columns=[target_col], errors="ignore")
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
