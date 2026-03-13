from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, build_time_features


def train_model(df: pd.DataFrame, target_col: str = "is_fraud"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    data = build_time_features(df)
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(data, target_col=target_col)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(class_weight="balanced", random_state=42))
        ]
    )

    pipeline.fit(X_train, y_train)
    valid_pred = pipeline.predict(X_valid)
    valid_proba = pipeline.predict_proba(X_valid)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_valid, valid_proba),
        "classification_report": classification_report(y_valid, valid_pred, output_dict=False),
    }

    return pipeline, metrics

def predict_scores(model: Pipeline, df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    data = build_time_features(df)
    proba = model.predict_proba(data)[:, 1]
    
    result = pd.DataFrame()
    result["id_user"] = df["id_user"]
    result["is_fraud"] = (proba >= threshold).astype(int)
    
    return result
