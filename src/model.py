from __future__ import annotations

import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


def find_best_threshold(y_true, y_proba, thresholds=None):
    """Find threshold that maximizes F1-score for the positive class."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)

    best_f1 = 0
    best_thresh = 0.5

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


def train_model(df: pd.DataFrame, target_col: str = "is_fraud"):
    """Train LightGBM with tuned hyperparameters and threshold optimization."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    X = df.drop(columns=[target_col, "id_user"], errors="ignore")
    y = df[target_col]

    feature_cols = X.columns.tolist()

    # Calculate scale_pos_weight for class imbalance
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_weight = n_neg / n_pos
    print(f"  Class ratio: {n_neg:,} neg / {n_pos:,} pos = {scale_weight:.1f}x")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Tuned LightGBM for fraud detection
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        scale_pos_weight=scale_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
    )

    # Predict probabilities on validation
    valid_proba = model.predict_proba(X_valid)[:, 1]

    # Find optimal threshold for F1
    best_thresh, best_f1 = find_best_threshold(y_valid, valid_proba)
    print(f"\n  Optimal threshold: {best_thresh:.2f} (F1={best_f1:.4f})")

    valid_pred = (valid_proba >= best_thresh).astype(int)

    roc_auc = roc_auc_score(y_valid, valid_proba)
    report = classification_report(y_valid, valid_pred, output_dict=False)

    metrics = {
        "roc_auc": roc_auc,
        "best_threshold": best_thresh,
        "best_f1": best_f1,
        "classification_report": report,
    }

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)

    return model, metrics, feature_cols, best_thresh, importance


def predict_scores(model, df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.5) -> pd.DataFrame:
    """Predict fraud labels on test data using optimal threshold."""
    X = df[feature_cols]
    proba = model.predict_proba(X)[:, 1]

    result = pd.DataFrame()
    result["id_user"] = df["id_user"]
    result["is_fraud"] = (proba >= threshold).astype(int)

    return result


def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load processed features
    print("Loading processed features...")
    train_df = pd.read_csv("data/processed/train_features.csv")
    test_df = pd.read_csv("data/processed/test_features.csv")

    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape:  {test_df.shape}")

    # 2. Train model
    print("\nTraining LightGBM model...")
    model, metrics, feature_cols, threshold, importance = train_model(train_df)

    print(f"\n{'='*50}")
    print(f"📊 ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"🎯 Best F1 (fraud): {metrics['best_f1']:.4f}")
    print(f"📏 Threshold:       {metrics['best_threshold']:.2f}")
    print(f"{'='*50}")
    print(f"\n{metrics['classification_report']}")

    # Feature importance top-10
    print("🏆 Top-10 Important Features:")
    for i, (feat, imp) in enumerate(importance.head(10).items(), 1):
        print(f"  {i:2d}. {feat}: {imp}")

    # 3. Predict on test
    print("\nGenerating predictions on test set...")
    predictions = predict_scores(model, test_df, feature_cols, threshold)
    predictions.to_csv("outputs/predictions.csv", index=False)

    print(f"\n✅ Predictions saved to outputs/predictions.csv")
    print(f"   Total test users: {len(predictions):,}")
    print(f"   Predicted fraud:  {predictions['is_fraud'].sum():,} ({predictions['is_fraud'].mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
