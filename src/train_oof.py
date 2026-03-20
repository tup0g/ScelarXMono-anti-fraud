import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import sys
import os
import json
import pickle
import csv
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import load_data, extract_features
from src.pipeline_utils import fold_oof_mean_encoding
from sklearn.metrics import f1_score, precision_score, recall_score

N_FOLDS = 5
RANDOM_STATE = 42
FORCE_REBUILD_BASE_CACHE = False  # Set True after changing features.py

# ENSEMBLE_SEEDS = [42, 123, 456] 
ENSEMBLE_SEEDS = [456] 

# ── Sample Weighting Config ──────────────────────────────────
USE_SAMPLE_WEIGHTING = False
THRESHOLD_STRATEGY = 'two_segment'  # 'global' or 'two_segment'

LOW_ACTIVITY_TX_THRESHOLD = 8
LOW_ACTIVITY_FRAUD_WEIGHT = 1.75
SINGLE_ATTEMPT_FRAUD_WEIGHT = 2.25

# ── Feature packages for ablation ────────────────────────────
# Toggle groups on/off by editing ACTIVE_PACKAGES.
FEATURE_PACKAGES = {
    'BASELINE_CORE': [
        'failed_tx_count', 'has_fraud_error', 'is_country_mismatch',
        'is_pay_mismatch', 'total_transactions', 'total_unique_cards',
        'success_rate',
        'has_name_email_match', 'cards_per_tx',
        'days_active', 'n_error_types',
        'n_card_brands',
        'hours_to_first_tx_log',
        'first_tx_failed',
        'fraud_error_rate', 'country_mismatch_rate',
        'success_rate_24h', 'failed_tx_count_24h', 'fraud_error_rate_24h',
        'unique_cards_24h',
        'first_tx_le_1h',
        'payment_country_entropy', 'error_group_entropy', 'transaction_type_entropy',
        'card_country_nunique', 'payment_country_nunique',
        'card_type_nunique',
        'first_tx_amount', 'amount_cv', 'share_small_amounts',
        'max_amount_24h', 'share_repeated_amounts',
        # encoded / fold-aware baseline
        'traffic_type_enc',
        'reg_country_risk_oof', 'gender_risk_oof',
        'shared_card_users_mean',
        'share_tx_on_shared_cards',
        'tx_time_span_hours', 'tx_hour_std', 'mean_hours_between_tx',
        'min_hours_between_tx', 'recurring_tx_ratio',
        'unique_holders', 'holder_change_rate', 'antifraud_error_count',
    ],
    'EMAIL_DOMAIN_PACKAGE': [
        'email_domain_risk_oof',
    ],
    'FIRST3_PACKAGE': [
        'success_rate_first3',
        'failed_tx_count_first3',
        'unique_cards_first3',
        'fraud_error_rate_first3',
    ],
    'PAY_CARDINIT_PACKAGE': [
        'pay_country_mismatch_rate',
    ],
    'AMOUNT_RATIO_PACKAGE': [
        'amount_ratio_first_vs_mean',
    ],
}

# ── Active packages config ───────────────────────────────────
# Comment out a line to disable that package in the next run.
ACTIVE_PACKAGES = [
    'BASELINE_CORE',
    'FIRST3_PACKAGE',
    'PAY_CARDINIT_PACKAGE',
    'AMOUNT_RATIO_PACKAGE',
]


def fold_shared_card_features(train_tx_fold, features_df, tx_df):
    card_user_counts = train_tx_fold.groupby('card_mask_hash')['id_user'].nunique()

    user_cards = tx_df[tx_df['id_user'].isin(features_df['id_user'])][['id_user', 'card_mask_hash']].drop_duplicates()
    user_cards['shared_users'] = user_cards['card_mask_hash'].map(card_user_counts).fillna(1) - 1

    shared_max = user_cards.groupby('id_user')['shared_users'].max().clip(lower=0)
    shared_mean = user_cards.groupby('id_user')['shared_users'].mean().clip(lower=0)
    n_shared = user_cards[user_cards['shared_users'] > 0].groupby('id_user').size()
    cards_shared_2 = user_cards[user_cards['shared_users'] >= 2].groupby('id_user').size()
    cards_shared_5 = user_cards[user_cards['shared_users'] >= 5].groupby('id_user').size()

    shared_card_hashes = set(user_cards[user_cards['shared_users'] > 0]['card_mask_hash'])
    tx_on_shared = tx_df[
        (tx_df['id_user'].isin(features_df['id_user'])) &
        (tx_df['card_mask_hash'].isin(shared_card_hashes))
    ].groupby('id_user').size()
    tx_total = tx_df[tx_df['id_user'].isin(features_df['id_user'])].groupby('id_user').size()
    tx_total = tx_total.reindex(features_df['id_user']).fillna(0)

    features_df = features_df.copy()
    features_df['shared_card_users_max'] = shared_max.reindex(features_df['id_user']).fillna(0).values
    features_df['shared_card_users_mean'] = shared_mean.reindex(features_df['id_user']).fillna(0).values
    features_df['n_shared_cards'] = n_shared.reindex(features_df['id_user']).fillna(0).values
    features_df['cards_shared_2plus'] = cards_shared_2.reindex(features_df['id_user']).fillna(0).values
    features_df['cards_shared_5plus'] = cards_shared_5.reindex(features_df['id_user']).fillna(0).values

    tx_on_s = tx_on_shared.reindex(features_df['id_user']).fillna(0)
    features_df['share_tx_on_shared_cards'] = np.where(tx_total > 0, tx_on_s / tx_total, 0)

    return features_df


def get_feature_cols(features_df):
    """Build feature list from active packages, filtered by available columns."""
    all_cols = []
    for pkg_name in ACTIVE_PACKAGES:
        all_cols.extend(FEATURE_PACKAGES[pkg_name])
    return [c for c in all_cols if c in features_df.columns]

def run_oof_pipeline(best_params=None, seed=42):
    print("=" * 60)
    print("FOLD-AWARE OOF TRAINING PIPELINE (train only)")
    print("=" * 60)

    # ── Завантажуємо ТІЛЬКИ train дані ──────────────────────────
    train_users, train_tx, _, _ = load_data()
    y = train_users['is_fraud'].values
    print(f"Train users: {len(train_users)}, fraud rate: {y.mean()*100:.2f}%")
    print(f"Active packages: {sorted(ACTIVE_PACKAGES)}")

    cache_path = Path('data/processed/train_base_features.pkl')
    if cache_path.exists() and not FORCE_REBUILD_BASE_CACHE:
        print("Loading cached base features...")
        with open(cache_path, 'rb') as f:
            all_base_features = pickle.load(f)
        all_base_features = all_base_features.set_index('id_user').loc[
            train_users['id_user']
        ].reset_index()
    else:
        print("Extracting base features (one-time, will be cached)...")
        all_base_features = extract_features(train_users, train_tx)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_base_features, f)
        print(f"✅ Cached to {cache_path}")

    if best_params is None:
        try:
            with open('models/best_params.json', 'r') as f:
                best_params = json.load(f)
            print("Loaded best parameters from models/best_params.json")
        except FileNotFoundError:
            print("best_params.json not found — using defaults")
            best_params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 8,
                'num_leaves': 64,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }

    best_params.update({
        'random_state': seed,
        'n_jobs': -1,
        'verbose': -1,
    })

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    oof_probas = np.zeros(len(train_users))
    feature_importances = []
    final_feature_cols = None

    # ── Змінна для збереження fold 5 моделі ─────────────────────
    shap_model = None
    X_val_shap = None
    y_val_shap = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_users, y)):
        print(f"\n{'='*40} FOLD {fold+1}/{N_FOLDS} {'='*40}")

        # Slice pre-computed base features by fold index
        fold_train_features = all_base_features.iloc[train_idx].copy()
        fold_val_features   = all_base_features.iloc[val_idx].copy()

        fold_train_user_ids = set(fold_train_features['id_user'])
        fold_train_tx = train_tx[train_tx['id_user'].isin(fold_train_user_ids)]

        # ── Fold-aware features only ────────────────────────────
        print(f"  Computing shared card features (fold-aware)...")
        fold_train_features = fold_shared_card_features(fold_train_tx, fold_train_features, train_tx)
        fold_val_features   = fold_shared_card_features(fold_train_tx, fold_val_features,   train_tx)

        print(f"  Applying target encoding (fold-aware OOF)...")
        fold_train_features, fold_val_features, _, _ = fold_oof_mean_encoding(
            fold_train_features, fold_val_features,
            target_col='is_fraud', cat_cols=['traffic_type'], suffix='_enc',
            alpha_map={'traffic_type': 10},
        )

        print(f"  Applying risk encoding (fold-aware OOF)...")
        fold_train_features, fold_val_features, _, _ = fold_oof_mean_encoding(
            fold_train_features, fold_val_features,
            target_col='is_fraud',
            cat_cols=['reg_country', 'gender', 'email_domain'],
            suffix='_risk_oof',
            alpha_map={'reg_country': 25, 'gender': 5, 'email_domain': 75},
            min_count_map={'reg_country': 5, 'email_domain': 20},
        )

        if final_feature_cols is None:
            final_feature_cols = get_feature_cols(fold_train_features)
            print(f"  Total features: {len(final_feature_cols)}")

        X_tr  = fold_train_features[final_feature_cols]
        y_tr  = fold_train_features['is_fraud']
        X_val = fold_val_features[final_feature_cols]
        y_val = fold_val_features['is_fraud']

        fit_kwargs = {}
        if USE_SAMPLE_WEIGHTING:
            sample_weight = np.ones(len(y_tr), dtype=float)
            tx_lt_5 = X_tr['total_transactions'].to_numpy() < LOW_ACTIVITY_TX_THRESHOLD
            single_attempt = X_tr['single_attempt'].to_numpy() == 1
            fraud_mask = y_tr.to_numpy() == 1

            sample_weight[fraud_mask & tx_lt_5] = LOW_ACTIVITY_FRAUD_WEIGHT
            sample_weight[fraud_mask & single_attempt] = SINGLE_ATTEMPT_FRAUD_WEIGHT
            fit_kwargs['sample_weight'] = sample_weight

        print(f"  Training LightGBM...")
        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_tr, y_tr,
            **fit_kwargs,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        oof_probas[val_idx] = model.predict_proba(X_val)[:, 1]

        fold_f1 = max(
            f1_score(y_val, (oof_probas[val_idx] >= t).astype(int))
            for t in np.arange(0.1, 0.9, 0.05)
        )
        print(f"  Fold {fold+1} best F1: {fold_f1:.4f}")

        feature_importances.append(
            pd.DataFrame({'feature': final_feature_cols,
                        'importance': model.feature_importances_})
        )

        # ── Зберігаємо fold 5 для shap_analysis ─────────────────
        if fold == N_FOLDS - 1:
            shap_model = model
            X_val_shap = X_val.copy()
            y_val_shap = y_val.copy()

    # ── OOF результати ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OOF RESULTS")
    print("=" * 60)

    thresholds = np.arange(0.05, 0.95, 0.01)
    
    if THRESHOLD_STRATEGY == 'global':
        f1_scores  = [f1_score(y, (oof_probas >= t).astype(int)) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1        = max(f1_scores)
        best_pred      = (oof_probas >= best_threshold).astype(int)
        t_low = best_threshold
        t_high = best_threshold
        print(f"\n*** GLOBAL OOF F1: {best_f1:.4f} @ threshold={best_threshold:.2f} ***")

    elif THRESHOLD_STRATEGY == 'two_segment':
        tx_lt_5 = all_base_features['total_transactions'].to_numpy() < LOW_ACTIVITY_TX_THRESHOLD
        
        y_low = y[tx_lt_5]
        oof_low = oof_probas[tx_lt_5]
        if len(y_low) > 0:
            f1_scores_low = [f1_score(y_low, (oof_low >= t).astype(int)) for t in thresholds]
            t_low = thresholds[np.argmax(f1_scores_low)]
        else:
            t_low = 0.5
            
        y_high = y[~tx_lt_5]
        oof_high = oof_probas[~tx_lt_5]
        if len(y_high) > 0:
            f1_scores_high = [f1_score(y_high, (oof_high >= t).astype(int)) for t in thresholds]
            t_high = thresholds[np.argmax(f1_scores_high)]
        else:
            t_high = 0.5
            
        best_pred = np.zeros_like(y)
        best_pred[tx_lt_5] = (oof_low >= t_low).astype(int)
        best_pred[~tx_lt_5] = (oof_high >= t_high).astype(int)
        
        best_f1 = f1_score(y, best_pred)
        best_threshold = t_high # fallback for exports logging
        print(f"\n*** TWO-SEGMENT OOF F1: {best_f1:.4f} @ t_low={t_low:.2f}, t_high={t_high:.2f} ***")

    fraud_mask = y == 1
    print(f"\nРозподіл oof_probas:")
    print(f"  Fraud:     mean={oof_probas[fraud_mask].mean():.4f}, "
        f"median={np.median(oof_probas[fraud_mask]):.4f}")
    print(f"  Non-fraud: mean={oof_probas[~fraud_mask].mean():.4f}, "
        f"median={np.median(oof_probas[~fraud_mask]):.4f}")
    print(f"  % fraud with proba > 0.5: "
        f"{(oof_probas[fraud_mask] > 0.5).mean()*100:.1f}%")

    prec = precision_score(y, best_pred)
    rec  = recall_score(y, best_pred)
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {best_f1:.4f}")

    # Feature importance
    imp_df = (pd.concat(feature_importances)
            .groupby('feature')['importance']
            .mean()
            .sort_values(ascending=False))
    print(f"\nТоп-15 фіч за importance:")
    print(imp_df.head(15))

    low_imp = imp_df[imp_df < 1].index.tolist()
    if low_imp:
        print(f"\n⚠️  Фічі з дуже низьким importance: {low_imp}")

    # ── Зберігаємо outputs ───────────────────────────────────────
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    # 1. OOF predictions (для error analysis)
    oof_pred = best_pred
    pd.DataFrame({
        'id_user':    train_users['id_user'],
        'is_fraud':   y,
        'oof_proba':  oof_probas,
        'oof_pred':   oof_pred,
    }).to_csv('outputs/oof_predictions.csv', index=False)
    print("\n✅ Збережено outputs/oof_predictions.csv")

    # 2. Fold 5 модель і дані для shap_analysis.py
    shap_model.booster_.save_model('models/fold5_model.txt')
    X_val_shap.to_csv('outputs/fold5_val_features.csv', index=False)
    y_val_shap.to_csv('outputs/fold5_val_labels.csv', index=False)
    print("✅ Збережено models/fold5_model.txt")
    print("✅ Збережено outputs/fold5_val_features.csv")
    print("✅ Збережено outputs/fold5_val_labels.csv")

    # 3. Best threshold для predict_test.py
    if THRESHOLD_STRATEGY == 'global':
        payload = {
            'strategy': 'global',
            'threshold': float(best_threshold),
        }
    else:
        payload = {
            'strategy': 'two_segment',
            'tx_threshold': int(LOW_ACTIVITY_TX_THRESHOLD),
            'low_tx_threshold': float(t_low),
            'high_tx_threshold': float(t_high),
        }

    with open('models/best_threshold.json', 'w') as f:
        json.dump(payload, f, indent=2)
    print("✅ Збережено models/best_threshold.json")

    # 4. Feature cols для використання в інших скриптах
    with open('models/feature_cols.json', 'w') as f:
        json.dump(final_feature_cols, f, indent=2)
    print("✅ Збережено models/feature_cols.json")

    # 5. Log experiment result v2
    log_path = Path("outputs/analysis/experiment_log_v2.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_exists = log_path.exists()

    packages_str = "+".join([p.replace('_PACKAGE', '') for p in ACTIVE_PACKAGES if p != 'BASELINE_CORE'])
    if not packages_str:
        packages_str = "BASELINE_ONLY"
        
    row = {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'active_packages': packages_str,
        'use_sample_weighting': USE_SAMPLE_WEIGHTING,
        'tx_threshold': LOW_ACTIVITY_TX_THRESHOLD,
        'low_activity_weight': LOW_ACTIVITY_FRAUD_WEIGHT,
        'single_attempt_weight': SINGLE_ATTEMPT_FRAUD_WEIGHT,
        'threshold_strategy': THRESHOLD_STRATEGY,
        'global_threshold': f"{best_threshold:.2f}" if THRESHOLD_STRATEGY == 'global' else "",
        'low_tx_threshold': f"{t_low:.2f}" if THRESHOLD_STRATEGY == 'two_segment' else "",
        'high_tx_threshold': f"{t_high:.2f}" if THRESHOLD_STRATEGY == 'two_segment' else "",
        'precision': f"{prec:.4f}",
        'recall': f"{rec:.4f}",
        'best_f1': f"{best_f1:.4f}",
        'n_features': len(final_feature_cols),
        'notes': ''
    }
    
    fieldnames = list(row.keys())
    
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not log_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"✅ Лог експерименту додано в {log_path}")

    print(f"\n{'='*60}")
    print(f"ГОТОВО. Наступний крок:")
    print(f"  python src/shap_analysis.py   ← аналіз фіч")
    print(f"  python src/predict_test.py    ← тільки коли модель фіналізована")
    print(f"{'='*60}")

    return oof_probas

if __name__ == '__main__':    
    try:
        with open('models/best_params.json', 'r') as f:
            best_params = json.load(f)
    except FileNotFoundError:
        best_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 64,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }

    all_seeds_oof_probas = []
    
    for seed in ENSEMBLE_SEEDS:
        print(f"\n{'#'*60}\n# SEED {seed}\n{'#'*60}")
        seed_probas = run_oof_pipeline(best_params=best_params.copy(), seed=seed)
        all_seeds_oof_probas.append(seed_probas)
    
    # Усереднення
    oof_probas = np.mean(all_seeds_oof_probas, axis=0)

    # Завантажуємо train_users для two_segment
    train_users_final = pd.read_csv('data/raw/train_users.csv')
    y = train_users_final['is_fraud'].values

    # Завантажуємо base features для two_segment threshold
    cache_path = Path('data/processed/train_base_features.pkl')
    with open(cache_path, 'rb') as f:
        all_base_features = pickle.load(f)
    all_base_features = all_base_features.set_index('id_user').loc[
        train_users_final['id_user']
    ].reset_index()

    thresholds = np.arange(0.05, 0.95, 0.01)
    tx_lt_5 = all_base_features['total_transactions'].to_numpy() < LOW_ACTIVITY_TX_THRESHOLD

    # two_segment
    y_low = y[tx_lt_5]
    oof_low = oof_probas[tx_lt_5]
    f1_scores_low = [f1_score(y_low, (oof_low >= t).astype(int)) for t in thresholds]
    t_low = thresholds[np.argmax(f1_scores_low)]

    y_high = y[~tx_lt_5]
    oof_high = oof_probas[~tx_lt_5]
    f1_scores_high = [f1_score(y_high, (oof_high >= t).astype(int)) for t in thresholds]
    t_high = thresholds[np.argmax(f1_scores_high)]

    best_pred = np.zeros_like(y)
    best_pred[tx_lt_5] = (oof_low >= t_low).astype(int)
    best_pred[~tx_lt_5] = (oof_high >= t_high).astype(int)

    best_f1 = f1_score(y, best_pred)
    prec = precision_score(y, best_pred)
    rec = recall_score(y, best_pred)

    print(f"\n*** ENSEMBLE TWO-SEGMENT OOF F1: {best_f1:.4f} @ t_low={t_low:.2f}, t_high={t_high:.2f} ***")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")

    # Зберігаємо ensemble threshold
    payload = {
        'strategy': 'two_segment',
        'tx_threshold': int(LOW_ACTIVITY_TX_THRESHOLD),
        'low_tx_threshold': float(t_low),
        'high_tx_threshold': float(t_high),
    }
    with open('models/best_threshold.json', 'w') as f:
        json.dump(payload, f, indent=2)
    print("✅ Ensemble threshold збережено в models/best_threshold.json")

    # Зберігаємо ensemble oof predictions
    pd.DataFrame({
        'id_user': train_users_final['id_user'],
        'is_fraud': y,
        'oof_proba': oof_probas,
        'oof_pred': best_pred,
    }).to_csv('outputs/oof_predictions.csv', index=False)
    print("✅ Ensemble OOF predictions збережено")