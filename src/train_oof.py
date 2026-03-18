import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from pathlib import Path
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import load_data, extract_features, compute_entropy

N_FOLDS = 5
RANDOM_STATE = 42

BASE_FEATURE_COLS = [
    'is_instant_registration', 'has_night_tx', 'is_country_mismatch',
    'is_pay_mismatch', 'total_transactions', 'total_unique_cards',
    'success_rate', 'failed_tx_count', 'has_fraud_error',
    'has_name_email_match', 'cards_per_tx',
    'single_attempt', 'days_active', 'n_error_types',
    'n_card_brands', 'n_tx_types',
    'tx_24h', 'tx_7d', 'hours_to_first_tx_log',
    'first_tx_failed', 'first_tx_fraud_error',
    'fraud_error_rate', 'country_mismatch_rate',
    'success_rate_24h', 'failed_tx_count_24h', 'fraud_error_rate_24h',
    'unique_cards_24h', 'country_mismatch_rate_24h', 'n_error_types_24h',
    'first_tx_le_1h', 'first_tx_le_6h', 'first_tx_le_24h', 'first_tx_gt_7d',
    'payment_country_entropy', 'error_group_entropy', 'transaction_type_entropy',
    'card_country_nunique', 'payment_country_nunique', 'currency_nunique',
    'card_type_nunique',
    'first_tx_amount', 'amount_cv', 'share_small_amounts',
    'max_amount_24h', 'share_repeated_amounts',
]

ENCODED_FEATURE_COLS = [
    'traffic_type_enc',
    'reg_country_risk_oof', 'gender_risk_oof',
    'shared_card_users_max', 'shared_card_users_mean',
    'n_shared_cards', 'cards_shared_2plus', 'cards_shared_5plus',
    'share_tx_on_shared_cards',
]


def fold_target_encoding(train_features, other_features, target_col='is_fraud', cat_col='traffic_type'):
    global_mean = train_features[target_col].mean()
    mapping = train_features.groupby(cat_col)[target_col].mean().to_dict()

    train_features[f'{cat_col}_enc'] = train_features[cat_col].map(mapping).fillna(global_mean)
    other_features[f'{cat_col}_enc'] = other_features[cat_col].map(mapping).fillna(global_mean)

    return train_features, other_features, mapping, global_mean


def fold_risk_encoding(train_features, other_features, target_col='is_fraud', cols=None):
    if cols is None:
        cols = ['reg_country', 'gender']

    global_mean = train_features[target_col].mean()
    encodings = {}

    for col in cols:
        col_means = train_features.groupby(col)[target_col].mean().to_dict()
        encodings[col] = col_means

        train_features[f'{col}_risk_oof'] = train_features[col].map(col_means).fillna(global_mean)
        other_features[f'{col}_risk_oof'] = other_features[col].map(col_means).fillna(global_mean)

    return train_features, other_features, encodings, global_mean


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
    all_possible = BASE_FEATURE_COLS + ENCODED_FEATURE_COLS
    return [c for c in all_possible if c in features_df.columns]


def run_oof_pipeline(best_params=None):
    print("=" * 60)
    print("FOLD-AWARE OOF TRAINING PIPELINE (train only)")
    print("=" * 60)

    # ── Завантажуємо ТІЛЬКИ train дані ──────────────────────────
    # test дані тут не потрібні — ми концентруємось на побудові
    # та оцінці моделі, а не на фінальних predictions
    train_users, train_tx, _, _ = load_data()
    y = train_users['is_fraud'].values
    print(f"Train users: {len(train_users)}, fraud rate: {y.mean()*100:.2f}%")

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
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
    })

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_probas = np.zeros(len(train_users))
    feature_importances = []
    final_feature_cols = None

    # ── Змінна для збереження fold 5 моделі ─────────────────────
    # Буде використана в shap_analysis.py без перенавчання
    shap_model = None
    X_val_shap = None
    y_val_shap = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_users, y)):
        print(f"\n{'='*40} FOLD {fold+1}/{N_FOLDS} {'='*40}")

        fold_train_users = train_users.iloc[train_idx].copy()
        fold_val_users   = train_users.iloc[val_idx].copy()

        fold_train_user_ids = set(fold_train_users['id_user'])
        fold_train_tx = train_tx[train_tx['id_user'].isin(fold_train_user_ids)]

        print(f"  Extracting features...")
        fold_train_features = extract_features(fold_train_users, train_tx)
        fold_val_features   = extract_features(fold_val_users,   train_tx)

        print(f"  Computing shared card features (fold-aware)...")
        fold_train_features = fold_shared_card_features(fold_train_tx, fold_train_features, train_tx)
        fold_val_features   = fold_shared_card_features(fold_train_tx, fold_val_features,   train_tx)

        print(f"  Applying target encoding (fold-aware)...")
        fold_train_features, fold_val_features, _, _ = fold_target_encoding(
            fold_train_features, fold_val_features)

        print(f"  Applying risk encoding (fold-aware)...")
        fold_train_features, fold_val_features, _, _ = fold_risk_encoding(
            fold_train_features, fold_val_features)

        if final_feature_cols is None:
            final_feature_cols = get_feature_cols(fold_train_features)
            print(f"  Total features: {len(final_feature_cols)}")

        X_tr  = fold_train_features[final_feature_cols]
        y_tr  = fold_train_features['is_fraud']
        X_val = fold_val_features[final_feature_cols]
        y_val = fold_val_features['is_fraud']

        print(f"  Training LightGBM...")
        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_tr, y_tr,
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
        # ЗАМІСТЬ того щоб рахувати SHAP тут (дублювання),
        # просто зберігаємо модель і дані на диск.
        # shap_analysis.py завантажить їх без перенавчання.
        if fold == N_FOLDS - 1:
            shap_model = model
            X_val_shap = X_val.copy()
            y_val_shap = y_val.copy()

        # ── ВИДАЛЕНО: predict на test всередині fold loop ────────
        # Раніше тут було:
        #   test_features = extract_features(test_users, test_tx)
        #   fold_test_probas = model.predict_proba(X_test)
        #   test_probas_sum += fold_test_probas
        # Це займало ~20% часу кожного fold і було зайвим
        # під час розробки моделі.
        # Всі test predictions перенесено в predict_test.py

    # ── OOF результати ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OOF RESULTS")
    print("=" * 60)

    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores  = [f1_score(y, (oof_probas >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1        = max(f1_scores)

    fraud_mask = y == 1
    print(f"\nРозподіл oof_probas:")
    print(f"  Fraud:     mean={oof_probas[fraud_mask].mean():.4f}, "
          f"median={np.median(oof_probas[fraud_mask]):.4f}")
    print(f"  Non-fraud: mean={oof_probas[~fraud_mask].mean():.4f}, "
          f"median={np.median(oof_probas[~fraud_mask]):.4f}")
    print(f"  % fraud with proba > 0.5: "
          f"{(oof_probas[fraud_mask] > 0.5).mean()*100:.1f}%")
    print(f"\n*** OOF F1: {best_f1:.4f} @ threshold={best_threshold:.2f} ***")

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
    pd.DataFrame({
        'id_user':    train_users['id_user'],
        'is_fraud':   y,
        'oof_proba':  oof_probas,
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
    with open('models/best_threshold.json', 'w') as f:
        json.dump({'threshold': float(best_threshold)}, f, indent=2)
    print("✅ Збережено models/best_threshold.json")

    # 4. Feature cols для використання в інших скриптах
    with open('models/feature_cols.json', 'w') as f:
        json.dump(final_feature_cols, f, indent=2)
    print("✅ Збережено models/feature_cols.json")

    print(f"\n{'='*60}")
    print(f"ГОТОВО. Наступний крок:")
    print(f"  python src/shap_analysis.py   ← аналіз фіч")
    print(f"  python src/predict_test.py    ← тільки коли модель фіналізована")
    print(f"{'='*60}")

    return best_f1, best_threshold, best_params, final_feature_cols


if __name__ == '__main__':
    best_f1, best_threshold, best_params, feature_cols = run_oof_pipeline()