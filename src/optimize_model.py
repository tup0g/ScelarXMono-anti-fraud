"""
Step 5: Fast Optuna Optimization for Fraud Detection with Fold-Aware Pipeline.
Precomputes CV folds one time, then optimizes LightGBM hyperparameters.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from pathlib import Path
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import load_data, extract_features
from src.train_oof import fold_target_encoding, fold_risk_encoding, fold_shared_card_features, get_feature_cols

# Constants
N_FOLDS = 5
RANDOM_STATE = 42
N_TRIALS = 100

def prepare_cv_folds():
    """Precompute the 5 folds with completely isolated feature engineering."""
    print("Pre-computing 5 completely isolated CV folds...")
    train_users, train_tx, _, _ = load_data()
    y = train_users['is_fraud'].values
    
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    folds_data = []
    final_feature_cols = None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_users, y)):
        print(f"  Preparing Fold {fold+1}...")
        fold_train_users = train_users.iloc[train_idx]
        fold_val_users = train_users.iloc[val_idx]
        
        fold_train_tx = train_tx[train_tx['id_user'].isin(set(fold_train_users['id_user']))]
        
        # 1. Base features
        fold_train_features = extract_features(fold_train_users, train_tx)
        fold_val_features = extract_features(fold_val_users, train_tx)
        
        # 2. Shared cards (fold-aware)
        fold_train_features = fold_shared_card_features(fold_train_tx, fold_train_features, train_tx)
        fold_val_features = fold_shared_card_features(fold_train_tx, fold_val_features, train_tx)
        
        # 3. Target encoding (fold-aware)
        fold_train_features, fold_val_features, _, _ = fold_target_encoding(
            fold_train_features, fold_val_features
        )
        
        # 4. Risk encoding (fold-aware)
        fold_train_features, fold_val_features, _, _ = fold_risk_encoding(
            fold_train_features, fold_val_features
        )
        
        if final_feature_cols is None:
            final_feature_cols = get_feature_cols(fold_train_features)
            
        X_tr = fold_train_features[final_feature_cols].values
        y_tr = fold_train_features['is_fraud'].values
        X_val = fold_val_features[final_feature_cols].values
        y_val = fold_val_features['is_fraud'].values
        
        folds_data.append({
            'X_tr': X_tr, 'y_tr': y_tr,
            'X_val': X_val, 'y_val': y_val,
            'val_idx': val_idx
        })
        
    print("Finished precomputing folds.\n")
    return folds_data, len(train_users), final_feature_cols, y

def run_optimization():
    print("=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    folds_data, total_samples, feature_cols, y_true = prepare_cv_folds()
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 30.0),
            'max_bin': trial.suggest_int('max_bin', 63, 511),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1,
        }
        
        if params['num_leaves'] > 2**params['max_depth'] - 1:
            params['num_leaves'] = 2**params['max_depth'] - 1

        oof_probas = np.zeros(total_samples)
        
        for fold_data in folds_data:
            model = lgb.LGBMClassifier(**params)
            model.fit(
                fold_data['X_tr'], fold_data['y_tr'],
                eval_set=[(fold_data['X_val'], fold_data['y_val'])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            )
            oof_probas[fold_data['val_idx']] = model.predict_proba(fold_data['X_val'])[:, 1]

        thresholds = np.arange(0.05, 0.95, 0.01)
        f1_scores = [f1_score(y_true, (oof_probas >= t).astype(int)) for t in thresholds]
        f1 = max(f1_scores)
        return f1
    
    # Restrict log output from optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="lgbm_fraud_cv")
    
    print(f"Running {N_TRIALS} trials...")
    # Wrap in progress bar or print statements manually since show_progress_bar in CLI might clutter
    def print_callback(study, trial):
        if trial.number % 10 == 0:
            print(f"Trial {trial.number}/{N_TRIALS} finished with F1: {trial.value:.4f} and parameters: {trial.params}")

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_callback])
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETED")
    print(f"Best OOF F1 (dynamic threshold): {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    Path("models").mkdir(parents=True, exist_ok=True)
    with open('models/best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
        print("\nBest parameters saved to models/best_params.json")

if __name__ == "__main__":
    run_optimization()
