import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def setup_directories():
    """Create processed data directory if it doesn't exist."""
    Path("data/processed").mkdir(parents=True, exist_ok=True)

def load_data():
    """Load raw data files."""
    print("Loading raw data...")
    train_users = pd.read_csv("data/raw/train_users.csv")
    train_tx = pd.read_csv("data/raw/train_transactions.csv")
    test_users = pd.read_csv("data/raw/test_users.csv")
    test_tx = pd.read_csv("data/raw/test_transactions.csv")
    
    # Pre-convert timestamps to datetime (UTC)
    for df in [train_users, test_users]:
        df['timestamp_reg'] = pd.to_datetime(df['timestamp_reg'], format='ISO8601', utc=True)
    for df in [train_tx, test_tx]:
        df['timestamp_tr'] = pd.to_datetime(df['timestamp_tr'], format='ISO8601', utc=True)
        
    return train_users, train_tx, test_users, test_tx


def compute_entropy(series):
    counts = series.value_counts(normalize=True)
    return -(counts * np.log(counts + 1e-10)).sum()

def check_name_email_match_vectorized(df):
    """
    Vectorized check if any word from card_holder (>2 chars) is in email.
    """
    working_df = df[['id_user', 'card_holder', 'email']].drop_duplicates().copy()
    working_df = working_df.dropna(subset=['card_holder', 'email'])
    
    working_df['card_holder'] = working_df['card_holder'].str.lower()
    working_df['email'] = working_df['email'].str.lower()
    
    tokens = working_df['card_holder'].str.split()
    working_df['tokens'] = tokens
    exploded = working_df.explode('tokens')
    exploded = exploded[exploded['tokens'].str.len() > 2].copy()
    
    if exploded.empty:
        return pd.Series(False, index=df['id_user'].unique())
    
    exploded['match'] = [t in e for t, e in zip(exploded['tokens'], exploded['email'])]
    return exploded.groupby('id_user')['match'].any()

def extract_features(users_df, tx_df):
    """
    Extract user-level features from transactions and user info.
    """
    print(f"Extracting features for {len(users_df)} users...")
    
    # 1. Merge basic info needed for features
    # Use inner join: only keep transactions for users in users_df
    # This is critical for fold-aware pipeline where users_df is a subset
    df = tx_df.merge(users_df[['id_user', 'reg_country', 'email', 'timestamp_reg']], on='id_user', how='inner')
    
    # Calculate hours since registration
    df['hours_since_reg'] = (df['timestamp_tr'] - df['timestamp_reg']).dt.total_seconds() / 3600.0
    
    # 2. Time-based features
    # Instant registration: first tx within 60s
    first_tx_df = df.sort_values('timestamp_tr').groupby('id_user').first()
    is_instant = (first_tx_df['hours_since_reg'] * 3600 < 60)
    
    # Night tx: 00:00 - 06:00
    df['is_night'] = df['timestamp_tr'].dt.hour.between(0, 5)
    has_night_tx = df.groupby('id_user')['is_night'].any()
    
    # NEW: Temporal features
    tx_24h = df[df['hours_since_reg'] <= 24].groupby('id_user').size()
    tx_7d = df[df['hours_since_reg'] <= 168].groupby('id_user').size()
    hours_to_first_tx_log = np.log1p(first_tx_df['hours_since_reg'].clip(lower=0))
    first_tx_failed = (first_tx_df['status'] != 'success').astype(int)
    first_tx_fraud_error = (first_tx_df['error_group'] == 'fraud').astype(int)
    
    # 3. Geo features
    df['country_mismatch'] = (df['card_country'] != df['reg_country']) & df['card_country'].notna()
    is_country_mismatch = df.groupby('id_user')['country_mismatch'].any()
    mismatch_rate = df.groupby('id_user')['country_mismatch'].sum() # to be divided by total
    
    df['pay_country_mismatch'] = (df['payment_country'] != df['reg_country']) & df['payment_country'].notna()
    is_pay_mismatch = df.groupby('id_user')['pay_country_mismatch'].any()
    
    # 4. Behavioral & Rate features
    stats = df.groupby('id_user').agg(
        total_transactions=('amount', 'count'),
        total_unique_cards=('card_mask_hash', 'nunique'),
        success_count=('status', lambda x: (x == 'success').sum()),
        has_fraud_error=('error_group', lambda x: (x == 'fraud').any()),
        fraud_error_count=('error_group', lambda x: (x == 'fraud').sum()),
        first_tx_time=('timestamp_tr', 'min'),
        last_tx_time=('timestamp_tr', 'max'),
        n_error_types=('error_group', 'nunique'),
        n_card_brands=('card_brand', 'nunique'),
        n_tx_types=('transaction_type', 'nunique')
    )
    
    stats['success_rate'] = stats['success_count'] / stats['total_transactions']
    stats['failed_tx_count'] = stats['total_transactions'] - stats['success_count']
    stats['cards_per_tx'] = stats['total_unique_cards'] / stats['total_transactions']
    
    stats['single_attempt'] = (stats['total_transactions'] == 1).astype(int)
    stats['days_active'] = (stats['last_tx_time'] - stats['first_tx_time']).dt.total_seconds() / 86400.0
    
    # NEW: Rates
    stats['fraud_error_rate'] = stats['fraud_error_count'] / stats['total_transactions']
    stats['country_mismatch_rate'] = mismatch_rate / stats['total_transactions']
    
    # 5. Token match features
    has_name_email_match = check_name_email_match_vectorized(df)
    
    # 6. Combine all features
    features = users_df[['id_user', 'traffic_type']].copy()
    if 'is_fraud' in users_df.columns:
        features['is_fraud'] = users_df['is_fraud']
    
    # Prepare gender and reg_country for OOF later
    features['gender'] = users_df['gender']
    features['reg_country'] = users_df['reg_country']
        
    features = features.set_index('id_user')
    
    features['is_instant_registration'] = is_instant.astype(int)
    features['has_night_tx'] = has_night_tx.astype(int)
    features['is_country_mismatch'] = is_country_mismatch.astype(int)
    features['is_pay_mismatch'] = is_pay_mismatch.astype(int)
    features['total_transactions'] = stats['total_transactions']
    features['total_unique_cards'] = stats['total_unique_cards']
    features['success_rate'] = stats['success_rate']
    features['failed_tx_count'] = stats['failed_tx_count']
    features['has_fraud_error'] = stats['has_fraud_error'].astype(int)
    features['cards_per_tx'] = stats['cards_per_tx']
    features['single_attempt'] = stats['single_attempt']
    features['days_active'] = stats['days_active']
    features['n_error_types'] = stats['n_error_types']
    features['n_card_brands'] = stats['n_card_brands']
    features['n_tx_types'] = stats['n_tx_types']
    features['has_name_email_match'] = has_name_email_match.reindex(features.index).fillna(False).astype(int)
    
    # NEW: First-window features (24h)
    tx_24h_df = df[df['hours_since_reg'] <= 24]
    
    # Reindex tx_24h and tx_7d to features.index for consistent shapes
    tx_24h = tx_24h.reindex(features.index).fillna(0)
    tx_7d = tx_7d.reindex(features.index).fillna(0)
    
    # success_count_24h
    success_24h = tx_24h_df[tx_24h_df['status'] == 'success'].groupby('id_user').size()
    success_24h = success_24h.reindex(features.index).fillna(0)
    features['success_rate_24h'] = np.where(tx_24h > 0, success_24h / tx_24h, 0)
    
    # failed_tx_count_24h
    features['failed_tx_count_24h'] = tx_24h - success_24h
    
    # fraud_error_rate_24h
    fraud_24h = tx_24h_df[tx_24h_df['error_group'] == 'fraud'].groupby('id_user').size()
    fraud_24h = fraud_24h.reindex(features.index).fillna(0)
    features['fraud_error_rate_24h'] = np.where(tx_24h > 0, fraud_24h / tx_24h, 0)
    
    # unique_cards_24h
    features['unique_cards_24h'] = tx_24h_df.groupby('id_user')['card_mask_hash'].nunique().reindex(features.index).fillna(0)
    
    # country_mismatch_rate_24h
    mismatch_24h = (tx_24h_df['card_country'] != tx_24h_df['reg_country']) & tx_24h_df['card_country'].notna()
    mismatch_24h_count = mismatch_24h.groupby(tx_24h_df['id_user']).sum().reindex(features.index).fillna(0)
    features['country_mismatch_rate_24h'] = np.where(tx_24h > 0, mismatch_24h_count / tx_24h, 0)
    
    # n_error_types_24h
    features['n_error_types_24h'] = tx_24h_df.groupby('id_user')['error_group'].nunique().reindex(features.index).fillna(0)

    # NEW: Time bucket features
    features['first_tx_le_1h'] = (first_tx_df['hours_since_reg'] <= 1).reindex(features.index).fillna(False).astype(int)
    features['first_tx_le_6h'] = (first_tx_df['hours_since_reg'] <= 6).reindex(features.index).fillna(False).astype(int)
    features['first_tx_le_24h'] = (first_tx_df['hours_since_reg'] <= 24).reindex(features.index).fillna(False).astype(int)
    features['first_tx_gt_7d'] = (first_tx_df['hours_since_reg'] > 168).reindex(features.index).fillna(False).astype(int)

    # NEW: Entropy / Diversity features
    features['payment_country_entropy'] = df.groupby('id_user')['payment_country'].apply(compute_entropy).reindex(features.index).fillna(0)
    features['error_group_entropy'] = df.groupby('id_user')['error_group'].apply(compute_entropy).reindex(features.index).fillna(0)
    features['transaction_type_entropy'] = df.groupby('id_user')['transaction_type'].apply(compute_entropy).reindex(features.index).fillna(0)
    
    features['card_country_nunique'] = df.groupby('id_user')['card_country'].nunique().reindex(features.index).fillna(0)
    features['payment_country_nunique'] = df.groupby('id_user')['payment_country'].nunique().reindex(features.index).fillna(0)
    features['currency_nunique'] = df.groupby('id_user')['currency'].nunique().reindex(features.index).fillna(0)
    features['card_type_nunique'] = df.groupby('id_user')['card_type'].nunique().reindex(features.index).fillna(0)

    # NEW: Amount behavior features
    features['first_tx_amount'] = first_tx_df['amount'].reindex(features.index).fillna(0)
    
    amount_std = df.groupby('id_user')['amount'].std().reindex(features.index).fillna(0)
    amount_mean = df.groupby('id_user')['amount'].mean().reindex(features.index).fillna(0)
    features['amount_cv'] = np.where(amount_mean > 0, amount_std / amount_mean, 0)
    
    small_amounts = (df['amount'] < 5).groupby(df['id_user']).sum().reindex(features.index).fillna(0)
    features['share_small_amounts'] = np.where(stats['total_transactions'] > 0, small_amounts / stats['total_transactions'], 0)
    
    features['max_amount_24h'] = tx_24h_df.groupby('id_user')['amount'].max().reindex(features.index).fillna(0)
    
    # share_repeated_amounts
    unique_amounts = df.groupby('id_user')['amount'].nunique().reindex(features.index).fillna(0)
    non_unique = stats['total_transactions'] - unique_amounts
    features['share_repeated_amounts'] = np.where(stats['total_transactions'] > 0, non_unique / stats['total_transactions'], 0)

    # Assign NEW temporal & rate features
    features['tx_24h'] = tx_24h
    features['tx_7d'] = tx_7d
    features['hours_to_first_tx_log'] = hours_to_first_tx_log
    features['first_tx_failed'] = first_tx_failed
    features['first_tx_fraud_error'] = first_tx_fraud_error
    features['fraud_error_rate'] = stats['fraud_error_rate']
    features['country_mismatch_rate'] = stats['country_mismatch_rate']
    
    # Fill NaNs for users with no transactions
    cols_to_fill = [
        'is_instant_registration', 'has_night_tx', 'is_country_mismatch', 
        'is_pay_mismatch', 'total_transactions', 'total_unique_cards', 
        'success_rate', 'failed_tx_count', 'has_fraud_error', 'has_name_email_match',
        'cards_per_tx', 'single_attempt', 'days_active',
        'n_error_types', 'n_card_brands', 'n_tx_types',
        'tx_24h', 'tx_7d', 'hours_to_first_tx_log', 
        'first_tx_failed', 'first_tx_fraud_error', 
        'fraud_error_rate', 'country_mismatch_rate',
        'success_rate_24h', 'failed_tx_count_24h', 'fraud_error_rate_24h',
        'unique_cards_24h', 'country_mismatch_rate_24h', 'n_error_types_24h',
        'first_tx_le_1h', 'first_tx_le_6h', 'first_tx_le_24h', 'first_tx_gt_7d',
        'payment_country_entropy', 'error_group_entropy', 'transaction_type_entropy',
        'card_country_nunique', 'payment_country_nunique', 'currency_nunique', 'card_type_nunique',
        'first_tx_amount', 'amount_cv', 'share_small_amounts', 'max_amount_24h', 'share_repeated_amounts'
    ]
    features[cols_to_fill] = features[cols_to_fill].fillna(0)
    
    return features.reset_index()

def apply_target_encoding(train_df, test_df, target_col='is_fraud', cat_col='traffic_type'):
    print(f"Applying Target Encoding to {cat_col}...")
    global_mean = train_df[target_col].mean()
    mapping = train_df.groupby(cat_col)[target_col].mean().to_dict()
    
    train_df[f'{cat_col}_enc'] = train_df[cat_col].map(mapping).fillna(global_mean)
    test_df[f'{cat_col}_enc'] = test_df[cat_col].map(mapping).fillna(global_mean)
    
    train_df = train_df.drop(columns=[cat_col])
    test_df = test_df.drop(columns=[cat_col])
    
    return train_df, test_df

def apply_oof_encoding(train_df, test_df, target_col='is_fraud', cat_cols=['reg_country', 'gender']):
    """Apply 5-fold OOF Target Encoding for categorical features to avoid leakage."""
    print(f"Applying OOF Target Encoding to {cat_cols}...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for col in cat_cols:
        # Check if feature has enough signal (correlation heuristic skip logic requested by user)
        # We encode it and then check correlation. If < 0.02 we drop it.
        train_df[f'{col}_risk_oof'] = np.nan
        
        global_mean = train_df[target_col].mean()
        col_means = train_df.groupby(col)[target_col].mean().to_dict()
        
        # Test mapping (uses full train data)
        test_df[f'{col}_risk_oof'] = test_df[col].map(col_means).fillna(global_mean)
        
        # Train OOF mapping
        for train_idx, val_idx in skf.split(train_df, train_df[target_col]):
            X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
            fold_means = X_tr.groupby(col)[target_col].mean().to_dict()
            train_df.loc[val_idx, f'{col}_risk_oof'] = X_val[col].map(fold_means).fillna(global_mean)
            
        train_df[f'{col}_risk_oof'] = train_df[f'{col}_risk_oof'].fillna(global_mean)
        
        # Check correlation
        corr = np.abs(np.corrcoef(train_df[f'{col}_risk_oof'], train_df[target_col])[0, 1])
        print(f"  {col}_risk_oof correlation with target: {corr:.4f}")
        if corr < 0.02:
            print(f"  Dropping {col}_risk_oof due to low correlation (< 0.02).")
            train_df = train_df.drop(columns=[f'{col}_risk_oof'])
            test_df = test_df.drop(columns=[f'{col}_risk_oof'])
            
        # Clean up original text column
        train_df = train_df.drop(columns=[col])
        test_df = test_df.drop(columns=[col])

    return train_df, test_df

def apply_shared_card_users_max(train_tx, test_tx, train_features, test_features):
    print("Applying shared_card_users_max...")
    # 1. Build card mapping from TRAIN only to avoid leakage
    card_user_counts = train_tx.groupby('card_mask_hash')['id_user'].nunique()
    
    def process_shared(features_df, tx_df):
        user_cards = tx_df[['id_user', 'card_mask_hash']].drop_duplicates()
        user_cards['shared_users'] = user_cards['card_mask_hash'].map(card_user_counts).fillna(1) - 1
        
        shared_max = user_cards.groupby('id_user')['shared_users'].max().clip(lower=0)
        shared_mean = user_cards.groupby('id_user')['shared_users'].mean().clip(lower=0)
        
        # Number of shared cards (shared with ANY other user)
        n_shared = user_cards[user_cards['shared_users'] > 0].groupby('id_user').size()
        cards_shared_2 = user_cards[user_cards['shared_users'] >= 2].groupby('id_user').size()
        cards_shared_5 = user_cards[user_cards['shared_users'] >= 5].groupby('id_user').size()
        
        # Transactions on shared cards
        shared_card_hashes = user_cards[user_cards['shared_users'] > 0]['card_mask_hash']
        tx_on_shared = tx_df[tx_df['card_mask_hash'].isin(shared_card_hashes)].groupby('id_user').size()
        tx_total = tx_df.groupby('id_user').size().reindex(features_df['id_user']).fillna(0)
        
        # Reindex
        features_df['shared_card_users_max'] = shared_max.reindex(features_df['id_user']).fillna(0).values
        features_df['shared_card_users_mean'] = shared_mean.reindex(features_df['id_user']).fillna(0).values
        features_df['n_shared_cards'] = n_shared.reindex(features_df['id_user']).fillna(0).values
        features_df['cards_shared_2plus'] = cards_shared_2.reindex(features_df['id_user']).fillna(0).values
        features_df['cards_shared_5plus'] = cards_shared_5.reindex(features_df['id_user']).fillna(0).values
        
        tx_on_shared = tx_on_shared.reindex(features_df['id_user']).fillna(0)
        import numpy as np
        features_df['share_tx_on_shared_cards'] = np.where(tx_total > 0, tx_on_shared / tx_total, 0)
        
        return features_df
        
    train_features = process_shared(train_features, train_tx)
    test_features = process_shared(test_features, test_tx)
    
    return train_features, test_features

def main():
    setup_directories()
    
    train_users, train_tx, test_users, test_tx = load_data()
    
    # 1. Extract common features
    train_features = extract_features(train_users, train_tx)
    test_features = extract_features(test_users, test_tx)
    
    # 2. Add shared card features (Train-Only Mapping)
    train_features, test_features = apply_shared_card_users_max(train_tx, test_tx, train_features, test_features)
    
    # 3. Categorical Encoding (Traffic Type)
    train_features, test_features = apply_target_encoding(train_features, test_features)
    
    # 4. OOF Risk Encoding (reg_country, gender)
    train_features, test_features = apply_oof_encoding(train_features, test_features)
    
    print("\nValidating no leakage...")
    if 'is_fraud' in test_features.columns:
        print("ERROR: is_fraud found in test features!")
    else:
        print("✅ No target leakage in test features.")
    
    # 5. Save
    print("Saving processed features...")
    train_features.to_csv("data/processed/train_features.csv", index=False)
    test_features.to_csv("data/processed/test_features.csv", index=False)
    
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    print("\n✅ Feature Engineering Complete!")

if __name__ == '__main__':
    main()
