import pandas as pd
import numpy as np
import os
from pathlib import Path

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

def check_name_email_match_vectorized(df):
    """
    Vectorized check if any word from card_holder (>2 chars) is in email.
    Returns a series with id_user as index and bool as value.
    """
    # Create a working copy with only necessary columns
    working_df = df[['id_user', 'card_holder', 'email']].drop_duplicates().copy()
    working_df = working_df.dropna(subset=['card_holder', 'email'])
    
    # Normalize
    working_df['card_holder'] = working_df['card_holder'].str.lower()
    working_df['email'] = working_df['email'].str.lower()
    
    # Tokenize card_holder
    tokens = working_df['card_holder'].str.split()
    working_df['tokens'] = tokens
    
    # Explode tokens to individual rows
    exploded = working_df.explode('tokens')
    
    # Filter tokens by length
    exploded = exploded[exploded['tokens'].str.len() > 2].copy()
    
    if exploded.empty:
        return pd.Series(False, index=df['id_user'].unique())
    
    # Check if token is in email
    # Note: str.contains with regex=False is faster but we need per-row match
    # Since we exploded, we can use a clever vectorized comparison if we hack it, 
    # but the most robust way in pandas without apply is iterating over unique tokens 
    # OR using a vectorized list comprehension if tokens count isn't massive.
    # Actually, a simple vectorized loop over exploded rows is still faster than apply on the whole df.
    
    exploded['match'] = [t in e for t, e in zip(exploded['tokens'], exploded['email'])]
    
    # Aggregate back to user
    user_match = exploded.groupby('id_user')['match'].any()
    return user_match

def _normalize_card_type(s):
    """Normalize card_type to a few categories."""
    if pd.isna(s):
        return 'UNKNOWN'
    s = str(s).upper().strip()
    if 'PREPAID' in s:
        return 'PREPAID'
    if 'CREDIT' in s and 'DEBIT' not in s:
        return 'CREDIT'
    if 'DEBIT' in s:
        return 'DEBIT'
    return 'OTHER'


def _normalize_card_brand(s):
    """Normalize card_brand to top brands."""
    if pd.isna(s):
        return 'UNKNOWN'
    s = str(s).upper().strip()
    if 'VISA' in s:
        return 'VISA'
    if 'MASTER' in s or s == 'MC' or 'MC_' in s:
        return 'MASTERCARD'
    if 'AMEX' in s or 'AMERICAN EXPRESS' in s:
        return 'AMEX'
    if 'DISCOVER' in s:
        return 'DISCOVER'
    if 'MAESTRO' in s:
        return 'MAESTRO'
    return 'OTHER'


def extract_features(users_df, tx_df):
    """
    Extract user-level features from transactions and user info.
    Common logic for both train and test.
    """
    print(f"Extracting features for {len(users_df)} users...")
    
    # 1. Merge basic info needed for features
    df = tx_df.merge(users_df[['id_user', 'reg_country', 'email', 'timestamp_reg']], on='id_user', how='left')
    
    # Normalize card_type and card_brand for later use
    df['card_type_norm'] = df['card_type'].apply(_normalize_card_type)
    df['card_brand_norm'] = df['card_brand'].apply(_normalize_card_brand)
    
    # 2. Time-based features
    # Instant registration: first tx within 60s
    first_tx = df.groupby('id_user')['timestamp_tr'].min().reset_index()
    first_tx = first_tx.merge(users_df[['id_user', 'timestamp_reg']], on='id_user')
    first_tx['seconds_to_first_tx'] = (first_tx['timestamp_tr'] - first_tx['timestamp_reg']).dt.total_seconds()
    is_instant = (first_tx.set_index('id_user')['seconds_to_first_tx'] < 60)
    
    # Night tx: 00:00 - 06:00
    df['hour'] = df['timestamp_tr'].dt.hour
    df['is_night'] = df['hour'].between(0, 5)
    has_night_tx = df.groupby('id_user')['is_night'].any()
    
    # NEW: Transaction velocity & timing features
    tx_times = df.sort_values(['id_user', 'timestamp_tr']).copy()
    tx_times['prev_ts'] = tx_times.groupby('id_user')['timestamp_tr'].shift(1)
    tx_times['gap_hours'] = (tx_times['timestamp_tr'] - tx_times['prev_ts']).dt.total_seconds() / 3600.0
    
    time_agg = df.groupby('id_user').agg(
        first_ts=('timestamp_tr', 'min'),
        last_ts=('timestamp_tr', 'max'),
        tx_hour_std=('hour', 'std'),
    )
    time_agg['tx_time_span_hours'] = (time_agg['last_ts'] - time_agg['first_ts']).dt.total_seconds() / 3600.0
    time_agg = time_agg.drop(columns=['first_ts', 'last_ts'])
    time_agg['tx_hour_std'] = time_agg['tx_hour_std'].fillna(0)
    
    gap_agg = tx_times.groupby('id_user')['gap_hours'].agg(
        mean_hours_between_tx='mean',
        min_hours_between_tx='min',
    )
    
    # Weekend ratio
    df['is_weekend'] = df['timestamp_tr'].dt.dayofweek.isin([5, 6]).astype(int)
    weekend_ratio = df.groupby('id_user')['is_weekend'].mean().rename('weekend_tx_ratio')
    
    # 3. Geo features
    df['country_mismatch'] = (df['card_country'] != df['reg_country']) & df['card_country'].notna()
    is_country_mismatch = df.groupby('id_user')['country_mismatch'].any()
    # NEW: continuous mismatch rates
    country_mismatch_rate = df.groupby('id_user')['country_mismatch'].mean().rename('country_mismatch_rate')
    
    df['pay_country_mismatch'] = (df['payment_country'] != df['reg_country']) & df['payment_country'].notna()
    is_pay_mismatch = df.groupby('id_user')['pay_country_mismatch'].any()
    pay_mismatch_rate = df.groupby('id_user')['pay_country_mismatch'].mean().rename('pay_country_mismatch_rate')
    
    # 4. Behavioral features
    stats = df.groupby('id_user').agg(
        total_transactions=('amount', 'count'),
        total_unique_cards=('card_mask_hash', 'nunique'),
        success_count=('status', lambda x: (x == 'success').sum()),
        has_fraud_error=('error_group', lambda x: (x == 'fraud').any()),
        max_amount=('amount', 'max'),
        std_amount=('amount', 'std'),
        unique_holders=('card_holder', 'nunique'),
        unique_pay_countries=('payment_country', 'nunique'),
        card_init_count=('transaction_type', lambda x: (x == 'card_init').sum()),
        failed_card_init_count=('transaction_type', lambda x: ((x == 'card_init') & (df.loc[x.index, 'status'] == 'fail')).sum()),
        # NEW: amount aggregations
        mean_amount=('amount', 'mean'),
        median_amount=('amount', 'median'),
        total_amount=('amount', 'sum'),
        min_amount=('amount', 'min'),
    )
    stats['success_rate'] = stats['success_count'] / stats['total_transactions']
    stats['failed_tx_count'] = stats['total_transactions'] - stats['success_count']
    stats['card_init_rate'] = stats['card_init_count'] / stats['total_transactions']
    stats['failed_card_init_rate'] = stats['failed_card_init_count'] / stats['total_transactions'].clip(lower=1)
    stats['cards_per_holder'] = stats['total_unique_cards'] / stats['unique_holders'].clip(lower=1)
    stats['holder_change_rate'] = stats['unique_holders'] / stats['total_transactions'].clip(lower=1)
    stats['std_amount'] = stats['std_amount'].fillna(0)
    # NEW: derived amount features
    stats['amount_range'] = stats['max_amount'] - stats['min_amount']
    stats['amount_cv'] = stats['std_amount'] / stats['mean_amount'].clip(lower=0.01)
    
    # NEW: Error pattern features
    error_stats = df.groupby('id_user').agg(
        unique_error_groups=('error_group', 'nunique'),
        antifraud_error_count=('error_group', lambda x: (x == 'antifraud').sum()),
        fraud_error_count=('error_group', lambda x: (x == 'fraud').sum()),
    )
    error_stats['antifraud_error_rate'] = error_stats['antifraud_error_count'] / stats['total_transactions'].clip(lower=1)
    
    # NEW: Card diversity features
    card_div = df.groupby('id_user').agg(
        unique_card_brands=('card_brand_norm', 'nunique'),
        unique_card_types=('card_type_norm', 'nunique'),
        unique_currencies=('currency', 'nunique'),
        prepaid_count=('card_type_norm', lambda x: (x == 'PREPAID').sum()),
    )
    card_div['prepaid_card_ratio'] = card_div['prepaid_count'] / stats['total_transactions'].clip(lower=1)
    card_div = card_div.drop(columns=['prepaid_count'])
    
    # NEW: Transaction type ratios
    tx_type_stats = df.groupby('id_user').agg(
        recurring_count=('transaction_type', lambda x: (x == 'card_recurring').sum()),
        digital_wallet_count=('transaction_type', lambda x: x.isin(['google-pay', 'apple-pay']).sum()),
    )
    tx_type_stats['recurring_tx_ratio'] = tx_type_stats['recurring_count'] / stats['total_transactions'].clip(lower=1)
    tx_type_stats['digital_wallet_ratio'] = tx_type_stats['digital_wallet_count'] / stats['total_transactions'].clip(lower=1)
    tx_type_stats = tx_type_stats.drop(columns=['recurring_count', 'digital_wallet_count'])
    
    # NEW: Mode categorical features per user (for target encoding later)
    mode_cats = df.groupby('id_user').agg(
        card_brand_main=('card_brand_norm', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'UNKNOWN'),
        card_type_main=('card_type_norm', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'UNKNOWN'),
        currency_main=('currency', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'UNKNOWN'),
    )
    
    # 5. Token match features (vectorized)
    has_name_email_match = check_name_email_match_vectorized(df)
    
    # 6. Combine all features
    high_fraud_countries = ['Indonesia', 'Ghana', 'Zimbabwe', 'Nigeria', 'Ukraine']
    users_df['email_domain'] = users_df['email'].str.split('@').str[1].str.lower()
    features = users_df[['id_user', 'traffic_type', 'gender']].copy()
    if 'is_fraud' in users_df.columns:
        features['is_fraud'] = users_df['is_fraud']
        
    features = features.set_index('id_user')
    
    # Original features
    features['is_instant_registration'] = is_instant.astype(int)
    features['has_night_tx'] = has_night_tx.astype(int)
    features['is_country_mismatch'] = is_country_mismatch.astype(int)
    features['is_pay_mismatch'] = is_pay_mismatch.astype(int)
    features['is_high_fraud_country'] = users_df.set_index('id_user')['reg_country'].isin(high_fraud_countries).astype(int)
    features['reg_country'] = users_df.set_index('id_user')['reg_country']
    features['email_domain'] = users_df.set_index('id_user')['email_domain']
    features['total_transactions'] = stats['total_transactions']
    features['total_unique_cards'] = stats['total_unique_cards']
    features['success_rate'] = stats['success_rate']
    features['failed_tx_count'] = stats['failed_tx_count']
    features['has_fraud_error'] = stats['has_fraud_error'].astype(int)
    features['has_name_email_match'] = has_name_email_match.reindex(features.index).fillna(False).astype(int)
    features['max_amount'] = stats['max_amount']
    features['std_amount'] = stats['std_amount']
    features['unique_holders'] = stats['unique_holders']
    features['unique_pay_countries'] = stats['unique_pay_countries']
    features['card_init_rate'] = stats['card_init_rate']
    features['failed_card_init_rate'] = stats['failed_card_init_rate']
    features['cards_per_holder'] = stats['cards_per_holder']
    features['holder_change_rate'] = stats['holder_change_rate']
    
    # NEW: Amount aggregations
    features['mean_amount'] = stats['mean_amount']
    features['median_amount'] = stats['median_amount']
    features['total_amount'] = stats['total_amount']
    features['amount_range'] = stats['amount_range']
    features['amount_cv'] = stats['amount_cv']
    
    # NEW: Transaction velocity & timing
    features['tx_time_span_hours'] = time_agg['tx_time_span_hours']
    features['tx_hour_std'] = time_agg['tx_hour_std']
    features['mean_hours_between_tx'] = gap_agg['mean_hours_between_tx']
    features['min_hours_between_tx'] = gap_agg['min_hours_between_tx']
    features['weekend_tx_ratio'] = weekend_ratio
    
    # NEW: Continuous mismatch rates
    features['country_mismatch_rate'] = country_mismatch_rate
    features['pay_country_mismatch_rate'] = pay_mismatch_rate
    
    # NEW: Error patterns
    features['unique_error_groups'] = error_stats['unique_error_groups']
    features['antifraud_error_count'] = error_stats['antifraud_error_count']
    features['antifraud_error_rate'] = error_stats['antifraud_error_rate']
    features['fraud_error_count'] = error_stats['fraud_error_count']
    
    # NEW: Card diversity
    features['unique_card_brands'] = card_div['unique_card_brands']
    features['unique_card_types'] = card_div['unique_card_types']
    features['unique_currencies'] = card_div['unique_currencies']
    features['prepaid_card_ratio'] = card_div['prepaid_card_ratio']
    
    # NEW: Transaction type ratios
    features['recurring_tx_ratio'] = tx_type_stats['recurring_tx_ratio']
    features['digital_wallet_ratio'] = tx_type_stats['digital_wallet_ratio']
    
    # NEW: Mode categoricals (for target encoding in main)
    features['card_brand_main'] = mode_cats['card_brand_main']
    features['card_type_main'] = mode_cats['card_type_main']
    features['currency_main'] = mode_cats['currency_main']
    
    # Fill NaNs for users with no transactions
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_fraud' in numeric_cols:
        numeric_cols.remove('is_fraud')
    features[numeric_cols] = features[numeric_cols].fillna(0)
    
    # Fill NaN categoricals
    for col in ['card_brand_main', 'card_type_main', 'currency_main']:
        features[col] = features[col].fillna('UNKNOWN')
    
    return features.reset_index()

def apply_target_encoding(train_df, test_df, target_col='is_fraud', cat_col='traffic_type'):
    """
    Fit target encoding on train data and map it to both train and test.
    Avoids data leakage by using only train signal.
    """
    print(f"Applying Target Encoding to {cat_col}...")
    
    # Calculate global mean for smoothing or filling missing categories in test
    global_mean = train_df[target_col].mean()
    
    # Calculate mean per category on train
    mapping = train_df.groupby(cat_col)[target_col].mean()
    mapping_dict = mapping.to_dict()
    
    # Transform
    train_df[f'{cat_col}_enc'] = train_df[cat_col].map(mapping_dict).fillna(global_mean)
    test_df[f'{cat_col}_enc'] = test_df[cat_col].map(mapping_dict).fillna(global_mean)
    
    # Drop original categorical column
    train_df = train_df.drop(columns=[cat_col])
    test_df = test_df.drop(columns=[cat_col])
    
    return train_df, test_df

def main():
    setup_directories()
    
    # 1. Load
    train_users, train_tx, test_users, test_tx = load_data()
    
    # 2. Extract common features (Leakage-free as they are based on internal profile/transactions)
    train_features = extract_features(train_users, train_tx)
    test_features = extract_features(test_users, test_tx)
    
    # 3. Handle categorical encoding (Strict Fit on Train)
    cat_cols = ['traffic_type', 'reg_country', 'email_domain',
                'gender', 'card_brand_main', 'card_type_main', 'currency_main']
    for col in cat_cols:
        train_features, test_features = apply_target_encoding(
            train_features, test_features, cat_col=col
        )
    
    # 4. Save
    print("Saving processed features...")
    train_features.to_csv("data/processed/train_features.csv", index=False)
    test_features.to_csv("data/processed/test_features.csv", index=False)
    
    print("\n✅ Feature Engineering Complete!")
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(f"Features saved to data/processed/")

if __name__ == "__main__":
    main()
