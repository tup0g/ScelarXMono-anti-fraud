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

def extract_features(users_df, tx_df):
    """
    Extract user-level features from transactions and user info.
    Common logic for both train and test.
    """
    print(f"Extracting features for {len(users_df)} users...")
    
    # 1. Merge basic info needed for features
    df = tx_df.merge(users_df[['id_user', 'reg_country', 'email', 'timestamp_reg']], on='id_user', how='left')
    
    # 2. Time-based features
    # Instant registration: first tx within 60s
    first_tx = df.groupby('id_user')['timestamp_tr'].min().reset_index()
    first_tx = first_tx.merge(users_df[['id_user', 'timestamp_reg']], on='id_user')
    first_tx['seconds_to_first_tx'] = (first_tx['timestamp_tr'] - first_tx['timestamp_reg']).dt.total_seconds()
    is_instant = (first_tx.set_index('id_user')['seconds_to_first_tx'] < 60)
    
    # Night tx: 00:00 - 06:00
    df['is_night'] = df['timestamp_tr'].dt.hour.between(0, 5)
    has_night_tx = df.groupby('id_user')['is_night'].any()
    
    # 3. Geo features
    df['country_mismatch'] = (df['card_country'] != df['reg_country']) & df['card_country'].notna()
    is_country_mismatch = df.groupby('id_user')['country_mismatch'].any()
    
    df['pay_country_mismatch'] = (df['payment_country'] != df['reg_country']) & df['payment_country'].notna()
    is_pay_mismatch = df.groupby('id_user')['pay_country_mismatch'].any()
    
    # 4. Behavioral features
    stats = df.groupby('id_user').agg(
        total_transactions=('amount', 'count'),
        total_unique_cards=('card_mask_hash', 'nunique'),
        success_count=('status', lambda x: (x == 'success').sum()),
        has_fraud_error=('error_group', lambda x: (x == 'fraud').any())
    )
    stats['success_rate'] = stats['success_count'] / stats['total_transactions']
    stats['failed_tx_count'] = stats['total_transactions'] - stats['success_count']
    
    # 5. Token match features (vectorized)
    has_name_email_match = check_name_email_match_vectorized(df)
    
    # 6. Combine all features
    features = users_df[['id_user', 'traffic_type']].copy()
    if 'is_fraud' in users_df.columns:
        features['is_fraud'] = users_df['is_fraud']
        
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
    features['has_name_email_match'] = has_name_email_match.reindex(features.index).fillna(False).astype(int)
    
    # Fill NaNs for users with no transactions
    cols_to_fill = [
        'is_instant_registration', 'has_night_tx', 'is_country_mismatch', 
        'is_pay_mismatch', 'total_transactions', 'total_unique_cards', 
        'success_rate', 'failed_tx_count', 'has_fraud_error', 'has_name_email_match'
    ]
    features[cols_to_fill] = features[cols_to_fill].fillna(0)
    
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
    train_features, test_features = apply_target_encoding(train_features, test_features)
    
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
