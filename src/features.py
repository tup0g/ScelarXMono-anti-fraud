import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path


def setup_directories():
    """Create processed data directory if it doesn't exist."""
    Path("data/processed").mkdir(parents=True, exist_ok=True)


def load_data_sqlite(db_path: str = "data/sql/train_data.db"):
    """Load training data from SQLite database."""
    print(f"Loading data from {db_path}...")
    conn = sqlite3.connect(db_path)

    users = pd.read_sql("SELECT * FROM users", conn)
    transactions = pd.read_sql("SELECT * FROM transactions", conn)

    conn.close()

    # Convert timestamps
    users['timestamp_reg'] = pd.to_datetime(users['timestamp_reg'], format='ISO8601', utc=True)
    transactions['timestamp_tr'] = pd.to_datetime(transactions['timestamp_tr'], format='ISO8601', utc=True)

    print(f"  Users: {len(users):,}, Transactions: {len(transactions):,}")
    return users, transactions


def load_data_csv():
    """Load data from CSV files (for test data without labels)."""
    print("Loading CSV data...")
    test_users = pd.read_csv("data/raw/test_users.csv")
    test_tx = pd.read_csv("data/raw/test_transactions.csv")

    test_users['timestamp_reg'] = pd.to_datetime(test_users['timestamp_reg'], format='ISO8601', utc=True)
    test_tx['timestamp_tr'] = pd.to_datetime(test_tx['timestamp_tr'], format='ISO8601', utc=True)

    print(f"  Test users: {len(test_users):,}, Test transactions: {len(test_tx):,}")
    return test_users, test_tx


def check_name_email_match_vectorized(df):
    """
    Vectorized check if any word from card_holder (>2 chars) is in email.
    Returns a series with id_user as index and bool as value.
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

    user_match = exploded.groupby('id_user')['match'].any()
    return user_match


def extract_features(users_df, tx_df):
    """
    Extract comprehensive user-level features from transactions and user info.
    Produces ~25 features for fraud detection.
    """
    print(f"Extracting features for {len(users_df):,} users...")

    # Merge user info into transactions for context
    df = tx_df.merge(
        users_df[['id_user', 'reg_country', 'email', 'timestamp_reg']],
        on='id_user', how='left'
    )

    # ========== TIME FEATURES ==========

    # Seconds to first transaction after registration
    first_tx = df.groupby('id_user')['timestamp_tr'].min().reset_index()
    first_tx = first_tx.merge(users_df[['id_user', 'timestamp_reg']], on='id_user')
    first_tx['seconds_to_first_tx'] = (first_tx['timestamp_tr'] - first_tx['timestamp_reg']).dt.total_seconds()
    is_instant = (first_tx.set_index('id_user')['seconds_to_first_tx'] < 60)

    # Night transactions (00:00 - 06:00)
    df['is_night'] = df['timestamp_tr'].dt.hour.between(0, 5)
    has_night_tx = df.groupby('id_user')['is_night'].any()
    pct_night_tx = df.groupby('id_user')['is_night'].mean()

    # Active days & tx velocity
    df['tx_date'] = df['timestamp_tr'].dt.date
    active_days = df.groupby('id_user')['tx_date'].nunique()
    total_tx = df.groupby('id_user').size()
    tx_per_day = total_tx / active_days.clip(lower=1)

    # Time span of activity
    time_span = df.groupby('id_user')['timestamp_tr'].agg(lambda x: (x.max() - x.min()).total_seconds() / 3600)

    # ========== GEO FEATURES ==========

    df['country_mismatch'] = (df['card_country'] != df['reg_country']) & df['card_country'].notna()
    is_country_mismatch = df.groupby('id_user')['country_mismatch'].any()
    pct_country_mismatch = df.groupby('id_user')['country_mismatch'].mean()

    df['pay_country_mismatch'] = (df['payment_country'] != df['reg_country']) & df['payment_country'].notna()
    is_pay_mismatch = df.groupby('id_user')['pay_country_mismatch'].any()

    n_card_countries = df.groupby('id_user')['card_country'].nunique()
    n_payment_countries = df.groupby('id_user')['payment_country'].nunique()

    # ========== AMOUNT FEATURES ==========

    amount_stats = df.groupby('id_user')['amount'].agg(
        avg_amount='mean',
        max_amount='max',
        std_amount='std',
        min_amount='min',
        total_amount='sum',
    )
    amount_stats['std_amount'] = amount_stats['std_amount'].fillna(0)
    amount_stats['amount_range'] = amount_stats['max_amount'] - amount_stats['min_amount']

    # ========== STATUS & ERROR FEATURES ==========

    stats = df.groupby('id_user').agg(
        total_transactions=('amount', 'count'),
        total_unique_cards=('card_mask_hash', 'nunique'),
        success_count=('status', lambda x: (x == 'success').sum()),
    )
    stats['success_rate'] = stats['success_count'] / stats['total_transactions']
    stats['failed_tx_count'] = stats['total_transactions'] - stats['success_count']

    # Error type features
    df['is_fraud_error'] = df['error_group'] == 'fraud'
    df['is_antifraud_error'] = df['error_group'] == 'antifraud'
    df['has_error'] = df['error_group'].notna()

    error_stats = df.groupby('id_user').agg(
        has_fraud_error=('is_fraud_error', 'any'),
        has_antifraud_error=('is_antifraud_error', 'any'),
        n_errors=('has_error', 'sum'),
        n_error_types=('error_group', 'nunique'),
        pct_fraud_errors=('is_fraud_error', 'mean'),
        pct_antifraud_errors=('is_antifraud_error', 'mean'),
    )
    error_stats['error_rate'] = error_stats['n_errors'] / stats['total_transactions']

    # ========== CARD & PAYMENT DIVERSITY ==========

    n_card_brands = df.groupby('id_user')['card_brand'].nunique()
    n_currencies = df.groupby('id_user')['currency'].nunique()
    n_transaction_types = df.groupby('id_user')['transaction_type'].nunique()

    # Transaction type ratios
    tx_type_counts = df.groupby(['id_user', 'transaction_type']).size().unstack(fill_value=0)
    tx_type_totals = tx_type_counts.sum(axis=1)
    pct_card_init = tx_type_counts.get('card_init', pd.Series(0, index=tx_type_counts.index)) / tx_type_totals
    pct_recurring = tx_type_counts.get('card_recurring', pd.Series(0, index=tx_type_counts.index)) / tx_type_totals

    # Card type diversity
    n_card_types = df.groupby('id_user')['card_type'].nunique()

    # ========== NAME-EMAIL MATCH ==========

    has_name_email_match = check_name_email_match_vectorized(df)

    # ========== USER PROFILE FEATURES ==========

    gender_enc = (users_df.set_index('id_user')['gender'] == 'male').astype(int)

    # ========== COMBINE ALL FEATURES ==========

    features = users_df[['id_user', 'traffic_type']].copy()
    if 'is_fraud' in users_df.columns:
        features['is_fraud'] = users_df['is_fraud']

    features = features.set_index('id_user')

    # Time features
    features['is_instant_registration'] = is_instant.astype(int)
    features['has_night_tx'] = has_night_tx.astype(int)
    features['pct_night_tx'] = pct_night_tx
    features['active_days'] = active_days
    features['tx_per_day'] = tx_per_day
    features['time_span_hours'] = time_span

    # Geo features
    features['is_country_mismatch'] = is_country_mismatch.astype(int)
    features['pct_country_mismatch'] = pct_country_mismatch
    features['is_pay_mismatch'] = is_pay_mismatch.astype(int)
    features['n_card_countries'] = n_card_countries
    features['n_payment_countries'] = n_payment_countries

    # Amount features
    features['avg_amount'] = amount_stats['avg_amount']
    features['max_amount'] = amount_stats['max_amount']
    features['std_amount'] = amount_stats['std_amount']
    features['total_amount'] = amount_stats['total_amount']
    features['amount_range'] = amount_stats['amount_range']

    # Transaction stats
    features['total_transactions'] = stats['total_transactions']
    features['total_unique_cards'] = stats['total_unique_cards']
    features['success_rate'] = stats['success_rate']
    features['failed_tx_count'] = stats['failed_tx_count']

    # Error features
    features['has_fraud_error'] = error_stats['has_fraud_error'].astype(int)
    features['has_antifraud_error'] = error_stats['has_antifraud_error'].astype(int)
    features['n_error_types'] = error_stats['n_error_types']
    features['pct_fraud_errors'] = error_stats['pct_fraud_errors']
    features['pct_antifraud_errors'] = error_stats['pct_antifraud_errors']
    features['error_rate'] = error_stats['error_rate']

    # Card & payment diversity
    features['n_card_brands'] = n_card_brands
    features['n_currencies'] = n_currencies
    features['n_transaction_types'] = n_transaction_types
    features['n_card_types'] = n_card_types
    features['pct_card_init'] = pct_card_init
    features['pct_recurring'] = pct_recurring

    # Name-email match
    features['has_name_email_match'] = has_name_email_match.reindex(features.index).fillna(False).astype(int)

    # User profile
    features['gender_enc'] = gender_enc

    # Fill NaNs
    numeric_cols = features.select_dtypes(include=['number']).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)

    return features.reset_index()


def apply_target_encoding(train_df, test_df, target_col='is_fraud', cat_col='traffic_type'):
    """
    Fit target encoding on train data and map it to both train and test.
    Avoids data leakage by using only train signal.
    """
    print(f"Applying Target Encoding to '{cat_col}'...")

    global_mean = train_df[target_col].mean()
    mapping = train_df.groupby(cat_col)[target_col].mean()
    mapping_dict = mapping.to_dict()

    train_df[f'{cat_col}_enc'] = train_df[cat_col].map(mapping_dict).fillna(global_mean)
    test_df[f'{cat_col}_enc'] = test_df[cat_col].map(mapping_dict).fillna(global_mean)

    train_df = train_df.drop(columns=[cat_col])
    test_df = test_df.drop(columns=[cat_col])

    return train_df, test_df


def main():
    setup_directories()

    # 1. Load from SQLite (train) and CSV (test)
    train_users, train_tx = load_data_sqlite("data/sql/train_data.db")
    test_users, test_tx = load_data_csv()

    # 2. Extract features
    train_features = extract_features(train_users, train_tx)
    test_features = extract_features(test_users, test_tx)

    print(f"\nTrain raw features: {train_features.shape}")
    print(f"Test raw features:  {test_features.shape}")

    # 3. Target encoding for categorical columns
    train_features, test_features = apply_target_encoding(train_features, test_features)

    # 4. Save
    print("Saving processed features...")
    train_features.to_csv("data/processed/train_features.csv", index=False)
    test_features.to_csv("data/processed/test_features.csv", index=False)

    print(f"\n✅ Feature Engineering Complete!")
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape:  {test_features.shape}")
    print(f"Feature columns: {[c for c in train_features.columns if c not in ['id_user', 'is_fraud']]}")


if __name__ == "__main__":
    main()
