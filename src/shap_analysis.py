import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Path("outputs/plots").mkdir(parents=True, exist_ok=True)
Path("outputs/analysis").mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════
# КРОК 1 — Завантаження моделі і даних з диску
# ═══════════════════════════════════════════════════════

def load_fold5_artifacts():
    """
    Завантажує fold 5 модель і дані збережені train_oof.py.
    Жодного перенавчання — просто читаємо з диску.
    """
    required_files = [
        'models/fold5_model.txt',
        'outputs/fold5_val_features.csv',
        'outputs/fold5_val_labels.csv',
        'models/best_threshold.json',
        'models/feature_cols.json',
    ]

    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print("❌ ПОМИЛКА: Відсутні файли:")
        for f in missing:
            print(f"   {f}")
        print("\nСпочатку запусти: python src/train_oof.py")
        sys.exit(1)

    print("Завантаження fold 5 артефактів з диску...")

    # Модель
    model = lgb.Booster(model_file='models/fold5_model.txt')

    # Дані валідації (fold 5 val — ~79,000 юзерів)
    X_val = pd.read_csv('outputs/fold5_val_features.csv')
    y_val = pd.read_csv('outputs/fold5_val_labels.csv').squeeze()

    # Threshold і feature cols
    with open('models/best_threshold.json', 'r') as f:
        best_threshold = json.load(f)['threshold']

    with open('models/feature_cols.json', 'r') as f:
        feature_cols = json.load(f)

    # Вирівнюємо колонки
    X_val = X_val[feature_cols]

    print(f"✅ Модель завантажена")
    print(f"✅ Val розмір: {len(X_val)} юзерів")
    print(f"✅ Фіч: {len(feature_cols)}")
    print(f"✅ Threshold: {best_threshold:.2f}")
    print(f"✅ Fraud юзерів у val: {y_val.sum()} "
          f"({y_val.mean()*100:.2f}%)")

    return model, X_val, y_val, best_threshold, feature_cols


# ═══════════════════════════════════════════════════════
# КРОК 2 — Базова оцінка моделі на val даних
# ═══════════════════════════════════════════════════════

def evaluate_model(model, X_val, y_val, best_threshold, feature_cols):
    """Оцінює поточну якість моделі на val fold 5."""
    print("\n" + "=" * 60)
    print("БАЗОВА ОЦІНКА МОДЕЛІ (fold 5 val)")
    print("=" * 60)

    val_proba = model.predict(X_val)  # LightGBM Booster повертає probas напряму

    # F1 при різних threshold
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_val, (val_proba >= t).astype(int)) for t in thresholds]
    best_t_local = thresholds[np.argmax(f1s)]
    best_f1_local = max(f1s)

    y_pred = (val_proba >= best_threshold).astype(int)

    print(f"\nПри OOF threshold ({best_threshold:.2f}):")
    print(f"  F1:        {f1_score(y_val, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"\nОптимальний threshold на цьому fold: "
          f"{best_t_local:.2f} → F1={best_f1_local:.4f}")

    # Розподіл probabilities
    fraud_proba = val_proba[y_val == 1]
    legit_proba = val_proba[y_val == 0]

    print(f"\nРозподіл probabilities:")
    print(f"  Fraud:  mean={fraud_proba.mean():.4f}, "
          f"median={np.median(fraud_proba):.4f}")
    print(f"  Legit:  mean={legit_proba.mean():.4f}, "
          f"median={np.median(legit_proba):.4f}")
    print(f"  % fraud > threshold:  "
          f"{(fraud_proba >= best_threshold).mean()*100:.1f}%  (recall)")
    print(f"  % legit > threshold:  "
          f"{(legit_proba >= best_threshold).mean()*100:.2f}%  (false positives)")

    # Сегментація fraud по впевненості моделі
    print(f"\nСегментація fraud за ймовірністю:")
    segments = [
        (0.0,  0.1,  "Confident miss   (proba < 0.10)"),
        (0.1,  best_threshold, f"Grey zone        (0.10 - {best_threshold:.2f})"),
        (best_threshold, 0.7,  f"Caught (low)     ({best_threshold:.2f} - 0.70)"),
        (0.7,  1.01, "Caught (strong)  (proba > 0.70)"),
    ]
    for lo, hi, label in segments:
        mask = (fraud_proba >= lo) & (fraud_proba < hi)
        print(f"  {label}: {mask.sum()} "
              f"({mask.mean()*100:.1f}%)")

    return val_proba, best_t_local


# ═══════════════════════════════════════════════════════
# КРОК 3 — SHAP аналіз
# ═══════════════════════════════════════════════════════

def compute_shap(model, X_val, feature_cols, sample_size=15000):
    """
    Рахує SHAP values на sample з val даних.
    15,000 юзерів — достатньо для стабільних результатів.
    """
    print(f"\n{'='*60}")
    print(f"SHAP АНАЛІЗ (sample={sample_size} юзерів)")
    print("=" * 60)

    X_shap = X_val.sample(n=min(sample_size, len(X_val)),
                           random_state=RANDOM_STATE)

    print("Обчислення SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    print("✅ SHAP values обчислено")
    return sv, X_shap


# ═══════════════════════════════════════════════════════
# КРОК 4 — Повна таблиця важливості фіч
# ═══════════════════════════════════════════════════════

def build_importance_table(sv, X_shap, feature_cols):
    """Будує повну таблицю з усіма метриками по кожній фічі."""
    print(f"\n{'='*60}")
    print("ПОВНА ТАБЛИЦЯ ВАЖЛИВОСТІ ФІЧ")
    print("=" * 60)

    shap_df = pd.DataFrame({
        'feature':       feature_cols,
        'mean_abs_shap': np.abs(sv).mean(axis=0),
        'mean_shap':     sv.mean(axis=0),
        'std_shap':      sv.std(axis=0),
        'max_shap':      sv.max(axis=0),
        'min_shap':      sv.min(axis=0),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    shap_df['direction'] = shap_df['mean_shap'].apply(
        lambda x: '→ FRAUD' if x > 0 else '→ LEGIT'
    )
    shap_df['rank'] = range(1, len(shap_df) + 1)

    # Друкуємо таблицю
    print(f"\n{'Rank':<5} {'Feature':<35} {'mean|SHAP|':<12} "
          f"{'Direction':<12} {'max_impact'}")
    print("-" * 72)
    for _, row in shap_df.iterrows():
        marker = (
            "✅" if row['mean_abs_shap'] >= 0.02 else
            "⚠️ " if row['mean_abs_shap'] >= 0.005 else
            "❌"
        )
        print(f"{int(row['rank']):<5} {row['feature']:<35} "
              f"{row['mean_abs_shap']:.5f}      "
              f"{row['direction']:<12} {row['max_shap']:.3f}  {marker}")

    return shap_df


# ═══════════════════════════════════════════════════════
# КРОК 5 — Категоризація фіч
# ═══════════════════════════════════════════════════════

def categorize_features(shap_df):
    """Ділить фічі на strong / medium / weak."""
    print(f"\n{'='*60}")
    print("КАТЕГОРІЇ ФІЧ")
    print("=" * 60)

    strong = shap_df[shap_df['mean_abs_shap'] >= 0.02]
    medium = shap_df[
        (shap_df['mean_abs_shap'] >= 0.005) &
        (shap_df['mean_abs_shap'] < 0.02)
    ]
    weak = shap_df[shap_df['mean_abs_shap'] < 0.005]

    print(f"\n✅ STRONG (>= 0.02): {len(strong)} фіч — зберігаємо обов'язково")
    for _, r in strong.iterrows():
        print(f"   {r['feature']}: {r['mean_abs_shap']:.5f} {r['direction']}")

    print(f"\n⚠️  MEDIUM (0.005-0.02): {len(medium)} фіч — зберігаємо")
    for _, r in medium.iterrows():
        print(f"   {r['feature']}: {r['mean_abs_shap']:.5f} {r['direction']}")

    print(f"\n❌ WEAK (< 0.005): {len(weak)} фіч — кандидати на видалення")
    weak_list = weak['feature'].tolist()
    for _, r in weak.iterrows():
        print(f"   {r['feature']}: {r['mean_abs_shap']:.5f}")

    print(f"\nВидалення {len(weak_list)} слабких фіч може:")
    print(f"  → Зменшити шум у моделі")
    print(f"  → Пришвидшити навчання")
    print(f"  → Потенційно покращити F1")

    return strong, medium, weak, weak_list


# ═══════════════════════════════════════════════════════
# КРОК 6 — Missed vs Caught аналіз
# ═══════════════════════════════════════════════════════

def missed_vs_caught_analysis(model, X_val, y_val,
                               best_threshold, feature_cols):
    """
    Порівнює профілі пропущених і спійманих шахраїв.
    Базується на ПОВНОМУ val fold (не тільки sample).
    """
    print(f"\n{'='*60}")
    print("MISSED vs CAUGHT АНАЛІЗ (повний val fold)")
    print("=" * 60)

    val_proba = model.predict(X_val)
    y_pred = (val_proba >= best_threshold).astype(int)

    caught = X_val[(y_val == 1) & (y_pred == 1)]
    missed = X_val[(y_val == 1) & (y_pred == 0)]

    print(f"\nСпіймано шахраїв:  {len(caught)} "
          f"({len(caught)/(len(caught)+len(missed))*100:.1f}%)")
    print(f"Пропущено шахраїв: {len(missed)} "
          f"({len(missed)/(len(caught)+len(missed))*100:.1f}%)")

    if len(missed) == 0 or len(caught) == 0:
        print("Недостатньо даних для порівняння")
        return

    comparison = pd.DataFrame({
        'missed_mean': missed[feature_cols].mean(),
        'caught_mean': caught[feature_cols].mean(),
    })
    comparison['abs_diff'] = abs(
        comparison['missed_mean'] - comparison['caught_mean']
    )
    comparison['ratio'] = (
        comparison['missed_mean'] /
        comparison['caught_mean'].replace(0, 1e-10)
    )
    comparison = comparison.sort_values('abs_diff', ascending=False)

    print(f"\nТоп-10 фіч які найбільше відрізняють "
          f"missed від caught:")
    print(f"{'Feature':<35} {'Missed':<12} {'Caught':<12} "
          f"{'Ratio':<10} {'Сигнал'}")
    print("-" * 75)
    for feat, row in comparison.head(10).iterrows():
        signal = (
            "missed БІЛЬШИЙ" if row['ratio'] > 1.2 else
            "caught БІЛЬШИЙ" if row['ratio'] < 0.8 else
            "схожі"
        )
        print(f"{feat:<35} {row['missed_mean']:.4f}       "
              f"{row['caught_mean']:.4f}       "
              f"{row['ratio']:.2f}x      {signal}")

    # Профіль missed шахраїв
    print(f"\nПрофіль пропущеного шахрая:")
    print(f"  success_rate:       "
          f"{missed['success_rate'].mean():.3f} vs "
          f"{caught['success_rate'].mean():.3f} (caught)")
    print(f"  total_transactions: "
          f"{missed['total_transactions'].mean():.1f} vs "
          f"{caught['total_transactions'].mean():.1f} (caught)")
    print(f"  total_unique_cards: "
          f"{missed['total_unique_cards'].mean():.2f} vs "
          f"{caught['total_unique_cards'].mean():.2f} (caught)")

    low_tx = (missed['total_transactions'] < 5).mean() * 100
    print(f"\n  % missed з < 5 транзакцій: {low_tx:.1f}% "
          f"← основна проблема")

    return comparison


# ═══════════════════════════════════════════════════════
# КРОК 7 — Графіки
# ═══════════════════════════════════════════════════════

def generate_plots(sv, X_shap, val_proba, y_val,
                   best_threshold, feature_cols, shap_df):
    """Генерує всі графіки для аналізу."""
    print(f"\n{'='*60}")
    print("ГЕНЕРАЦІЯ ГРАФІКІВ")
    print("=" * 60)

    # Plot 1 — Beeswarm (топ 20 фіч)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(sv, X_shap, max_display=20, show=False)
    plt.title("SHAP Beeswarm — напрямок впливу фіч (топ 20)",
              fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig('outputs/plots/shap_beeswarm.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ outputs/plots/shap_beeswarm.png")

    # Plot 2 — Bar chart (всі фічі)
    n_features = len(feature_cols)
    plt.figure(figsize=(12, max(8, n_features * 0.28)))
    shap.summary_plot(sv, X_shap, plot_type='bar',
                      max_display=n_features, show=False)
    plt.title("SHAP Bar — важливість всіх фіч", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig('outputs/plots/shap_bar_all.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ outputs/plots/shap_bar_all.png")

    # Plot 3 — Розподіл probabilities
    fraud_p = val_proba[y_val == 1]
    legit_p = val_proba[y_val == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(legit_p, bins=60, alpha=0.6,
                 color='green', label='Legit', density=True)
    axes[0].hist(fraud_p, bins=60, alpha=0.6,
                 color='red', label='Fraud', density=True)
    axes[0].axvline(best_threshold, color='black',
                    linestyle='--', linewidth=2,
                    label=f'Threshold={best_threshold:.2f}')
    axes[0].set_title('Розподіл ймовірностей: Fraud vs Legit')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].legend()

    axes[1].hist(fraud_p, bins=60, alpha=0.7,
                 color='red', density=True)
    axes[1].axvline(best_threshold, color='black',
                    linestyle='--', linewidth=2,
                    label=f'Threshold={best_threshold:.2f}')
    axes[1].set_title('Розподіл ймовірностей — тільки Fraud')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('outputs/plots/probability_distribution.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ outputs/plots/probability_distribution.png")

    # Plot 4 — Strong vs Weak фічі
    colors = shap_df['mean_abs_shap'].apply(
        lambda x: '#2ecc71' if x >= 0.02
        else '#f39c12' if x >= 0.005
        else '#e74c3c'
    )
    fig, ax = plt.subplots(figsize=(12, max(6, n_features * 0.28)))
    bars = ax.barh(shap_df['feature'][::-1],
                   shap_df['mean_abs_shap'][::-1],
                   color=colors[::-1])
    ax.axvline(0.005, color='orange', linestyle='--',
               alpha=0.7, label='Weak threshold (0.005)')
    ax.axvline(0.02, color='green', linestyle='--',
               alpha=0.7, label='Strong threshold (0.02)')
    ax.set_title('Категорії фіч: Strong / Medium / Weak')
    ax.set_xlabel('mean |SHAP value|')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/plots/feature_categories.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ outputs/plots/feature_categories.png")


# ═══════════════════════════════════════════════════════
# КРОК 8 — Рекомендації для покращення F1
# ═══════════════════════════════════════════════════════

def print_recommendations(weak_list, val_proba, y_val,
                           best_threshold, shap_df):
    """Конкретні рекомендації що робити далі."""
    print(f"\n{'='*60}")
    print("РЕКОМЕНДАЦІЇ ДЛЯ ПОКРАЩЕННЯ F1")
    print("=" * 60)

    fraud_p = val_proba[y_val == 1]
    grey_pct = ((fraud_p >= 0.1) &
                (fraud_p < best_threshold)).mean() * 100
    confident_miss_pct = (fraud_p < 0.1).mean() * 100
    caught_pct = (fraud_p >= best_threshold).mean() * 100

    print(f"\nПоточний стан:")
    print(f"  Спіймано:         {caught_pct:.1f}% шахраїв")
    print(f"  Сіра зона:        {grey_pct:.1f}% (майже спіймані)")
    print(f"  Впевнені промахи: {confident_miss_pct:.1f}% (no signal)")

    print(f"\n{'─'*50}")
    priority = 1

    if weak_list:
        print(f"\n{priority}. ВИДАЛИТИ {len(weak_list)} слабких фіч → "
              f"re-run Optuna")
        print(f"   {weak_list}")
        print(f"   Очікуваний приріст F1: +0.01-0.03")
        priority += 1

    if grey_pct > 15:
        print(f"\n{priority}. СІРА ЗОНА {grey_pct:.1f}% → нові фічі")
        print(f"   Ці шахраї МАЙЖЕ спіймані — потрібен додатковий сигнал")
        print(f"   Ідеї: email_domain_risk, device_pattern, session_features")
        print(f"   Очікуваний приріст F1: +0.02-0.05")
        priority += 1

    if confident_miss_pct > 10:
        print(f"\n{priority}. ВПЕВНЕНІ ПРОМАХИ {confident_miss_pct:.1f}% → "
              f"інший підхід")
        print(f"   Модель не має жодного сигналу для цих шахраїв")
        print(f"   Це 'тихі шахраї' з 1-3 транзакціями")
        print(f"   Ідеї: rule-based pre-filter для single_attempt юзерів")
        priority += 1

    print(f"\n{priority}. CatBoost benchmark")
    print(f"   Запусти CatBoost з дефолтними params → порівняй F1")
    print(f"   Очікуваний приріст F1: +0.01-0.03")
    priority += 1

    print(f"\n{priority}. Seed ensemble (найпростіший буст)")
    print(f"   5x LightGBM з різними random_state → усереднити probas")
    print(f"   Очікуваний приріст F1: +0.01-0.02 без змін в коді")

    # Топ-5 для Notion документу
    print(f"\n{'─'*50}")
    print(f"ТОП-5 ФІCH ДЛЯ NOTION ДОКУМЕНТУ:")
    for i, (_, row) in enumerate(shap_df.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']}: "
              f"{row['mean_abs_shap']:.5f} {row['direction']}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def run_shap_analysis():
    print("=" * 60)
    print("DEEP SHAP ANALYSIS")
    print("(базується на збереженій fold 5 моделі)")
    print("=" * 60)

    # 1. Завантажуємо готову модель — без перенавчання
    model, X_val, y_val, best_threshold, feature_cols = \
        load_fold5_artifacts()

    # 2. Базова оцінка
    val_proba, _ = evaluate_model(
        model, X_val, y_val, best_threshold, feature_cols)

    # 3. SHAP values
    sv, X_shap = compute_shap(model, X_val, feature_cols, sample_size=15000)

    # 4. Таблиця важливості
    shap_df = build_importance_table(sv, X_shap, feature_cols)

    # 5. Категорії
    strong, medium, weak, weak_list = categorize_features(shap_df)

    # 6. Missed vs Caught
    missed_vs_caught_analysis(
        model, X_val, y_val, best_threshold, feature_cols)

    # 7. Графіки
    generate_plots(sv, X_shap, val_proba, y_val,
                   best_threshold, feature_cols, shap_df)

    # 8. Рекомендації
    print_recommendations(
        weak_list, val_proba, y_val, best_threshold, shap_df)

    # 9. Зберігаємо CSV з повним аналізом
    shap_df.to_csv('outputs/analysis/shap_full_analysis.csv', index=False)
    print(f"\n✅ Збережено outputs/analysis/shap_full_analysis.csv")

    print(f"\n{'='*60}")
    print(f"АНАЛІЗ ЗАВЕРШЕНО")
    print(f"Наступний крок:")
    if weak_list:
        print(f"  1. Видали слабкі фічі з features.py: {weak_list[:3]}...")
        print(f"  2. python src/optimize_model.py")
        print(f"  3. python src/train_oof.py")
        print(f"  4. python src/shap_analysis.py  ← порівняй результат")
    else:
        print(f"  Слабких фіч немає → спробуй CatBoost або ensemble")
    print(f"{'='*60}")

    return shap_df, weak_list


if __name__ == "__main__":
    shap_df, weak_features = run_shap_analysis()