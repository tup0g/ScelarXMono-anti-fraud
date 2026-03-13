# Anti-Fraud Project

Базова архітектура проєкту для задачі anti-fraud (EDA → feature engineering → modeling → predictions).

## Структура

- `data/raw/` — сирі CSV (не зберігаються в git)
- `data/processed/` — підготовлені датасети
- `notebooks/01_eda.ipynb` — розвідковий аналіз
- `notebooks/02_features.ipynb` — генерація та перевірка фічей
- `notebooks/03_model.ipynb` — навчання моделі + SHAP
- `src/features.py` — функції для фічей і препроцесингу
- `src/model.py` — базовий пайплайн навчання та інференсу
- `outputs/predictions.csv` — файл із фінальними скорингами

## Швидкий старт

1. Створити віртуальне середовище
2. Встановити залежності:
   - `pip install -r requirements.txt`
3. Покласти CSV у `data/raw/`
4. Відкрити ноутбук `notebooks/01_eda.ipynb`
