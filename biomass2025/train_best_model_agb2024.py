import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
import joblib

DATA_PATH = 'FEI data/opt_means_cleaned.csv'
TARGET_COL_OPTIONS = ['AGB_2024', 'AGB_2017']

MODELS = {
    'Random Forest': RandomForestRegressor(n_estimators=400, random_state=42),
    'LightGBM': LGBMRegressor(random_state=42, verbose=-1, n_estimators=600, num_leaves=64, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0, n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, gamma=0.0),
    'SVR': SVR(C=10, epsilon=0.1, kernel='rbf')
}


def main():
    print('📥 Loading data from', DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Determine target column
    target_col = next((c for c in TARGET_COL_OPTIONS if c in df.columns), None)
    if target_col is None:
        raise ValueError('No AGB target column found. Expected one of: ' + ', '.join(TARGET_COL_OPTIONS))

    print('🎯 Using target:', target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for name, model in MODELS.items():
        print(f'🚀 Training {name}...')
        start = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test_scaled)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))

        results[name] = {
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae,
            'Training Time (s)': train_time,
            'model': model
        }
        print(f'   ✅ {name}: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}, time={train_time:.2f}s')

    # Find best model
    best_name = min(results.keys(), key=lambda n: results[n]['RMSE'])
    best = results[best_name]
    print('\n🏆 Best model:', best_name)
    print(f"   RMSE={best['RMSE']:.4f}, R²={best['R2']:.4f}, MAE={best['MAE']:.4f}")

    # Save outputs
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/AGB2024_{best_name.replace(" ", "_")}_{ts}.pkl'
    scaler_path = f'models/AGB2024_Scaler_{ts}.pkl'
    summary_path = f'models/AGB2024_Model_Results_{ts}.csv'

    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(best['model'], model_path)
    joblib.dump(scaler, scaler_path)

    summary_rows = []
    for n, r in results.items():
        summary_rows.append({'Model': n, 'RMSE': r['RMSE'], 'R2': r['R2'], 'MAE': r['MAE'], 'Training Time (s)': r['Training Time (s)']})
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print('\n💾 Saved:')
    print('   Model  ->', model_path)
    print('   Scaler ->', scaler_path)
    print('   Summary->', summary_path)


if __name__ == '__main__':
    main()
