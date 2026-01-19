import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load models
rf_model = joblib.load('AGB_RandomForest_Model.pkl')
lgbm_model = joblib.load('AGB_LGBM_Model.pkl')
xgb_model = joblib.load('AGB_XGBoost_Model.pkl')
svr_model = joblib.load('AGB_Tuned_SVR_Model.pkl')
blended_model_info = joblib.load('AGB_Blended_Model.pkl')
scaler = joblib.load('AGB_Scaler.pkl')

# Load data
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['AGB_2017'])
y = data['AGB_2017']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model predictions
rf_pred = rf_model.predict(X_test_scaled)
lgbm_pred = lgbm_model.predict(X_test_scaled)
xgb_pred = xgb_model.predict(X_test_scaled)
svr_pred = svr_model.predict(X_test_scaled)

# Create blended predictions using the best weights
best_weights = blended_model_info['best_weights']
blended_pred = (
    best_weights[0] * xgb_pred +
    best_weights[1] * rf_pred +
    best_weights[2] * lgbm_pred +
    best_weights[3] * svr_pred
)

# Performance metrics
models = ['Random Forest', 'LightGBM', 'XGBoost', 'SVR', 'Blended Model']  # Exclude 'DNN'
rmse_values = [
    mean_squared_error(y_test, rf_pred, squared=False),
    mean_squared_error(y_test, lgbm_pred, squared=False),
    mean_squared_error(y_test, xgb_pred, squared=False),
    mean_squared_error(y_test, svr_pred, squared=False),
    mean_squared_error(y_test, blended_pred, squared=False)  # Blended model performance
]

r2_values = [
    r2_score(y_test, rf_pred),
    r2_score(y_test, lgbm_pred),
    r2_score(y_test, xgb_pred),
    r2_score(y_test, svr_pred),
    r2_score(y_test, blended_pred)  # Blended model performance
]

# 1. Model Performance Comparison
x = np.arange(len(models))

fig, ax1 = plt.subplots(figsize=(14, 7))

# RMSE bar plot
bars_rmse = ax1.bar(x - 0.2, rmse_values, width=0.4, label='RMSE', color='skyblue')
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models)

for bar in bars_rmse:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

# R² bar plot
ax2 = ax1.twinx()
bars_r2 = ax2.bar(x + 0.2, r2_values, width=0.4, label='R²', color='green', alpha=0.5)
ax2.set_ylabel('R²')

for bar in bars_r2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

# 2. Feature Importance (Tree-based models)
for model, model_name in zip([rf_model, lgbm_model, xgb_model], ['Random Forest', 'LightGBM', 'XGBoost']):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(12, 8))
    plt.title(f"{model_name} Feature Importances", fontsize=16)
    plt.bar(range(X.shape[1]), importances[indices], align="center", color='skyblue')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# 3. SHAP Analysis for Tree-based models
for model, model_name in zip([rf_model, lgbm_model, xgb_model], ['Random Forest', 'LightGBM', 'XGBoost']):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print(f"SHAP Summary Plot for {model_name}:")
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    print(f"SHAP Dependence Plot for {model_name} (First Feature):")
    shap.dependence_plot(X.columns[0], shap_values, X_test)

# 4. LIME Analysis (use for SVR and RF)
lime_explainer = LimeTabularExplainer(X_train_scaled, feature_names=X.columns, class_names=['AGB_2017'], mode='regression')

# Explain a single instance (example: first instance)
for model, model_name in zip([svr_model, rf_model], ['SVR', 'Random Forest']):
    lime_exp = lime_explainer.explain_instance(X_test_scaled[0], model.predict, num_features=10)
    print(f"LIME Explanation for {model_name}:")
    lime_exp.as_pyplot_figure()
    plt.show()
