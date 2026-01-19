import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data = pd.read_csv(file_path)

# Features and target
X = data.drop(columns=['AGB_2017'])
y = data['AGB_2017']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bagging with Random Forest
rf_model = RandomForestRegressor(
    n_estimators=1000,
    min_samples_split=2,
    min_samples_leaf=4,
    max_features='sqrt',
    max_depth=None,
    bootstrap=True,
    random_state=42
)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate
rf_pred = rf_model.predict(X_test_scaled)
rmse_rf = mean_squared_error(y_test, rf_pred, squared=False)
r2_rf = r2_score(y_test, rf_pred)

print("Random Forest Performance (Bagging):")
print(f"  Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"  R-squared (R2): {r2_rf}")

# Save the model
joblib.dump(rf_model, 'AGB_RandomForest_Model.pkl')
from sklearn.ensemble import AdaBoostRegressor

# Boosting with AdaBoost
ada_model = AdaBoostRegressor(
    n_estimators=500,
    learning_rate=0.01,
    random_state=42
)

# Train the model
ada_model.fit(X_train_scaled, y_train)

# Predict and evaluate
ada_pred = ada_model.predict(X_test_scaled)
rmse_ada = mean_squared_error(y_test, ada_pred, squared=False)
r2_ada = r2_score(y_test, ada_pred)

print("AdaBoost Performance (Boosting):")
print(f"  Root Mean Squared Error (RMSE): {rmse_ada}")
print(f"  R-squared (R2): {r2_ada}")

# Save the model
joblib.dump(ada_model, 'AGB_AdaBoost_Model.pkl')

from xgboost import XGBRegressor

# Boosting with XGBoost
xgb_model = XGBRegressor(
    subsample=0.6,
    reg_lambda=1,
    reg_alpha=0.5,
    n_estimators=500,
    min_child_weight=1,
    max_depth=3,
    learning_rate=0.01,
    gamma=0,
    colsample_bytree=1.0,
    objective='reg:squarederror',
    random_state=42
)

# Train the model
xgb_model.fit(X_train_scaled, y_train)

# Predict and evaluate
xgb_pred = xgb_model.predict(X_test_scaled)
rmse_xgb = mean_squared_error(y_test, xgb_pred, squared=False)
r2_xgb = r2_score(y_test, xgb_pred)

print("XGBoost Performance (Boosting):")
print(f"  Root Mean Squared Error (RMSE): {rmse_xgb}")
print(f"  R-squared (R2): {r2_xgb}")

# Save the model
joblib.dump(xgb_model, 'AGB_XGBoost_Model.pkl')

from catboost import CatBoostRegressor

# Boosting with CatBoost
catboost_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=0  # Set to 0 to disable output, increase to get training details
)

# Train the model
catboost_model.fit(X_train_scaled, y_train)

# Predict and evaluate
catboost_pred = catboost_model.predict(X_test_scaled)
rmse_catboost = mean_squared_error(y_test, catboost_pred, squared=False)
r2_catboost = r2_score(y_test, catboost_pred)

print("CatBoost Performance (Boosting):")
print(f"  Root Mean Squared Error (RMSE): {rmse_catboost}")
print(f"  R-squared (R2): {r2_catboost}")

# Save the model
joblib.dump(catboost_model, 'AGB_CatBoost_Model.pkl')


import matplotlib.pyplot as plt
import numpy as np

# 1. Bar Plot for RMSE and R² Values
rmse_values = [rmse_rf, rmse_ada, rmse_xgb, rmse_catboost]
r2_values = [r2_rf, r2_ada, r2_xgb, r2_catboost]
labels = ['Random Forest', 'AdaBoost', 'XGBoost', 'CatBoost']

x = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot for RMSE
bars_rmse = ax1.bar(x - 0.2, rmse_values, width=0.4, label='RMSE', color='b', alpha=0.7)
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# Adding the RMSE values on top of the bars
for bar in bars_rmse:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 3), ha='center', va='bottom')

# Bar plot for R²
ax2 = ax1.twinx()
bars_r2 = ax2.bar(x + 0.2, r2_values, width=0.4, label='R²', color='g', alpha=0.7)
ax2.set_ylabel('R²')

# Adding the R² values on top of the bars
for bar in bars_r2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 3), ha='center', va='bottom')

# Positioning the legends outside the plot area
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.9))

plt.tight_layout()
plt.show()

# 2. Scatter Plot of True vs. Predicted Values for Each Model
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest')

plt.subplot(2, 2, 2)
plt.scatter(y_test, ada_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('AdaBoost')

plt.subplot(2, 2, 3)
plt.scatter(y_test, xgb_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost')

plt.subplot(2, 2, 4)
plt.scatter(y_test, catboost_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('CatBoost')

plt.tight_layout()
plt.show()

# 3. Residuals Plot for Each Model
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(rf_pred, y_test - rf_pred, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Random Forest Residuals')

plt.subplot(2, 2, 2)
plt.scatter(ada_pred, y_test - ada_pred, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('AdaBoost Residuals')

plt.subplot(2, 2, 3)
plt.scatter(xgb_pred, y_test - xgb_pred, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('XGBoost Residuals')

plt.subplot(2, 2, 4)
plt.scatter(catboost_pred, y_test - catboost_pred, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('CatBoost Residuals')

plt.tight_layout()
plt.show()

# 4. Feature Importance for Tree-based Models
# Random Forest Feature Importance
importances_rf = rf_model.feature_importances_
sorted_indices_rf = np.argsort(importances_rf)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), importances_rf[sorted_indices_rf], align='center')
plt.xticks(range(X_train.shape[1]), X.columns[sorted_indices_rf], rotation=90)
plt.title('Random Forest Feature Importance')
plt.show()

# XGBoost Feature Importance
importances_xgb = xgb_model.feature_importances_
sorted_indices_xgb = np.argsort(importances_xgb)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), importances_xgb[sorted_indices_xgb], align='center')
plt.xticks(range(X_train.shape[1]), X.columns[sorted_indices_xgb], rotation=90)
plt.title('XGBoost Feature Importance')
plt.show()

# CatBoost Feature Importance
importances_catboost = catboost_model.get_feature_importance()
sorted_indices_catboost = np.argsort(importances_catboost)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), importances_catboost[sorted_indices_catboost], align='center')
plt.xticks(range(X_train.shape[1]), X.columns[sorted_indices_catboost], rotation=90)
plt.title('CatBoost Feature Importance')
plt.show()

# 5. Model Comparison Table
import pandas as pd

results = {
    'Model': ['Random Forest', 'AdaBoost', 'XGBoost', 'CatBoost'],
    'RMSE': [rmse_rf, rmse_ada, rmse_xgb, rmse_catboost],
    'R²': [r2_rf, r2_ada, r2_xgb, r2_catboost]
}

results_df = pd.DataFrame(results)
print(results_df)
