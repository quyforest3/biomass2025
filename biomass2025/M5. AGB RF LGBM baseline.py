# AGB_Prediction_NewForest.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data = pd.read_csv(file_path)

# Check for missing values
if data.isnull().sum().any():
    print("Missing values detected!")
    print(data.isnull().sum())
else:
    print("No missing values detected.")

# Features and target
X = data.drop(columns=['AGB_2017'])
y = data['AGB_2017']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Train an LGBM Regressor
lgbm_model = LGBMRegressor(random_state=42)
lgbm_model.fit(X_train_scaled, y_train)

# Make predictions with both models
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_lgbm = lgbm_model.predict(X_test_scaled)

# Model evaluation
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm, squared=False)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print("Random Forest Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"  R-squared (R2): {r2_rf}\n")

print("LGBM Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_lgbm}")
print(f"  R-squared (R2): {r2_lgbm}")

# Feature importance analysis for LGBM
importances_lgbm = lgbm_model.feature_importances_
features = X.columns
indices_lgbm = importances_lgbm.argsort()[::-1]

# Plot LGBM feature importances
plt.figure(figsize=(12, 8))
plt.title("LGBM Feature Importances", fontsize=16)
plt.bar(range(X.shape[1]), importances_lgbm[indices_lgbm], align="center")
plt.xticks(range(X.shape[1]), features[indices_lgbm], rotation=90)
plt.tight_layout()
plt.show()

# Save the models and the scaler if needed
import joblib
joblib.dump(rf_model, 'AGB_RandomForest_Model.pkl')
joblib.dump(lgbm_model, 'AGB_LGBM_Model.pkl')
joblib.dump(scaler, 'AGB_Scaler.pkl')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Scatter Plot of True vs. Predicted Values for Both Models
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest: True vs Predicted Values')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lgbm, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('LGBM: True vs Predicted Values')
plt.grid(True)

plt.tight_layout()
plt.show()

# 2. Residuals Plot for Both Models
residuals_rf = y_test - y_pred_rf
residuals_lgbm = y_test - y_pred_lgbm

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_pred_rf, residuals_rf, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Random Forest: Residuals vs Predicted Values')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_pred_lgbm, residuals_lgbm, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('LGBM: Residuals vs Predicted Values')
plt.grid(True)

plt.tight_layout()
plt.show()

# 3. Model Comparison Bar Plot for RMSE and R²
rmse_values = [rmse_rf, rmse_lgbm]
r2_values = [r2_rf, r2_lgbm]
labels = ['Random Forest', 'LGBM']

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

# 4. Prediction Error Distribution for Both Models
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(residuals_rf, kde=True, color='blue', bins=30)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Random Forest: Distribution of Prediction Errors')
plt.grid(True)

plt.subplot(1, 2, 2)
sns.histplot(residuals_lgbm, kde=True, color='green', bins=30)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('LGBM: Distribution of Prediction Errors')
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. Feature Importance for Random Forest (Additional)
importances_rf = rf_model.feature_importances_
indices_rf = importances_rf.argsort()[::-1]

plt.figure(figsize=(12, 8))
plt.title("Random Forest Feature Importances", fontsize=16)
plt.bar(range(X.shape[1]), importances_rf[indices_rf], align="center", color="orange")
plt.xticks(range(X.shape[1]), features[indices_rf], rotation=90)
plt.tight_layout()
plt.show()
