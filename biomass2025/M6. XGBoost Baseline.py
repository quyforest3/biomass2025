import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
file_path = 'FEI data/opt_means_cleaned.csv'
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

# Baseline model using default parameters of XGBoost
baseline_xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Fit the baseline model
baseline_xgb_model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred_baseline_xgb = baseline_xgb_model.predict(X_test_scaled)
rmse_baseline_xgb = mean_squared_error(y_test, y_pred_baseline_xgb) ** 0.5
r2_baseline_xgb = r2_score(y_test, y_pred_baseline_xgb)

print("Baseline XGBoost Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_baseline_xgb}")
print(f"  R-squared (R2): {r2_baseline_xgb}")

# Feature importance analysis for baseline XGBoost
importances_baseline_xgb = baseline_xgb_model.feature_importances_
features = X.columns
indices_baseline_xgb = importances_baseline_xgb.argsort()[::-1]

# Set up a color palette related to environmental context
colors = sns.color_palette("BrBG", len(features))

plt.figure(figsize=(12, 8))
plt.title("Baseline XGBoost Feature Importances", fontsize=16)
plt.bar(range(X.shape[1]), importances_baseline_xgb[indices_baseline_xgb], align="center", color=colors)
plt.xticks(range(X.shape[1]), features[indices_baseline_xgb], rotation=90)
plt.tight_layout()
plt.show()

# Save the baseline model and the scaler
joblib.dump(baseline_xgb_model, 'AGB_Baseline_XGBoost_Model.pkl')
joblib.dump(scaler, 'AGB_Scaler.pkl')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Scatter Plot of True vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_baseline_xgb, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Baseline XGBoost)')
plt.grid(True)
plt.show()

# 2. Residuals Plot
residuals_baseline = y_test - y_pred_baseline_xgb

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_baseline_xgb, residuals_baseline, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values (Baseline XGBoost)')
plt.grid(True)
plt.show()

# 3. Learning Curve (for baseline model)
train_sizes = np.linspace(0.1, 0.9, 5)
train_scores, test_scores = [], []

for train_size in train_sizes:
    X_train_partial, _, y_train_partial, _ = train_test_split(X_train_scaled, y_train, train_size=train_size, random_state=42)
    baseline_xgb_model.fit(X_train_partial, y_train_partial)
    train_pred = baseline_xgb_model.predict(X_train_partial)
    test_pred = baseline_xgb_model.predict(X_test_scaled)
    train_scores.append(mean_squared_error(y_train_partial, train_pred) ** 0.5)
    test_scores.append(mean_squared_error(y_test, test_pred) ** 0.5)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores, label='Train RMSE', marker='o')
plt.plot(train_sizes, test_scores, label='Test RMSE', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.title('Learning Curve (Baseline XGBoost)')
plt.legend()
plt.grid(True)
plt.show()

# 4. Prediction Error Distribution
plt.figure(figsize=(8, 6))
sns.histplot(residuals_baseline, kde=True, color='green', bins=30)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Distribution of Prediction Errors (Baseline XGBoost)')
plt.grid(True)
plt.show()
