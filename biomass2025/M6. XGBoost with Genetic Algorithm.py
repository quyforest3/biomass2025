import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
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

# Hyperparameter tuning for XGBoost using GridSearchCV
xgb_param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 2]
}

xgb_grid_search = GridSearchCV(estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
                               param_grid=xgb_param_grid,
                               cv=3, n_jobs=-1, verbose=2,
                               scoring='neg_mean_squared_error')

try:
    xgb_grid_search.fit(X_train_scaled, y_train)
    best_xgb_model = xgb_grid_search.best_estimator_
    print(f"XGBoost Best Parameters: {xgb_grid_search.best_params_}")
except Exception as e:
    print(f"Error during XGBoost grid search: {e}")
    best_xgb_model = xgb_grid_search

# Model evaluation
y_pred_xgb = best_xgb_model.predict(X_test_scaled)
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_xgb}")
print(f"  R-squared (R2): {r2_xgb}")

# Feature importance analysis for XGBoost
importances_xgb = best_xgb_model.feature_importances_
features = X.columns
indices_xgb = importances_xgb.argsort()[::-1]

# Set up a color palette related to environmental context
colors = sns.color_palette("BrBG", len(features))

plt.figure(figsize=(12, 8))
plt.title("XGBoost Feature Importances", fontsize=16)
plt.bar(range(X.shape[1]), importances_xgb[indices_xgb], align="center", color=colors)
plt.xticks(range(X.shape[1]), features[indices_xgb], rotation=90)
plt.tight_layout()
plt.show()

# Save the model and the scaler
joblib.dump(best_xgb_model, 'AGB_XGBoost_Model.pkl')
joblib.dump(scaler, 'AGB_Scaler.pkl')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Scatter Plot of True vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (XGBoost)')
plt.grid(True)
plt.show()

# 2. Residuals Plot
residuals = y_test - y_pred_xgb

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_xgb, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values (XGBoost)')
plt.grid(True)
plt.show()

# 3. Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
train_scores, test_scores = [], []

for train_size in train_sizes:
    X_train_partial, _, y_train_partial, _ = train_test_split(X_train_scaled, y_train, train_size=train_size, random_state=42)
    best_xgb_model.fit(X_train_partial, y_train_partial)
    train_pred = best_xgb_model.predict(X_train_partial)
    test_pred = best_xgb_model.predict(X_test_scaled)
    train_scores.append(mean_squared_error(y_train_partial, train_pred, squared=False))
    test_scores.append(mean_squared_error(y_test, test_pred, squared=False))

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores, label='Train RMSE', marker='o')
plt.plot(train_sizes, test_scores, label='Test RMSE', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.title('Learning Curve (XGBoost)')
plt.legend()
plt.grid(True)
plt.show()

# 4. Prediction Error Distribution
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='green', bins=30)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Distribution of Prediction Errors (XGBoost)')
plt.grid(True)
plt.show()
