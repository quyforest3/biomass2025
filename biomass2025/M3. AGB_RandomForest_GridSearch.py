import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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

# Hyperparameter tuning for Random Forest using GridSearchCV
rf_param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                              param_grid=rf_param_grid,
                              cv=5, n_jobs=-1, verbose=2,
                              scoring='neg_mean_squared_error')

try:
    rf_grid_search.fit(X_train_scaled, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    print(f"Random Forest Best Parameters: {rf_grid_search.best_params_}")
except Exception as e:
    print(f"Error during Random Forest grid search: {e}")
    best_rf_model = rf_grid_search

# Model evaluation
y_pred_rf = best_rf_model.predict(X_test_scaled)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"  R-squared (R2): {r2_rf}\n")

# Feature importance analysis for Random Forest
importances_rf = best_rf_model.feature_importances_
features = X.columns
indices_rf = importances_rf.argsort()[::-1]

# Set up a color palette related to environmental context
colors = sns.color_palette("BrBG", len(features))

plt.figure(figsize=(12, 8))
plt.title("Random Forest Feature Importances", fontsize=16)
plt.bar(range(X.shape[1]), importances_rf[indices_rf], align="center", color=colors)
plt.xticks(range(X.shape[1]), features[indices_rf], rotation=90)
plt.tight_layout()
plt.show()

# Save the model and the scaler
joblib.dump(best_rf_model, 'AGB_RandomForest_Model.pkl')
joblib.dump(scaler, 'AGB_Scaler.pkl')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Scatter Plot of True vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Random Forest)')
plt.grid(True)
plt.show()

# 2. Residuals Plot
residuals_rf = y_test - y_pred_rf

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_rf, residuals_rf, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values (Random Forest)')
plt.grid(True)
plt.show()

# 3. Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
train_scores, test_scores = [], []

for train_size in train_sizes:
    X_train_partial, _, y_train_partial, _ = train_test_split(X_train_scaled, y_train, train_size=train_size, random_state=42)
    best_rf_model.fit(X_train_partial, y_train_partial)
    train_pred = best_rf_model.predict(X_train_partial)
    test_pred = best_rf_model.predict(X_test_scaled)
    train_scores.append(mean_squared_error(y_train_partial, train_pred, squared=False))
    test_scores.append(mean_squared_error(y_test, test_pred, squared=False))

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores, label='Train RMSE', marker='o')
plt.plot(train_sizes, test_scores, label='Test RMSE', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.title('Learning Curve (Random Forest)')
plt.legend()
plt.grid(True)
plt.show()

# 4. Prediction Error Distribution
plt.figure(figsize=(8, 6))
sns.histplot(residuals_rf, kde=True, color='blue', bins=30)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Distribution of Prediction Errors (Random Forest)')
plt.grid(True)
plt.show()

# 5. Model Comparison Bar Plot (If comparing multiple models)
# Assuming you have RMSE and R² values for multiple models, you can compare them
# rmse_values = [rmse_rf, other_rmse_1, other_rmse_2]
# r2_values = [r2_rf, other_r2_1, other_r2_2]
# labels = ['Random Forest', 'Model 2', 'Model 3']

# x = np.arange(len(labels))

# fig, ax1 = plt.subplots(figsize=(12, 6))

# # Bar plot for RMSE
# bars_rmse = ax1.bar(x - 0.2, rmse_values, width=0.4, label='RMSE', color='b', alpha=0.7)
# ax1.set_xlabel('Model')
# ax1.set_ylabel('RMSE')
# ax1.set_title('Model Performance Comparison')
# ax1.set_xticks(x)
# ax1.set_xticklabels(labels)

# # Adding the RMSE values on top of the bars
# for bar in bars_rmse:
#     yval = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 3), ha='center', va='bottom')

# # Bar plot for R²
# ax2 = ax1.twinx()
# bars_r2 = ax2.bar(x + 0.2, r2_values, width=0.4, label='R²', color='g', alpha=0.7)
# ax2.set_ylabel('R²')

# # Adding the R² values on top of the bars
# for bar in bars_r2:
#     yval = bar.get_height()
#     ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 3), ha='center', va='bottom')

# # Positioning the legends outside the plot area
# ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
# ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.9))

# plt.tight_layout()
# plt.show()
