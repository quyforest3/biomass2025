import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

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

# Hyperparameter tuning for LGBM
lgbm_param_distributions = {
    'num_leaves': [31, 50, 70, 100],
    'max_depth': [5, 7, 9, 12, -1],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500, 1000],
    'min_data_in_leaf': [20, 30, 50, 100],
    'lambda_l1': [0, 0.1, 0.5, 1.0],
    'lambda_l2': [0, 0.1, 0.5, 1.0],
    'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
    'feature_fraction': [0.7, 0.8, 0.9, 1.0],
    'min_split_gain': [0, 0.1, 0.2],
    'subsample_for_bin': [20000, 50000, 100000],
    'cat_smooth': [1, 10, 20],
    'max_bin': [128, 256, 512],
    'objective': ['regression', 'huber', 'fair']
}

lgbm_random_search = RandomizedSearchCV(estimator=LGBMRegressor(random_state=42),
                                        param_distributions=lgbm_param_distributions,
                                        n_iter=100, cv=5, n_jobs=-1, verbose=2,
                                        scoring='neg_mean_squared_error', random_state=42)

try:
    lgbm_random_search.fit(X_train_scaled, y_train)
    best_lgbm_model = lgbm_random_search.best_estimator_
    print(f"LGBM Best Parameters: {lgbm_random_search.best_params_}")
except Exception as e:
    print(f"Error during LGBM random search: {e}")
    best_lgbm_model = lgbm_random_search

# Model evaluation
y_pred_lgbm = best_lgbm_model.predict(X_test_scaled)
rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm) ** 0.5
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print("LGBM Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_lgbm}")
print(f"  R-squared (R2): {r2_lgbm}")

# Feature importance analysis for LGBM
importances_lgbm = best_lgbm_model.feature_importances_
features = X.columns
indices_lgbm = importances_lgbm.argsort()[::-1]

# Set up a color palette related to environmental context
colors = sns.color_palette("BrBG", len(features))

plt.figure(figsize=(12, 8))
plt.title("LGBM Feature Importances", fontsize=16)
plt.bar(range(X.shape[1]), importances_lgbm[indices_lgbm], align="center", color=colors)
plt.xticks(range(X.shape[1]), features[indices_lgbm], rotation=90)
plt.tight_layout()
plt.show()

# Save the model and the scaler
joblib.dump(best_lgbm_model, 'AGB_LGBM_Model.pkl')
joblib.dump(scaler, 'AGB_Scaler.pkl')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Scatter Plot of True vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lgbm, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (LGBM)')
plt.grid(True)
plt.show()

# 2. Residuals Plot
residuals_lgbm = y_test - y_pred_lgbm

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_lgbm, residuals_lgbm, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values (LGBM)')
plt.grid(True)
plt.show()

# 3. Learning Curve
train_sizes = np.linspace(0.1, 0.9, 5)  # Changed train_sizes to a valid range
train_scores, test_scores = [], []

for train_size in train_sizes:
    X_train_partial, _, y_train_partial, _ = train_test_split(X_train_scaled, y_train, train_size=train_size, random_state=42)
    best_lgbm_model.fit(X_train_partial, y_train_partial)
    train_pred = best_lgbm_model.predict(X_train_partial)
    test_pred = best_lgbm_model.predict(X_test_scaled)
    train_scores.append(mean_squared_error(y_train_partial, train_pred, squared=False))
    test_scores.append(mean_squared_error(y_test, test_pred, squared=False))

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores, label='Train RMSE', marker='o')
plt.plot(train_sizes, test_scores, label='Test RMSE', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.title('Learning Curve (LGBM)')
plt.legend()
plt.grid(True)
plt.show()

# 4. Prediction Error Distribution
plt.figure(figsize=(8, 6))
sns.histplot(residuals_lgbm, kde=True, color='green', bins=30)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Distribution of Prediction Errors (LGBM)')
plt.grid(True)
plt.show()

# 5. Model Comparison Bar Plot (If comparing multiple models)
# Assuming you have RMSE and R² values for multiple models, you can compare them
# rmse_values = [rmse_lgbm, other_rmse_1, other_rmse_2]
# r2_values = [r2_lgbm, other_r2_1, other_r2_2]
# labels = ['LGBM', 'Model 2', 'Model 3']

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
# ax2.legend(loc='upper left', bbox_to_anchor=(
