import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# Hyperparameter tuning for XGBoost using RandomizedSearchCV
xgb_param_distributions = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

xgb_random_search = RandomizedSearchCV(estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
                                       param_distributions=xgb_param_distributions,
                                       n_iter=50, cv=5, n_jobs=-1, verbose=2,
                                       scoring='neg_mean_squared_error', random_state=42)

try:
    xgb_random_search.fit(X_train_scaled, y_train)
    best_xgb_model = xgb_random_search.best_estimator_
    print(f"XGBoost Best Parameters: {xgb_random_search.best_params_}")
except Exception as e:
    print(f"Error during XGBoost random search: {e}")
    best_xgb_model = xgb_random_search

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
train_sizes = np.linspace(0.1, 0.9, 5)  # Changed train_sizes to a valid range
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

# SHAP Analysis
explainer = shap.TreeExplainer(best_xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

# Print the SHAP values summary
shap_mean_abs_values = np.abs(shap_values).mean(axis=0)
shap_values_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean SHAP Value': shap_mean_abs_values
}).sort_values(by='Mean SHAP Value', ascending=False)
print("SHAP Values for XGBoost Regressor:")
print(shap_values_df)

# Plot SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)
plt.title('SHAP Summary Plot for XGBoost Regressor')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the model and data have already been loaded and the model has been trained
# Feature importance analysis for XGBoost
importances_xgb = best_xgb_model.feature_importances_
features = X.columns
indices_xgb = importances_xgb.argsort()[::-1]

# Print feature importances
print("Feature Importance for XGBoost Regressor:")
feature_importance_df = pd.DataFrame({
    'Feature': features[indices_xgb],
    'Importance': importances_xgb[indices_xgb]
})
print(feature_importance_df)

# Plotting the feature importance
plt.figure(figsize=(12, 8))
plt.title("XGBoost Feature Importances", fontsize=16)
sns.barplot(x=importances_xgb[indices_xgb], y=features[indices_xgb], palette="viridis")
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming the model and data have already been loaded and the model has been trained
# Load the trained model and scaler (if required)
# best_xgb_model = joblib.load('AGB_XGBoost_Model.pkl')

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns,
    class_names=['AGB_2017'],
    mode='regression'
)

# Explain a specific instance (first instance in the test set)
i = 0  # Index of the instance to explain
exp = explainer.explain_instance(
    data_row=X_test_scaled[i],
    predict_fn=best_xgb_model.predict,
    num_features=5
)

# Print the explanation in text format
print(f"LIME Explanation for XGBoost Model (Test Instance {i}):")
print(exp.as_list())

# Display the explanation as a plot
plt.figure(figsize=(10, 6))
exp.as_pyplot_figure()
plt.title(f'LIME Explanation for XGBoost Model (Test Instance {i})')
plt.show()

import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Assuming SHAP values have been calculated and stored in `shap_values_df`
# Identify the top 3 features with the highest SHAP values
top_3_features = shap_values_df.head(3)['Feature'].tolist()

print(f"Top 3 Features based on SHAP values: {top_3_features}")

# Create a mapping from feature names to their indices for PartialDependenceDisplay
feature_indices = [list(X.columns).index(feature) for feature in top_3_features]

# Generate Partial Dependence Plots for the top 3 features
fig, ax = plt.subplots(figsize=(15, 10))

PartialDependenceDisplay.from_estimator(
    best_xgb_model, 
    X_train_scaled, 
    features=feature_indices, 
    feature_names=X.columns,
    grid_resolution=50,
    ax=ax
)

plt.suptitle(f'Partial Dependence Plots for Top 3 Features ({", ".join(top_3_features)})', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()
