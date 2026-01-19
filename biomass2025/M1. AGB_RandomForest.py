import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
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

# Hyperparameter tuning for Random Forest
rf_param_distributions = {
    'n_estimators': [100, 200, 500, 1000],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_random_search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42),
                                      param_distributions=rf_param_distributions,
                                      n_iter=100, cv=5, n_jobs=-1, verbose=2,
                                      scoring='neg_mean_squared_error', random_state=42)

try:
    rf_random_search.fit(X_train_scaled, y_train)
    best_rf_model = rf_random_search.best_estimator_
    print(f"Random Forest Best Parameters: {rf_random_search.best_params_}")
except Exception as e:
    print(f"Error during Random Forest random search: {e}")
    best_rf_model = rf_random_search

# Model evaluation
y_pred_rf = best_rf_model.predict(X_test_scaled)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"  R-squared (R2): {r2_rf}\n")

# Feature importance analysis for Random Forest
# Calculate feature importances
importances = best_rf_model.feature_importances_

# Create a DataFrame to hold the feature names and their importance values
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print the feature importance values
print("Feature Importance for Random Forest Regressor:")
print(feature_importance_df)

# Plot the feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
plt.title("Traditional Feature Importance for Random Forest Regressor", fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# Save the model and the scaler
joblib.dump(best_rf_model, 'AGB_RandomForest_Model.pkl')
joblib.dump(scaler, 'AGB_Scaler.pkl')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Calculate residuals
residuals = y_test - y_pred_rf

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_rf, y=residuals, color='dodgerblue', s=60)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Random Forest Regressor')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf, color='darkorange', s=60)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Plot for Random Forest Regressor')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple', bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals for Random Forest Regressor')
plt.grid(True)
plt.show()


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_rf_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="darkblue", label="Training Error")
plt.plot(train_sizes, test_scores_mean, 'o-', color="darkgreen", label="Cross-Validation Error")
plt.xlabel("Training Set Size")
plt.ylabel("RMSE")
plt.title("Learning Curve for Random Forest Regressor")
plt.legend(loc="best")
plt.grid(True)
plt.show()

import shap

# Initialize the SHAP explainer
explainer = shap.TreeExplainer(best_rf_model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_scaled)

# Print the mean absolute SHAP values for each feature
shap_mean_abs_values = np.abs(shap_values).mean(axis=0)

# Create a DataFrame to hold the features and their corresponding mean SHAP values
shap_values_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean SHAP Value': shap_mean_abs_values
}).sort_values(by='Mean SHAP Value', ascending=False)

# Print the SHAP values
print("SHAP Values for Random Forest Regressor:")
print(shap_values_df)

# Plot SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)
plt.title('SHAP Summary Plot for Random Forest Regressor')
plt.show()

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns,
    class_names=['AGB_2017'],
    mode='regression'
)

# Select an instance from the test set to explain
i = 0  # You can change this index to explain a different instance
exp = explainer.explain_instance(
    data_row=X_test_scaled[i],
    predict_fn=best_rf_model.predict,
    num_features=5
)

# Print the LIME explanation in text format
print(f"LIME Explanation for Random Forest Model (Test Instance {i}):")
lime_explanation = exp.as_list()
for feature, value in lime_explanation:
    print(f"{feature}: {value}")

# Visualize the LIME explanation
plt.figure(figsize=(10, 6))
exp.as_pyplot_figure()
plt.title(f'LIME Explanation for Random Forest Model (Test Instance {i})')
plt.show()

import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import shap

# Assuming the SHAP values have been calculated and stored in `shap_values_df`
# If not, you can calculate them using the previous code

# Extract the top 3 features based on SHAP values
top_3_features = shap_values_df.head(3)['Feature'].tolist()

print(f"Top 3 Features based on SHAP values: {top_3_features}")

# Generate Partial Dependence Plots for the top 3 features
fig, ax = plt.subplots(figsize=(15, 10))

PartialDependenceDisplay.from_estimator(
    best_rf_model, 
    X_train_scaled, 
    features=top_3_features, 
    feature_names=X.columns,
    grid_resolution=50,
    ax=ax
)

plt.suptitle(f'Partial Dependence Plots for Top 3 Features ({top_3_features})', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()


