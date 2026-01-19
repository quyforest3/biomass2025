import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the cleaned data
cleaned_file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data_cleaned = pd.read_csv(cleaned_file_path)

# Define features and target variable
X = data_cleaned.drop(columns=['AGB_2017'])  # Features
y = data_cleaned['AGB_2017']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Set up the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=100,  # Number of parameter settings sampled
    scoring='neg_mean_squared_error',  # Use negative MSE as scoring metric
    cv=3,  # Number of cross-validation folds
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit the search model
random_search.fit(X_train, y_train)

# Get the best model
best_rf_model = random_search.best_estimator_

# Best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Score (negative MSE):", random_search.best_score_)

# Predict using the best model
y_pred = best_rf_model.predict(X_test)

# Calculate evaluation metrics
rf_mse = mean_squared_error(y_test, y_pred)
rf_r2 = r2_score(y_test, y_pred)

print(f"Tuned Random Forest Test Set Mean Squared Error: {rf_mse}")
print(f"Tuned Random Forest Test Set R-squared: {rf_r2}")

# Save the tuned Random Forest model to a file
rf_model_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\tuned_random_forest_model.pkl'
joblib.dump(best_rf_model, rf_model_path)
print(f"Tuned Random Forest model saved to: {rf_model_path}")

# Optionally, you can save the predictions to a CSV file
rf_predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
rf_predictions_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\tuned_rf_predictions.csv'
rf_predictions_df.to_csv(rf_predictions_path, index=False)
print(f"Tuned Random Forest predictions saved to: {rf_predictions_path}")
