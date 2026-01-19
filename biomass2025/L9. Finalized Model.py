import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import joblib  # To save the model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV file from the specified path
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means.csv'
data = pd.read_csv(file_path)

# Drop the unnecessary columns: 'Unnamed: 0', 'index', and 'ID'
data_cleaned = data.drop(columns=['Unnamed: 0', 'index', 'ID'])

# Define features and target variable
X = data_cleaned.drop(columns=['AGB_2017'])
y = data_cleaned['AGB_2017']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use the best parameters from the RandomizedSearchCV
best_params = {
    'subsample_for_bin': 100000, 'objective': 'huber', 'num_leaves': 31, 'n_estimators': 1000,
    'min_split_gain': 0.2, 'min_data_in_leaf': 20, 'max_depth': 5, 'max_bin': 128,
    'learning_rate': 0.1, 'lambda_l2': 0.1, 'lambda_l1': 0.1, 'feature_fraction': 0.9,
    'cat_smooth': 20, 'bagging_fraction': 0.9
}

# Initialize the LGBM model with the best parameters
final_model = lgb.LGBMRegressor(**best_params)

# Train the final model on the entire training set
final_model.fit(X_train, y_train)

# Predict using the final model
y_pred = final_model.predict(X_test)

# Calculate evaluation metrics
final_mse = mean_squared_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)

print(f"Final Model Test Set Mean Squared Error: {final_mse}")
print(f"Final Model Test Set R-squared: {final_r2}")

# Save the final model to a file
model_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\final_lightgbm_model.pkl'
joblib.dump(final_model, model_path)
print(f"Final model saved to: {model_path}")

# Optionally, save the predictions and the test set to a CSV file
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
predictions_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to: {predictions_path}")

# Optionally, save the cleaned data
cleaned_file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data_cleaned.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to: {cleaned_file_path}")

