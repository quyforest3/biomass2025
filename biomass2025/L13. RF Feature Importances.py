import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the cleaned data
cleaned_file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data_cleaned = pd.read_csv(cleaned_file_path)

# Define features and target variable
X = data_cleaned.drop(columns=['AGB_2017'])  # Features
y = data_cleaned['AGB_2017']  # Target

# Load the saved Random Forest model
rf_model_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\tuned_random_forest_model.pkl'
rf_model = joblib.load(rf_model_path)

# Extract feature importances from the model
feature_importances = rf_model.feature_importances_
features = X.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Print the feature importances
print("Feature Importances:")
print(importance_df)

# Plot the feature importances
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance - Tuned Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.show()

# Select the top N features (e.g., top 10)
top_n = 10
selected_features = importance_df['Feature'].head(top_n).tolist()

# Subset the data to include only the selected features
X_selected = X[selected_features]

# Split the selected data into training and testing sets
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Retrain the Random Forest model using only the selected features
rf_selected_model = RandomForestRegressor(**rf_model.get_params())  # No need to specify random_state again
rf_selected_model.fit(X_train_sel, y_train_sel)

# Predict using the model with selected features
y_pred_sel = rf_selected_model.predict(X_test_sel)

# Calculate evaluation metrics
rf_sel_mse = mean_squared_error(y_test_sel, y_pred_sel)
rf_sel_r2 = r2_score(y_test_sel, y_pred_sel)

print(f"Random Forest with Selected Features - Test Set Mean Squared Error: {rf_sel_mse}")
print(f"Random Forest with Selected Features - Test Set R-squared: {rf_sel_r2}")

# Save the model with selected features
rf_selected_model_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\rf_selected_features_model.pkl'
joblib.dump(rf_selected_model, rf_selected_model_path)
print(f"Random Forest model with selected features saved to: {rf_selected_model_path}")

# Save the selected features dataset
selected_features_df = pd.DataFrame(X_selected)
selected_features_df['AGB_2017'] = y
selected_features_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\selected_features_data.csv'
selected_features_df.to_csv(selected_features_path, index=False)
print(f"Selected features data saved to: {selected_features_path}")
