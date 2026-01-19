import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import matplotlib.pyplot as plt

# Load the CSV file from the specified path
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means.csv'
data = pd.read_csv(file_path)

# Drop the unnecessary columns: 'Unnamed: 0', 'index', and 'ID'
data_cleaned = data.drop(columns=['Unnamed: 0', 'index', 'ID'])

# Define features and target variable
X = data_cleaned.drop(columns=['AGB_2017'])
y = data_cleaned['AGB_2017']

# Set up the feature selection pipeline
pipeline = Pipeline([
    ('select', SelectKBest(f_regression, k='all'))  # Use 'all' to keep all features initially
])

# Fit and transform the features
X_selected = pipeline.fit_transform(X, y)

# Split the selected features into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Initialize the LGBM model with the best parameters
best_params = {
    'subsample_for_bin': 100000, 'objective': 'huber', 'num_leaves': 31, 'n_estimators': 1000,
    'min_split_gain': 0.2, 'min_data_in_leaf': 20, 'max_depth': 5, 'max_bin': 128,
    'learning_rate': 0.1, 'lambda_l2': 0.1, 'lambda_l1': 0.1, 'feature_fraction': 0.9,
    'cat_smooth': 20, 'bagging_fraction': 0.9
}
lgbm = lgb.LGBMRegressor(**best_params)

# Fit the model on the original training set (with all features)
lgbm.fit(X_train, y_train)

# Predict using the original model
y_pred_original = lgbm.predict(X_test)

# Calculate evaluation metrics for the original model
mse_original = mean_squared_error(y_test, y_pred_original)
r2_original = r2_score(y_test, y_pred_original)

# Get feature importances
importances = lgbm.feature_importances_

# Get feature names
feature_names = pipeline.named_steps['select'].get_feature_names_out(input_features=X.columns)

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Set a threshold to remove features with low importance
threshold = 10  # You can adjust this value based on your data

# Identify important features
important_features = importance_df[importance_df['Importance'] > threshold]['Feature'].tolist()

# Filter the dataset to keep only important features
X_important = X[important_features]

# Split the important features into training and testing sets
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_important, y, test_size=0.2, random_state=42)

# Retrain the model using only the important features
lgbm.fit(X_train_imp, y_train_imp)

# Predict using the model with important features
y_pred_imp = lgbm.predict(X_test_imp)

# Calculate evaluation metrics for the model with important features
mse_imp = mean_squared_error(y_test_imp, y_pred_imp)
r2_imp = r2_score(y_test_imp, y_pred_imp)

# Compare results
print(f"Original Test Set Mean Squared Error: {mse_original}")
print(f"Original Test Set R-squared: {r2_original}")

print(f"Test Set Mean Squared Error with Important Features: {mse_imp}")
print(f"Test Set R-squared with Important Features: {r2_imp}")

# Optionally, you can save the important features dataset to a new CSV file
important_file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_important_features.csv'
X_important.to_csv(important_file_path, index=False)

print(f"Important features data saved to: {important_file_path}")
