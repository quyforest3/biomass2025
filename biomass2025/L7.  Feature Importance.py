import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    ('select', SelectKBest(f_regression, k='all'))  # Use 'all' to keep all features for now
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

# Fit the model on the training set
lgbm.fit(X_train, y_train)

# Get feature importances
importances = lgbm.feature_importances_

# Get feature names (since we used SelectKBest, they are the selected features)
feature_names = pipeline.named_steps['select'].get_feature_names_out(input_features=X.columns)

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance - LightGBM')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.show()
