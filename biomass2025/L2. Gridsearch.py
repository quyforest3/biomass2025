import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

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

# Initialize the LGBM model
lgbm = lgb.LGBMRegressor()

# Define the parameter grid
param_grid = {
    'num_leaves': [31, 50, 70],
    'max_depth': [5, 7, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'min_data_in_leaf': [20, 30, 50],
    'lambda_l1': [0, 0.1, 0.5],
    'lambda_l2': [0, 0.1, 0.5],
    'bagging_fraction': [0.7, 0.8, 0.9],
    'feature_fraction': [0.7, 0.8, 0.9]
}

# Set up the GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Use negative MSE as scoring metric
    cv=3,  # Number of cross-validation folds
    verbose=1,
    n_jobs=-1  # Use all available cores
)

# Fit the grid search model
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score (negative MSE):", grid_search.best_score_)

# Predict using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test Set Mean Squared Error: {mse}")

# Optionally, you can save the cleaned data to a new CSV file
cleaned_file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to: {cleaned_file_path}")
