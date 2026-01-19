import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
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

# Initialize the LGBM model with best found parameters
best_params = {
    'subsample_for_bin': 100000, 'objective': 'huber', 'num_leaves': 31, 'n_estimators': 1000,
    'min_split_gain': 0.2, 'min_data_in_leaf': 20, 'max_depth': 5, 'max_bin': 128,
    'learning_rate': 0.1, 'lambda_l2': 0.1, 'lambda_l1': 0.1, 'feature_fraction': 0.9,
    'cat_smooth': 20, 'bagging_fraction': 0.9
}
lgbm = lgb.LGBMRegressor(**best_params)

# Fit the model on the training set
lgbm.fit(X_train, y_train)

# Predict on both training and test sets
y_train_pred = lgbm.predict(X_train)
y_test_pred = lgbm.predict(X_test)

# Calculate MSE for both training and test sets
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Calculate R-squared for both training and test sets
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training Set Mean Squared Error: {train_mse}")
print(f"Test Set Mean Squared Error: {test_mse}")
print(f"Training Set R-squared: {train_r2}")
print(f"Test Set R-squared: {test_r2}")

# Cross-Validation Score (optional for further verification)
cv_scores = cross_val_score(lgbm, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE Scores: {-cv_scores}")
print(f"Average Cross-Validation MSE: {-cv_scores.mean()}")
