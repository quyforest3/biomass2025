import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error
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

# Plot learning curves
train_sizes, train_scores, test_scores = learning_curve(
    lgbm, X_train, y_train, cv=5, scoring='neg_mean_squared_error', 
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Calculate mean and standard deviation of training and test scores
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, color='r', alpha=0.1)
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, color='g', alpha=0.1)

plt.title('Learning Curves (LightGBM)')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend(loc='best')
plt.grid()
plt.show()
