import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Load the training and test datasets
output_dir = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed'
training_data_path = os.path.join(output_dir, 'training_data.csv')

# Load the data
df = pd.read_csv(training_data_path)

# Separate features and target
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # The last column is the target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert the data into LightGBM Dataset format with free_raw_data=False
train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)

# Define parameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Manual early stopping implementation
best_rmse = float('inf')
early_stopping_rounds = 10
patience = 0

# Train the model manually with early stopping
num_boost_round = 25
best_model = None
best_iteration = 0

for i in range(1, num_boost_round + 1):
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=i,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        init_model=best_model if i > 1 else None
    )
    
    rmse = np.sqrt(mean_squared_error(y_test, gbm.predict(X_test, num_iteration=i)))
    
    if rmse < best_rmse:
        best_rmse = rmse
        patience = 0
        best_iteration = i
        best_model = gbm
    else:
        patience += 1
    
    print(f"Iteration {i}, RMSE: {rmse:.4f}, Best RMSE: {best_rmse:.4f}")

    if patience >= early_stopping_rounds:
        print(f"Early stopping at iteration {best_iteration} with RMSE: {best_rmse:.4f}")
        break

# Use the best model
gbm = best_model

# Predict on the test set
y_pred = gbm.predict(X_test, num_iteration=best_iteration)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance on Test Set:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² (Coefficient of Determination): {r2:.2f}")

# Save the Model
import joblib
model_path = os.path.join(output_dir, 'lightgbm_agbd_model.pkl')
joblib.dump(gbm, model_path)
print(f"Model saved to {model_path}")
