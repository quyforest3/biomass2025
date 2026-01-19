import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and preprocess your data
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['AGB_2017'])
y = data['AGB_2017']

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the models
rf_model = joblib.load('AGB_RandomForest_Model.pkl')
dnn_model = tf.keras.models.load_model('AGB_Best_Tuned_DNN_Model.h5', compile=False)
dnn_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

# Predict on the test set with each model
rf_pred = rf_model.predict(X_test_scaled)
dnn_pred = dnn_model.predict(X_test_scaled).flatten()

# Define the function to optimize
def optimize_weights(weights):
    blended_pred = weights[0] * rf_pred + weights[1] * dnn_pred
    rmse = mean_squared_error(y_test, blended_pred, squared=False)
    return rmse

# Constraints: The weights should sum to 1
constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}

# Initial guess for the weights
initial_weights = [0.5, 0.5]

# Bounds: The weights should be between 0 and 1
bounds = [(0, 1), (0, 1)]

# Optimize the weights
result = minimize(optimize_weights, initial_weights, bounds=bounds, constraints=constraints)

# Get the optimal weights
optimal_weights = result.x
print(f"Optimal Weights: {optimal_weights}")

# Calculate the blended predictions using the optimal weights
blended_pred_optimized = optimal_weights[0] * rf_pred + optimal_weights[1] * dnn_pred

# Evaluate the optimized blended model
rmse_optimized = mean_squared_error(y_test, blended_pred_optimized, squared=False)
r2_optimized = r2_score(y_test, blended_pred_optimized)

print("Optimized Blended Model Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_optimized}")
print(f"  R-squared (R2): {r2_optimized}")
