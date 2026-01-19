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

import matplotlib.pyplot as plt


# Generate a range of weights from 0 to 1
weights_rf = np.linspace(0, 1, 100)
weights_dnn = 1 - weights_rf

rmse_values = []

for w_rf, w_dnn in zip(weights_rf, weights_dnn):
    blended_pred = w_rf * rf_pred + w_dnn * dnn_pred
    rmse = mean_squared_error(y_test, blended_pred, squared=False)
    rmse_values.append(rmse)

# Plot RMSE against weight for Random Forest
plt.figure(figsize=(10, 6))
plt.plot(weights_rf, rmse_values, label='RMSE')
plt.xlabel('Weight for Random Forest')
plt.ylabel('RMSE')
plt.title('RMSE vs. Weight for Random Forest in Blending')
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 6))

# Scatter plot for Random Forest predictions
plt.subplot(1, 3, 1)
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Predictions')

# Scatter plot for DNN predictions
plt.subplot(1, 3, 2)
plt.scatter(y_test, dnn_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('DNN Predictions')

# Scatter plot for Optimized Blended predictions
plt.subplot(1, 3, 3)
plt.scatter(y_test, blended_pred_optimized, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Optimized Blended Predictions')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Assuming you have these variables from your previous code
rmse_values = [
    mean_squared_error(y_test, rf_pred, squared=False),
    mean_squared_error(y_test, dnn_pred, squared=False),
    rmse_blended,  # Simple average blending
    rmse_optimized  # Optimized blending
]

r2_values = [
    r2_score(y_test, rf_pred),
    r2_score(y_test, dnn_pred),
    r2_blended,  # Simple average blending
    r2_optimized  # Optimized blending
]

labels = ['Random Forest', 'DNN', 'Blended (Simple)', 'Blended (Optimized)']

x = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot for RMSE
bars_rmse = ax1.bar(x - 0.2, rmse_values, width=0.4, label='RMSE', color='b', alpha=0.7)
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# Adding the RMSE values on top of the bars
for bar in bars_rmse:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

# Bar plot for R²
ax2 = ax1.twinx()
bars_r2 = ax2.bar(x + 0.2, r2_values, width=0.4, label='R²', color='g', alpha=0.7)
ax2.set_ylabel('R²')

# Adding the R² values on top of the bars
for bar in bars_r2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

# Positioning the legends outside the plot area
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.9))

plt.tight_layout()
plt.show()

def analyze_distribution(y_true, y_pred):
    residuals = y_true - y_pred
    spread = np.std(residuals)
    bias = np.mean(residuals)
    print(f"Spread (Standard Deviation of Residuals): {spread:.3f}")
    print(f"Bias (Mean of Residuals): {bias:.3f}")
    return spread, bias

# Analyzing the residuals of the optimized blended model
spread, bias = analyze_distribution(y_test, blended_pred_optimized)

def detect_outliers(y_true, y_pred, threshold=2):
    residuals = y_true - y_pred
    std_residuals = np.std(residuals)
    mean_residuals = np.mean(residuals)
    outliers = np.abs(residuals - mean_residuals) > threshold * std_residuals
    num_outliers = np.sum(outliers)
    print(f"Number of Outliers: {num_outliers}")
    print(f"Outliers Percentage: {100 * num_outliers / len(y_true):.2f}%")

    # Visualizing outliers
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label='Data Points')
    plt.scatter(y_true[outliers], y_pred[outliers], color='red', label='Outliers', alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values with Outliers')
    plt.legend()
    plt.grid(True)
    plt.show()

    return outliers

# Detecting outliers in the optimized blended model
outliers = detect_outliers(y_test, blended_pred_optimized)
def analyze_trends(y_true, y_pred):
    residuals = y_true - y_pred
    
    # Residuals vs Predicted Values plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True)
    plt.show()

    # Calculating correlation between predicted values and residuals
    correlation = np.corrcoef(y_pred, residuals)[0, 1]
    print(f"Correlation between predicted values and residuals: {correlation:.3f}")

    return correlation

# Analyzing trends in the residuals of the optimized blended model
correlation = analyze_trends(y_test, blended_pred_optimized)

