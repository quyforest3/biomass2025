import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
# Load the dataset
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data = pd.read_csv(file_path)

# Features and target
X = data.drop(columns=['AGB_2017'])
y = data['AGB_2017']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Parameters for bootstrapping
n_bootstraps = 10  # Number of bootstrap samples
n_samples = X_train_scaled.shape[0]  # Number of samples in each bootstrap

# Store predictions from each model
bootstrap_predictions = []

for i in range(n_bootstraps):
    # Create a bootstrap sample
    X_train_bootstrap, y_train_bootstrap = resample(X_train_scaled, y_train, n_samples=n_samples, random_state=i)
    
    # Initialize the LGBM model
    model = LGBMRegressor(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        min_data_in_leaf=20,
        max_depth=7,
        lambda_l1=1.0,
        lambda_l2=0.5,
        bagging_fraction=1.0,
        feature_fraction=0.9,
        random_state=42
    )
    
    # Train the model on the bootstrap sample
    model.fit(X_train_bootstrap, y_train_bootstrap)
    
    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Store the predictions
    bootstrap_predictions.append(y_pred)

# Average the predictions from all bootstrapped models
final_predictions = np.mean(bootstrap_predictions, axis=0)
# Evaluate the averaged predictions
rmse_bootstrap = mean_squared_error(y_test, final_predictions, squared=False)
r2_bootstrap = r2_score(y_test, final_predictions)

print("Bootstrapped LGBM Model Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_bootstrap}")
print(f"  R-squared (R2): {r2_bootstrap}")

import matplotlib.pyplot as plt

# Bar plot of RMSE and R²
rmse_values = [rmse_bootstrap]
r2_values = [r2_bootstrap]
labels = ['Bootstrapped LGBM']

x = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(8, 5))

# Bar plot for RMSE
bars_rmse = ax1.bar(x - 0.2, rmse_values, width=0.4, label='RMSE', color='b', alpha=0.7)
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE')
ax1.set_title('Bootstrapped LGBM Model Performance')
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

plt.figure(figsize=(8, 6))
plt.scatter(y_test, final_predictions, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Bootstrapped LGBM)')
plt.grid(True)
plt.show()

# Calculate RMSE for each bootstrap sample
rmse_per_sample = [mean_squared_error(y_test, preds, squared=False) for preds in bootstrap_predictions]

plt.figure(figsize=(10, 6))
plt.plot(range(1, n_bootstraps + 1), rmse_per_sample, marker='o')
plt.xlabel('Bootstrap Sample')
plt.ylabel('RMSE')
plt.title('RMSE Across Bootstrap Samples (LGBM)')
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Assuming you have true values `y_test` and predicted values `final_predictions` from your previous model

# Plotting the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, final_predictions, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.grid(True)
plt.show()

# 1. General Distribution of Points
def analyze_distribution(y_true, y_pred):
    residuals = y_true - y_pred
    spread = np.std(residuals)
    print(f"Spread (Standard Deviation of Residuals): {spread:.3f}")
    
    # Checking the central tendency
    bias = np.mean(residuals)
    print(f"Bias (Mean of Residuals): {bias:.3f}")

    return spread, bias

# 2. Outliers Detection
def detect_outliers(y_true, y_pred, threshold=2):
    residuals = y_true - y_pred
    # Outliers are defined as points where the residual is more than `threshold` standard deviations from the mean
    std_residuals = np.std(residuals)
    mean_residuals = np.mean(residuals)
    outliers = np.abs(residuals - mean_residuals) > threshold * std_residuals
    num_outliers = np.sum(outliers)
    
    print(f"Number of Outliers: {num_outliers}")
    print(f"Outliers Percentage: {100 * num_outliers / len(y_true):.2f}%")

    # Optional: Highlight outliers in the scatter plot
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

# 3. Patterns and Trends
def analyze_trends(y_true, y_pred):
    residuals = y_true - y_pred
    
    # Check for a linear trend in residuals
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True)
    plt.show()

    # Optionally, you can calculate correlation between predicted values and residuals
    correlation = np.corrcoef(y_pred, residuals)[0, 1]
    print(f"Correlation between predicted values and residuals: {correlation:.3f}")

    # A correlation close to 0 suggests no linear trend, while a high positive or negative correlation suggests a pattern.

    return correlation

# Running the analyses
spread, bias = analyze_distribution(y_test, final_predictions)
outliers = detect_outliers(y_test, final_predictions)
correlation = analyze_trends(y_test, final_predictions)
