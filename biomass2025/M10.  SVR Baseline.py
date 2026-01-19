import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = 'FEI data/opt_means_cleaned.csv'
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

# Define the baseline SVR model with default parameters
svr_baseline_model = SVR()

# Train the baseline model
svr_baseline_model.fit(X_train_scaled, y_train)

# Evaluate the baseline model
svr_baseline_pred = svr_baseline_model.predict(X_test_scaled)
rmse_svr_baseline = mean_squared_error(y_test, svr_baseline_pred) ** 0.5
r2_svr_baseline = r2_score(y_test, svr_baseline_pred)

print("Baseline Support Vector Regressor (SVR) Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_svr_baseline}")
print(f"  R-squared (R2): {r2_svr_baseline}")

# Save the baseline model
joblib.dump(svr_baseline_model, 'AGB_Baseline_SVR_Model.pkl')

# Visualizations for the baseline model

# 1. Scatter Plot of True vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, svr_baseline_pred, alpha=0.5, label='SVR Baseline Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Prediction Line')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Baseline SVR)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Residuals Plot
residuals_baseline = y_test - svr_baseline_pred

plt.figure(figsize=(8, 6))
plt.scatter(svr_baseline_pred, residuals_baseline, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (True - Predicted)')
plt.title('Residuals vs Predicted Values (Baseline SVR)')
plt.grid(True)
plt.show()

# 3. Bar Plot for RMSE and R²
rmse_values = [rmse_svr_baseline]
r2_values = [r2_svr_baseline]
labels = ['Baseline SVR']

x = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(8, 5))

# Bar plot for RMSE
bars_rmse = ax1.bar(x - 0.2, rmse_values, width=0.4, label='RMSE', color='b', alpha=0.7)
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE')
ax1.set_title('SVR Baseline Model Performance')
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
