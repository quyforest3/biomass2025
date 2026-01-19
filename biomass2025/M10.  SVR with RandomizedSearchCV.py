import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

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

# Define the SVR model
svr_model = SVR()

# Define the parameter distribution for randomized search
param_distributions = {
    'C': np.logspace(-3, 3, 10),  # Explore a wider range for C
    'epsilon': np.linspace(0.01, 0.5, 10),  # Explore more values for epsilon
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Include more kernel options
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 10)),  # Explore a wider range for gamma
    'degree': [2, 3, 4, 5, 6]  # For 'poly' kernel, explore more degrees
}

# Set up RandomizedSearchCV
svr_random_search = RandomizedSearchCV(
    estimator=svr_model,
    param_distributions=param_distributions,
    n_iter=100,  # Number of parameter settings sampled
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,
    verbose=2,
    scoring='neg_mean_squared_error',
    random_state=42
)

# Fit the RandomizedSearchCV model
svr_random_search.fit(X_train_scaled, y_train)

# Best parameters
best_svr_model = svr_random_search.best_estimator_
print(f"Best parameters from RandomizedSearchCV: {svr_random_search.best_params_}")

# Evaluate the model
svr_pred = best_svr_model.predict(X_test_scaled)
rmse_svr = mean_squared_error(y_test, svr_pred, squared=False)
r2_svr = r2_score(y_test, svr_pred)

print("Tuned Support Vector Regressor (SVR) Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_svr}")
print(f"  R-squared (R2): {r2_svr}")

# Save the best model
joblib.dump(best_svr_model, 'AGB_Tuned_SVR_Model.pkl')

# 1. Bar Plot for RMSE and R²
rmse_values = [rmse_svr]
r2_values = [r2_svr]
labels = ['Tuned SVR']

x = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(8, 5))

# Bar plot for RMSE
bars_rmse = ax1.bar(x - 0.2, rmse_values, width=0.4, label='RMSE', color='b', alpha=0.7)
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE')
ax1.set_title('SVR Model Performance')
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

# 2. Scatter Plot of True vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, svr_pred, alpha=0.5, label='SVR Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Prediction Line')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Tuned SVR)')
plt.legend()
plt.grid(True)
plt.show()

# 3. Residuals Plot
residuals = y_test - svr_pred

plt.figure(figsize=(8, 6))
plt.scatter(svr_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (True - Predicted)')
plt.title('Residuals vs Predicted Values (Tuned SVR)')
plt.grid(True)
plt.show()

# 4. Plot of SVR Model's Learning Curves
train_sizes = np.linspace(0.1, 0.9, 5)  # Adjusted range to avoid 1.0
train_scores, test_scores = [], []

for train_size in train_sizes:
    X_train_partial, _, y_train_partial, _ = train_test_split(X_train_scaled, y_train, train_size=train_size, random_state=42)
    best_svr_model.fit(X_train_partial, y_train_partial)
    train_pred = best_svr_model.predict(X_train_partial)
    test_pred = best_svr_model.predict(X_test_scaled)
    train_scores.append(mean_squared_error(y_train_partial, train_pred, squared=False))
    test_scores.append(mean_squared_error(y_test, test_pred, squared=False))

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores, label='Train RMSE', marker='o')
plt.plot(train_sizes, test_scores, label='Test RMSE', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.title('Learning Curves for SVR Model')
plt.legend()
plt.grid(True)
plt.show()

# 5. Displaying the Best Hyperparameters
best_params = svr_random_search.best_params_
print("\nBest Hyperparameters from RandomizedSearchCV:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
