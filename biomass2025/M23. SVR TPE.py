import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import joblib

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

# Define the objective function for hyperparameter tuning
def objective(params):
    model = SVR(
        C=params['C'],
        epsilon=params['epsilon'],
        kernel=params['kernel'],
        degree=int(params['degree']),
        gamma=params['gamma'],
        coef0=params['coef0']
    )

    # Define the RMSE scorer
    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    
    # Perform 5-fold cross-validation
    cv_rmse = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=rmse_scorer)
    mean_cv_rmse = np.mean(cv_rmse)
    
    return {'loss': mean_cv_rmse, 'status': STATUS_OK}

# Define the hyperparameter space
space = {
    'C': hp.loguniform('C', np.log(1e-3), np.log(1e3)),
    'epsilon': hp.uniform('epsilon', 0.01, 1.0),
    'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    'degree': hp.quniform('degree', 2, 5, 1),  # Only relevant for 'poly' kernel
    'gamma': hp.choice('gamma', ['scale', 'auto']),
    'coef0': hp.uniform('coef0', 0, 1)  # Only relevant for 'poly' and 'sigmoid' kernels
}

# Run the optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)

# Map the best choice for kernel and gamma to actual values
kernel_map = {0: 'linear', 1: 'poly', 2: 'rbf', 3: 'sigmoid'}
gamma_map = {0: 'scale', 1: 'auto'}

best_params = {
    'C': best['C'],
    'epsilon': best['epsilon'],
    'kernel': kernel_map[best['kernel']],
    'degree': int(best['degree']),
    'gamma': gamma_map[best['gamma']],
    'coef0': best['coef0']
}

# Print the best hyperparameters
print("Best Hyperparameters found by Tree-structured Parzen Estimator (TPE):")
for param_name, param_value in best_params.items():
    print(f"{param_name}: {param_value}")

# Train the final model with the best hyperparameters
best_svr_model = SVR(
    C=best_params['C'],
    epsilon=best_params['epsilon'],
    kernel=best_params['kernel'],
    degree=best_params['degree'],
    gamma=best_params['gamma'],
    coef0=best_params['coef0']
)

best_svr_model.fit(X_train_scaled, y_train)
y_pred_svr = best_svr_model.predict(X_test_scaled)
rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
r2_svr = r2_score(y_test, y_pred_svr)

print(f"Final SVR Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_svr}")
print(f"  R-squared (R2): {r2_svr}")

# Save the final model
joblib.dump(best_svr_model, 'AGB_SVR_Model_TPE.pkl')

import matplotlib.pyplot as plt
import seaborn as sns

residuals = y_test - y_pred_svr
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_svr, y=residuals, color='dodgerblue', s=60)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for SVR with TPE Hyperparameter Tuning')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_svr, color='darkorange', s=60)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Plot for SVR with TPE Hyperparameter Tuning')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple', bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals for SVR with TPE Hyperparameter Tuning')
plt.grid(True)
plt.show()

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_svr_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="darkblue", label="Training Error")
plt.plot(train_sizes, test_scores_mean, 'o-', color="darkgreen", label="Cross-Validation Error")
plt.xlabel("Training Set Size")
plt.ylabel("RMSE")
plt.title("Learning Curve for SVR with TPE Hyperparameter Tuning")
plt.legend(loc="best")
plt.grid(True)
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

# If you're using an older version that still supports squared=False
# rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)

# Instead, use the following if your version of scikit-learn supports it:
from sklearn.metrics import mean_squared_error

# Calculate RMSE directly
rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)

# Or if the `root_mean_squared_error` function is available:
# from sklearn.metrics import root_mean_squared_error
# rmse_svr = root_mean_squared_error(y_test, y_pred_svr)
try:
    from sklearn.inspection import PartialDependenceDisplay
except ImportError:
    from sklearn.ensemble import PartialDependenceDisplay  # In case of older versions

# Now you can use it as follows:
features = ['B12', 'NDMI', 'B8A']  # example feature indices
PartialDependenceDisplay.from_estimator(best_svr_model, X_train_scaled, features=features, feature_names=X.columns)
plt.suptitle('Partial Dependence Plots for SVR with TPE Hyperparameter Tuning', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()

import shap
import pandas as pd

# Assuming best_svr_model is already trained and X_train_scaled, X_test_scaled, X are defined
explainer = shap.KernelExplainer(best_svr_model.predict, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

# Convert SHAP values into a DataFrame for better readability
shap_values_df = pd.DataFrame(shap_values, columns=X.columns)

# Print the SHAP values for the first few instances
print("SHAP values for the first few instances:\n")
print(shap_values_df.head())

# Optionally, print the mean absolute SHAP values across all instances (i.e., feature importance)
mean_abs_shap_values = shap_values_df.abs().mean().sort_values(ascending=False)
print("\nMean absolute SHAP values (feature importance):\n")
print(mean_abs_shap_values)

# Plot the SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)
plt.title('SHAP Summary Plot for SVR with TPE Hyperparameter Tuning')
plt.show()



import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled, feature_names=X.columns, class_names=['AGB_2017'], discretize_continuous=True, mode='regression')

i = 0  # Index of the instance to explain
exp = explainer.explain_instance(X_test_scaled[i], best_svr_model.predict, num_features=5)
plt.figure(figsize=(10, 6))
exp.as_pyplot_figure()
plt.title('LIME Explanation for SVR with TPE Hyperparameter Tuning')
plt.show()

# Print the LIME explanation in a human-readable format
print(f"\nLIME Explanation for SVR with TPE (Instance {i}):")
for feature, contribution in exp.as_list():
    print(f"{feature}: {contribution:.4f}")

