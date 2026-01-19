import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
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

# Define the objective map at the global level
objective_map = ['regression', 'huber', 'fair']

# Define the objective function for hyperparameter tuning
def objective(params):
    # Map objective index to actual objective string
    params['objective'] = objective_map[int(params['objective'])]

    model = lgb.LGBMRegressor(
        n_estimators=int(params['n_estimators']),
        num_leaves=int(params['num_leaves']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        min_data_in_leaf=int(params['min_data_in_leaf']),
        lambda_l1=params['lambda_l1'],
        lambda_l2=params['lambda_l2'],
        bagging_fraction=params['bagging_fraction'],
        feature_fraction=params['feature_fraction'],
        min_split_gain=params['min_split_gain'],
        subsample_for_bin=int(params['subsample_for_bin']),
        cat_smooth=int(params['cat_smooth']),
        max_bin=int(params['max_bin']),
        objective=params['objective'],
        random_state=42
    )

    # Define the RMSE scorer
    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    
    # Perform 5-fold cross-validation
    cv_rmse = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=rmse_scorer)
    mean_cv_rmse = np.mean(cv_rmse)
    
    return {'loss': mean_cv_rmse, 'status': STATUS_OK}

# Define the hyperparameter space
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'max_depth': hp.quniform('max_depth', 5, 50, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 20, 100, 1),
    'lambda_l1': hp.uniform('lambda_l1', 0, 1.0),
    'lambda_l2': hp.uniform('lambda_l2', 0, 1.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
    'min_split_gain': hp.uniform('min_split_gain', 0, 0.3),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 100000, 1),
    'cat_smooth': hp.quniform('cat_smooth', 1, 100, 1),
    'max_bin': hp.quniform('max_bin', 128, 512, 1),
    'objective': hp.choice('objective', [0, 1, 2])  # We map these indices to actual objective names
}

# Run the optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)

# Get the best hyperparameters
best_params = {
    'n_estimators': int(best['n_estimators']),
    'num_leaves': int(best['num_leaves']),
    'max_depth': int(best['max_depth']),
    'learning_rate': best['learning_rate'],
    'min_data_in_leaf': int(best['min_data_in_leaf']),
    'lambda_l1': best['lambda_l1'],
    'lambda_l2': best['lambda_l2'],
    'bagging_fraction': best['bagging_fraction'],
    'feature_fraction': best['feature_fraction'],
    'min_split_gain': best['min_split_gain'],
    'subsample_for_bin': int(best['subsample_for_bin']),
    'cat_smooth': int(best['cat_smooth']),
    'max_bin': int(best['max_bin']),
    'objective': objective_map[best['objective']]  # Map the best index to the actual objective string
}

print("Best Hyperparameters found by Tree-structured Parzen Estimator (TPE):")
for param_name, param_value in best_params.items():
    print(f"{param_name}: {param_value}")

# Train the final model with the best hyperparameters
best_lgbm_model = lgb.LGBMRegressor(
    n_estimators=best_params['n_estimators'],
    num_leaves=best_params['num_leaves'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    min_data_in_leaf=best_params['min_data_in_leaf'],
    lambda_l1=best_params['lambda_l1'],
    lambda_l2=best_params['lambda_l2'],
    bagging_fraction=best_params['bagging_fraction'],
    feature_fraction=best_params['feature_fraction'],
    min_split_gain=best_params['min_split_gain'],
    subsample_for_bin=best_params['subsample_for_bin'],
    cat_smooth=best_params['cat_smooth'],
    max_bin=best_params['max_bin'],
    objective=best_params['objective'],
    random_state=42
)

best_lgbm_model.fit(X_train_scaled, y_train)
y_pred_lgbm = best_lgbm_model.predict(X_test_scaled)
rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm, squared=False)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print(f"Final LightGBM Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_lgbm}")
print(f"  R-squared (R2): {r2_lgbm}")

# Save the final model
joblib.dump(best_lgbm_model, 'AGB_LGBM_Model_TPE.pkl')

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.inspection import PartialDependenceDisplay

# 1. Feature Importance
importances_lgbm = best_lgbm_model.feature_importances_
features = X.columns
indices_lgbm = importances_lgbm.argsort()[::-1]

# Print feature importances
print("Feature Importance for LightGBM Regressor:")
feature_importance_df = pd.DataFrame({
    'Feature': features[indices_lgbm],
    'Importance': importances_lgbm[indices_lgbm]
})
print(feature_importance_df)

# Plotting the feature importance
plt.figure(figsize=(12, 8))
plt.title("LightGBM Feature Importances", fontsize=16)
sns.barplot(x=importances_lgbm[indices_lgbm], y=features[indices_lgbm], palette="viridis")
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 2. SHAP Analysis
explainer = shap.TreeExplainer(best_lgbm_model)
shap_values = explainer.shap_values(X_test_scaled)

# Print the SHAP values summary
shap_mean_abs_values = np.abs(shap_values).mean(axis=0)
shap_values_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean SHAP Value': shap_mean_abs_values
}).sort_values(by='Mean SHAP Value', ascending=False)
print("SHAP Values for LightGBM Regressor:")
print(shap_values_df)

# Plot SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)
plt.title('SHAP Summary Plot for LightGBM Regressor')
plt.show()

# 3. Residual Plot
residuals = y_test - y_pred_lgbm
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_lgbm, y=residuals, color='dodgerblue', s=60)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for LightGBM Regressor')
plt.grid(True)
plt.show()

# 4. Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_lgbm, color='darkorange', s=60)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Plot for LightGBM Regressor')
plt.grid(True)
plt.show()

# 5. Distribution of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple', bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals for LightGBM Regressor')
plt.grid(True)
plt.show()

# 6. Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    best_lgbm_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="darkblue", label="Training Error")
plt.plot(train_sizes, test_scores_mean, 'o-', color="darkgreen", label="Cross-Validation Error")
plt.xlabel("Training Set Size")
plt.ylabel("RMSE")
plt.title("Learning Curve for LightGBM Regressor")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# 7. Partial Dependence Plots (for the 3 highest SHAP features)
top_3_shap_features = shap_values_df.head(3)['Feature'].tolist()
feature_indices = [list(X.columns).index(feature) for feature in top_3_shap_features]

fig, ax = plt.subplots(figsize=(15, 10))
PartialDependenceDisplay.from_estimator(
    best_lgbm_model, 
    X_train_scaled, 
    features=feature_indices, 
    feature_names=X.columns,
    grid_resolution=50,
    ax=ax
)
plt.suptitle(f'Partial Dependence Plots for Top 3 Features ({", ".join(top_3_shap_features)})', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()
