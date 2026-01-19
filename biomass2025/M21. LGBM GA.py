import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return {'loss': rmse, 'status': STATUS_OK}

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
