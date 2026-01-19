import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
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

# Define the objective function for hyperparameter tuning with 5-fold cross-validation
def objective(params):
    model = xgb.XGBRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        min_child_weight=params['min_child_weight'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=42,
        objective='reg:squarederror'
    )

    # Define RMSE scorer
    rmse_scorer = make_scorer(mean_squared_error, squared=False)

    # Perform 5-fold cross-validation
    cv_rmse = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=rmse_scorer)
    mean_cv_rmse = np.mean(cv_rmse)
    
    return {'loss': mean_cv_rmse, 'status': STATUS_OK}

# Define the hyperparameter space
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'gamma': hp.uniform('gamma', 0, 5),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1.0)
}

# Run the optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)

# Get the best hyperparameters
best_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'learning_rate': best['learning_rate'],
    'subsample': best['subsample'],
    'colsample_bytree': best['colsample_bytree'],
    'gamma': best['gamma'],
    'min_child_weight': int(best['min_child_weight']),
    'reg_alpha': best['reg_alpha'],
    'reg_lambda': best['reg_lambda']
}

# Print the best hyperparameters
print("Best Hyperparameters found by Tree-structured Parzen Estimator (TPE):")
for param_name, param_value in best_params.items():
    print(f"{param_name}: {param_value}")

# Train the final model with the best hyperparameters
best_xgb_model = xgb.XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    gamma=best_params['gamma'],
    min_child_weight=best_params['min_child_weight'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    random_state=42,
    objective='reg:squarederror'
)

# Train the model on the entire training set
best_xgb_model.fit(X_train_scaled, y_train)

# Predict and evaluate on the test set
y_pred_xgb = best_xgb_model.predict(X_test_scaled)
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"Final XGBoost Regressor Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse_xgb}")
print(f"  R-squared (R2): {r2_xgb}")

# Save the final model
joblib.dump(best_xgb_model, 'AGB_XGBoost_Model_TPE.pkl')
