import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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
    if params['bootstrap']:
        model = RandomForestRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            max_features=params['max_features'],
            bootstrap=params['bootstrap'],
            max_samples=params['max_samples'],
            random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            max_features=params['max_features'],
            bootstrap=params['bootstrap'],
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
    'max_depth': hp.quniform('max_depth', 5, 50, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_features': hp.choice('max_features', ['sqrt', 'log2']),
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'max_samples': hp.uniform('max_samples', 0.5, 1.0)  # Max samples for bootstrap
}

# Run the optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)

# Get the best hyperparameters
best_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'min_samples_split': int(best['min_samples_split']),
    'min_samples_leaf': int(best['min_samples_leaf']),
    'max_features': ['sqrt', 'log2'][best['max_features']],
    'bootstrap': [True, False][best['bootstrap']],
}

# Only add max_samples if bootstrap is True
if best_params['bootstrap']:
    best_params['max_samples'] = best['max_samples']

# Train the final model with the best hyperparameters
if best_params['bootstrap']:
    best_rf_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        bootstrap=best_params['bootstrap'],
        max_samples=best_params.get('max_samples', None),
        random_state=42
    )
else:
    best_rf_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        bootstrap=best_params['bootstrap'],
        random_state=42
    )

best_rf_model.fit(X_train_scaled, y_train)
y_pred_rf = best_rf_model.predict(X_test_scaled)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Final Random Forest Regressor Performance:")
print(f"  Best Hyperparameters: {best_params}")
print(f"  Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"  R-squared (R2): {r2_rf}")

# Save the final model
joblib.dump(best_rf_model, 'AGB_RandomForest_Model_TPE.pkl')
