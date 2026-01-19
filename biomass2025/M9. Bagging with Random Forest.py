import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
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

# Bagging with Random Forest
rf_model = RandomForestRegressor(
    n_estimators=1000,
    min_samples_split=2,
    min_samples_leaf=4,
    max_features='sqrt',
    max_depth=None,
    bootstrap=True,
    random_state=42
)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate
rf_pred = rf_model.predict(X_test_scaled)
rmse_rf = mean_squared_error(y_test, rf_pred, squared=False)
r2_rf = r2_score(y_test, rf_pred)

print("Random Forest Performance (Bagging):")
print(f"  Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"  R-squared (R2): {r2_rf}")

# Save the model
joblib.dump(rf_model, 'AGB_RandomForest_Model.pkl')
from sklearn.ensemble import AdaBoostRegressor

# Boosting with AdaBoost
ada_model = AdaBoostRegressor(
    n_estimators=500,
    learning_rate=0.01,
    random_state=42
)

# Train the model
ada_model.fit(X_train_scaled, y_train)

# Predict and evaluate
ada_pred = ada_model.predict(X_test_scaled)
rmse_ada = mean_squared_error(y_test, ada_pred, squared=False)
r2_ada = r2_score(y_test, ada_pred)

print("AdaBoost Performance (Boosting):")
print(f"  Root Mean Squared Error (RMSE): {rmse_ada}")
print(f"  R-squared (R2): {r2_ada}")

# Save the model
joblib.dump(ada_model, 'AGB_AdaBoost_Model.pkl')

from xgboost import XGBRegressor

# Boosting with XGBoost
xgb_model = XGBRegressor(
    subsample=0.6,
    reg_lambda=1,
    reg_alpha=0.5,
    n_estimators=500,
    min_child_weight=1,
    max_depth=3,
    learning_rate=0.01,
    gamma=0,
    colsample_bytree=1.0,
    objective='reg:squarederror',
    random_state=42
)

# Train the model
xgb_model.fit(X_train_scaled, y_train)

# Predict and evaluate
xgb_pred = xgb_model.predict(X_test_scaled)
rmse_xgb = mean_squared_error(y_test, xgb_pred, squared=False)
r2_xgb = r2_score(y_test, xgb_pred)

print("XGBoost Performance (Boosting):")
print(f"  Root Mean Squared Error (RMSE): {rmse_xgb}")
print(f"  R-squared (R2): {r2_xgb}")

# Save the model
joblib.dump(xgb_model, 'AGB_XGBoost_Model.pkl')

from catboost import CatBoostRegressor

# Boosting with CatBoost
catboost_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=0  # Set to 0 to disable output, increase to get training details
)

# Train the model
catboost_model.fit(X_train_scaled, y_train)

# Predict and evaluate
catboost_pred = catboost_model.predict(X_test_scaled)
rmse_catboost = mean_squared_error(y_test, catboost_pred, squared=False)
r2_catboost = r2_score(y_test, catboost_pred)

print("CatBoost Performance (Boosting):")
print(f"  Root Mean Squared Error (RMSE): {rmse_catboost}")
print(f"  R-squared (R2): {r2_catboost}")

# Save the model
joblib.dump(catboost_model, 'AGB_CatBoost_Model.pkl')
