import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from itertools import product
import joblib

# Load the dataset
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data = pd.read_csv(file_path)

# Features and target
X = data.drop(columns=['AGB_2017'])
y = data['AGB_2017']

# Split the data into training and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled_full = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# Initialize the models with best parameters
xgb_model = XGBRegressor(
    n_estimators=387,
    max_depth=7,
    learning_rate=0.16004345992599733,
    subsample=0.87411590644101,
    colsample_bytree=0.5103073194589992,
    gamma=2.2422157770245095,
    min_child_weight=4,
    reg_alpha=0.6248414337767971,
    reg_lambda=0.6030426354072311,
    objective='reg:squarederror',
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=231,
    max_features=0.01,
    min_samples_split=5,
    min_samples_leaf=1,
    max_depth=40,
    bootstrap=False,
    random_state=42
)

lgbm_model = LGBMRegressor(
    n_estimators=865,
    num_leaves=135,
    max_depth=32,
    learning_rate=0.07923424599538162,
    min_data_in_leaf=36,
    lambda_l1=0.09677246749678747,
    lambda_l2=0.5200536824833271,
    bagging_fraction=0.8966332578803505,
    feature_fraction=0.9145160723605901,
    min_split_gain=0.25665154804447565,
    subsample_for_bin=32522,
    cat_smooth=40,
    max_bin=488,
    objective='huber',
    random_state=42
)

svr_model = SVR(
    C=4.804117454943375,
    epsilon=0.8775449854750186,
    kernel='poly',
    degree=5,
    gamma='scale',
    coef0=0.2555541795435947
)

# Prepare K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Grid search for best weights across folds
best_rmse = float('inf')
best_weights = (0.25, 0.25, 0.25, 0.25)  # Default weights

for w1, w2, w3, w4 in product(np.arange(0, 1.1, 0.1), repeat=4):
    if w1 + w2 + w3 + w4 == 1.0:  # Weights must sum to 1
        rmse_scores = []
        for train_index, val_index in kf.split(X_train_scaled_full):
            X_train, X_val = X_train_scaled_full[train_index], X_train_scaled_full[val_index]
            y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]

            # Train models
            xgb_model.fit(X_train, y_train)
            rf_model.fit(X_train, y_train)
            lgbm_model.fit(X_train, y_train)
            svr_model.fit(X_train, y_train)

            # Predictions on validation set
            xgb_pred_val = xgb_model.predict(X_val)
            rf_pred_val = rf_model.predict(X_val)
            lgbm_pred_val = lgbm_model.predict(X_val)
            svr_pred_val = svr_model.predict(X_val)

            # Blended prediction
            blended_pred_val = w1 * xgb_pred_val + w2 * rf_pred_val + w3 * lgbm_pred_val + w4 * svr_pred_val

            # Calculate RMSE for this fold
            rmse_val = mean_squared_error(y_val, blended_pred_val, squared=False)
            rmse_scores.append(rmse_val)

        # Average RMSE across all folds
        avg_rmse = np.mean(rmse_scores)
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_weights = (w1, w2, w3, w4)

# Train final models on full training set
xgb_model.fit(X_train_scaled_full, y_train_full)
rf_model.fit(X_train_scaled_full, y_train_full)
lgbm_model.fit(X_train_scaled_full, y_train_full)
svr_model.fit(X_train_scaled_full, y_train_full)

# Generate final blended predictions for the test set
xgb_pred = xgb_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test_scaled)
lgbm_pred = lgbm_model.predict(X_test_scaled)
svr_pred = svr_model.predict(X_test_scaled)

blended_pred_best = (
    best_weights[0] * xgb_pred +
    best_weights[1] * rf_pred +
    best_weights[2] * lgbm_pred +
    best_weights[3] * svr_pred
)

# Evaluate the final blended model
rmse_blended_best = mean_squared_error(y_test, blended_pred_best, squared=False)
r2_blended_best = r2_score(y_test, blended_pred_best)

print("Best Blended Model Performance:")
print(f"  Best Weights: {best_weights}")
print(f"  Root Mean Squared Error (RMSE): {rmse_blended_best}")
print(f"  R-squared (R2): {r2_blended_best}")

# Save the models, scaler, and blended model weights
joblib.dump(xgb_model, 'AGB_XGBoost_Model.pkl')
joblib.dump(rf_model, 'AGB_RandomForest_Model.pkl')
joblib.dump(lgbm_model, 'AGB_LGBM_Model.pkl')
joblib.dump(svr_model, 'AGB_SVR_Model.pkl')
joblib.dump(scaler, 'AGB_Scaler.pkl')

# Save the blended model weights and performance
blended_model_info = {
    'best_weights': best_weights,
    'rmse_blended_best': rmse_blended_best,
    'r2_blended_best': r2_blended_best
}
joblib.dump(blended_model_info, 'AGB_Blended_Model.pkl')

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.inspection import PartialDependenceDisplay

# 1. Residual Plot
residuals = y_test - blended_pred_best
plt.figure(figsize=(10, 6))
sns.scatterplot(x=blended_pred_best, y=residuals, color='dodgerblue', s=60)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Blended Model')
plt.grid(True)
plt.show()

# 2. Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=blended_pred_best, color='darkorange', s=60)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Plot for Blended Model')
plt.grid(True)
plt.show()

# 3. Distribution of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple', bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals for Blended Model')
plt.grid(True)
plt.show()





# Average feature importances across all models
xgb_importances = xgb_model.feature_importances_
rf_importances = rf_model.feature_importances_
lgbm_importances = lgbm_model.feature_importances_

# Calculate average feature importance
avg_importances = np.mean([xgb_importances, rf_importances, lgbm_importances], axis=0)

# Create a DataFrame to hold the feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Average Importance': avg_importances
}).sort_values(by='Average Importance', ascending=False)

# Print feature importances
print("Average Feature Importance for Blended Model:")
print(feature_importance_df)

# Plot the average feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Average Importance', y='Feature', data=feature_importance_df, palette="viridis")
plt.title("Average Feature Importance for Blended Model", fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Learning Curve for the XGBoost Model (as an example)
train_sizes, train_scores, test_scores = learning_curve(
    xgb_model, X_train_scaled_full, y_train_full, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="darkblue", label="Training Error")
plt.plot(train_sizes, test_scores_mean, 'o-', color="darkgreen", label="Cross-Validation Error")
plt.xlabel("Training Set Size")
plt.ylabel("RMSE")
plt.title("Learning Curve for Blended Model (XGBoost)")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# SHAP Analysis for XGBoost in the Blended Model
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

# Print the SHAP values summary
shap_mean_abs_values = np.abs(shap_values).mean(axis=0)
shap_values_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean SHAP Value': shap_mean_abs_values
}).sort_values(by='Mean SHAP Value', ascending=False)
print("SHAP Values for XGBoost Regressor in Blended Model:")
print(shap_values_df)

# Plot SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)
plt.title('SHAP Summary Plot for XGBoost in Blended Model')
plt.show()


from sklearn.inspection import PartialDependenceDisplay

# Top 3 SHAP Features for XGBoost
top_3_shap_features = shap_values_df.head(3)['Feature'].tolist()
feature_indices = [list(X.columns).index(feature) for feature in top_3_shap_features]

fig, ax = plt.subplots(figsize=(15, 10))
PartialDependenceDisplay.from_estimator(
    xgb_model, 
    X_train_scaled_full, 
    features=feature_indices, 
    feature_names=X.columns,
    grid_resolution=50,
    ax=ax
)
plt.suptitle(f'Partial Dependence Plots for Top 3 Features ({", ".join(top_3_shap_features)}) in XGBoost', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()


import lime
import lime.lime_tabular

# LIME Analysis for Blended Model (using XGBoost as an example model)
def lime_analysis(model, model_name, X_train, X_test, instance_index=0):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['AGB_2017'],
        discretize_continuous=True,
        mode='regression'
    )

    exp = explainer.explain_instance(X_test.values[instance_index], model.predict, num_features=5)

    # Print LIME explanation in a human-readable format
    print(f"\nLIME Explanation for {model_name} (Instance {instance_index}):")
    lime_explanation = exp.as_list()
    for feature, contribution in lime_explanation:
        print(f"{feature}: {contribution:.4f}")

    # Plot LIME explanation
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.title(f'LIME Explanation for {model_name}', fontsize=16)
    plt.show()

# Run LIME Analysis for the XGBoost model as part of the blended model
lime_analysis(xgb_model, "XGBoost", pd.DataFrame(X_train_scaled_full, columns=X.columns), pd.DataFrame(X_test_scaled, columns=X.columns))
