import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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

# Base models with best parameters
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

# Fit the models on the entire training data
xgb_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)
lgbm_model.fit(X_train_scaled, y_train)

# Define the stacking regressor
estimators = [
    ('xgb', xgb_model),
    ('rf', rf_model),
    ('lgbm', lgbm_model),
    ('svr', svr_model)
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV(),
    cv=5,
    n_jobs=-1
)

# Apply 5-fold cross-validation to evaluate the model
cv_rmse_scores = cross_val_score(stacking_model, X_train_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error')
cv_r2_scores = cross_val_score(stacking_model, X_train_scaled, y_train, cv=5, scoring='r2')

# Average the scores
mean_rmse = -cv_rmse_scores.mean()
mean_r2 = cv_r2_scores.mean()

print("Cross-Validated Stacking Model Performance:")
print(f"  Average CV Root Mean Squared Error (RMSE): {mean_rmse}")
print(f"  Average CV R-squared (R2): {mean_r2}")

# Train the stacking model on the full training data
stacking_model.fit(X_train_scaled, y_train)

# Generate predictions for the test set
stacking_pred = stacking_model.predict(X_test_scaled)

# Evaluate the stacking model on the test set
rmse_stacking = mean_squared_error(y_test, stacking_pred, squared=False)
r2_stacking = r2_score(y_test, stacking_pred)

print("Final Stacking Model Performance on Test Set:")
print(f"  Root Mean Squared Error (RMSE): {rmse_stacking}")
print(f"  R-squared (R2): {r2_stacking}")

# SHAP Analysis for the Stacking Model
explainer = shap.KernelExplainer(stacking_model.predict, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

# Print the SHAP values summary
shap_mean_abs_values = np.abs(shap_values).mean(axis=0)
shap_values_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean SHAP Value': shap_mean_abs_values
}).sort_values(by='Mean SHAP Value', ascending=False)
print("SHAP Values for Stacking Model:")
print(shap_values_df)

# Plot SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)
plt.title('SHAP Summary Plot for Stacking Model')
plt.show()

# Partial Dependence Plots for top 3 features based on SHAP
top_features = shap_values_df['Feature'].head(3).tolist()

# Importing the correct partial dependence function based on your scikit-learn version
try:
    from sklearn.inspection import PartialDependenceDisplay
    # Use PartialDependenceDisplay for newer versions
    PartialDependenceDisplay.from_estimator(stacking_model, X_test_scaled, features=top_features, feature_names=X.columns, grid_resolution=50)
except ImportError:
    from sklearn.ensemble import PartialDependenceDisplay  # Use this for older versions
    PartialDependenceDisplay.from_estimator(stacking_model, X_test_scaled, features=top_features, feature_names=X.columns, grid_resolution=50)

plt.suptitle('Partial Dependence Plots for Top 3 SHAP Features (Stacking Model)', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()


import lime
import lime.lime_tabular

# LIME Analysis for the Stacking Model
def lime_analysis(model, X_train, X_test, instance_index=0):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=X.columns,
        class_names=['AGB_2017'],
        mode='regression'
    )

    # Explain a specific instance (e.g., the first instance in the test set)
    exp = explainer.explain_instance(
        data_row=X_test[instance_index],
        predict_fn=model.predict,
        num_features=5
    )

    # Print LIME explanation
    print(f"\nLIME Explanation for Stacking Model (Test Instance {instance_index}):")
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight:.4f}")

    # Plot LIME explanation
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.title(f'LIME Explanation for Stacking Model (Test Instance {instance_index})')
    plt.show()

# Perform LIME analysis on the first instance in the test set
lime_analysis(stacking_model, X_train_scaled, X_test_scaled, instance_index=0)



import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Assuming the stacking model has already been trained and the SHAP values have been calculated
explainer = shap.KernelExplainer(stacking_model.predict, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

# Print the SHAP values summary
shap_mean_abs_values = np.abs(shap_values).mean(axis=0)
shap_values_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean SHAP Value': shap_mean_abs_values
}).sort_values(by='Mean SHAP Value', ascending=False)

print("SHAP Values for Stacking Model:")
print(shap_values_df)

# If you want to print SHAP values for individual predictions:
shap_values_per_instance = pd.DataFrame(
    shap_values, columns=X.columns
)

print("\nSHAP Values for individual instances in the test set (first 5 instances shown):")
print(shap_values_per_instance.head())

# Plot SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)
plt.title('SHAP Summary Plot for Stacking Model')
plt.show()
import os
from sklearn.model_selection import learning_curve

# Function to plot and save learning curves
def plot_and_save_learning_curve(model, X, y, model_name, save_path):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="darkblue", label="Training Error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="darkgreen", label="Cross-Validation Error")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.title(f"Learning Curve for {model_name}")
    plt.legend(loc="best")
    plt.grid(True)

    # Save the plot to the specified directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plot_filename = os.path.join(save_path, f"{model_name}_Learning_Curve.png")
    plt.savefig(plot_filename)
    plt.close()

    print(f"Learning curve saved to {plot_filename}")

# Save path
save_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\pics\stacking'

# Plot and save learning curve for the stacking model
plot_and_save_learning_curve(stacking_model, X_train_scaled, y_train, "Stacking Model", save_path)
