import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
rf_model = joblib.load('AGB_RandomForest_Model.pkl')
lgbm_model = joblib.load('AGB_LGBM_Model.pkl')
xgb_model = joblib.load('AGB_XGBoost_Model.pkl')

# Load data
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['AGB_2017'])
y = data['AGB_2017']

# Function to plot feature importance
def plot_feature_importance(model, model_name):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
    plt.title(f"Traditional Feature Importance for {model_name}", fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Return the DataFrame for further analysis if needed
    return feature_importance_df

# SHAP Analysis
def shap_analysis(model, model_name, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP Summary Plot
    plt.title(f"SHAP Summary Plot for {model_name}", fontsize=16, color='darkred')
    shap.summary_plot(shap_values, X_sample, feature_names=X.columns)
    plt.show()

    # Calculate mean absolute SHAP value for each feature
    shap_mean_abs_values = np.abs(shap_values).mean(axis=0)
    
    # Create a DataFrame to hold features and their corresponding mean SHAP values
    shap_values_df = pd.DataFrame({
        'Feature': X.columns,
        'Mean SHAP Value': shap_mean_abs_values
    }).sort_values(by='Mean SHAP Value', ascending=False)
    
    # Print the most important features based on SHAP values
    print(f"\nMost Important Features for {model_name} (based on SHAP values):")
    print(shap_values_df.head(10))  # Show top 10 features
    
    return shap_values_df

# Run SHAP Analysis for each model and traditional feature importance
for model, model_name in zip([rf_model, lgbm_model, xgb_model], ['Random Forest', 'LightGBM', 'XGBoost']):
    print(f"\nRunning analysis for {model_name}...")

    # Plot traditional feature importance
    print(f"\nTraditional Feature Importance for {model_name}:")
    plot_feature_importance(model, model_name)

    # Run SHAP analysis
    shap_values_df = shap_analysis(model, model_name, X)

    # Optionally, save SHAP values for further analysis
    shap_values_df.to_csv(f'SHAP_Values_{model_name}.csv', index=False)
