import joblib
import shap
import pandas as pd
import lightgbm as lgb
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
# Load the final model
model_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\final_lightgbm_model.pkl'
final_model = joblib.load(model_path)

# Load the cleaned data
cleaned_file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data_cleaned = pd.read_csv(cleaned_file_path)

# Define features and target variable
X = data_cleaned.drop(columns=['AGB_2017'])  # Features
y = data_cleaned['AGB_2017']  # Target

# Create a SHAP explainer for the LightGBM model
shap_explainer = shap.TreeExplainer(final_model)

# Calculate SHAP values for the dataset
shap_values = shap_explainer.shap_values(X)

# Capture SHAP summary statistics for the top 10 features
shap_summary = pd.DataFrame({
    'Feature': X.columns,
    'Mean |SHAP Value|': np.mean(np.abs(shap_values), axis=0)
}).sort_values(by='Mean |SHAP Value|', ascending=False).head(10)

# Print SHAP summary statistics
print("Top 10 features by mean |SHAP value|:")
print(shap_summary)

# Plot summary plot of SHAP values
shap.summary_plot(shap_values, X)

# Example: Plot dependence plot for a specific feature, say 'B06'
shap.dependence_plot('B06', shap_values, X)

# Create a LIME explainer
lime_explainer = LimeTabularExplainer(X.values, feature_names=X.columns, class_names=['AGB_2017'], mode='regression')

# Choose a data point to explain (e.g., the first data point in the test set)
i = 0
lime_exp = lime_explainer.explain_instance(X.values[i], final_model.predict, num_features=10)

# Display the explanation
lime_exp.show_in_notebook(show_table=True)

# Capture LIME explanation results
lime_exp_as_list = lime_exp.as_list()
lime_exp_df = pd.DataFrame(lime_exp_as_list, columns=['Feature', 'LIME Weight'])

# Print LIME explanation results
print("\nLIME Explanation for Instance 0:")
print(lime_exp_df)
