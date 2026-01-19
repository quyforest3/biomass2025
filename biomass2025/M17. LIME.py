import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer  # Add this line

# Load models
rf_model = joblib.load('AGB_RandomForest_Model.pkl')
lgbm_model = joblib.load('AGB_LGBM_Model.pkl')
xgb_model = joblib.load('AGB_XGBoost_Model.pkl')

# Load data
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['AGB_2017'])
y = data['AGB_2017']

# Split data for the explainer
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (assuming you used StandardScaler before)
scaler = joblib.load('AGB_Scaler.pkl')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LIME Analysis function
def lime_analysis(model, model_name, X_sample, X_scaled_sample, idx):
    lime_explainer = LimeTabularExplainer(X_train_scaled, feature_names=X.columns, class_names=['AGB_2017'], mode='regression')
    
    # Explain a single instance
    lime_exp = lime_explainer.explain_instance(X_scaled_sample[idx], model.predict, num_features=10)
    
    # Display LIME explanation
    print(f"LIME Explanation for {model_name} (Instance {idx}):")
    lime_exp.show_in_notebook(show_table=True)
    
    # Plot the explanation
    lime_exp.as_pyplot_figure()
    plt.title(f"LIME Explanation for {model_name} (Instance {idx})", fontsize=16, color='darkblue')
    plt.show()

    # Print LIME results
    print(f"\nLIME Importance for {model_name} (Instance {idx}):")
    for feature, weight in lime_exp.as_list():
        print(f"{feature}: {weight}")

# Run LIME Analysis for each model on the first instance of the test set
lime_analysis(rf_model, "Random Forest", X_test, X_test_scaled, 0)
lime_analysis(lgbm_model, "LightGBM", X_test, X_test_scaled, 0)
lime_analysis(xgb_model, "XGBoost", X_test, X_test_scaled, 0)
