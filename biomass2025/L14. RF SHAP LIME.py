import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load the cleaned data
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data = pd.read_csv(file_path)

# Define features and target variable
X = data.drop(columns=['AGB_2017'])
y = data['AGB_2017']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the model and the scaler using the current version of scikit-learn
joblib.dump(rf_model, 'tuned_random_forest_model_new.pkl')
joblib.dump(scaler, 'AGB_Scaler.pkl')
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Load the saved Random Forest model
rf_model_path = r'tuned_random_forest_model_new.pkl'
best_rf_model = joblib.load(rf_model_path)

# Load the cleaned data
cleaned_file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data_cleaned = pd.read_csv(cleaned_file_path)

# Define features and target variable
X = data_cleaned.drop(columns=['AGB_2017'])
y = data_cleaned['AGB_2017']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SHAP Analysis
explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
print("SHAP Summary Plot:")
shap.summary_plot(shap_values, X_test)

# SHAP dependence plot for a specific feature (e.g., 'ChlRe')
print("SHAP Dependence Plot for 'ChlRe':")
shap.dependence_plot('ChlRe', shap_values, X_test)

# LIME Analysis
lime_explainer = LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=['AGB_2017'], mode='regression')

# Choose a data point to explain (e.g., the first data point in the test set)
i = 0
lime_exp = lime_explainer.explain_instance(X_test.values[i], best_rf_model.predict, num_features=10)

# Display the LIME explanation
print("LIME Explanation for Instance 0:")
lime_exp.show_in_notebook(show_table=True)

# Optionally, plot the LIME explanation in a more readable format
print("LIME Explanation Plot for Instance 0:")
lime_exp.as_pyplot_figure()
plt.show()

