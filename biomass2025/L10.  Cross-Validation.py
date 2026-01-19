import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import lightgbm as lgb
import joblib

# Load the final model
model_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\final_lightgbm_model.pkl'
final_model = joblib.load(model_path)

# Load the cleaned data
cleaned_file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data_cleaned = pd.read_csv(cleaned_file_path)

# Define features and target variable
X = data_cleaned.drop(columns=['AGB_2017'])
y = data_cleaned['AGB_2017']

# Perform cross-validation
# Here we use negative MSE and R2 as scoring metrics
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score)

# 5-fold cross-validation
cv_mse_scores = cross_val_score(final_model, X, y, cv=5, scoring=mse_scorer, n_jobs=-1)
cv_r2_scores = cross_val_score(final_model, X, y, cv=5, scoring=r2_scorer, n_jobs=-1)

# Convert negative MSE to positive for interpretation
cv_mse_scores = -cv_mse_scores

# Print the results
print(f"Cross-Validation MSE Scores: {cv_mse_scores}")
print(f"Mean Cross-Validation MSE: {cv_mse_scores.mean()}")
print(f"Cross-Validation R-squared Scores: {cv_r2_scores}")
print(f"Mean Cross-Validation R-squared: {cv_r2_scores.mean()}")

# Box Plot for MSE
plt.figure(figsize=(12, 6))
sns.boxplot(data=[cv_mse_scores, cv_r2_scores], palette="Set2")
plt.xticks([0, 1], ['MSE', 'R²'])
plt.title('Cross-Validation Scores: MSE and R²')
plt.ylabel('Score')
plt.show()

# Line Plot for MSE and R² across folds
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, len(cv_mse_scores) + 1), cv_mse_scores, marker='o', label='MSE')
plt.plot(np.arange(1, len(cv_r2_scores) + 1), cv_r2_scores, marker='o', label='R²')
plt.title('Cross-Validation Scores Across Folds')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()

# Violin Plot for MSE and R²
plt.figure(figsize=(12, 6))
sns.violinplot(data=[cv_mse_scores, cv_r2_scores], palette="Set2")
plt.xticks([0, 1], ['MSE', 'R²'])
plt.title('Cross-Validation Scores Distribution: MSE and R²')
plt.ylabel('Score')
plt.show()