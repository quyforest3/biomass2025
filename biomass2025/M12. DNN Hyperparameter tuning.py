import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib
import shutil
import os
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess your data
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\FEI data\opt_means_cleaned.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['AGB_2017'])
y = data['AGB_2017']

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Clear the tuner directory if it exists
tuner_dir = 'tuner_dir/AGB_DNN_Tuning'
if os.path.exists(tuner_dir):
    shutil.rmtree(tuner_dir)
print(f"Tuner directory {tuner_dir} has been cleared. Re-running the tuning process...")

# Define a function to build the model
def build_model(hp):
    model = Sequential()
    
    # Add first hidden layer with tunable number of units
    model.add(Dense(units=hp.Int('units_layer_1', min_value=32, max_value=512, step=32),
                    activation='relu', input_dim=X_train_scaled.shape[1]))
    
    # Add additional layers with tunable number of layers and units
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(Dense(units=hp.Int(f'units_layer_{i+2}', min_value=32, max_value=512, step=32),
                        activation='relu'))
        model.add(Dropout(rate=hp.Float(f'dropout_{i+2}', min_value=0.0, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='mse')
    
    return model

# Initialize the tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,  # Number of models to try
    executions_per_trial=2,  # Number of times to evaluate each model
    directory='tuner_dir',
    project_name='AGB_DNN_Tuning'
)

# Display search space summary
tuner.search_space_summary()

# Perform the search
tuner.search(X_train_scaled, y_train, epochs=50, validation_split=0.2, verbose=2)

# Retrieve the best model and hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

# Display the best hyperparameters
print(f"Best Hyperparameters: {best_hps.values}")

# Evaluate the best model on the test data
y_pred = best_model.predict(X_test_scaled).flatten()
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Best Tuned Deep Neural Network Performance:")
print(f"  Root Mean Squared Error (RMSE): {rmse}")
print(f"  R-squared (R2): {r2}")

# Save the best model
best_model.save('AGB_Best_Tuned_DNN_Model.h5')

# Save the scaler
joblib.dump(scaler, 'AGB_Scaler.pkl')

# Visualizations

# Bar plot of RMSE and R²
rmse_values = [rmse]
r2_values = [r2]
labels = ['Tuned DNN']

x = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(8, 5))

# Bar plot for RMSE
bars_rmse = ax1.bar(x - 0.2, rmse_values, width=0.4, label='RMSE', color='b', alpha=0.7)
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE')
ax1.set_title('Tuned DNN Model Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# Adding the RMSE values on top of the bars
for bar in bars_rmse:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

# Bar plot for R²
ax2 = ax1.twinx()
bars_r2 = ax2.bar(x + 0.2, r2_values, width=0.4, label='R²', color='g', alpha=0.7)
ax2.set_ylabel('R²')

# Adding the R² values on top of the bars
for bar in bars_r2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

# Positioning the legends outside the plot area
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.9))

plt.tight_layout()
plt.show()

# Scatter plot of true vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Tuned DNN Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Prediction Line')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values (Tuned DNN Model)')
plt.legend()
plt.grid(True)
plt.show()

# Analyze residuals
def analyze_distribution(y_true, y_pred):
    residuals = y_true - y_pred
    spread = np.std(residuals)
    bias = np.mean(residuals)
    print(f"Spread (Standard Deviation of Residuals): {spread:.3f}")
    print(f"Bias (Mean of Residuals): {bias:.3f}")
    return spread, bias

spread, bias = analyze_distribution(y_test, y_pred)

# Identify outliers
def detect_outliers(y_true, y_pred, threshold=2):
    residuals = y_true - y_pred
    std_residuals = np.std(residuals)
    mean_residuals = np.mean(residuals)
    outliers = np.abs(residuals - mean_residuals) > threshold * std_residuals
    num_outliers = np.sum(outliers)
    print(f"Number of Outliers: {num_outliers}")
    print(f"Outliers Percentage: {100 * num_outliers / len(y_true):.2f}%")

    # Visualizing outliers
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label='Data Points')
    plt.scatter(y_true[outliers], y_pred[outliers], color='red', label='Outliers', alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values with Outliers')
    plt.legend()
    plt.grid(True)
    plt.show()

    return outliers

outliers = detect_outliers(y_test, y_pred)

# Analyze trends in residuals
def analyze_trends(y_true, y_pred):
    residuals = y_true - y_pred
    
    # Residuals vs Predicted Values plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residuals vs Predicted Values (Tuned DNN)')
    plt.grid(True)
    plt.show()

    # Calculating correlation between predicted values and residuals
    correlation = np.corrcoef(y_pred, residuals)[0, 1]
    print(f"Correlation between predicted values and residuals: {correlation:.3f}")

    return correlation

correlation = analyze_trends(y_test, y_pred)
