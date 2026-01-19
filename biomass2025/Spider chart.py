import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\pics\Book1.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet3')

# Updating the x-axis labels to include both Model and Hyperparameter Tuning Method
df['Model_Hyperparameter'] = df['Model'] + " (" + df['Hyperparameter Tuning Method'] + ")"

# Define the data for the radar chart
models = df['Model_Hyperparameter']
categories = ['RMSE', 'R²']

# Number of models
N = len(models)

# What will be the angle of each axis in the plot? (we divide the plot / number of models)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

# The radar chart is circular, so we need to "complete the loop"
angles += angles[:1]

# Prepare the data
rmse_values = df['RMSE'].tolist()
r2_values = df['R²'].tolist()

# Complete the loop for both values
rmse_values += rmse_values[:1]
r2_values += r2_values[:1]

# Initialize the spider plot
fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], models, color='black', size=10, fontweight='bold')



# Draw ylabels
ax.set_rlabel_position(30)
plt.yticks([0, 1, 2, 3], ["0", "1", "2", "3"], color="grey", size=10)
plt.ylim(0, max(max(rmse_values), max(r2_values)) + 0.5)

# Plot data with custom colors and line styles
ax.plot(angles, rmse_values, linewidth=2, linestyle='solid', label='RMSE', color='#FF6347', marker='o')
ax.fill(angles, rmse_values, '#FF6347', alpha=0.3)

ax.plot(angles, r2_values, linewidth=2, linestyle='solid', label='R²', color='#4682B4', marker='o')
ax.fill(angles, r2_values, '#4682B4', alpha=0.3)

# Add a legend with custom labels for highlighted models
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12, title="Metrics", title_fontsize='13')

# Add a title
plt.title('Model Performance Comparison (RMSE & R²)', size=15, color='black', weight='bold', position=(0.5, 1.1))

# Annotate data points for better clarity
for i in range(N):
    ax.text(angles[i], rmse_values[i] + 0.1, f'{rmse_values[i]:.2f}', horizontalalignment='center', size=10, color='#FF6347', weight='semibold')
    ax.text(angles[i], r2_values[i] + 0.1, f'{r2_values[i]:.2f}', horizontalalignment='center', size=10, color='#4682B4', weight='semibold')

# Show the plot
plt.show()
