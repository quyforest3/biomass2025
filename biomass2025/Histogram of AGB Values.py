import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('gedi_agb_data_within_roi.csv')

# Plot histogram of AGB values
plt.figure(figsize=(10, 6))
plt.hist(df['AGB_L4A'], bins=50, color='green', edgecolor='black')
plt.xlabel('AGB Values')
plt.ylabel('Frequency')
plt.title('Histogram of AGB Values')
plt.grid(True)

# Save the plot as an image file
plt.savefig('histogram_agb_values.png')
