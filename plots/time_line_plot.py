import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')

# File path - replace with your CSV file path
file_path = 'stream_features.csv'

# Load the data
df = pd.read_csv(file_path, sep=";", header=0)


# Assuming the first column is the time/x-axis
# If your CSV has a different structure, modify this part
x_column = df.columns[0]
feature_columns = df.columns[1:]  # All columns except the first one

# If you want to select specific columns instead of all:
# feature_columns = [
#     'temporal_dependency',
#     'case_concurrency',
#     # 'concept_stability',
#     # 'case_throughput_stability',
#     # 'parallel_activity_ratio',
#     # 'activity_duration_stability',
#     # 'case_priority_dynamics',
#     # 'avg_distinct_activities_increase',
#     # 'concept_drift',
#     'long_term_dependencies'
# ]  # Replace with your column names

# Create a custom color palette (more professional than default)
colors = sns.color_palette("viridis", len(feature_columns))

# Create a bigger figure with higher DPI for better quality
plt.figure(figsize=(12, 8), dpi=100)

# Plot each feature with custom styling
for i, column in enumerate(feature_columns):
    plt.plot(list(range(0,20)), df[column],
             linewidth=2.5,  # Thicker lines
             color=colors[i],  # Custom color
             alpha=0.9,  # Slight transparency
             marker='o',  # Add markers
             markersize=5,  # Marker size
             markerfacecolor=colors[i],  # Marker fill color
             markeredgecolor='white',  # Marker edge color
             markeredgewidth=1,  # Marker edge width
             label=column)  # Add label for legend

# Customize the plot
plt.title('Feature Values over Time', fontsize=20, pad=20)
plt.xlabel('Time', fontsize=14, labelpad=10)
plt.ylabel('Feature Value', fontsize=14, labelpad=10)

# Add grid but make it subtle
plt.grid(True, linestyle='--', alpha=0.7)

# Customize tick parameters
plt.tick_params(axis='both', which='major', labelsize=12)

# Add legend with custom styling
plt.legend(loc='upper center',
           bbox_to_anchor=(0.5, -0.15),
           ncol=3,  # Number of columns in the legend
           frameon=True,
           fancybox=True,
           shadow=True,
           fontsize=12)

# Customize the layout to make sure everything fits
plt.tight_layout()

# Add a box around the plot
plt.box(True)

# If your x-axis values are dates, uncomment this:
# plt.gcf().autofmt_xdate()

# Save the plot as a high-resolution image
# plt.savefig('feature_values_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()