import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# 1. Load Data
file_path = 'Summary_performance_DMS_substitutions_Spearman.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    # Stopping execution if file is missing to avoid further errors
    exit()

# 2. Classification Logic
# If 'Model type' contains the word 'Structure', it is Structure-based
# Otherwise, it is Sequence-based
df['Category'] = df['Model type'].apply(
    lambda x: 'Structure-based' if 'Structure' in str(x) else 'Sequence-based'
)

# Extract Spearman scores for both groups
# Using 'Average_Spearman' column as requested
struct_scores = df[df['Category'] == 'Structure-based']['Average_Spearman'].dropna().tolist()
seq_scores = df[df['Category'] == 'Sequence-based']['Average_Spearman'].dropna().tolist()

# 3. Calculate Means
means = [np.mean(struct_scores), np.mean(seq_scores)]
labels = ['Structure-based', 'Sequence-based']
colors = ['#3498db', '#e67e22'] # Blue for structure, Orange for sequence

# 4. Create the Plot
plt.figure(figsize=(8, 7), dpi=100)

# Draw the two bars (Averages)
# alpha=0.3 makes them semi-transparent so points are visible
plt.bar(labels, means, color=colors, alpha=0.3, edgecolor='black', width=0.6)

# 5. Add Scatter Points (Individual Models)
# jitter spreads the points horizontally so they don't overlap on a straight line
def plot_jittered_points(x_index, data, color):
    jitter = np.random.uniform(-0.12, 0.12, size=len(data))
    plt.scatter(np.repeat(x_index, len(data)) + jitter, data, 
                color=color, s=60, alpha=0.7, edgecolor='black', zorder=3)

plot_jittered_points(0, struct_scores, '#2980b9') # Darker blue points
plot_jittered_points(1, seq_scores, '#d35400')    # Darker orange points

# 6. Add Mean Score Text Labels
for i, mean in enumerate(means):
    plt.text(i, mean + 0.01, f'Mean: {mean:.3f}', ha='center', 
             va='bottom', fontweight='bold', fontsize=11)

# 7. Chart Decoration (English Only)
plt.title('AbiBench: Model Performance Comparison (Spearman)', fontsize=14, pad=20)
plt.ylabel('Average Spearman Correlation', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Adjust Y-axis limits based on data range
all_vals = struct_scores + seq_scores
plt.ylim(min(all_vals) - 0.05, max(all_vals) + 0.1)

# 8. Create a Clean Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markersize=10, label='Individual Model Score', markeredgecolor='black'),
    mpatches.Patch(color='gray', alpha=0.3, label='Group Average')
]
plt.legend(handles=legend_elements, loc='upper right', frameon=True)

plt.tight_layout()

# 9. Show the Plot
plt.show()




