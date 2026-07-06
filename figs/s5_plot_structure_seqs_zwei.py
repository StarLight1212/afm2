import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Data Preparation
structure_data = [
    0.27, 0.27, 0.19, 0.13, 0.13, 0.12, 0.08, 
    0.03, 0.02, -0.01, -0.01, -0.05, -0.05, -0.07
]
sequence_data = [
    0.08, 0.02, -0.02, -0.11, -0.14
]

# Calculate Means
struct_mean = np.mean(structure_data)
seq_mean = np.mean(sequence_data)

# 2. Setup Plot
plt.figure(figsize=(8, 7), dpi=100)
labels = ['Structure-based', 'Sequence-based']
means = [struct_mean, seq_mean]
colors = ['#3498db', '#e67e22']

# 3. Draw Bars (Averages)
bars = plt.bar(labels, means, color=colors, alpha=0.3, edgecolor='black', width=0.6, label='Mean Score')

# 4. Draw Scatter Points (Individual Models)
# We use a "jitter" to spread points horizontally so they don't overlap
def add_jittered_points(x_pos, data, color):
    jitter = np.random.normal(0, 0.04, size=len(data))  # Small horizontal random spread
    plt.scatter(np.repeat(x_pos, len(data)) + jitter, data, 
                color=color, edgecolor='white', s=60, alpha=0.9, zorder=3)

add_jittered_points(0, structure_data, '#2980b9') # Darker blue points
add_jittered_points(1, sequence_data, '#d35400')  # Darker orange points

# 5. Add Value Labels for the Means
for i, mean in enumerate(means):
    plt.text(i, mean + (0.01 if mean > 0 else -0.02), f'Mean: {mean:.3f}', 
             ha='center', va='bottom' if mean > 0 else 'top', 
             fontweight='bold', fontsize=12)

# 6. Chart Decoration (English Only)
plt.title('AbiBench Performance: Structure vs. Sequence Models', fontsize=14, pad=20)
plt.ylabel('Score (Average Value)', fontsize=12)
plt.axhline(0, color='black', linewidth=1, alpha=0.5) # Zero line
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.ylim(min(sequence_data) - 0.05, max(structure_data) + 0.1)

# Adding a custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Individual Model Score'),
    plt.Rectangle((0,0), 1, 1, color='gray', alpha=0.3, label='Group Average')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()

# 7. Show Plot
plt.show()