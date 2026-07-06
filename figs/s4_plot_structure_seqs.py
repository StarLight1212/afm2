import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

# 1. Data Preparation
# 'IsStructure' identifies models marked with an asterisk in your list
data = [
    ("ProteinMPNN", 0.27, True), ("ESMIF1", 0.27, True), ("Antifold", 0.19, True),
    ("diffab", 0.13, True), ("-ΔG", 0.13, True), ("diffab_fixbb", 0.12, True),
    ("SaProt", 0.08, True), ("ESM3", 0.03, True), ("dyMEAN_fixbb", 0.02, True),
    ("ProSST", -0.01, True), ("dyMEAN", -0.01, True), ("MEAN_fixbb", -0.05, True),
    ("-SASA", -0.05, True), ("MEAN", -0.07, True),
    ("CurrAb", 0.08, False), ("ESM2", 0.02, False), ("ProtGPT2", -0.02, False),
    ("progen2-large", -0.11, False), ("AntiBERTy", -0.14, False)
]

df = pd.DataFrame(data, columns=['Model', 'Value', 'IsStructure'])

# 2. Sorting Logic
# Group structure-based models on the left, sequence-based on the right
# Within each group, models are sorted by value in descending order
df_struct = df[df['IsStructure']].sort_values(by='Value', ascending=False)
df_seq = df[~df['IsStructure']].sort_values(by='Value', ascending=False)

# Concatenate to form the final plotting order
df_final = pd.concat([df_struct, df_seq]).reset_index(drop=True)

# 3. Plotting Configuration
plt.figure(figsize=(14, 8), dpi=100)

# Define Colors
struct_color = '#3498db'  # Blue for Structure-based
seq_color = '#e67e22'     # Orange for Sequence-based
colors = [struct_color if is_struct else seq_color for is_struct in df_final['IsStructure']]

# Create Bar Chart
bars = plt.bar(df_final['Model'], df_final['Value'], color=colors, edgecolor='black', alpha=0.8)

# 4. Add Value Labels
for bar in bars:
    yval = bar.get_height()
    # Adjust label position based on positive or negative value
    offset = 0.005 if yval >= 0 else -0.015
    plt.text(bar.get_x() + bar.get_width()/2, yval + offset, 
             f'{yval:.2f}', va='center', ha='center', fontsize=9, fontweight='bold')

# 5. Visual Dividers and Annotations
# Draw a vertical line between the two categories
split_idx = len(df_struct) - 0.5
plt.axvline(x=split_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

# Label the groups in the plot area
max_y = df['Value'].max()
plt.text(split_idx - 0.5, max_y * 0.9, 'Structure-based Models →', 
         color=struct_color, ha='right', fontsize=12, fontweight='bold')
plt.text(split_idx + 0.5, max_y * 0.9, '← Sequence-based Models', 
         color=seq_color, ha='left', fontsize=12, fontweight='bold')

# 6. Chart Decoration (English Only)
plt.title('AbiBench: Protein Model Performance Comparison', fontsize=16, pad=25)
plt.ylabel('Average Value', fontsize=12)
plt.xlabel('Model / Metric Name', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.axhline(0, color='black', linewidth=1)  # Baseline at zero
plt.grid(axis='y', linestyle=':', alpha=0.5)

# 7. Legend
struct_patch = mpatches.Patch(color=struct_color, label='Structure-based')
seq_patch = mpatches.Patch(color=seq_color, label='Sequence-based')
plt.legend(handles=[struct_patch, seq_patch], loc='upper right', frameon=True)

plt.tight_layout()

# 8. Display
plt.show()