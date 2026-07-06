import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 定义文件路径
file_paths = [
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_single_protein_design/result/split_vis_protein_name/KCN2_with_fitness_with_esm2.csv",
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_single_protein_design/result/split_vis_protein_name/HIS7_with_fitness_with_esm2.csv",
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_single_protein_design/result/split_vis_protein_name/AAV2_with_fitness_with_esm2.csv"
]

# 2. 加载并合并数据
all_dfs = []
for path in file_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_dfs.append(df)
    else:
        print(f"文件未找到: {path}")

full_df = pd.concat(all_dfs, ignore_index=True)

# 3. 数据清洗与计算
# 计算总分: PTM + ESM2_Naturalness
full_df['Total_Score'] = full_df['PTM'] + full_df['ESM2_Naturalness']

# 过滤出需要的三个模型（防止 CSV 中有其他无关模型）
target_models = ['Hallucination', 'MPNN', 'ESM_IF1']
plot_df = full_df[full_df['model_name'].isin(target_models)].copy()

# 4. 设置绘图风格
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

# 5. 绘制分组柱状图
# ci="sd" 表示误差棒显示标准差 (Standard Deviation)
# 如果你一定要显示方差，可以预先计算好再传给 ax.bar，但通常学术图表用标准差
ax = sns.barplot(
    data=plot_df, 
    x='protein_name', 
    y='Total_Score', 
    hue='model_name', 
    capsize=.1,      # 误差棒顶端的小横线
    errorbar="sd",   # 显式指定标准差
    palette="viridis" # 颜色方案
)

# 6. 图表美化
plt.title('Protein Design Benchmark: PTM + ESM2 Naturalness', fontsize=16)
plt.xlabel('Protein Name', fontsize=14)
plt.ylabel('Score (PTM + ESM2_Naturalness)', fontsize=14)
plt.legend(title='Model Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 7. 保存图片
output_fig = "protein_design_benchmark_scores.png"
plt.tight_layout()
plt.savefig(output_fig, dpi=300)
print(f"柱状图已保存至: {output_fig}")

# 8. 打印统计数据供核对
stats = plot_df.groupby(['protein_name', 'model_name'])['Total_Score'].agg(['mean', 'std', 'var']).reset_index()
print("\n统计汇总 (Mean & Variance):")
print(stats)

plt.show()