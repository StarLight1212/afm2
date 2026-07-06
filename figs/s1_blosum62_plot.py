import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio.Align import substitution_matrices
from Bio import Align

# =========================================================
# 1. 路径配置
# =========================================================
agent_csv = "./data/generated_seqs_iter_140_with_fitness_his7.csv"
baseline_csv = "./data/designed_sequences_baseline.csv"

output_score_csv = "./data/his7_blosum62_scores_all_sequences.csv"
output_summary_csv = "./data/his7_blosum62_summary.csv"
output_fig = "./data/his7_blosum62_bar_scatter_errorbar.png"

# =========================================================
# 2. WT 序列
# =========================================================
WT_SEQUENCE = (
    "MTEQKALVKRITNETKIQIAISLKGGPLAIEHSIFPEKEAEAVAEQATQSQVINVHTGIGFLDHMIHALAKHSGWSLIVECIGDLHIDDHHTTEDCGIALGQAFKEALGAVRGVKRFGSGFAPLDEALSRAVVDLSNRPYAVVELGLQREKVGDLSCEMIPHFLESFAEASRITLHVDCLRGKNDHHRSESAFKALAVAIREATSPNGTNDVPSTKGVLM"
)

# =========================================================
# 3. 配置全局比对器 (PairwiseAligner)
# =========================================================
blosum62 = substitution_matrices.load("BLOSUM62")
aligner = Align.PairwiseAligner()
aligner.substitution_matrix = blosum62
# 设置常用的 Gap 罚分 (可以根据需要调整)
aligner.open_gap_score = -10.0
aligner.extend_gap_score = -0.5
aligner.mode = 'global' # 全局比对

def calc_blosum62_alignment_score(seq, wt_seq):
    """
    使用全局比对计算两条序列的 BLOSUM62 得分，允许长度不一致。
    """
    seq = str(seq).strip().upper()
    wt_seq = str(wt_seq).strip().upper()
    
    # 过滤异常的空序列
    if not seq or not wt_seq:
        return np.nan
        
    try:
        # 进行序列比对并获取最高得分
        score = aligner.align(wt_seq, seq).score
        return score
    except Exception as e:
        print(f"比对失败: {e} (seq: {seq[:10]}...)")
        return np.nan

# =========================================================
# 4. 读取 Agent 数据
# =========================================================
agent_df = pd.read_csv(agent_csv)
if "AASeq" not in agent_df.columns:
    raise ValueError("Agent CSV 缺少 AASeq 列")

agent_df = agent_df.rename(columns={"AASeq": "sequence"}).copy()
agent_df["model_name"] = "Agent"
agent_df["sequence"] = agent_df["sequence"].astype(str).str.strip().str.upper()

# =========================================================
# 5. 读取 Baseline 数据并筛选 HIS7
# =========================================================
baseline_df = pd.read_csv(baseline_csv)
required_baseline_cols = {"protein_name", "model_name", "sequence"}
missing_baseline_cols = required_baseline_cols - set(baseline_df.columns)
if missing_baseline_cols:
    raise ValueError(f"Baseline CSV 缺少必要列: {missing_baseline_cols}")

baseline_df = baseline_df.copy()
baseline_df["sequence"] = baseline_df["sequence"].astype(str).str.strip().str.upper()
baseline_df = baseline_df[baseline_df["protein_name"] == "HIS7"].copy()

print("Baseline 中检测到的 model_name：", sorted(baseline_df["model_name"].dropna().unique().tolist()))

# =========================================================
# 6. 仅保留目标模型并合并
# =========================================================
keep_models = ["Agent", "MPNN", "Hallucination", "ESM_IF1"]

baseline_df = baseline_df[baseline_df["model_name"].isin(["MPNN", "Hallucination", "ESM_IF1"])].copy()
agent_df = agent_df[agent_df["model_name"] == "Agent"].copy()

all_cols = sorted(set(agent_df.columns).union(set(baseline_df.columns)))
agent_df = agent_df.reindex(columns=all_cols)
baseline_df = baseline_df.reindex(columns=all_cols)

merged_df = pd.concat([agent_df, baseline_df], ignore_index=True)

# =========================================================
# 7. 长度诊断 (仅作汇报，不再因长度不同而抛出异常)
# =========================================================
merged_df["seq_len"] = merged_df["sequence"].str.len()
wt_len = len(WT_SEQUENCE)

print(f"\nWT 长度：{wt_len}")
print("\n合并后各模型长度分布：")
for model in keep_models:
    sub = merged_df[merged_df["model_name"] == model]
    if len(sub) == 0:
        print(f"{model}: 无数据")
    else:
        print(f"{model}:")
        print(sub["seq_len"].value_counts().sort_index())

if len(merged_df) == 0:
    raise ValueError("合并后没有任何序列，请检查输入文件。")

print("\n由于使用了全局比对算法(Alignment)，已允许生成的序列与WT长度不一致。")

# =========================================================
# 8. 计算 BLOSUM62 全局比对分数
# =========================================================
print("\n正在计算 BLOSUM62 Alignment 得分，请稍候...")
merged_df["blosum62_score"] = merged_df["sequence"].apply(
    lambda x: calc_blosum62_alignment_score(x, WT_SEQUENCE)
)

# 过滤掉无法计算得分的无效序列
merged_df = merged_df.dropna(subset=["blosum62_score"]).copy()

merged_df.to_csv(output_score_csv, index=False)
print(f"已保存逐条序列得分表：{output_score_csv}")

# =========================================================
# 9. 汇总统计
# =========================================================
summary_df = (
    merged_df.groupby("model_name")["blosum62_score"]
    .agg(["count", "mean", "std"])
    .reset_index()
)
summary_df["sem"] = summary_df["std"] / np.sqrt(summary_df["count"])
summary_df["model_name"] = pd.Categorical(summary_df["model_name"], categories=keep_models, ordered=True)
summary_df = summary_df.sort_values("model_name").reset_index(drop=True)

summary_df.to_csv(output_summary_csv, index=False)
print(f"已保存汇总统计表：{output_summary_csv}")
print("\n汇总结果：")
print(summary_df)

# =========================================================
# 10. 绘图
# =========================================================
plot_df = merged_df.copy()
plot_df["model_name"] = pd.Categorical(plot_df["model_name"], categories=keep_models, ordered=True)
plot_df = plot_df.sort_values("model_name")

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

x = np.arange(len(keep_models))
bar_heights = []
bar_errors = []

for model in keep_models:
    row = summary_df[summary_df["model_name"] == model]
    if len(row) == 0 or pd.isna(row["mean"].values[0]):
        bar_heights.append(0)
        bar_errors.append(0)
    else:
        bar_heights.append(row["mean"].values[0])
        bar_errors.append(row["sem"].values[0])

ax.bar(
    x,
    bar_heights,
    yerr=bar_errors,
    capsize=5,
    alpha=0.75,
    width=0.65,
    edgecolor="black",
    linewidth=1.2
)

rng = np.random.default_rng(42)
for i, model in enumerate(keep_models):
    sub = plot_df[plot_df["model_name"] == model]
    y = sub["blosum62_score"].values
    if len(y) == 0:
        continue
    jitter = rng.normal(0, 0.06, size=len(y))
    ax.scatter(
        np.full(len(y), x[i]) + jitter,
        y,
        alpha=0.75,
        s=28,
        edgecolors="black",
        linewidths=0.4
    )

ax.set_xticks(x)
ax.set_xticklabels(keep_models, fontsize=12)
ax.set_ylabel("BLOSUM62 Alignment Score vs WT", fontsize=13)
ax.set_xlabel("Model", fontsize=13)
ax.set_title("BLOSUM62 Global Alignment Scores relative to WT (HIS7)", fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(output_fig, bbox_inches="tight")
plt.show()

print(f"\n图片已保存：{output_fig}")