import os
import math
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. 路径配置
# =========================================================
agent_csv = "./data/generated_seqs_iter_140_with_fitness_his7.csv"
baseline_csv = "./data/designed_sequences_baseline.csv"

output_seq_metric_csv = "./data/his7_sequence_level_similarity_metrics.csv"
output_pair_metric_csv = "./data/his7_pairwise_within_group_similarity_metrics.csv"
output_summary_csv = "./data/his7_similarity_summary.csv"
output_fig = "./data/his7_similarity_barplots_2x2.png"

# =========================================================
# 2. WT 序列
# =========================================================
WT_SEQUENCE = (
    "MTEQKALVKRITNETKIQIAISLKGGPLAIEHSIFPEKEAEAVAEQATQSQVINVHTGIGFLDHMIHALAKHSGWSLIVECIGDLHIDDHHTTEDCGIALGQAFKEALGAVRGVKRFGSGFAPLDEALSRAVVDLSNRPYAVVELGLQREKVGDLSCEMIPHFLESFAEASRITLHVDCLRGKNDHHRSESAFKALAVAIREATSPNGTNDVPSTKGVLM"
)

# =========================================================
# 3. 参数配置
# =========================================================
keep_models = ["Agent", "MPNN", "Hallucination", "ESM_IF1"]
NGRAM_N = 3  # 可改成 2、3、4 等

# =========================================================
# 4. 工具函数
# =========================================================
def clean_sequence(seq):
    """标准化序列字符串。"""
    if pd.isna(seq):
        return ""
    return str(seq).strip().upper()

def get_ngrams(seq, n=3):
    """
    获取序列的 n-gram 集合。
    若序列长度 < n，则返回整个序列作为一个 token（避免空集导致无意义结果）。
    """
    seq = clean_sequence(seq)
    if not seq:
        return set()
    if len(seq) < n:
        return {seq}
    return {seq[i:i+n] for i in range(len(seq) - n + 1)}

def ngram_jaccard_similarity(seq1, seq2, n=3):
    """计算两个序列的 n-gram Jaccard similarity。"""
    grams1 = get_ngrams(seq1, n)
    grams2 = get_ngrams(seq2, n)

    if len(grams1) == 0 and len(grams2) == 0:
        return np.nan
    if len(grams1) == 0 or len(grams2) == 0:
        return np.nan

    union = grams1 | grams2
    inter = grams1 & grams2

    if len(union) == 0:
        return np.nan
    return len(inter) / len(union)

def lcs_length(seq1, seq2):
    """
    计算两个序列的最长公共子序列（LCS）长度。
    使用动态规划，空间复杂度 O(min(m, n))。
    """
    seq1 = clean_sequence(seq1)
    seq2 = clean_sequence(seq2)

    if not seq1 or not seq2:
        return 0

    # 为了节省内存，让 seq2 更短
    if len(seq1) < len(seq2):
        short, long_ = seq1, seq2
    else:
        short, long_ = seq2, seq1

    prev = [0] * (len(short) + 1)

    for i in range(1, len(long_) + 1):
        curr = [0] * (len(short) + 1)
        c1 = long_[i - 1]
        for j in range(1, len(short) + 1):
            c2 = short[j - 1]
            if c1 == c2:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    return prev[-1]

def lcs_ratio(seq1, seq2):
    """
    LCS ratio = LCS长度 / max(len(seq1), len(seq2))
    """
    seq1 = clean_sequence(seq1)
    seq2 = clean_sequence(seq2)

    if not seq1 or not seq2:
        return np.nan

    denom = max(len(seq1), len(seq2))
    if denom == 0:
        return np.nan

    return lcs_length(seq1, seq2) / denom

def sem(series):
    """计算标准误。"""
    series = pd.Series(series).dropna()
    n = len(series)
    if n <= 1:
        return np.nan
    return series.std(ddof=1) / np.sqrt(n)

def summarize_metric(df, group_col, value_col):
    """按组汇总 count / mean / std / sem。"""
    out = (
        df.groupby(group_col)[value_col]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["count"])
    return out

# =========================================================
# 5. 读取 Agent 数据
# =========================================================
agent_df = pd.read_csv(agent_csv)
if "AASeq" not in agent_df.columns:
    raise ValueError("Agent CSV 缺少 AASeq 列")

agent_df = agent_df.rename(columns={"AASeq": "sequence"}).copy()
agent_df["model_name"] = "Agent"
agent_df["sequence"] = agent_df["sequence"].apply(clean_sequence)

# =========================================================
# 6. 读取 Baseline 数据并筛选 HIS7
# =========================================================
baseline_df = pd.read_csv(baseline_csv)
required_baseline_cols = {"protein_name", "model_name", "sequence"}
missing_baseline_cols = required_baseline_cols - set(baseline_df.columns)
if missing_baseline_cols:
    raise ValueError(f"Baseline CSV 缺少必要列: {missing_baseline_cols}")

baseline_df = baseline_df.copy()
baseline_df["sequence"] = baseline_df["sequence"].apply(clean_sequence)
baseline_df = baseline_df[baseline_df["protein_name"] == "HIS7"].copy()

print("Baseline 中检测到的 model_name：", sorted(baseline_df["model_name"].dropna().unique().tolist()))

# =========================================================
# 7. 合并目标模型
# =========================================================
baseline_df = baseline_df[baseline_df["model_name"].isin(["MPNN", "Hallucination", "ESM_IF1"])].copy()
agent_df = agent_df[agent_df["model_name"] == "Agent"].copy()

all_cols = sorted(set(agent_df.columns).union(set(baseline_df.columns)))
agent_df = agent_df.reindex(columns=all_cols)
baseline_df = baseline_df.reindex(columns=all_cols)

merged_df = pd.concat([agent_df, baseline_df], ignore_index=True)
merged_df["seq_len"] = merged_df["sequence"].str.len()

if len(merged_df) == 0:
    raise ValueError("合并后没有任何序列，请检查输入文件。")

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

# 去掉空序列
merged_df = merged_df[merged_df["sequence"].str.len() > 0].copy()
merged_df.reset_index(drop=True, inplace=True)

# 给每条序列一个唯一ID，便于追踪
merged_df["seq_id"] = [f"seq_{i}" for i in range(len(merged_df))]

# =========================================================
# 8. 计算“与WT”的两类指标（逐条序列）
# =========================================================
print("\n正在计算 sequence vs WT 的 n-gram similarity 和 LCS ratio，请稍候...")

merged_df["ngram_similarity_vs_wt"] = merged_df["sequence"].apply(
    lambda x: ngram_jaccard_similarity(x, WT_SEQUENCE, n=NGRAM_N)
)
merged_df["lcs_ratio_vs_wt"] = merged_df["sequence"].apply(
    lambda x: lcs_ratio(x, WT_SEQUENCE)
)

# 保存逐条序列结果
merged_df.to_csv(output_seq_metric_csv, index=False)
print(f"已保存逐条序列指标表：{output_seq_metric_csv}")

# =========================================================
# 9. 计算“各组内序列两两之间”的两类指标
# =========================================================
print("\n正在计算各组内 pairwise n-gram similarity 和 LCS ratio，请稍候...")

pairwise_records = []

for model in keep_models:
    sub = merged_df[merged_df["model_name"] == model].copy()
    seqs = sub["sequence"].tolist()
    ids = sub["seq_id"].tolist()

    if len(seqs) < 2:
        print(f"{model}: 序列数少于2，无法计算组内两两相似性。")
        continue

    for (id1, seq1), (id2, seq2) in combinations(zip(ids, seqs), 2):
        record = {
            "model_name": model,
            "seq_id_1": id1,
            "seq_id_2": id2,
            "ngram_similarity_within_group": ngram_jaccard_similarity(seq1, seq2, n=NGRAM_N),
            "lcs_ratio_within_group": lcs_ratio(seq1, seq2),
        }
        pairwise_records.append(record)

pairwise_df = pd.DataFrame(pairwise_records)

if len(pairwise_df) == 0:
    print("警告：没有生成任何组内 pairwise 结果。")
else:
    pairwise_df.to_csv(output_pair_metric_csv, index=False)
    print(f"已保存组内两两指标表：{output_pair_metric_csv}")

# =========================================================
# 10. 汇总统计
# =========================================================
summary_parts = []

# 10.1 与 WT 的逐条序列指标汇总
for metric in ["ngram_similarity_vs_wt", "lcs_ratio_vs_wt"]:
    tmp = summarize_metric(merged_df, "model_name", metric)
    tmp["metric"] = metric
    summary_parts.append(tmp)

# 10.2 各组内 pairwise 指标汇总
if len(pairwise_df) > 0:
    for metric in ["ngram_similarity_within_group", "lcs_ratio_within_group"]:
        tmp = summarize_metric(pairwise_df, "model_name", metric)
        tmp["metric"] = metric
        summary_parts.append(tmp)

summary_df = pd.concat(summary_parts, ignore_index=True)

summary_df["model_name"] = pd.Categorical(
    summary_df["model_name"],
    categories=keep_models,
    ordered=True
)
summary_df = summary_df.sort_values(["metric", "model_name"]).reset_index(drop=True)
summary_df.to_csv(output_summary_csv, index=False)

print(f"已保存汇总统计表：{output_summary_csv}")
print("\n汇总结果：")
print(summary_df)

# =========================================================
# 11. 绘图函数
# =========================================================
def plot_bar_scatter(ax, raw_df, summary_df_metric, x_order, model_col, value_col, title, ylabel):
    """
    柱状图 + 散点 + SEM误差线
    """
    x = np.arange(len(x_order))
    means = []
    sems = []

    for model in x_order:
        row = summary_df_metric[summary_df_metric["model_name"] == model]
        if len(row) == 0:
            means.append(0)
            sems.append(0)
        else:
            means.append(row["mean"].values[0] if not pd.isna(row["mean"].values[0]) else 0)
            sems.append(row["sem"].values[0] if not pd.isna(row["sem"].values[0]) else 0)

    ax.bar(
        x,
        means,
        yerr=sems,
        capsize=5,
        alpha=0.75,
        width=0.65,
        edgecolor="black",
        linewidth=1.2
    )

    rng = np.random.default_rng(42)
    for i, model in enumerate(x_order):
        sub = raw_df[raw_df[model_col] == model]
        y = sub[value_col].dropna().values
        if len(y) == 0:
            continue
        jitter = rng.normal(0, 0.06, size=len(y))
        ax.scatter(
            np.full(len(y), x[i]) + jitter,
            y,
            alpha=0.75,
            s=26,
            edgecolors="black",
            linewidths=0.35
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_order, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

# =========================================================
# 12. 准备绘图数据
# =========================================================
plot_seq_df = merged_df.copy()
plot_seq_df["model_name"] = pd.Categorical(plot_seq_df["model_name"], categories=keep_models, ordered=True)
plot_seq_df = plot_seq_df.sort_values("model_name")

if len(pairwise_df) > 0:
    plot_pair_df = pairwise_df.copy()
    plot_pair_df["model_name"] = pd.Categorical(plot_pair_df["model_name"], categories=keep_models, ordered=True)
    plot_pair_df = plot_pair_df.sort_values("model_name")
else:
    plot_pair_df = pd.DataFrame(columns=["model_name", "ngram_similarity_within_group", "lcs_ratio_within_group"])

# =========================================================
# 13. 绘制 2×2 图
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

# 13.1 n-gram similarity vs WT
metric = "ngram_similarity_vs_wt"
summary_metric = summary_df[summary_df["metric"] == metric].copy()
plot_bar_scatter(
    ax=axes[0, 0],
    raw_df=plot_seq_df,
    summary_df_metric=summary_metric,
    x_order=keep_models,
    model_col="model_name",
    value_col=metric,
    title=f"{NGRAM_N}-gram Similarity vs WT",
    ylabel=f"{NGRAM_N}-gram Jaccard Similarity"
)

# 13.2 within-group n-gram similarity
metric = "ngram_similarity_within_group"
summary_metric = summary_df[summary_df["metric"] == metric].copy()
plot_bar_scatter(
    ax=axes[0, 1],
    raw_df=plot_pair_df,
    summary_df_metric=summary_metric,
    x_order=keep_models,
    model_col="model_name",
    value_col=metric,
    title=f"Within-group {NGRAM_N}-gram Similarity",
    ylabel=f"{NGRAM_N}-gram Jaccard Similarity"
)

# 13.3 LCS ratio vs WT
metric = "lcs_ratio_vs_wt"
summary_metric = summary_df[summary_df["metric"] == metric].copy()
plot_bar_scatter(
    ax=axes[1, 0],
    raw_df=plot_seq_df,
    summary_df_metric=summary_metric,
    x_order=keep_models,
    model_col="model_name",
    value_col=metric,
    title="LCS Ratio vs WT",
    ylabel="LCS Ratio"
)

# 13.4 within-group LCS ratio
metric = "lcs_ratio_within_group"
summary_metric = summary_df[summary_df["metric"] == metric].copy()
plot_bar_scatter(
    ax=axes[1, 1],
    raw_df=plot_pair_df,
    summary_df_metric=summary_metric,
    x_order=keep_models,
    model_col="model_name",
    value_col=metric,
    title="Within-group LCS Ratio",
    ylabel="LCS Ratio"
)

plt.suptitle("HIS7 Sequence Similarity Analysis", fontsize=15, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(output_fig, bbox_inches="tight")
plt.show()

print(f"\n图片已保存：{output_fig}")

# =========================================================
# 14. 额外输出：更直观的四张汇总表
# =========================================================
def get_metric_summary_table(summary_df, metric_name, keep_models):
    sub = summary_df[summary_df["metric"] == metric_name].copy()
    sub["model_name"] = pd.Categorical(sub["model_name"], categories=keep_models, ordered=True)
    sub = sub.sort_values("model_name").reset_index(drop=True)
    return sub

print("\n================ 汇总表（1）与WT的 n-gram similarity ================")
print(get_metric_summary_table(summary_df, "ngram_similarity_vs_wt", keep_models))

print("\n================ 汇总表（2）组内 n-gram similarity ================")
print(get_metric_summary_table(summary_df, "ngram_similarity_within_group", keep_models))

print("\n================ 汇总表（3）与WT的 LCS ratio ================")
print(get_metric_summary_table(summary_df, "lcs_ratio_vs_wt", keep_models))

print("\n================ 汇总表（4）组内 LCS ratio ================")
print(get_metric_summary_table(summary_df, "lcs_ratio_within_group", keep_models))