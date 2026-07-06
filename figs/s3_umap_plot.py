import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap.umap_ as umap


# =========================================================
# 1. 路径配置
# =========================================================
agent_csv = "./data/generated_seqs_iter_140_with_fitness_his7.csv"
baseline_csv = "./data/designed_sequences_baseline.csv"
output_dir = "./result_umap/"

os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 2. WT 序列
# =========================================================
WT_SEQUENCE = (
    "MTEQKALVKRITNETKIQIAISLKGGPLAIEHSIFPEKEAEAVAEQATQSQVINVHTGIGFLDHMIHALAKHSGWSLIVECIGDLHIDDHHTTEDCGIALGQAFKEALGAVRGVKRFGSGFAPLDEALSRAVVDLSNRPYAVVELGLQREKVGDLSCEMIPHFLESFAEASRITLHVDCLRGKNDHHRSESAFKALAVAIREATSPNGTNDVPSTKGVLM"
)

# =========================================================
# 3. 基础配置
# =========================================================
keep_models = ["Agent", "MPNN", "Hallucination", "ESM_IF1"]

color_map = {
    "Agent": "#D62728",         # 红
    "MPNN": "#1F77B4",          # 蓝
    "Hallucination": "#2CA02C", # 绿
    "ESM_IF1": "#9467BD",       # 紫
    "WT": "#000000"             # 黑
}

# 氨基酸字典
alphabet = ['<PAD>', 'X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'V', 'W', 'Y']
word_dict = {aa: i for i, aa in enumerate(alphabet)}
vocab_size = len(alphabet)


# =========================================================
# 4. 工具函数
# =========================================================
def clean_sequence(seq):
    """清洗序列字符串。"""
    if pd.isna(seq):
        return ""
    return str(seq).strip().upper()


def sequence_to_numeric(sequence, max_len):
    """
    将蛋白序列转成整数编码。
    未知字符映射到 X。
    长度不足补 PAD。
    """
    sequence = clean_sequence(sequence)
    output = []

    for aa in sequence:
        if aa in word_dict:
            output.append(word_dict[aa])
        else:
            output.append(word_dict['X'])

    if len(output) < max_len:
        output.extend([word_dict['<PAD>']] * (max_len - len(output)))

    return np.array(output[:max_len], dtype=np.int32)


def numeric_to_binary_features(numeric_matrix, vocab_size):
    """
    将 [N, L] 的整数矩阵转换为 [N, L*vocab_size] 的二值特征矩阵。
    每个位点的氨基酸类型做 one-hot。
    这样更适合 UMAP + jaccard。
    """
    n_samples, seq_len = numeric_matrix.shape
    binary = np.zeros((n_samples, seq_len * vocab_size), dtype=np.uint8)

    for i in range(n_samples):
        for pos in range(seq_len):
            aa_idx = numeric_matrix[i, pos]
            feature_idx = pos * vocab_size + aa_idx
            binary[i, feature_idx] = 1

    return binary


def load_and_prepare_data():
    """
    读取 Agent / Baseline 数据，并加入 WT。
    返回：
        feature_matrix: 用于 UMAP 的二值矩阵
        plot_df: 包含 sequence / model_name 等信息的 DataFrame
        group_info: [(group_name, count), ...]
    """
    # -------------------------
    # 读取 Agent 数据
    # -------------------------
    agent_df = pd.read_csv(agent_csv)
    if "AASeq" not in agent_df.columns:
        raise ValueError("Agent CSV 缺少 AASeq 列")

    agent_df = agent_df.rename(columns={"AASeq": "sequence"}).copy()
    agent_df["model_name"] = "Agent"
    agent_df["sequence"] = agent_df["sequence"].apply(clean_sequence)

    # -------------------------
    # 读取 Baseline 数据
    # -------------------------
    baseline_df = pd.read_csv(baseline_csv)
    required_baseline_cols = {"protein_name", "model_name", "sequence"}
    missing_cols = required_baseline_cols - set(baseline_df.columns)
    if missing_cols:
        raise ValueError(f"Baseline CSV 缺少必要列: {missing_cols}")

    baseline_df = baseline_df.copy()
    baseline_df["sequence"] = baseline_df["sequence"].apply(clean_sequence)
    baseline_df = baseline_df[baseline_df["protein_name"] == "HIS7"].copy()
    baseline_df = baseline_df[baseline_df["model_name"].isin(["MPNN", "Hallucination", "ESM_IF1"])].copy()

    print("Baseline 中检测到的 model_name：", sorted(baseline_df["model_name"].dropna().unique().tolist()))

    # -------------------------
    # 合并
    # -------------------------
    all_cols = sorted(set(agent_df.columns).union(set(baseline_df.columns)))
    agent_df = agent_df.reindex(columns=all_cols)
    baseline_df = baseline_df.reindex(columns=all_cols)

    merged_df = pd.concat([agent_df, baseline_df], ignore_index=True)
    merged_df = merged_df[merged_df["model_name"].isin(keep_models)].copy()
    merged_df = merged_df[merged_df["sequence"].str.len() > 0].copy()

    if len(merged_df) == 0:
        raise ValueError("合并后没有任何序列，请检查输入文件。")

    # -------------------------
    # 加入 WT
    # -------------------------
    wt_df = pd.DataFrame({
        "sequence": [WT_SEQUENCE],
        "model_name": ["WT"]
    })

    wt_df = wt_df.reindex(columns=merged_df.columns.union(["sequence", "model_name"]))
    merged_df = merged_df.reindex(columns=wt_df.columns)
    plot_df = pd.concat([merged_df, wt_df], ignore_index=True)

    # -------------------------
    # 长度信息
    # -------------------------
    plot_df["seq_len"] = plot_df["sequence"].str.len()
    max_len = plot_df["seq_len"].max()

    print(f"\nWT 长度: {len(WT_SEQUENCE)}")
    print(f"全局最大序列长度: {max_len}")
    print("\n各组序列数量：")
    for name in keep_models + ["WT"]:
        n = (plot_df["model_name"] == name).sum()
        print(f"{name}: {n}")

    # -------------------------
    # 编码
    # -------------------------
    numeric_data = np.array([sequence_to_numeric(seq, max_len) for seq in plot_df["sequence"]])
    feature_matrix = numeric_to_binary_features(numeric_data, vocab_size=vocab_size)

    # -------------------------
    # group_info（按绘图顺序）
    # -------------------------
    ordered_groups = keep_models + ["WT"]
    group_info = []
    for g in ordered_groups:
        group_info.append((g, (plot_df["model_name"] == g).sum()))

    return feature_matrix, plot_df, group_info


def plot_and_save(embedding, plot_df, out_path, title):
    """
    将 embedding 按组绘图并保存。
    """
    plt.figure(figsize=(10, 8), dpi=300)

    ordered_groups = keep_models + ["WT"]

    for group in ordered_groups:
        sub = plot_df[plot_df["model_name"] == group]
        idx = sub.index.to_numpy()

        if len(idx) == 0:
            continue

        if group == "WT":
            plt.scatter(
                embedding[idx, 0],
                embedding[idx, 1],
                c=color_map[group],
                label=group,
                s=140,
                alpha=1.0,
                edgecolors="gold",
                linewidths=1.5,
                marker="*",
                zorder=10
            )
        else:
            plt.scatter(
                embedding[idx, 0],
                embedding[idx, 1],
                c=color_map[group],
                label=group,
                s=14,
                alpha=0.70,
                edgecolors="black",
                linewidths=0.2
            )

    plt.legend(loc="best", frameon=False)
    plt.title(title, fontsize=12)
    plt.xlabel("UMAP1", fontsize=11)
    plt.ylabel("UMAP2", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# =========================================================
# 5. 主程序：UMAP 网格搜索
# =========================================================
if __name__ == "__main__":
    print("正在准备数据...")
    input_array, plot_df, group_info = load_and_prepare_data()

    # -----------------------------------------------------
    # UMAP 网格搜索参数
    # -----------------------------------------------------
    neighbors_list = [2, 5, 7, 9, 11, 16, 20, 25, 30, 35]
    min_dists_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    rand_states_list = [k for k in range(0, 80, 3)]
    epochs_list = [30, 50]
    lr = 0.3

    total_jobs = (
        len(neighbors_list)
        * len(min_dists_list)
        * len(rand_states_list)
        * len(epochs_list)
    )

    print(f"\n开始搜索，预计生成 {total_jobs} 张图...")
    print(f"输出目录：{output_dir}")

    for n_nb, m_dist, r_state, ep in itertools.product(
        neighbors_list, min_dists_list, rand_states_list, epochs_list
    ):
        file_name = f"umap_nb{n_nb}_dist{m_dist}_rs{r_state}_ep{ep}.png"
        out_path = os.path.join(output_dir, file_name)

        # 跳过已存在文件，支持断点续跑
        if os.path.exists(out_path):
            continue

        try:
            reducer = umap.UMAP(
                n_neighbors=n_nb,
                min_dist=m_dist,
                metric="jaccard",
                n_epochs=ep,
                learning_rate=lr,
                random_state=r_state,
                init="pca",
                n_components=2,
                verbose=False
            )

            embedding = reducer.fit_transform(input_array)

            title = (
                f"HIS7 UMAP | n_neighbors={n_nb}, min_dist={m_dist}, "
                f"random_state={r_state}, n_epochs={ep}"
            )

            plot_and_save(
                embedding=embedding,
                plot_df=plot_df,
                out_path=out_path,
                title=title
            )

            print(f"已完成: {file_name}")

        except Exception as e:
            print(f"渲染错误 {file_name}: {e}")

    print("\n全部任务完成。")