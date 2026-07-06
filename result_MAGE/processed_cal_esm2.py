import pandas as pd
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import os
from tqdm import tqdm

# 1. 配置参数
model_path = "/home/data2/public/guoweis/pretrain/agent/ESM2"
file_paths = [
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_single_protein_design/result/split_vis_protein_name/KCN2_with_fitness.csv",
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_single_protein_design/result/split_vis_protein_name/HIS7_with_fitness.csv",
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_single_protein_design/result/split_vis_protein_name/AAV2_with_fitness.csv"
]

# 2. 加载模型和分词器
print(f"正在从本地加载 ESM2 模型: {model_path}...")
tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmForMaskedLM.from_pretrained(model_path)

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def calculate_esm2_naturalness(sequence):
    """
    计算序列的平均对数似然 (Average Log-Likelihood) 作为天然性指标
    """
    if not isinstance(sequence, str) or len(sequence) == 0:
        return None
    
    # 将序列中的非标准氨基酸或空格处理（ESM2通常处理20种标准氨基酸）
    sequence = sequence.upper().replace(" ", "")
    
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()
    
    # 对于 ESM2 这种 Masked LM，我们通过不 mask 任何 token 直接 forward
    # 得到每个位置上原始氨基酸的预测概率
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        # loss 是 CrossEntropyLoss，即 -log(p)
        # 这里返回的是序列所有 token 的平均 negative log likelihood
        neg_log_likelihood = outputs.loss.item()
    
    # 我们取负数，使得得分越高（越接近0）代表越“自然”
    naturalness_score = -neg_log_likelihood
    return naturalness_score

# 3. 循环处理文件
for path in file_paths:
    if not os.path.exists(path):
        print(f"跳过不存在的文件: {path}")
        continue
    
    print(f"正在处理文件: {os.path.basename(path)}")
    df = pd.read_csv(path)
    
    # 确保存储得分的列表
    scores = []
    
    # 遍历序列进行计算
    for seq in tqdm(df['sequence'], desc="计算分数"):
        try:
            score = calculate_esm2_naturalness(seq)
            scores.append(score)
        except Exception as e:
            print(f"计算序列时出错: {e}")
            scores.append(None)
    
    # 将结果添加到新列
    df['ESM2_Naturalness'] = scores
    
    # 保存结果（建议另存为新文件或覆盖原文件）
    output_path = path.replace(".csv", "_with_esm2.csv")
    df.to_csv(output_path, index=False)
    print(f"结果已保存至: {output_path}")

print("\n所有任务处理完成！")