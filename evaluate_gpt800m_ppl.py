import os
import sys
import math
import json
import random
import logging
import csv
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Environment & Path Configurations
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 确保在指定路径下运行
ROOT_DIR = "/home/data2/public/guoweis/pretrain/agent"
sys.path.append(ROOT_DIR)

from models.GPTModel.GPT import GPTModel, GPTConfig
from models.GPTModel.tokenizer import AminoAcidTokenizer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==========================================
# 2. Data Processing Modules
# ==========================================
def sample_uniref_sequences(dataset, num_samples: int, seed: int, seq_column: str = "sequence") -> List[str]:
    """
    从Huggingface Dataset中随机抽取指定数量的序列，避免全量加载到内存
    """
    total_len = len(dataset)
    random.seed(seed)
    
    actual_samples = min(num_samples, total_len)
    logger.info(f"Randomly sampling {actual_samples} sequences (Seed: {seed})...")
    
    # 随机生成索引并提取序列
    indices = random.sample(range(total_len), actual_samples)
    
    # 批量根据索引提取以加快速度
    sampled_data = dataset.select(indices)
    sampled_sequences = [str(item[seq_column]).strip() for item in sampled_data]
    
    return sampled_sequences


class SingleChainDataset(Dataset):
    """Dataset class for individual protein chains."""
    def __init__(self, sequences: List[str], tokenizer: AminoAcidTokenizer, max_len: int):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        seq = self.sequences[idx]
        tokenized = self.tokenizer.encode(seq, add_special_tokens=True)
        if len(tokenized) > self.max_len:
            tokenized = tokenized[:self.max_len]
        return {"input_ids": tokenized}


class ProteinDataCollator:
    """Collator for dynamic padding and calculating Causal LM labels."""
    def __init__(self, tokenizer: AminoAcidTokenizer, max_len: int):
        self.pad_token_id = tokenizer.pad_id
        self.max_len = max_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features if f]
        
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        if padded_input_ids.size(1) > self.max_len:
            padded_input_ids = padded_input_ids[:, :self.max_len]

        attention_mask = (padded_input_ids != self.pad_token_id).bool()
        
        labels = padded_input_ids.clone()
        labels[labels == self.pad_token_id] = -100 

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# ==========================================
# 3. Model Wrapper (Loss Calculation)
# ==========================================
class GPTForCausalLMWrapper(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.model = GPTModel(config)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tie_weights()

    def tie_weights(self) -> None:
        self.model.lm_head.weight = self.model.transformer.wte.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss}

# ==========================================
# 4. Evaluation Helper Function
# ==========================================
def evaluate_dataset(dataloader: DataLoader, model: nn.Module, device: torch.device, desc: str) -> float:
    """Evaluates and returns perplexity."""
    total_loss_sum = 0.0
    total_valid_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            batch_loss = outputs["loss"]
            
            shift_labels = labels[..., 1:].contiguous()
            valid_tokens_in_batch = (shift_labels != -100).sum().item()
            
            if valid_tokens_in_batch > 0:
                total_loss_sum += batch_loss.item() * valid_tokens_in_batch
                total_valid_tokens += valid_tokens_in_batch

    if total_valid_tokens > 0:
        global_avg_loss = total_loss_sum / total_valid_tokens
        try:
            perplexity = math.exp(global_avg_loss)
        except OverflowError:
            perplexity = float("inf")
    else:
        perplexity = float("inf")
        
    return perplexity

# ==========================================
# 5. Plotting Function
# ==========================================
def plot_ppl_trend(steps, means, stds, save_path):
    plt.figure(figsize=(10, 6))
    
    # 绘制带误差棒的折线图
    plt.errorbar(steps, means, yerr=stds, fmt='-o', color='b', ecolor='r', 
                 capsize=5, capthick=2, markersize=8, label='PPL ± StdDev')
    
    # 图像美化
    plt.title('GPT_800M Perplexity vs. Training Steps (UniRef90)', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Perplexity (PPL)', fontsize=12)
    plt.xticks(steps)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved successfully to: {save_path}")
    plt.close()

# ==========================================
# 6. Main Execution
# ==========================================
def main():
    # 路径配置
    DATA_PATH = "/home/data2/public/guoweis/pretrain/ProtGPT/data/uniref90_no_rank_hf"
    VOCAB_PATH = f"{ROOT_DIR}/models/GPTModel/vocab.json"
    
    # 注意：你需要指定一个GPT_800M的config文件路径。此处预留位置，请确保路径正确。
    CONFIG_PATH = f"{ROOT_DIR}/models/weights/GPT_800M/model_config.json"  # <--- 请检查这里是否正确
    
    # Checkpoint 映射字典
    CHECKPOINTS = {
        43000: f"{ROOT_DIR}/models/weights/GPT_800M/pytorch_model_0314.bin",
        60000: f"{ROOT_DIR}/models/weights/GPT_800M/pytorch_model.bin",
        82000: f"{ROOT_DIR}/models/weights/GPT_800M/pytorch_model_step_82000.bin",
        100000: f"{ROOT_DIR}/models/weights/GPT_800M/pytorch_model_step100000.bin"
    }
    
    SEEDS = [42, 43, 44]
    NUM_SAMPLES = 5000
    BATCH_SIZE = 4 # 根据显存大小可调节 (800M模型通常可以开到4或8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. 加载 Config 和 Tokenizer
    with open(CONFIG_PATH, "r") as f:
        config_dict = json.load(f)
    config = GPTConfig(**config_dict)
    tokenizer = AminoAcidTokenizer(VOCAB_PATH)
    collator = ProteinDataCollator(tokenizer, max_len=config.max_seq_len)

    # 2. 加载 Dataset (这里只加载元数据，不会爆内存)
    logger.info(f"Loading HF dataset from {DATA_PATH}...")
    hf_dataset = load_from_disk(DATA_PATH)
    # 检查UniRef90中表示序列的列名（一般是'sequence'，如果是其他名字请修改）
    seq_col = "sequence" if "sequence" in hf_dataset.column_names else hf_dataset.column_names[0]

    # 用于保存结果的数据结构
    results = {step: [] for step in CHECKPOINTS.keys()}

    # 3. 循环遍历每个 Checkpoint
    for step, weight_path in CHECKPOINTS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating Model Step: {step}")
        logger.info(f"Loading weights from: {weight_path}")
        
        # 初始化模型并加载权重
        model_wrapper = GPTForCausalLMWrapper(config)
        pretrained_weights = torch.load(weight_path, map_location="cpu")
        model_wrapper.model.load_state_dict(pretrained_weights)
        model_wrapper.to(device)
        model_wrapper.eval()

        # 4. 对该模型进行3次不同seed的采样和测试
        for i, seed in enumerate(SEEDS):
            logger.info(f"  --- Run {i+1}/3 | Seed: {seed} ---")
            
            # 抽样
            sampled_seqs = sample_uniref_sequences(hf_dataset, num_samples=NUM_SAMPLES, seed=seed, seq_column=seq_col)
            
            # 构建 Dataloader
            dataset = SingleChainDataset(sampled_seqs, tokenizer, max_len=config.max_seq_len)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=4, pin_memory=True)
            
            # 测试 PPL
            ppl = evaluate_dataset(dataloader, model_wrapper, device, desc=f"Eval Step {step} Seed {seed}")
            results[step].append(ppl)
            logger.info(f"  Result -> PPL: {ppl:.4f}")

        # 释放显存，准备加载下一个模型
        del model_wrapper
        del pretrained_weights
        torch.cuda.empty_cache()

    # 5. 打印汇总报告并画图
    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION REPORT")
    logger.info("=" * 50)
    
    steps_list = sorted(list(results.keys()))
    means = []
    stds = []
    
    for step in steps_list:
        ppls = results[step]
        mean_ppl = np.mean(ppls)
        std_ppl = np.std(ppls)
        means.append(mean_ppl)
        stds.append(std_ppl)
        logger.info(f"Step {step:6d} | PPL Mean: {mean_ppl:.4f} | PPL StdDev: {std_ppl:.4f} | Runs: {ppls}")

    # 绘制并保存图片
    save_fig_path = f"{ROOT_DIR}/figs/gpt800m_ppl_trend.png"
    plot_ppl_trend(steps_list, means, stds, save_fig_path)

if __name__ == "__main__":
    main()