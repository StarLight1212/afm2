import os
import sys
import re
import warnings
import torch
import pandas as pd
from typing import List, Set
from transformers import PreTrainedTokenizerFast, GenerationMixin

warnings.filterwarnings("ignore")


# =========================
# 1. 基础配置
# =========================
MODEL_DIR = "/home/data2/public/guoweis/pretrain/Progen2/progen2-small"
OUTPUT_CSV = "generated_progen2_small_unconditional_128.csv"

TARGET_COUNT = 128           # 需要最终保留的序列数
MIN_LENGTH = 128             # 序列长度至少 128
MAX_LENGTH = 512             # 生成上限，可按需调整
BATCH_SIZE = 32              # 显存允许可继续增大
TEMPERATURE = 1.0
TOP_P = 0.95

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# =========================
# 2. 加入工程路径，导入本地 ProGen2 类
# =========================
def setup_import_path(model_dir: str):
    """
    假设你的目录结构类似：
    /home/data2/public/guoweis/pretrain/Progen2/
        ├── models/
        ├── progen2-small/
        └── ...
    那么要把 Progen2 根目录加进 sys.path
    """
    progen_root = os.path.dirname(model_dir)   # -> /home/data2/public/guoweis/pretrain/Progen2
    if progen_root not in sys.path:
        sys.path.insert(0, progen_root)


def get_model_class():
    """
    动态补上 GenerationMixin，避免某些 transformers 版本里 generate 不可用。
    """
    from models.modeling_progen import ProGenForCausalLM

    if not issubclass(ProGenForCausalLM, GenerationMixin):
        print("检测到 ProGenForCausalLM 缺少 GenerationMixin，正在动态注入...")
        class PatchedProGenForCausalLM(ProGenForCausalLM, GenerationMixin):
            pass
        return PatchedProGenForCausalLM
    return ProGenForCausalLM


# =========================
# 3. 清洗生成结果
# =========================
def clean_sequence(seq: str) -> str:
    """
    清理生成文本，只保留标准氨基酸字符。
    同时去掉可能残留的控制 token、空格、特殊符号。
    """
    seq = seq.strip()

    # 去空格和换行
    seq = seq.replace(" ", "").replace("\n", "").replace("\r", "")

    # 仅保留标准20种氨基酸字符
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())

    return seq


# =========================
# 4. 加载 tokenizer 和 model
# =========================
def load_tokenizer_and_model(model_dir: str, device: str):
    print(f"Loading tokenizer from: {model_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_dir,
        local_files_only=True
    )

    print(f"Loading model from: {model_dir}")
    ModelClass = get_model_class()
    model = ModelClass.from_pretrained(
        model_dir,
        local_files_only=True
    )
    model.to(device)
    model.eval()

    # 这里非常关键：
    # 你前面的代码把 "1" 作为起始 token，所以 eos/pad 也通常要和对应 special token 一致。
    # 这里优先尝试 "2"，若没有则退回 tokenizer.eos_token_id / pad_token_id。
    eos_id = tokenizer.convert_tokens_to_ids("2")
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        eos_id = tokenizer.eos_token_id

    pad_id = eos_id if eos_id is not None else tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    return tokenizer, model, eos_id, pad_id


# =========================
# 5. 无条件生成
# =========================
def generate_unconditional_sequences(
    tokenizer,
    model,
    eos_id: int,
    pad_id: int,
    target_count: int = 128,
    min_length: int = 128,
    max_length: int = 512,
    batch_size: int = 32,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cuda:0"
) -> List[str]:
    """
    ProGen2 的“无条件生成”：
    使用最小起始 prompt = "1"
    不附加任何蛋白前缀。
    """
    unique_seqs: Set[str] = set()

    # ProGen2 常用起始控制 token
    prompt = "1"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print(f"开始生成，目标保留 {target_count} 条长度 >= {min_length} 的唯一序列...")

    round_id = 0
    while len(unique_seqs) < target_count:
        round_id += 1

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=batch_size,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        accepted_this_round = 0
        for seq in decoded:
            seq = clean_sequence(seq)

            if len(seq) < min_length:
                continue

            if seq not in unique_seqs:
                unique_seqs.add(seq)
                accepted_this_round += 1

            if len(unique_seqs) >= target_count:
                break

        print(
            f"Round {round_id:03d} | 本轮新增 {accepted_this_round} 条 | "
            f"累计 {len(unique_seqs)}/{target_count}"
        )

    return list(unique_seqs)


# =========================
# 6. 保存结果
# =========================
def save_sequences_to_csv(seqs: List[str], output_csv: str):
    df = pd.DataFrame({
        "id": [f"seq_{i+1:03d}" for i in range(len(seqs))],
        "sequence": seqs,
        "length": [len(s) for s in seqs]
    })
    df.to_csv(output_csv, index=False)
    print(f"已保存到: {output_csv}")


# =========================
# 7. 主程序
# =========================
def main():
    print(f"Using device: {DEVICE}")

    setup_import_path(MODEL_DIR)
    tokenizer, model, eos_id, pad_id = load_tokenizer_and_model(MODEL_DIR, DEVICE)

    seqs = generate_unconditional_sequences(
        tokenizer=tokenizer,
        model=model,
        eos_id=eos_id,
        pad_id=pad_id,
        target_count=TARGET_COUNT,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        device=DEVICE
    )

    save_sequences_to_csv(seqs, OUTPUT_CSV)


if __name__ == "__main__":
    main()