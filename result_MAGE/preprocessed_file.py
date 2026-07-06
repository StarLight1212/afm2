import os
import re
import pandas as pd

root_dir = "result_MAGE"
task_names = ["1mhp", "1mlc", "1n8z", "2fjg", "4fqi", "aay149"]

# 更宽松、更稳妥的正则
# 格式目标：antigen[SEP]HC[LC]LC<|pad|>
pattern = re.compile(r"^(.*?)\[SEP\](.*?)\[LC\](.*?)(?:<\|pad\|>)*$")

for task in task_names:
    input_csv = os.path.join(root_dir, task, "seq.csv")
    output_csv = os.path.join(root_dir, task, "seq_clean.csv")

    if not os.path.exists(input_csv):
        print(f"[跳过] 文件不存在: {input_csv}")
        continue

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"[错误] 无法读取 {input_csv}: {e}")
        continue

    # 自动识别列名 0 / "0"
    if "0" in df.columns:
        seq_col = "0"
    elif 0 in df.columns:
        seq_col = 0
    else:
        print(f"[跳过] {input_csv} 中不存在列名 0 或 '0'")
        print(f"实际列名: {list(df.columns)}")
        continue

    results = []
    unmatched_rows = []

    for idx, seq in df[seq_col].items():
        if pd.isna(seq):
            unmatched_rows.append((idx, "NaN"))
            continue

        seq = str(seq).strip()

        # 先做轻微清洗
        seq = seq.replace("\n", "").replace("\r", "").strip()

        match = pattern.match(seq)
        if match:
            antigen, hc, lc = match.groups()

            # 再保险清理一下末尾 pad
            antigen = antigen.strip()
            hc = hc.strip()
            lc = re.sub(r"(?:<\|pad\|>)+$", "", lc).strip()

            results.append({
                "antigen": antigen,
                "HC": hc,
                "LC": lc
            })
        else:
            unmatched_rows.append((idx, seq))

    clean_df = pd.DataFrame(results)
    clean_df.to_csv(output_csv, index=False)

    print(f"[完成] {input_csv} -> {output_csv}")
    print(f"成功提取: {len(clean_df)} 条")
    print(f"未匹配: {len(unmatched_rows)} 条")

    # 打印前几条未匹配样本，方便排查
    if unmatched_rows:
        print("前 5 条未匹配示例：")
        for row_idx, bad_seq in unmatched_rows[:5]:
            print(f"  行 {row_idx}: {repr(bad_seq[:200])}")