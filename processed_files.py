import os
import json
import csv
from statistics import mean

# =========================
# 配置区
# =========================
input_dir = "result_0330_batch"
output_csv = "result_0330_batch_summary.csv"

# 仅处理这类文件
target_suffix = "seed_000.json"

# =========================
# 主逻辑
# =========================
rows = []

for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if not file_name.endswith(target_suffix):
            continue

        file_path = os.path.join(root, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 文件名去后缀，作为 json_name_stem
            json_name_stem = os.path.splitext(file_name)[0]

            # 提取 plddt 并求均值
            plddt_list = data.get("plddt", [])
            if isinstance(plddt_list, list) and len(plddt_list) > 0:
                avg_plddt = mean(plddt_list)
            else:
                avg_plddt = ""

            # 提取 ptm / iptm
            ptm = data.get("ptm", "")
            iptm = data.get("iptm", "")

            rows.append([
                json_name_stem,
                avg_plddt,
                ptm,
                iptm
            ])

        except Exception as e:
            print(f"处理失败: {file_path}")
            print(f"错误信息: {e}")

# 可选：按文件名排序
rows.sort(key=lambda x: x[0])

# =========================
# 写出 CSV
# =========================
with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["json_name_stem", "plddt", "PTM", "iPTM"])
    writer.writerows(rows)

print(f"完成，共提取 {len(rows)} 个 JSON 文件的数据。")
print(f"输出文件: {output_csv}")