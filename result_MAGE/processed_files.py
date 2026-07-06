import pandas as pd
import json
import os
from collections import defaultdict

# 1. 定义文件列表
file_paths = [
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_antibody_design/result/design_benchmark/split_vis_protein_name/1mhp.csv",
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_antibody_design/result/design_benchmark/split_vis_protein_name/1mlc.csv",
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_antibody_design/result/design_benchmark/split_vis_protein_name/1n8z.csv",
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_antibody_design/result/design_benchmark/split_vis_protein_name/2fjg.csv",
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_antibody_design/result/design_benchmark/split_vis_protein_name/4fqi.csv",
    "/home/data2/public/guoweis/pretrain/agent/figs/main_fig_antibody_design/result/design_benchmark/split_vis_protein_name/aay149.csv"
]

output_file = "merged_antibody_design.jsonl"
final_data = []

# 用于记录每个组合出现的次数
# key 为 "protein_name_model_name", value 为当前出现的次数
name_counter = defaultdict(int)

# 2. 循环读取并处理
for path in file_paths:
    if not os.path.exists(path):
        print(f"警告: 文件不存在 {path}")
        continue
    
    # 读取CSV
    df = pd.read_csv(path)
    
    # 3. 按照格式转换每一行
    for _, row in df.iterrows():
        # 获取基础 ID
        base_id = f"{row['protein_name']}_{row['model_name']}"
        
        # 计数器累加：如果该 base_id 是第一次出现，则变为 1
        name_counter[base_id] += 1
        current_count = name_counter[base_id]
        
        # 拼接最终 ID：使用 :03d 格式化为 3 位数字（如 001, 002）
        final_id = f"{base_id}_{current_count:03d}"
        
        # 组装 sequence
        # 处理可能存在的 NaN 值（如果有空链的情况）
        antigen = str(row['antigen']) if pd.notna(row['antigen']) else ""
        hc = str(row['hc']) if pd.notna(row['hc']) else ""
        lc = str(row['lc']) if pd.notna(row['lc']) else ""
        
        entry = {
            "id": final_id,
            "sequence": f"{antigen}:{hc}:{lc}"
        }
        final_data.append(entry)

# 4. 保存为 JSONL 格式
with open(output_file, 'w', encoding='utf-8') as f:
    for item in final_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"处理完成！")
print(f"共合并 {len(final_data)} 条数据。")
print(f"结果已保存至: {output_file}")

# 打印前 5 条示例查看 ID 效果
print("\n前 5 条数据示例：")
for sample in final_data[:5]:
    print(sample)