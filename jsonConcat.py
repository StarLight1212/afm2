import json

input_file = "./prepared_fasta/mpnn_binder_designs.jsonl"
output_file = "./prepared_fasta/mpnn_binder_designs_processed_output.jsonl"

# 需要拼接的序列
append_seq = ":HHEVVKFMDVYQRSYCHPIETLVDIFQEYPDEIEYIFKPSCVPLMRCGGCCNDEGLECVPTEESNITMQIMRIKPHQGQHIGEMSFLQHNKCECRPKKD"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)

        # 拼接序列
        data["sequence"] = data["sequence"] + append_seq

        # 写入新文件
        fout.write(json.dumps(data) + "\n")

print(f"处理完成，输出文件：{output_file}")
