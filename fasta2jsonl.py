import json

# input_fasta = "./prepared_fasta/mpnn_binder_designs.fasta"
# output_jsonl = "./prepared_fasta/mpnn_binder_designs.jsonl"
input_fasta = "./design_0331.fasta"
output_jsonl = "./design_0331.jsonl"

def parse_fasta_to_jsonl(fasta_path, output_path):
    with open(fasta_path, "r") as f, open(output_path, "w") as out:
        seq_id = None
        seq_lines = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            # 遇到新的header
            if line.startswith(">"):
                # 先写入上一个
                if seq_id is not None:
                    sequence = "".join(seq_lines).replace("/", ":")
                    json_obj = {
                        "id": seq_id,
                        "sequence": sequence
                    }
                    out.write(json.dumps(json_obj) + "\n")

                # 更新新的ID
                seq_id = line[1:]  # 去掉 >
                seq_lines = []

            else:
                seq_lines.append(line)

        # 写入最后一个
        if seq_id is not None:
            sequence = "".join(seq_lines).replace("/", ":")
            json_obj = {
                "id": seq_id,
                "sequence": sequence
            }
            out.write(json.dumps(json_obj) + "\n")


if __name__ == "__main__":
    parse_fasta_to_jsonl(input_fasta, output_jsonl)
    print(f"转换完成：{output_jsonl}")