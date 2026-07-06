input_file = "./design_PDL1_binder_rf_mpnn.fasta"
output_file = "./design_PDL1_binder_rf_mpnn_clean.fasta"

def clean_fasta(infile, outfile):
    with open(infile, "r") as f:
        lines = f.readlines()

    sequences = []
    current_seq = ""
    header = None

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if header and current_seq:
                sequences.append((header, current_seq))
            header = line
            current_seq = ""
        else:
            current_seq += line

    if header and current_seq:
        sequences.append((header, current_seq))

    # 写入新 fasta
    with open(outfile, "w") as f:
        for i, (h, seq) in enumerate(sequences):
            seq = seq.replace("/", ":")   # ⭐ 核心修改
            f.write(f"{h}\n")
            f.write(f"{seq}\n")

    print(f"处理完成：{outfile}")

clean_fasta(input_file, output_file)