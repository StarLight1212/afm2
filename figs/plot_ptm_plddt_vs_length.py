"""
PTM 与 pLDDT 随 sequence length 变化：每个 length 取 PTM、pLDDT 各自的最大值，
滑动窗口平滑后分别出图；散点仅保留每个 length 上分数最高的样本（同分并列都保留），
以轻微抖动画在曲线周围，保存为 SVG（无网格线）。
数据：../results_0423.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results_0423.csv"
OUT_DIR = Path(__file__).resolve().parent

# 滑动窗口大小（奇数更利于 center=True 对称平滑，可按需要改）
ROLLING_WINDOW = 7
# 散点抖动：围绕真实 (length, score) 小幅偏移，避免与平滑线完全重合
JITTER_X = 0.45
JITTER_Y_PTM = 0.012
JITTER_Y_PLDDT = 0.012
SCATTER_SEED = 42
SCATTER_KW = dict(s=10, alpha=0.28, linewidths=0, zorder=1)


def rolling_smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


df = pd.read_csv(CSV_PATH)
agg = df.groupby("length", as_index=False).agg({"PTM": "max", "pLDDT": "max"})
agg = agg.sort_values("length")

w = ROLLING_WINDOW
agg["PTM_sm"] = rolling_smooth(agg["PTM"], w)
agg["pLDDT_sm"] = rolling_smooth(agg["pLDDT"], w)

# 每个 length 只保留 PTM / pLDDT 等于该组最大值的行
df_ptm_only = df[df.groupby("length")["PTM"].transform("max") == df["PTM"]]
df_plddt_only = df[df.groupby("length")["pLDDT"].transform("max") == df["pLDDT"]]

rng = np.random.default_rng(SCATTER_SEED)
jx_ptm = rng.uniform(-JITTER_X, JITTER_X, size=len(df_ptm_only))
jy_ptm = rng.normal(0.0, JITTER_Y_PTM, size=len(df_ptm_only))
jx_plddt = rng.uniform(-JITTER_X, JITTER_X, size=len(df_plddt_only))
jy_plddt = rng.normal(0.0, JITTER_Y_PLDDT, size=len(df_plddt_only))

# --- PTM ---
fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
c_ptm = "#1A4D8C"
ax1.scatter(
    df_ptm_only["length"] + jx_ptm,
    df_ptm_only["PTM"] + jy_ptm,
    color=c_ptm,
    label="PTM (group max per length)",
    **SCATTER_KW,
)
ax1.plot(
    agg["length"],
    agg["PTM_sm"],
    color=c_ptm,
    linewidth=1.5,
    label="PTM (smoothed)",
    zorder=3,
)
ax1.set_xlabel("Length", fontsize=12)
ax1.set_ylabel("PTM (max per length, rolling mean)", fontsize=12)
ax1.set_title("PTM vs. length", fontsize=13, pad=12)
ax1.legend(loc="best", frameon=True)
ax1.grid(False)
fig1.tight_layout()
ptm_svg = OUT_DIR / "ptm_vs_length.svg"
fig1.savefig(ptm_svg, format="svg", bbox_inches="tight")
plt.close(fig1)

# --- pLDDT ---
fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=100)
c_plddt = "#76B7B2"
ax2.scatter(
    df_plddt_only["length"] + jx_plddt,
    df_plddt_only["pLDDT"] + jy_plddt,
    color=c_plddt,
    label="pLDDT (group max per length)",
    **SCATTER_KW,
)
ax2.plot(
    agg["length"],
    agg["pLDDT_sm"],
    color=c_plddt,
    linewidth=1.5,
    label="pLDDT (smoothed)",
    zorder=3,
)
ax2.set_xlabel("Length", fontsize=12)
ax2.set_ylabel("pLDDT (max per length, rolling mean)", fontsize=12)
ax2.set_title("pLDDT vs. length", fontsize=13, pad=12)
ax2.legend(loc="best", frameon=True)
ax2.grid(False)
fig2.tight_layout()
plddt_svg = OUT_DIR / "plddt_vs_length.svg"
fig2.savefig(plddt_svg, format="svg", bbox_inches="tight")
plt.close(fig2)

print(f"Saved: {ptm_svg}")
print(f"Saved: {plddt_svg}")
