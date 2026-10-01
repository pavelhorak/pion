#!/usr/bin/env python3
"""Generate publication-quality figures for UMA attention MLSys paper."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # TrueType (editable in Illustrator)
    "ps.fonttype": 42,
})

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close(fig)
    print(f"  saved {name}.pdf + .png")


# ===== Figure 1: Metal Dispatch Phase Breakdown (pie) ======================
def fig1():
    labels = ["Encode\n2.5 \u00b5s", "Commit\n1.4 \u00b5s", "Wait\n114.0 \u00b5s"]
    sizes = [2.5, 1.4, 114.0]
    colors = ["#d0d0d0", "#b8b8b8", "#e05038"]
    explode = (0, 0, 0.06)

    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=140, pctdistance=0.55,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title("Metal Compute Dispatch Breakdown (M4)")
    ax.text(0, -1.35, f"Total dispatch: 117.6 \u00b5s",
            ha="center", fontsize=10, style="italic")
    _save(fig, "fig1_dispatch_breakdown")


# ===== Figure 2: CPU vs GPU GEMV (dual y-axis) =============================
def fig2():
    N = [512, 1024, 4096, 16384, 32768, 65536, 131072]
    cpu_lat = [1.2, 1.9, 7.7, 30.5, 206, 582, 1011]
    gpu_lat = [145, 167, 182, 224, 355, 579, 933]
    cpu_bw = [225, 274, 274, 275, 82, 58, 66]
    gpu_bw = [1.8, 3.1, 11.6, 37.5, 47.3, 58.0, 71.9]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    c_cpu, c_gpu = "#1f77b4", "#ff7f0e"

    ax1.plot(N, cpu_lat, "o-", color=c_cpu, label="CPU latency", linewidth=1.8, markersize=5)
    ax1.plot(N, gpu_lat, "s-", color=c_gpu, label="GPU latency", linewidth=1.8, markersize=5)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("N (columns)")
    ax1.set_ylabel("Latency (\u00b5s)")
    ax1.set_xticks(N)
    ax1.set_xticklabels([f"{n//1024}K" if n >= 1024 else str(n) for n in N], fontsize=9)

    # dispatch floor
    ax1.axhline(118, color=c_gpu, ls="--", lw=0.8, alpha=0.6)
    ax1.annotate("dispatch floor (118 \u00b5s)", xy=(600, 118),
                 fontsize=8, color=c_gpu, va="bottom")

    # SLC cliff
    ax1.axvline(32768, color="gray", ls=":", lw=0.9, alpha=0.7)
    ax1.annotate("SLC cliff", xy=(32768, 400),
                 fontsize=8, color="gray", ha="right", rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(N, cpu_bw, "^--", color=c_cpu, alpha=0.45, label="CPU BW", markersize=4, linewidth=1.2)
    ax2.plot(N, gpu_bw, "v--", color=c_gpu, alpha=0.45, label="GPU BW", markersize=4, linewidth=1.2)
    ax2.set_ylabel("Effective bandwidth (GB/s)")
    ax2.set_ylim(0, 310)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8, frameon=True)

    ax1.set_title("CPU (AMX) vs GPU (Metal) GEMV \u2014 1\u00d7128 @ 128\u00d7N")
    fig.tight_layout()
    _save(fig, "fig2_cpu_vs_gpu_gemv")


# ===== Figure 3: Hybrid Pipeline (grouped bar) =============================
def fig3():
    N_labels = ["1024", "4096", "16384", "65536", "131072"]
    cpu_only = [8, 18, 53, 551, 1062]
    sequential = [215, 198, 248, 1035, 1303]
    pipelined = [147, 144, 144, 621, 1054]

    x = np.arange(len(N_labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w, cpu_only, w, label="CPU-only", color="#1f77b4")
    b2 = ax.bar(x,     sequential, w, label="Sequential Hybrid", color="#ff7f0e")
    b3 = ax.bar(x + w, pipelined, w, label="Pipelined Hybrid", color="#2ca02c")

    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Latency (\u00b5s)")
    ax.set_xticks(x)
    ax.set_xticklabels(N_labels)
    ax.legend(fontsize=9)

    # annotation at N=131072
    ax.annotate("1.01\u00d7", xy=(x[-1] + w, pipelined[-1]),
                xytext=(x[-1] + w + 0.15, pipelined[-1] * 1.35),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7))

    ax.set_title("Hybrid Attention Pipeline vs CPU-only")
    fig.tight_layout()
    _save(fig, "fig3_hybrid_pipeline")


# ===== Figure 4: Multi-Head GPU Scaling =====================================
def fig4():
    H = [1, 8, 16, 32]
    n1024  = [0.06, 0.22, 0.33, 0.63]
    n4096  = [0.10, 0.63, 1.15, 1.74]
    n16384 = [0.19, 1.67, 1.85, 2.27]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.fill_between([0.5, 33], 1.0, 2.5, color="#c8e6c8", alpha=0.35, label="_nolegend_")

    ax.plot(H, n1024,  "o-", label="N=1024",  linewidth=1.8, markersize=5)
    ax.plot(H, n4096,  "s-", label="N=4096",  linewidth=1.8, markersize=5)
    ax.plot(H, n16384, "D-", label="N=16384", linewidth=1.8, markersize=5)

    ax.axhline(1.0, color="gray", ls="--", lw=0.9)
    ax.annotate("break-even", xy=(0.8, 1.02), fontsize=8, color="gray")

    ax.set_xlabel("Number of heads (H)")
    ax.set_ylabel("Speedup vs CPU")
    ax.set_xticks(H)
    ax.set_xlim(0.5, 33)
    ax.set_ylim(0, 2.5)
    ax.legend(loc="upper left")
    ax.set_title("MLX GPU Speedup vs CPU \u2014 Multi-Head Scaling")
    fig.tight_layout()
    _save(fig, "fig4_multihead_scaling")


# ===== Figure 5: SLC Contention ============================================
def fig5():
    buf_mb     = [1, 4, 8, 16, 32, 64, 128]
    separate   = [2.3, -0.7, 4.9, 4.9, 5.2, 14.5, 29.4]
    same_mb    = [4, 16, 64]
    same_degr  = [-1.4, -0.3, 12.2]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(buf_mb, separate, "o-", label="Separate buffers", linewidth=1.8, markersize=5,
            color="#1f77b4")
    ax.plot(same_mb, same_degr, "s--", label="Same buffer", linewidth=1.8, markersize=5,
            color="#ff7f0e")

    ax.axvline(16, color="gray", ls=":", lw=0.9)
    ax.annotate("SLC capacity", xy=(16, max(separate) * 0.85),
                fontsize=8, color="gray", ha="left",
                xytext=(18, max(separate) * 0.85))

    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel("Buffer size (MB)")
    ax.set_ylabel("GPU bandwidth degradation (%)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.legend()
    ax.set_title("GPU Read Bandwidth Degradation Under CPU Contention")
    fig.tight_layout()
    _save(fig, "fig5_slc_contention")


# ===== Figure 6: Framework Comparison (horizontal bar) ======================
def fig6():
    labels = ["CPU (C + BLAS)", "MLX SDPA (dense)", "Custom Metal (fused)", "MLX matmul (sparse)"]
    latency = [7.79, 5.54, 4.661, 3.43]
    colors = ["#1f77b4", "#98df8a", "#ff7f0e", "#2ca02c"]

    speedup = [latency[0] / v for v in latency]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, latency, color=colors, edgecolor="white", height=0.55)

    for bar, sp in zip(bars, speedup):
        w = bar.get_width()
        txt = f"{sp:.2f}\u00d7" if sp > 1.005 else "baseline"
        ax.text(w + 0.12, bar.get_y() + bar.get_height() / 2,
                txt, va="center", fontsize=9)

    ax.set_xlabel("Latency (ms)")
    ax.set_xlim(0, max(latency) * 1.25)
    ax.set_title("Framework Comparison \u2014 H=32, N=16K, D=128")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "fig6_framework_comparison")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating UMA attention figures...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("Done.")
