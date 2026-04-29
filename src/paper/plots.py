"""Bar / comparison plots for paper-style ASR (Section V, Table 2)."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_asr_bars(results: list, out_path: str, title: str = "Attack Success Rate (%)"):
    """
    results: list of dicts with keys 'dataset', 'attack', 'asr_percent'
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels = [f"{r['dataset']}\n{r['attack'].upper()}" for r in results]
    values = [r["asr_percent"] for r in results]
    colors = plt.cm.tab10(range(len(values)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ASR (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(100, max(values) * 1.1 if values else 100))
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results_json(results, path: str):
    """Write list or dict to JSON (used for tables and fast-benchmark metadata)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def load_results_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_radar_fgsm_pgd(results: list, out_path: str):
    """Paper Figure 3 style: radar for FGSM vs PGD on MNIST and CIFAR-10."""
    datasets = sorted(set(r["dataset"] for r in results))
    categories = datasets
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fgsm_vals = []
    pgd_vals = []
    for d in datasets:
        fg = next((r["asr_percent"] for r in results if r["dataset"] == d and r["attack"] == "fgsm"), 0)
        pg = next((r["asr_percent"] for r in results if r["dataset"] == d and r["attack"] == "pgd"), 0)
        fgsm_vals.append(fg)
        pgd_vals.append(pg)
    fgsm_vals += fgsm_vals[:1]
    pgd_vals += pgd_vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, fgsm_vals, "o-", linewidth=2, label="FGSM", color="tab:orange")
    ax.fill(angles, fgsm_vals, alpha=0.15, color="tab:orange")
    ax.plot(angles, pgd_vals, "o-", linewidth=2, label="PGD", color="tab:blue")
    ax.fill(angles, pgd_vals, alpha=0.15, color="tab:blue")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("FGSM vs PGD ASR (%)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
