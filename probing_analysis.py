"""
Enhanced Analysis and Visualization for CoT Success Prediction
UPDATED: Bilingual comparison (English vs Multilingual) across all capabilities
- Side-by-side heatmaps for both conditions
- Delta heatmap showing where multilingual underperforms English
- Worst-performing layer identification per capability/condition
- Comprehensive comparison report
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# ==================== Result Loader ====================

def load_results_for_condition(base_dir: str) -> Dict[str, List[Dict]]:
    """
    Load all result JSON files from a base directory organized by capability subdirs.
    Returns dict: {capability_name: [result_dicts]}
    """
    base_path = Path(base_dir)
    results_by_capability = defaultdict(list)

    for capability_dir in base_path.glob("**/tasks/*"):
        if not capability_dir.is_dir():
            continue
        capability_name = capability_dir.name

        for json_file in capability_dir.glob("results_*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    data['filename'] = json_file.name
                    data['capability'] = capability_name
                    results_by_capability[capability_name].append(data)
            except Exception as e:
                print(f"  Warning: Could not load {json_file}: {e}")

    # Also accept flat structure: base_dir/capability/results_*.json
    for capability_dir in base_path.glob("*"):
        if not capability_dir.is_dir():
            continue
        capability_name = capability_dir.name
        if capability_name == "tasks":
            continue

        for json_file in capability_dir.glob("results_*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    data['filename'] = json_file.name
                    data['capability'] = capability_name
                    # Avoid duplicates
                    if not any(r['filename'] == data['filename']
                               for r in results_by_capability[capability_name]):
                        results_by_capability[capability_name].append(data)
            except Exception as e:
                print(f"  Warning: Could not load {json_file}: {e}")

    return dict(results_by_capability)


def build_layer_accuracy_matrix(
    results_by_capability: Dict[str, List[Dict]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two DataFrames:
      - accuracy_matrix: rows=capabilities, cols=layers, values=mean accuracy
      - f1_matrix:       rows=capabilities, cols=layers, values=mean macro-F1
    """
    layer_data: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    f1_data:    Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))

    for capability, results in results_by_capability.items():
        for r in results:
            if r.get('is_summary', False):
                continue
            layer = r['args']['layer_idx']
            layer_data[capability][layer].append(r['test_accuracy'])
            f1_data[capability][layer].append(r.get('test_f1_macro', 0.0))

    # Aggregate
    all_layers = sorted({
        layer
        for cap_data in layer_data.values()
        for layer in cap_data.keys()
    })
    all_capabilities = sorted(layer_data.keys())

    acc_matrix = pd.DataFrame(index=all_capabilities, columns=all_layers, dtype=float)
    f1_matrix  = pd.DataFrame(index=all_capabilities, columns=all_layers, dtype=float)

    for cap in all_capabilities:
        for layer in all_layers:
            if layer in layer_data[cap] and layer_data[cap][layer]:
                acc_matrix.loc[cap, layer] = np.mean(layer_data[cap][layer])
                f1_matrix.loc[cap, layer]  = np.mean(f1_data[cap][layer])

    return acc_matrix, f1_matrix


# ==================== Bilingual Analyzer ====================

class BilingualAnalyzer:
    """
    Compare English vs Multilingual probing results across all capabilities and layers.
    """

    CAPABILITIES = [
        "grounding", "instruction_following", "planning",
        "reasoning", "safety", "theory_of_mind"
    ]

    def __init__(self, english_dir: str, multilingual_dir: str):
        print(f"\nLoading English results from:      {english_dir}")
        self.en_results  = load_results_for_condition(english_dir)
        print(f"  Found capabilities: {sorted(self.en_results.keys())}")

        print(f"Loading Multilingual results from: {multilingual_dir}")
        self.ml_results  = load_results_for_condition(multilingual_dir)
        print(f"  Found capabilities: {sorted(self.ml_results.keys())}")

        # Build layer-accuracy matrices
        self.en_acc, self.en_f1 = build_layer_accuracy_matrix(self.en_results)
        self.ml_acc, self.ml_f1 = build_layer_accuracy_matrix(self.ml_results)

        # Unified layer / capability sets (union)
        self.all_layers = sorted(set(self.en_acc.columns) | set(self.ml_acc.columns))
        self.all_caps   = sorted(set(self.en_acc.index)   | set(self.ml_acc.index))

        # Reindex to shared space (NaN where missing)
        self.en_acc = self.en_acc.reindex(index=self.all_caps, columns=self.all_layers)
        self.ml_acc = self.ml_acc.reindex(index=self.all_caps, columns=self.all_layers)
        self.en_f1  = self.en_f1.reindex(index=self.all_caps,  columns=self.all_layers)
        self.ml_f1  = self.ml_f1.reindex(index=self.all_caps,  columns=self.all_layers)

        # Delta: multilingual minus english  (negative = multilingual worse)
        self.delta_acc = self.ml_acc - self.en_acc
        self.delta_f1  = self.ml_f1  - self.en_f1

        print(f"\nShared capabilities : {self.all_caps}")
        print(f"Shared layer range  : {min(self.all_layers)} – {max(self.all_layers)}")

    # ------------------------------------------------------------------
    # 1. Side-by-side + delta layer heatmaps
    # ------------------------------------------------------------------
    def plot_layer_heatmaps(self, output_path: str = "layer_heatmaps_bilingual.png"):
        """
        3-panel heatmap:  English | Multilingual | Delta (ML - EN)
        """
        caps_display = [c.replace("_", " ").title() for c in self.all_caps]

        fig, axes = plt.subplots(1, 3, figsize=(26, max(5, len(self.all_caps) * 0.9 + 2)))

        vmin, vmax = 0.2, 1.0  # 0.2 = 5-class random baseline

        # --- English ---
        mask_en = self.en_acc.isna()
        sns.heatmap(
            self.en_acc.astype(float), ax=axes[0],
            cmap="RdYlGn", vmin=vmin, vmax=vmax,
            mask=mask_en, annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.3, cbar_kws={"label": "Accuracy"},
            xticklabels=self.all_layers, yticklabels=caps_display
        )
        axes[0].set_title("English", fontsize=14, fontweight="bold", pad=10)
        axes[0].set_xlabel("Layer Index", fontsize=11)
        axes[0].set_ylabel("Capability", fontsize=11)
        axes[0].tick_params(axis='x', rotation=90)

        # --- Multilingual ---
        mask_ml = self.ml_acc.isna()
        sns.heatmap(
            self.ml_acc.astype(float), ax=axes[1],
            cmap="RdYlGn", vmin=vmin, vmax=vmax,
            mask=mask_ml, annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.3, cbar_kws={"label": "Accuracy"},
            xticklabels=self.all_layers, yticklabels=caps_display
        )
        axes[1].set_title("Multilingual", fontsize=14, fontweight="bold", pad=10)
        axes[1].set_xlabel("Layer Index", fontsize=11)
        axes[1].set_ylabel("")
        axes[1].tick_params(axis='x', rotation=90)

        # --- Delta (ML - EN) ---
        abs_max = float(self.delta_acc.abs().max().max())
        if np.isnan(abs_max) or abs_max == 0:
            abs_max = 0.2
        mask_delta = self.delta_acc.isna()
        sns.heatmap(
            self.delta_acc.astype(float), ax=axes[2],
            cmap="RdYlGn", vmin=-abs_max, vmax=abs_max,
            mask=mask_delta, annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.3, cbar_kws={"label": "Δ Accuracy (ML − EN)"},
            xticklabels=self.all_layers, yticklabels=caps_display,
            center=0
        )
        axes[2].set_title("Δ Accuracy  (Multilingual − English)\nRed = ML worse, Green = ML better",
                          fontsize=13, fontweight="bold", pad=10)
        axes[2].set_xlabel("Layer Index", fontsize=11)
        axes[2].set_ylabel("")
        axes[2].tick_params(axis='x', rotation=90)

        plt.suptitle("Layer-wise Probe Accuracy: English vs Multilingual (5-class)",
                     fontsize=16, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved bilingual layer heatmap  →  {output_path}")

    # ------------------------------------------------------------------
    # 2. Best-layer comparison bar chart
    # ------------------------------------------------------------------
    def plot_best_layer_comparison(self, output_path: str = "best_layer_comparison.png"):
        """
        For each capability: best accuracy in English vs Multilingual,
        and which layer achieved it.
        """
        records = []
        for cap in self.all_caps:
            for cond, acc_df in [("English", self.en_acc), ("Multilingual", self.ml_acc)]:
                row = acc_df.loc[cap]
                if row.isna().all():
                    continue
                best_layer = int(row.idxmax())
                best_acc   = float(row.max())
                records.append({
                    "capability": cap.replace("_", " ").title(),
                    "condition": cond,
                    "best_layer": best_layer,
                    "best_acc": best_acc
                })

        df = pd.DataFrame(records)
        if df.empty:
            print("  No data for best-layer comparison plot.")
            return

        caps_display = sorted(df["capability"].unique())
        x = np.arange(len(caps_display))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # --- Panel 1: best accuracy ---
        for i, (cond, color) in enumerate([("English", "steelblue"), ("Multilingual", "coral")]):
            sub = df[df["condition"] == cond].set_index("capability")
            vals = [sub.loc[c, "best_acc"] if c in sub.index else np.nan for c in caps_display]
            bars = ax1.bar(x + i * width - width/2, vals, width,
                           label=cond, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                             f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax1.axhline(0.2, color="red", linestyle="--", alpha=0.6, label="Random baseline")
        ax1.set_xticks(x)
        ax1.set_xticklabels(caps_display, rotation=30, ha="right")
        ax1.set_ylabel("Best Accuracy", fontsize=12)
        ax1.set_title("Best Probe Accuracy per Capability: English vs Multilingual",
                      fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis="y", alpha=0.3)

        # --- Panel 2: best layer index ---
        for i, (cond, color) in enumerate([("English", "steelblue"), ("Multilingual", "coral")]):
            sub = df[df["condition"] == cond].set_index("capability")
            vals = [sub.loc[c, "best_layer"] if c in sub.index else np.nan for c in caps_display]
            bars = ax2.bar(x + i * width - width/2, vals, width,
                           label=cond, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                             f"L{int(v)}", ha="center", va="bottom", fontsize=8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(caps_display, rotation=30, ha="right")
        ax2.set_ylabel("Best Layer Index", fontsize=12)
        ax2.set_title("Layer Index of Best Performance: English vs Multilingual",
                      fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved best-layer comparison     →  {output_path}")

    # ------------------------------------------------------------------
    # 3. Worst-performing layers identification
    # ------------------------------------------------------------------
    def identify_worst_layers(self, output_path: str = "worst_layers_analysis.png",
                               bottom_k: int = 5):
        """
        For each capability × condition, identify the bottom-k worst layers
        and visualise which layer regions consistently underperform.
        """
        rows = []
        for cap in self.all_caps:
            for cond, acc_df in [("English", self.en_acc), ("Multilingual", self.ml_acc)]:
                series = acc_df.loc[cap].dropna().astype(float)
                if series.empty:
                    continue
                worst = series.nsmallest(bottom_k)
                for layer, acc in worst.items():
                    rows.append({
                        "capability": cap.replace("_", " ").title(),
                        "condition": cond,
                        "layer": int(layer),
                        "accuracy": acc
                    })

        df = pd.DataFrame(rows)
        if df.empty:
            print("  No data for worst-layer analysis.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, cond, color in [
            (axes[0], "English",      "steelblue"),
            (axes[1], "Multilingual", "coral")
        ]:
            sub = df[df["condition"] == cond]
            if sub.empty:
                ax.set_visible(False)
                continue

            # Pivot: rows=capabilities, cols=rank (worst #1 … #k)
            pivoted = sub.groupby("capability").apply(
                lambda g: g.nsmallest(bottom_k, "accuracy")["layer"].values
            )
            cap_labels = pivoted.index.tolist()
            matrix = np.full((len(cap_labels), bottom_k), np.nan)
            for i, cap in enumerate(cap_labels):
                vals = pivoted[cap]
                matrix[i, :len(vals)] = vals

            sns.heatmap(
                matrix, ax=ax,
                cmap="Reds_r", annot=True, fmt=".0f", annot_kws={"size": 9},
                linewidths=0.4,
                xticklabels=[f"Worst #{k+1}" for k in range(bottom_k)],
                yticklabels=cap_labels,
                cbar_kws={"label": "Layer Index"}
            )
            ax.set_title(f"{cond}: Worst {bottom_k} Layers per Capability",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Rank (worst → less-worst)", fontsize=11)
            ax.set_ylabel("Capability", fontsize=11)

        plt.suptitle("Worst-Performing Layers Identification",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved worst-layer analysis      →  {output_path}")

        # Also print table
        print("\n" + "="*70)
        print(f"WORST {bottom_k} LAYERS PER CAPABILITY & CONDITION")
        print("="*70)
        for cond in ["English", "Multilingual"]:
            print(f"\n{cond.upper()}:")
            sub = df[df["condition"] == cond]
            for cap in sorted(sub["capability"].unique()):
                worst_layers = (sub[sub["capability"] == cap]
                                .nsmallest(bottom_k, "accuracy")[["layer", "accuracy"]]
                                .values.tolist())
                layers_str = ", ".join(f"L{int(l)}({a:.3f})" for l, a in worst_layers)
                print(f"  {cap:<30} {layers_str}")
        print("="*70)

    # ------------------------------------------------------------------
    # 4. Per-capability delta profile (line plot)
    # ------------------------------------------------------------------
    def plot_delta_profiles(self, output_path: str = "delta_profiles.png"):
        """
        One subplot per capability showing Δ accuracy (ML - EN) across layers,
        so you can see which layer regions are most hurt in multilingual setting.
        """
        n_caps = len(self.all_caps)
        ncols = 3
        nrows = (n_caps + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(6 * ncols, 4 * nrows),
                                  sharey=False)
        axes_flat = axes.flatten() if n_caps > 1 else [axes]

        for idx, cap in enumerate(sorted(self.all_caps)):
            ax = axes_flat[idx]
            delta_row = self.delta_acc.loc[cap].dropna().astype(float)

            if delta_row.empty:
                ax.set_visible(False)
                continue

            layers = delta_row.index.astype(int).tolist()
            deltas = delta_row.values.tolist()

            colors = ["green" if d >= 0 else "red" for d in deltas]
            ax.bar(layers, deltas, color=colors, alpha=0.75, width=0.8)
            ax.axhline(0, color="black", linewidth=1.2)

            # Mark worst-drop layer
            if deltas:
                worst_idx = int(np.argmin(deltas))
                ax.annotate(f"L{layers[worst_idx]}\n{deltas[worst_idx]:.3f}",
                            xy=(layers[worst_idx], deltas[worst_idx]),
                            xytext=(layers[worst_idx], deltas[worst_idx] - 0.02),
                            ha="center", va="top", fontsize=8,
                            color="darkred", fontweight="bold")

            ax.set_title(cap.replace("_", " ").title(), fontsize=12, fontweight="bold")
            ax.set_xlabel("Layer", fontsize=10)
            ax.set_ylabel("Δ Accuracy (ML − EN)", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

        # Hide unused
        for idx in range(n_caps, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.suptitle("Per-Layer Accuracy Drop: Multilingual vs English\n"
                     "(Red bars = multilingual performs worse)",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved delta profiles            →  {output_path}")

    # ------------------------------------------------------------------
    # 5. Overall summary bar chart
    # ------------------------------------------------------------------
    def plot_overall_summary(self, output_path: str = "overall_summary.png"):
        """
        For each capability: mean accuracy across all layers, both conditions.
        """
        records = []
        for cap in self.all_caps:
            for cond, acc_df in [("English", self.en_acc), ("Multilingual", self.ml_acc)]:
                row = acc_df.loc[cap].dropna().astype(float)
                if row.empty:
                    continue
                records.append({
                    "capability": cap.replace("_", " ").title(),
                    "condition": cond,
                    "mean_acc": float(row.mean()),
                    "std_acc":  float(row.std()),
                    "best_acc": float(row.max()),
                })

        df = pd.DataFrame(records)
        if df.empty:
            print("  No data for overall summary.")
            return

        caps_display = sorted(df["capability"].unique())
        x = np.arange(len(caps_display))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))

        for i, (cond, color) in enumerate([("English", "steelblue"), ("Multilingual", "coral")]):
            sub = df[df["condition"] == cond].set_index("capability")
            means = [sub.loc[c, "mean_acc"] if c in sub.index else np.nan for c in caps_display]
            stds  = [sub.loc[c, "std_acc"]  if c in sub.index else 0      for c in caps_display]
            bests = [sub.loc[c, "best_acc"] if c in sub.index else np.nan for c in caps_display]

            bars = ax.bar(x + i * width - width/2, means, width,
                          label=f"{cond} (mean)", color=color, alpha=0.8)
            ax.errorbar(x + i * width - width/2, means, yerr=stds,
                        fmt="none", ecolor="black", capsize=4, alpha=0.5)

            # Overlay best as a dot
            ax.scatter(x + i * width - width/2, bests,
                       marker="*", s=80, color=color,
                       edgecolors="black", linewidths=0.5, zorder=5,
                       label=f"{cond} (best)" if i == 0 else "_nolegend_")

        ax.axhline(0.2, color="red", linestyle="--", alpha=0.6, label="Random baseline (5-class)")
        ax.set_xticks(x)
        ax.set_xticklabels(caps_display, rotation=30, ha="right", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Mean & Best Probe Accuracy: English vs Multilingual",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved overall summary           →  {output_path}")

    # ------------------------------------------------------------------
    # 6. Comprehensive text report
    # ------------------------------------------------------------------
    def generate_report(self, output_path: str = "bilingual_report.txt"):
        """Write a comprehensive text comparison report."""

        with open(output_path, "w") as f:
            def w(line=""):
                f.write(line + "\n")

            w("=" * 80)
            w("BILINGUAL CoT PROBING - COMPARISON REPORT (English vs Multilingual)")
            w("=" * 80)
            w()

            # Summary table
            w("SUMMARY TABLE  (best accuracy per capability)")
            w("-" * 80)
            w(f"{'Capability':<28} {'EN best':>9} {'ML best':>9} {'Δ (ML−EN)':>11}  {'Winner'}")
            w("-" * 80)

            for cap in sorted(self.all_caps):
                en_row = self.en_acc.loc[cap].dropna().astype(float)
                ml_row = self.ml_acc.loc[cap].dropna().astype(float)
                en_best = float(en_row.max()) if not en_row.empty else float("nan")
                ml_best = float(ml_row.max()) if not ml_row.empty else float("nan")
                delta   = ml_best - en_best if not (np.isnan(en_best) or np.isnan(ml_best)) else float("nan")
                winner  = ("English" if delta < 0 else "Multilingual") if not np.isnan(delta) else "N/A"
                w(f"{cap.replace('_',' ').title():<28} "
                  f"{en_best:>9.4f} {ml_best:>9.4f} {delta:>+11.4f}  {winner}")

            w()
            w("=" * 80)
            w("WORST-PERFORMING LAYERS  (bottom-3 per capability & condition)")
            w("=" * 80)

            for cond, acc_df in [("English", self.en_acc), ("Multilingual", self.ml_acc)]:
                w(f"\n{cond.upper()}:")
                for cap in sorted(self.all_caps):
                    row = acc_df.loc[cap].dropna().astype(float)
                    if row.empty:
                        w(f"  {cap:<30}  no data")
                        continue
                    worst = row.nsmallest(3)
                    entry = ", ".join(f"Layer {int(l)} ({v:.3f})" for l, v in worst.items())
                    w(f"  {cap.replace('_',' ').title():<30}  {entry}")

            w()
            w("=" * 80)
            w("LAYER-LEVEL ANALYSIS  (mean accuracy per layer per capability)")
            w("=" * 80)

            for cap in sorted(self.all_caps):
                w(f"\n{cap.replace('_',' ').upper()}")
                w(f"  {'Layer':>7}  {'English':>10}  {'Multilingual':>14}  {'Δ (ML−EN)':>11}")
                w("  " + "-" * 48)
                for layer in self.all_layers:
                    en_v = self.en_acc.loc[cap, layer]
                    ml_v = self.ml_acc.loc[cap, layer]
                    if pd.isna(en_v) and pd.isna(ml_v):
                        continue
                    en_s  = f"{en_v:.4f}" if not pd.isna(en_v) else "  N/A  "
                    ml_s  = f"{ml_v:.4f}" if not pd.isna(ml_v) else "  N/A  "
                    d = ml_v - en_v if (not pd.isna(ml_v) and not pd.isna(en_v)) else float("nan")
                    d_s  = f"{d:+.4f}" if not np.isnan(d) else "   N/A  "
                    flag = " ◄ ML worse" if (not np.isnan(d) and d < -0.02) else ""
                    w(f"  {layer:>7}  {en_s:>10}  {ml_s:>14}  {d_s:>11}{flag}")

        print(f"✓ Saved bilingual report          →  {output_path}")

    # ------------------------------------------------------------------
    # 7. Metrics CSV
    # ------------------------------------------------------------------
    def save_metrics_csv(self, output_path: str = "bilingual_metrics.csv"):
        rows = []
        for cap in self.all_caps:
            for cond, acc_df, f1_df in [
                ("English",      self.en_acc, self.en_f1),
                ("Multilingual", self.ml_acc, self.ml_f1),
            ]:
                for layer in self.all_layers:
                    acc = acc_df.loc[cap, layer]
                    f1  = f1_df.loc[cap, layer]
                    if pd.isna(acc):
                        continue
                    rows.append({
                        "capability": cap,
                        "condition":  cond,
                        "layer":      int(layer),
                        "accuracy":   float(acc),
                        "f1_macro":   float(f1) if not pd.isna(f1) else None,
                    })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"✓ Saved metrics CSV               →  {output_path}")
        return df


# ==================== Command Line Interface ====================

def main():
    parser = argparse.ArgumentParser(
        description="Bilingual CoT probing analysis: English vs Multilingual"
    )
    parser.add_argument(
        "--english_dir", type=str, required=True,
        help="Base output directory from the English probing run "
             "(e.g. /mnt/storage/.../english)"
    )
    parser.add_argument(
        "--multilingual_dir", type=str, required=True,
        help="Base output directory from the Multilingual probing run "
             "(e.g. /mnt/storage/.../multilingual)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./bilingual_analysis",
        help="Directory to save all analysis outputs"
    )
    parser.add_argument(
        "--bottom_k", type=int, default=5,
        help="How many worst layers to highlight (default: 5)"
    )
    parser.add_argument(
        "--plots", nargs="+",
        choices=["all", "heatmaps", "best_layers", "worst_layers",
                 "delta", "summary", "report", "csv"],
        default=["all"],
        help="Which outputs to generate (default: all)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("BILINGUAL PROBING ANALYSIS  (English vs Multilingual)")
    print("=" * 80)
    print(f"English dir     : {args.english_dir}")
    print(f"Multilingual dir: {args.multilingual_dir}")
    print(f"Output dir      : {args.output_dir}")
    print("=" * 80)

    analyzer = BilingualAnalyzer(args.english_dir, args.multilingual_dir)

    do_all = "all" in args.plots

    if do_all or "heatmaps" in args.plots:
        analyzer.plot_layer_heatmaps(
            str(output_dir / "layer_heatmaps_bilingual.png"))

    if do_all or "best_layers" in args.plots:
        analyzer.plot_best_layer_comparison(
            str(output_dir / "best_layer_comparison.png"))

    if do_all or "worst_layers" in args.plots:
        analyzer.identify_worst_layers(
            str(output_dir / "worst_layers_analysis.png"),
            bottom_k=args.bottom_k)

    if do_all or "delta" in args.plots:
        analyzer.plot_delta_profiles(
            str(output_dir / "delta_profiles.png"))

    if do_all or "summary" in args.plots:
        analyzer.plot_overall_summary(
            str(output_dir / "overall_summary.png"))

    if do_all or "report" in args.plots:
        analyzer.generate_report(
            str(output_dir / "bilingual_report.txt"))

    if do_all or "csv" in args.plots:
        analyzer.save_metrics_csv(
            str(output_dir / "bilingual_metrics.csv"))

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("""
Generated files:
  layer_heatmaps_bilingual.png  — English | Multilingual | Delta side-by-side
  best_layer_comparison.png     — Best accuracy & best layer per capability
  worst_layers_analysis.png     — Worst-performing layers (ranked heatmap)
  delta_profiles.png            — Per-capability layer-by-layer accuracy drop
  overall_summary.png           — Mean/best accuracy across all layers
  bilingual_report.txt          — Full text comparison report
  bilingual_metrics.csv         — All metrics in tabular form
""")


if __name__ == "__main__":
    main()