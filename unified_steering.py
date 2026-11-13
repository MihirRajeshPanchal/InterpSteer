"""
Unified Activation Steering — combines steering vectors across ALL capabilities
before applying to a single LLM instance.

Combination modes:
  sum     — add all per-layer vectors element-wise
  average — average all per-layer vectors element-wise

For layers that appear in multiple capabilities the vectors are combined
(summed or averaged) before injection; single-capability layers are used as-is.

Output directories:
  <base_output_dir>/unified_sum/
  <base_output_dir>/unified_average/

Usage example:
  python unified_steering.py \
      --data_path   data/biggen_bench_translated.csv \
      --model_name  /mnt/storage/deeksha/models/sarvamai/sarvam-1 \
      --analysis_dir /mnt/storage/deeksha/interpsteer/sarvam-1/results \
      --output_dir   /mnt/storage/deeksha/interpsteer/sarvam-1 \
      --combine_mode sum \
      --alpha 0.5
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import gc
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ALL_CAPABILITIES = [
    "grounding",
    "instruction_following",
    "planning",
    "reasoning",
    "safety",
    "theory_of_mind",
]

DEFAULT_WORST_LAYERS = [0, 1, 2, 3, 4]


# ──────────────────────────────────────────────────────────────────────────────
# Worst-layer loaders  (identical to original script)
# ──────────────────────────────────────────────────────────────────────────────

def load_worst_layers_from_csv(
    csv_path: str,
    capability: Optional[str],
    condition: str = "Multilingual",
    top_n: int = 5,
) -> Dict[str, List[int]]:
    df = pd.read_csv(csv_path)
    df = df[df["condition"] == condition]
    if capability:
        df = df[df["capability"] == capability]
    worst = {}
    for cap, grp in df.groupby("capability"):
        worst[cap] = grp.nsmallest(top_n, "accuracy")["layer"].astype(int).tolist()
    return worst


def load_worst_layers_from_report(
    report_path: str,
    capability: Optional[str],
    condition: str = "Multilingual",
    top_n: int = 3,
) -> Dict[str, List[int]]:
    with open(report_path) as f:
        content = f.read()
    cond_upper = condition.upper()
    section_pattern = rf"{cond_upper}:\n(.*?)(?=\n[A-Z]{{2,}}:|\Z)"
    m = re.search(section_pattern, content, re.DOTALL)
    if not m:
        print(f"  Warning: '{cond_upper}' section not found in report.")
        return {}
    section = m.group(1)
    worst = {}
    for line in section.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) < 2:
            continue
        cap_raw = parts[0].strip().lower().replace(" ", "_")
        if capability and cap_raw != capability.lower():
            continue
        layers = [int(x) for x in re.findall(r"Layer\s+(\d+)", parts[1])[:top_n]]
        if layers:
            worst[cap_raw] = layers
    return worst


def load_worst_layers_auto(
    analysis_dir: str,
    capability: Optional[str],
    condition: str = "Multilingual",
    top_n: int = 5,
) -> Dict[str, List[int]]:
    p = Path(analysis_dir)
    csv_path    = p / "bilingual_metrics.csv"
    report_path = p / "bilingual_report.txt"
    if csv_path.exists():
        print(f"  Loading worst layers from CSV: {csv_path}")
        return load_worst_layers_from_csv(str(csv_path), capability, condition, top_n)
    if report_path.exists():
        print(f"  Loading worst layers from report: {report_path}")
        return load_worst_layers_from_report(str(report_path), capability, condition, top_n)
    print(f"  Warning: no analysis files in {analysis_dir}. Using defaults.")
    if capability:
        return {capability: DEFAULT_WORST_LAYERS[:top_n]}
    return {"all": DEFAULT_WORST_LAYERS[:top_n]}


# ──────────────────────────────────────────────────────────────────────────────
# Vector combination
# ──────────────────────────────────────────────────────────────────────────────

def combine_vectors(
    cap_layer_vectors: Dict[str, Dict[int, np.ndarray]],
    mode: str = "sum",
) -> Tuple[Dict[int, np.ndarray], Dict[int, List[str]]]:
    """
    Merge per-capability steering vectors into a single per-layer vector.

    Parameters
    ----------
    cap_layer_vectors : { capability: { layer_idx: ndarray } }
    mode              : 'sum' or 'average'

    Returns
    -------
    combined   : { layer_idx: combined_ndarray }
    layer_caps : { layer_idx: [capability, ...] }   — which caps contributed
    """
    # Collect all vectors per layer
    layer_to_vecs: Dict[int, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    for cap, layer_sv in cap_layer_vectors.items():
        for layer_idx, sv in layer_sv.items():
            layer_to_vecs[layer_idx].append((cap, sv))

    combined: Dict[int, np.ndarray] = {}
    layer_caps: Dict[int, List[str]] = {}

    for layer_idx, cap_vecs in layer_to_vecs.items():
        caps  = [cv[0] for cv in cap_vecs]
        vecs  = [cv[1] for cv in cap_vecs]
        stacked = np.stack(vecs, axis=0)          # (N, hidden_dim)
        combined[layer_idx] = (
            stacked.sum(axis=0) if mode == "sum"
            else stacked.mean(axis=0)
        )
        layer_caps[layer_idx] = caps

    return combined, layer_caps


def print_combination_summary(
    cap_layer_map: Dict[str, List[int]],
    combined_vectors: Dict[int, np.ndarray],
    layer_caps: Dict[int, List[str]],
    mode: str,
) -> None:
    """Print the human-readable summary shown in the task description."""
    SEP = "=" * 70
    print(f"\n{SEP}")
    print(f"Combination mode: {mode}")
    print(SEP)
    for cap, layers in sorted(cap_layer_map.items()):
        print(f"  ✓ {cap:<28} Layers: {layers}")

    print(f"\n{SEP}")
    print("Combining vectors for shared layers:")
    print(SEP)
    for layer_idx in sorted(combined_vectors.keys()):
        caps = layer_caps[layer_idx]
        cap_str = ", ".join(caps)
        tag = "single" if len(caps) == 1 else mode + "d"
        print(f"  Layer {layer_idx:>2}: {cap_str:<50} ({tag})")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Steering vector extractor  (identical to original, kept self-contained)
# ──────────────────────────────────────────────────────────────────────────────

class SteeringVectorExtractor:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={i: "10GiB" for i in range(torch.cuda.device_count())},
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_device = next(self.model.parameters()).device
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded. Layers: {self.num_layers}, Device: {self.model_device}")

    # ------------------------------------------------------------------
    def get_hidden_state(self, text: str, layer_idx: int, position: str = "last") -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048, padding=False
        )
        inputs = {k: v.to(self.model_device) for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            outputs = self.model(**inputs)
            hs = outputs.hidden_states[layer_idx]
            vec = (hs[0, -1, :] if position == "last" else hs[0].mean(dim=0)).cpu().float().numpy()
            del outputs, hs
            torch.cuda.empty_cache()
        return vec

    # ------------------------------------------------------------------
    def compute_steering_vector(
        self,
        df: pd.DataFrame,
        layer_idx: int,
        instruction_col: str = "translated_instruction",
        response_col: str = "translated_response",
        reference_col: str = "translated_reference_answer",
        score_col: str = "orig_score",
        system_prompt_col: str = "system_prompt",
        high_score: float = 5.0,
        low_score: float = 1.0,
        max_samples: int = 50,
        position: str = "last",
    ) -> np.ndarray:
        from tqdm import tqdm

        high_df = df[df[score_col] == high_score].head(max_samples)
        low_df  = df[df[score_col] == low_score].head(max_samples)
        ref_df  = df.head(max_samples)

        print(
            f"    Layer {layer_idx}: "
            f"{len(high_df)} high-score + {len(ref_df)} reference (pos) | "
            f"{len(low_df)} low-score (neg)"
        )

        high_hs, low_hs = [], []

        for _, row in tqdm(high_df.iterrows(), total=len(high_df),
                           desc=f"    L{layer_idx} high-response", leave=False):
            try:
                sp = row.get(system_prompt_col, "")
                high_hs.append(self.get_hidden_state(
                    f"{sp}\n\n{row[instruction_col]}\n\n{row[response_col]}", layer_idx, position
                ))
            except Exception as e:
                print(f"      Error (high-resp): {e}")

        for _, row in tqdm(ref_df.iterrows(), total=len(ref_df),
                           desc=f"    L{layer_idx} reference", leave=False):
            try:
                sp = row.get(system_prompt_col, "")
                high_hs.append(self.get_hidden_state(
                    f"{sp}\n\n{row[instruction_col]}\n\n{row[reference_col]}", layer_idx, position
                ))
            except Exception as e:
                print(f"      Error (ref): {e}")

        for _, row in tqdm(low_df.iterrows(), total=len(low_df),
                           desc=f"    L{layer_idx} low-response", leave=False):
            try:
                sp = row.get(system_prompt_col, "")
                low_hs.append(self.get_hidden_state(
                    f"{sp}\n\n{row[instruction_col]}\n\n{row[response_col]}", layer_idx, position
                ))
            except Exception as e:
                print(f"      Error (low-resp): {e}")

        if not high_hs or not low_hs:
            raise ValueError(
                f"Not enough samples at layer {layer_idx}: "
                f"{len(high_hs)} positives, {len(low_hs)} negatives"
            )

        sv = np.mean(high_hs, axis=0) - np.mean(low_hs, axis=0)
        print(f"    Layer {layer_idx} SV norm: {np.linalg.norm(sv):.4f}  "
              f"({len(high_hs)} pos, {len(low_hs)} neg)")
        return sv

    # ------------------------------------------------------------------
    def generate_with_multi_layer_steering(
        self,
        prompt: str,
        steering_vectors: Dict[int, np.ndarray],
        alpha: float = 20.0,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        if alpha < 0.1:
            return self.generate_without_steering(prompt, max_new_tokens, temperature, do_sample)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024, padding=False
        )
        inputs = {k: v.to(self.model_device) for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}

        sv_tensors: Dict[int, torch.Tensor] = {}
        for layer_idx, sv in steering_vectors.items():
            sv_t = torch.tensor(sv, dtype=torch.float16).to(self.model_device)
            if torch.isnan(sv_t).any() or torch.isinf(sv_t).any():
                print(f"  Warning: invalid values in SV layer {layer_idx}, skipping")
                continue
            if sv_t.norm().item() > 100:
                print(f"  Warning: large norm at layer {layer_idx}, skipping")
                continue
            sv_tensors[layer_idx] = sv_t

        if not sv_tensors:
            return self.generate_without_steering(prompt, max_new_tokens, temperature, do_sample)

        def make_hook(sv_t, a):
            def hook_fn(module, input, output):
                try:
                    hidden = output[0] if isinstance(output, tuple) else output
                    sv_norm = sv_t / (sv_t.norm() + 1e-8)
                    hidden_mean_abs = hidden.abs().mean()
                    max_steering = min(hidden_mean_abs * 0.1, hidden.abs().max() * 0.05)
                    steering = torch.clamp(
                        a * sv_norm.unsqueeze(0).unsqueeze(0), -max_steering, max_steering
                    )
                    if steering.abs().max() > hidden.abs().max() * 0.2:
                        return output
                    steered = hidden + steering
                    if torch.isnan(steered).any() or torch.isinf(steered).any():
                        return output
                    return (steered,) + output[1:] if isinstance(output, tuple) else steered
                except Exception:
                    return output
            return hook_fn

        hooks = []
        for layer_idx, sv_t in sv_tensors.items():
            if layer_idx < len(self.model.model.layers):
                hooks.append(
                    self.model.model.layers[layer_idx].register_forward_hook(
                        make_hook(sv_t, alpha)
                    )
                )

        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        except RuntimeError as e:
            if "CUDA" in str(e) or "assert" in str(e):
                print("Warning: CUDA error during steering, falling back to baseline")
                for h in hooks:
                    h.remove()
                hooks = []
                return self.generate_without_steering(
                    prompt, max_new_tokens, temperature=0.0, do_sample=False
                )
            raise
        finally:
            for h in hooks:
                h.remove()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return text

    # ------------------------------------------------------------------
    def generate_without_steering(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024, padding=False
            )
            inputs = {k: v.to(self.model_device) for k, v in inputs.items()
                      if k in ("input_ids", "attention_mask")}
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 0.1),
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    top_p=0.95,
                    repetition_penalty=1.1,
                )
            input_len = inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        except RuntimeError as e:
            if "CUDA" in str(e) or "assert" in str(e):
                print("CUDA error in baseline generation, attempting recovery...")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                try:
                    inputs = self.tokenizer(
                        prompt, return_tensors="pt", truncation=True,
                        max_length=1024, padding=False
                    )
                    inputs = {k: v.to(self.model_device) for k, v in inputs.items()
                              if k in ("input_ids", "attention_mask")}
                    with torch.no_grad():
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    input_len = inputs["input_ids"].shape[1]
                    text = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
                except Exception as e2:
                    print(f"Recovery failed: {e2}")
                    return "[Generation failed due to CUDA error]"
            else:
                raise
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return text


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_unified_steering(args):
    """
    1. Load worst-N layers per capability from bilingual analysis.
    2. Compute per-capability steering vectors at each of those layers.
    3. Combine all vectors layer-wise (sum or average).
    4. Generate baseline + unified-steered responses from a single LLM instance.
    5. Save to <output_dir>/unified_<mode>/
    """
    from tqdm import tqdm

    mode = args.combine_mode                          # "sum" or "average"
    output_dir = Path(args.output_dir) / f"unified_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 70
    print(f"\n{SEP}")
    print("Unified Activation Steering Experiment")
    print(f"Model:          {args.model_name}")
    print(f"Analysis dir:   {args.analysis_dir}")
    print(f"Condition:      {args.condition}")
    print(f"Combine mode:   {mode}")
    print(f"Top-N worst:    {args.top_n_worst}")
    print(f"Alpha:          {args.alpha}")
    print(f"Output:         {output_dir}")
    print(f"{SEP}\n")

    # ── 1. Worst layers per capability ─────────────────────────────────
    print("Loading worst-performing layers...")
    cap_layer_map: Dict[str, List[int]] = {}
    for cap in ALL_CAPABILITIES:
        wl = load_worst_layers_auto(
            args.analysis_dir, cap, args.condition, args.top_n_worst
        )
        if cap in wl:
            cap_layer_map[cap] = wl[cap]
        elif wl:
            # fallback: take first available key
            cap_layer_map[cap] = next(iter(wl.values()))

    if not cap_layer_map:
        print("Error: could not determine worst layers. Exiting.")
        return None

    with open(output_dir / "resolved_worst_layers.json", "w") as f:
        json.dump(cap_layer_map, f, indent=2)

    # ── 2. Load & (optionally) filter dataset ──────────────────────────
    print("\nLoading dataset...")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} samples. Columns: {df.columns.tolist()}")

    required = [args.instruction_col, args.response_col,
                args.reference_col, args.system_prompt_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return None

    eval_df = df.sample(
        n=min(args.num_eval_samples, len(df)), random_state=42
    ).reset_index(drop=True)
    print(f"Evaluation set: {len(eval_df)} samples")

    # ── 3. Initialise model (single instance) ──────────────────────────
    extractor = SteeringVectorExtractor(args.model_name)

    # ── 4. Compute per-capability steering vectors ─────────────────────
    cap_layer_vectors: Dict[str, Dict[int, np.ndarray]] = {}

    for cap in ALL_CAPABILITIES:
        if cap not in cap_layer_map:
            print(f"\n  Skipping {cap}: no layer info.")
            continue

        target_layers = [l for l in cap_layer_map[cap] if l < extractor.num_layers]
        cap_df = df[df["capability"] == cap] if "capability" in df.columns else df
        if len(cap_df) == 0:
            print(f"\n  Skipping {cap}: no data rows.")
            continue

        print(f"\nComputing steering vectors — '{cap}' — layers: {target_layers}")
        cap_layer_vectors[cap] = {}

        for layer_idx in target_layers:
            try:
                sv = extractor.compute_steering_vector(
                    df=cap_df,
                    layer_idx=layer_idx,
                    instruction_col=args.instruction_col,
                    response_col=args.response_col,
                    reference_col=args.reference_col,
                    score_col="orig_score",
                    system_prompt_col=args.system_prompt_col,
                    high_score=args.high_score_threshold,
                    low_score=args.low_score_threshold,
                    max_samples=args.max_steering_samples,
                    position=args.position,
                )
                cap_layer_vectors[cap][layer_idx] = sv
                np.save(output_dir / f"sv_{cap}_layer{layer_idx}.npy", sv)
            except Exception as e:
                print(f"  Error cap={cap} layer={layer_idx}: {e}")

    if not cap_layer_vectors:
        print("Error: no steering vectors computed.")
        return None

    # ── 5. Combine vectors layer-wise ──────────────────────────────────
    combined_vectors, layer_caps = combine_vectors(cap_layer_vectors, mode=mode)

    print_combination_summary(cap_layer_map, combined_vectors, layer_caps, mode)

    # Save combined vectors
    for layer_idx, sv in combined_vectors.items():
        np.save(output_dir / f"unified_{mode}_layer{layer_idx}.npy", sv)

    combo_meta = {
        layer_idx: {
            "capabilities": caps,
            "norm": float(np.linalg.norm(combined_vectors[layer_idx])),
        }
        for layer_idx, caps in layer_caps.items()
    }
    with open(output_dir / f"combined_vector_meta_{mode}.json", "w") as f:
        json.dump(combo_meta, f, indent=2)

    # ── 6. Generate responses ──────────────────────────────────────────
    results = []
    print(f"\nGenerating responses for {len(eval_df)} samples...")

    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Generating"):
        try:
            instruction   = row[args.instruction_col]
            orig_score    = row.get("orig_score", None)
            cap           = str(row.get("capability", ""))
            system_prompt = row.get(args.system_prompt_col, "Let's think step by step:")
            prompt        = f"{system_prompt}\n\n{instruction}"

            if idx > 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Baseline
            try:
                baseline = extractor.generate_without_steering(
                    prompt, args.max_new_tokens, args.temperature, args.do_sample
                )
            except Exception as e:
                print(f"  Baseline error sample {idx}: {e}")
                continue

            # Unified steered (all capabilities combined)
            try:
                unified = extractor.generate_with_multi_layer_steering(
                    prompt, combined_vectors, args.alpha,
                    args.max_new_tokens, args.temperature, args.do_sample,
                )
            except Exception as e:
                print(f"  Unified steering error sample {idx}: {e}")
                unified = baseline

            results.append({
                "idx":                  int(idx),
                "capability":           cap,
                "orig_score":           float(orig_score) if orig_score is not None else None,
                "instruction":          instruction,
                "system_prompt":        system_prompt,
                "baseline_response":    baseline,
                "unified_response":     unified,
                "combine_mode":         mode,
                "layers_steered":       sorted(combined_vectors.keys()),
            })

            print(
                f"  [{idx}] cap={cap:20s} orig={orig_score} | "
                f"baseline_len={len(baseline)} unified_len={len(unified)}"
            )

        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            import traceback; traceback.print_exc()
            if "CUDA" in str(e) and "assert" in str(e):
                print("\nCRITICAL: CUDA device in corrupted state — saving partial results.")
                if results:
                    pd.DataFrame(results).to_csv(
                        output_dir / f"partial_results_unified_{mode}.csv", index=False
                    )
                return None
            continue

        if idx % 5 == 0:
            try:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"CUDA error during cleanup at sample {idx}, stopping early.")
                    break

    # ── 7. Save outputs ────────────────────────────────────────────────
    if not results:
        print("No results collected.")
        return None

    results_df = pd.DataFrame(results)
    csv_path  = output_dir / f"unified_{mode}_results.csv"
    json_path = output_dir / f"unified_{mode}_summary.json"

    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV  → {csv_path}")

    summary = {
        "model":            args.model_name,
        "combine_mode":     mode,
        "alpha":            args.alpha,
        "condition":        args.condition,
        "top_n_worst":      args.top_n_worst,
        "capabilities":     list(cap_layer_vectors.keys()),
        "worst_layers_map": {k: v for k, v in cap_layer_map.items()},
        "combined_layers":  sorted(combined_vectors.keys()),
        "layer_caps":       {str(k): v for k, v in layer_caps.items()},
        "num_samples":      len(results),
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON → {json_path}")

    SEP = "=" * 70
    print(f"\n{SEP}")
    print("GENERATION COMPLETE")
    print(f"Mode:           {mode}")
    print(f"Samples:        {len(results)}")
    print(f"Layers steered: {sorted(combined_vectors.keys())}")
    print(f"{SEP}")

    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            print(f"CUDA initialised: {torch.cuda.device_count()} device(s)")
        except Exception as e:
            print(f"Warning: CUDA reset failed: {e}")

    parser = argparse.ArgumentParser(
        description="Unified Activation Steering — combines all capabilities into one LLM pass"
    )

    # Data
    parser.add_argument("--data_path",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./unified_steering_output",
                        help="Base output dir; results go to <output_dir>/unified_<mode>/")

    # Analysis
    parser.add_argument("--analysis_dir", type=str, required=True,
                        help="Dir containing bilingual_metrics.csv or bilingual_report.txt")
    parser.add_argument("--condition",    type=str, default="Multilingual",
                        choices=["English", "Multilingual"])
    parser.add_argument("--top_n_worst",  type=int, default=5)

    # Model
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")

    # Column names
    parser.add_argument("--instruction_col",   type=str, default="translated_input")
    parser.add_argument("--response_col",      type=str, default="translated_orig_response")
    parser.add_argument("--reference_col",     type=str, default="translated_reference_answer")
    parser.add_argument("--system_prompt_col", type=str, default="translated_system_prompt")

    # Combination mode
    parser.add_argument("--combine_mode", type=str, default="sum",
                        choices=["sum", "average"],
                        help="How to merge per-capability vectors at shared layers")

    # Steering
    parser.add_argument("--alpha",    type=float, default=0.5)
    parser.add_argument("--position", type=str,   default="last", choices=["last", "mean"])

    # Contrastive thresholds
    parser.add_argument("--high_score_threshold", type=float, default=5.0)
    parser.add_argument("--low_score_threshold",  type=float, default=1.0)
    parser.add_argument("--max_steering_samples", type=int,   default=50)

    # Generation
    parser.add_argument("--num_eval_samples", type=int,   default=50)
    parser.add_argument("--max_new_tokens",   type=int,   default=512)
    parser.add_argument("--temperature",      type=float, default=0.7)
    parser.add_argument("--do_sample",        action="store_true", default=False)

    args = parser.parse_args()
    run_unified_steering(args)


if __name__ == "__main__":
    main()