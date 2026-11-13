"""
Activation Steering for Multilingual CoT Quality Improvement

Methodology:
- Load worst-N layers per capability from bilingual analysis output
- Extract steering vectors at each of those layers using:
    Positive: system_prompt + translated_instruction + translated_response (score==5)
              system_prompt + translated_instruction + translated_reference_answer (always score 5)
    Negative: system_prompt + translated_instruction + translated_response (score==1)
- Apply multi-layer steering (additive injection at each worst layer)
- Store full baseline and steered responses (no truncation)

NOTE: Steering vector is computed as negative - positive (reversed).
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import gc
import warnings
warnings.filterwarnings('ignore')


# ==================== Worst Layer Loader ====================

def load_worst_layers_from_csv(
    csv_path: str,
    capability: Optional[str],
    condition: str = "Multilingual",
    top_n: int = 5
) -> Dict[str, List[int]]:
    """
    Load worst-performing layers from bilingual_metrics.csv.

    Args:
        csv_path: Path to bilingual_metrics.csv
        capability: Specific capability to filter (None = all)
        condition: 'English' or 'Multilingual'
        top_n: Number of worst layers to return per capability

    Returns:
        Dict mapping capability -> [worst layer indices, ascending accuracy order]
    """
    df = pd.read_csv(csv_path)
    df = df[df["condition"] == condition]

    if capability:
        df = df[df["capability"] == capability]

    worst_by_cap = {}
    for cap, group in df.groupby("capability"):
        sorted_group = group.nsmallest(top_n, "accuracy")
        worst_by_cap[cap] = sorted_group["layer"].astype(int).tolist()

    return worst_by_cap


def load_worst_layers_from_report(
    report_path: str,
    capability: Optional[str],
    condition: str = "Multilingual",
    top_n: int = 3
) -> Dict[str, List[int]]:
    """
    Parse worst layers from bilingual_report.txt.

    Falls back to this when CSV is not available.
    Returns Dict mapping capability -> [worst layer indices]
    """
    with open(report_path) as f:
        content = f.read()

    cond_upper = condition.upper()
    section_pattern = rf"{cond_upper}:\n(.*?)(?=\n[A-Z]{{2,}}:|\Z)"
    section_match = re.search(section_pattern, content, re.DOTALL)
    if not section_match:
        print(f"  Warning: Could not find '{cond_upper}' section in report.")
        return {}

    section = section_match.group(1)

    worst_by_cap = {}
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
        layer_matches = re.findall(r"Layer\s+(\d+)", parts[1])
        layers = [int(l) for l in layer_matches[:top_n]]
        if layers:
            worst_by_cap[cap_raw] = layers

    return worst_by_cap


def load_worst_layers_auto(
    analysis_dir: str,
    capability: Optional[str],
    condition: str = "Multilingual",
    top_n: int = 5
) -> Dict[str, List[int]]:
    """
    Auto-detect available analysis outputs and load worst layers.
    Priority: bilingual_metrics.csv > bilingual_report.txt > fallback

    Args:
        analysis_dir: Directory containing bilingual analysis outputs
        capability: Specific capability (None = all)
        condition: 'English' or 'Multilingual'
        top_n: Number of worst layers per capability

    Returns:
        Dict: capability -> [worst layer indices]
    """
    analysis_path = Path(analysis_dir)

    csv_path    = analysis_path / "bilingual_metrics.csv"
    report_path = analysis_path / "bilingual_report.txt"

    if csv_path.exists():
        print(f"  Loading worst layers from CSV: {csv_path}")
        return load_worst_layers_from_csv(str(csv_path), capability, condition, top_n)

    elif report_path.exists():
        print(f"  Loading worst layers from report: {report_path}")
        return load_worst_layers_from_report(str(report_path), capability, condition, top_n)

    else:
        print(f"  Warning: No analysis files found in {analysis_dir}")
        print(f"  Falling back to default worst layers: {DEFAULT_WORST_LAYERS}")
        if capability:
            return {capability: DEFAULT_WORST_LAYERS[:top_n]}
        return {"all": DEFAULT_WORST_LAYERS[:top_n]}


# Fallback if no analysis files are present
DEFAULT_WORST_LAYERS = [0, 1, 2, 3, 4]


# ==================== Steering Vector Extractor ====================

class SteeringVectorExtractor:
    """Extracts steering vectors and applies them during generation."""

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
            max_memory={i: "10GiB" for i in range(torch.cuda.device_count())}
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_device = next(self.model.parameters()).device
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded. Layers: {self.num_layers}, Device: {self.model_device}")

    def get_hidden_state(
        self, text: str, layer_idx: int, position: str = "last"
    ) -> np.ndarray:
        """Extract hidden state from a given layer."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=2048, padding=False
        )
        inputs = {k: v.to(self.model_device) for k, v in inputs.items()
                  if k in ['input_ids', 'attention_mask']}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hs = outputs.hidden_states[layer_idx]
            vec = (hs[0, -1, :] if position == "last"
                   else hs[0].mean(dim=0)).cpu().float().numpy()
            del outputs, hs
            torch.cuda.empty_cache()

        return vec

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
        position: str = "last"
    ) -> np.ndarray:
        """
        Steering vector = mean(negative hidden states) - mean(positive hidden states)
        (REVERSED: negative minus positive)

        Positive samples (high quality):
          1. Rows where orig_score == high_score:
             text = system_prompt + translated_instruction + translated_response
          2. All rows (reference answers are always score 5):
             text = system_prompt + translated_instruction + translated_reference_answer

        Negative samples (low quality):
          - Rows where orig_score == low_score:
            text = system_prompt + translated_instruction + translated_response
        """
        from tqdm import tqdm

        high_df = df[df[score_col] == high_score].head(max_samples)
        low_df  = df[df[score_col] == low_score].head(max_samples)
        ref_df  = df.head(max_samples)  # reference answers always treated as score-5 positives

        print(f"    Layer {layer_idx}: "
              f"{len(high_df)} high-score responses + "
              f"{len(ref_df)} reference answers (positives) | "
              f"{len(low_df)} low-score responses (negatives)")

        high_hs, low_hs = [], []

        # --- Positive set 1: high-scoring translated_response ---
        for _, row in tqdm(high_df.iterrows(), total=len(high_df),
                           desc=f"    L{layer_idx} high-response", leave=False):
            try:
                sp   = row.get(system_prompt_col, "")
                text = f"{sp}\n\n{row[instruction_col]}\n\n{row[response_col]}"
                high_hs.append(self.get_hidden_state(text, layer_idx, position))
            except Exception as e:
                print(f"      Error (high-resp): {e}")

        # --- Positive set 2: translated_reference_answer (always score 5) ---
        for _, row in tqdm(ref_df.iterrows(), total=len(ref_df),
                           desc=f"    L{layer_idx} reference", leave=False):
            try:
                sp   = row.get(system_prompt_col, "")
                text = f"{sp}\n\n{row[instruction_col]}\n\n{row[reference_col]}"
                high_hs.append(self.get_hidden_state(text, layer_idx, position))
            except Exception as e:
                print(f"      Error (ref): {e}")

        # --- Negative set: low-scoring translated_response ---
        for _, row in tqdm(low_df.iterrows(), total=len(low_df),
                           desc=f"    L{layer_idx} low-response", leave=False):
            try:
                sp   = row.get(system_prompt_col, "")
                text = f"{sp}\n\n{row[instruction_col]}\n\n{row[response_col]}"
                low_hs.append(self.get_hidden_state(text, layer_idx, position))
            except Exception as e:
                print(f"      Error (low-resp): {e}")

        if not high_hs or not low_hs:
            raise ValueError(
                f"Not enough samples at layer {layer_idx}: "
                f"{len(high_hs)} positives, {len(low_hs)} negatives"
            )

        # REVERSED: negative - positive
        sv = np.mean(low_hs, axis=0) - np.mean(high_hs, axis=0)
        print(f"    Layer {layer_idx} SV norm (reversed): {np.linalg.norm(sv):.4f}  "
              f"({len(high_hs)} pos, {len(low_hs)} neg)")
        return sv

    def generate_with_multi_layer_steering(
        self,
        prompt: str,
        steering_vectors: Dict[int, np.ndarray],
        alpha: float = 20.0,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Generate with steering vectors injected at multiple layers simultaneously.
        """
        if alpha < 0.1:
            print("Alpha too low, skipping steering")
            return self.generate_without_steering(prompt, max_new_tokens, temperature, do_sample)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=1024, padding=False
        )
        inputs = {k: v.to(self.model_device) for k, v in inputs.items()
                  if k in ['input_ids', 'attention_mask']}

        sv_tensors = {}
        for layer_idx, sv in steering_vectors.items():
            sv_t = torch.tensor(sv, dtype=torch.float16).to(self.model_device)
            if torch.isnan(sv_t).any() or torch.isinf(sv_t).any():
                print(f"Warning: Invalid values in steering vector at layer {layer_idx}, skipping")
                continue
            sv_norm = sv_t.norm().item()
            if sv_norm > 100:
                print(f"Warning: Steering vector at layer {layer_idx} has large norm {sv_norm:.2f}, skipping")
                continue
            sv_tensors[layer_idx] = sv_t

        if not sv_tensors:
            print("Warning: No valid steering vectors, falling back to baseline")
            return self.generate_without_steering(prompt, max_new_tokens, temperature, do_sample)

        def make_hook(sv_t, a):
            def hook_fn(module, input, output):
                try:
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    sv_norm = sv_t / (sv_t.norm() + 1e-8)
                    hidden_mean_abs = hidden.abs().mean()
                    max_steering = hidden_mean_abs * 0.1
                    hidden_max = hidden.abs().max()
                    max_steering = min(max_steering, hidden_max * 0.05)

                    steering = a * sv_norm.unsqueeze(0).unsqueeze(0)
                    steering = torch.clamp(steering, -max_steering, max_steering)

                    if steering.abs().max() > hidden.abs().max() * 0.2:
                        return output

                    steered_hidden = hidden + steering

                    if torch.isnan(steered_hidden).any() or torch.isinf(steered_hidden).any():
                        return output

                    if isinstance(output, tuple):
                        return (steered_hidden,) + output[1:]
                    else:
                        return steered_hidden
                except Exception:
                    return output
            return hook_fn

        hooks = []
        for layer_idx, sv_t in sv_tensors.items():
            if layer_idx < len(self.model.model.layers):
                h = self.model.model.layers[layer_idx].register_forward_hook(
                    make_hook(sv_t, alpha)
                )
                hooks.append(h)

        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            input_len = inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        except RuntimeError as e:
            if "CUDA" in str(e) or "assert" in str(e):
                print(f"Warning: Steering caused CUDA error, falling back to baseline")
                for h in hooks:
                    h.remove()
                hooks = []
                return self.generate_without_steering(
                    prompt, max_new_tokens, temperature=0.0, do_sample=False
                )
            else:
                raise
        finally:
            for h in hooks:
                h.remove()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        return text

    def generate_without_steering(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """Baseline generation with no steering."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=1024, padding=False
            )
            inputs = {k: v.to(self.model_device) for k, v in inputs.items()
                      if k in ['input_ids', 'attention_mask']}

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 0.1),
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    top_p=0.95,
                    repetition_penalty=1.1
                )
            input_len = inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        except RuntimeError as e:
            if "CUDA" in str(e) or "assert" in str(e):
                print(f"CUDA error in baseline generation, attempting recovery...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                try:
                    inputs = self.tokenizer(
                        prompt, return_tensors="pt", truncation=True,
                        max_length=1024, padding=False
                    )
                    inputs = {k: v.to(self.model_device) for k, v in inputs.items()
                              if k in ['input_ids', 'attention_mask']}
                    with torch.no_grad():
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    input_len = inputs["input_ids"].shape[1]
                    text = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
                except Exception as e2:
                    print(f"Recovery failed: {e2}")
                    return "[Generation failed due to CUDA error]"
            else:
                raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text


# ==================== Main Experiment ====================

def run_worst_layer_steering(args):
    """
    Full experiment:
    1. Load worst-N layers per capability from bilingual analysis
    2. Compute steering vectors at each of those layers (REVERSED: negative - positive)
       Positives: high-score translated_response + translated_reference_answer
       Negatives: low-score translated_response
    3. Generate baseline + per-layer + combined multi-layer steered responses
    4. Store full responses (no truncation)
    """
    from tqdm import tqdm

    print(f"\n{'='*65}")
    print("Worst-Layer Activation Steering Experiment (REVERSED SV)")
    print(f"Model:          {args.model_name}")
    print(f"Analysis dir:   {args.analysis_dir}")
    print(f"Condition:      {args.condition}")
    print(f"Worst N layers: {args.top_n_worst}")
    print(f"Steering alpha: {args.alpha}")
    print(f"Capability:     {args.filter_capability or 'ALL'}")
    print(f"Instr col:      {args.instruction_col}")
    print(f"Response col:   {args.response_col}")
    print(f"Reference col:  {args.reference_col}")
    print(f"System prompt:  Using '{args.system_prompt_col}' column from dataframe")
    print(f"{'='*65}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load worst layers ──────────────────────────────────────────
    print("Loading worst-performing layers from analysis...")
    worst_layers_map = load_worst_layers_auto(
        analysis_dir=args.analysis_dir,
        capability=args.filter_capability,
        condition=args.condition,
        top_n=args.top_n_worst
    )

    if not worst_layers_map:
        print("Error: Could not determine worst layers. Exiting.")
        return None

    print("\nWorst layers per capability:")
    for cap, layers in worst_layers_map.items():
        print(f"  {cap:<28}: {layers}")

    wl_path = output_dir / "resolved_worst_layers.json"
    with open(wl_path, "w") as f:
        json.dump(worst_layers_map, f, indent=2)
    print(f"\nSaved resolved worst layers → {wl_path}")

    # ── 2. Load data ──────────────────────────────────────────────────
    print("\nLoading dataset...")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} samples. Columns: {df.columns.tolist()}")

    # Validate required columns
    required_cols = [
        args.instruction_col,
        args.response_col,
        args.reference_col,
        args.system_prompt_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    if args.filter_capability and "capability" in df.columns:
        df = df[df["capability"] == args.filter_capability].copy()
        print(f"Filtered to {len(df)} samples for: {args.filter_capability}")

    if len(df) == 0:
        print("Error: No samples after filtering.")
        return None

    eval_df = df.sample(
        n=min(args.num_eval_samples, len(df)), random_state=42
    ).reset_index(drop=True)
    print(f"Evaluation set: {len(eval_df)} samples")

    # ── 3. Initialize model ───────────────────────────────────────────
    extractor = SteeringVectorExtractor(args.model_name)

    # ── 4. Compute steering vectors at worst layers ───────────────────
    all_steering_vectors: Dict[str, Dict[int, np.ndarray]] = {}

    caps_to_steer = (
        [args.filter_capability] if args.filter_capability
        else list(worst_layers_map.keys())
    )

    for cap in caps_to_steer:
        target_layers = worst_layers_map.get(cap, worst_layers_map.get("all", []))
        if not target_layers:
            print(f"  No worst layers found for {cap}, skipping.")
            continue

        target_layers = [l for l in target_layers if l < extractor.num_layers]
        print(f"\nComputing steering vectors for '{cap}' at layers: {target_layers}")

        cap_df = df[df["capability"] == cap] if "capability" in df.columns else df
        if len(cap_df) == 0:
            print(f"  No data for capability '{cap}', skipping.")
            continue

        all_steering_vectors[cap] = {}
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
                    position=args.position
                )
                all_steering_vectors[cap][layer_idx] = sv
                sv_path = output_dir / f"sv_{cap}_layer{layer_idx}.npy"
                np.save(sv_path, sv)
            except Exception as e:
                print(f"  Error at layer {layer_idx}: {e}")
                continue

    if not all_steering_vectors:
        print("Error: No steering vectors computed.")
        return None

    # ── 5. Generate responses ─────────────────────────────────────────
    results = []
    print(f"\nGenerating responses for {len(eval_df)} samples...")

    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Generating"):
        try:
            instruction    = row[args.instruction_col]
            orig_score     = row.get("orig_score", None)
            cap            = str(row.get("capability", ""))
            system_prompt  = row.get(args.system_prompt_col, "Let's think step by step:")
            prompt         = f"{system_prompt}\n\n{instruction}"

            sv_map = (
                all_steering_vectors.get(cap)
                or next(iter(all_steering_vectors.values()), None)
            )
            if sv_map is None:
                continue

            if idx > 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # --- Baseline ---
            try:
                baseline_response = extractor.generate_without_steering(
                    prompt, args.max_new_tokens, args.temperature, args.do_sample
                )
            except Exception as e:
                print(f"  Error in baseline for sample {idx}: {e}")
                continue

            # --- Per-layer steering ---
            per_layer_responses = {}
            for layer_idx, sv in sv_map.items():
                try:
                    steered = extractor.generate_with_multi_layer_steering(
                        prompt, {layer_idx: sv}, args.alpha,
                        args.max_new_tokens, args.temperature, args.do_sample
                    )
                    per_layer_responses[layer_idx] = steered
                except Exception as e:
                    print(f"  Error layer {layer_idx} sample {idx}: {e}")
                    per_layer_responses[layer_idx] = baseline_response

            # --- Combined multi-layer steering ---
            try:
                combined_response = extractor.generate_with_multi_layer_steering(
                    prompt, sv_map, args.alpha,
                    args.max_new_tokens, args.temperature, args.do_sample
                )
            except Exception as e:
                print(f"  Error combined steering sample {idx}: {e}")
                combined_response = baseline_response

            result = {
                "idx":               int(idx),
                "capability":        cap,
                "orig_score":        float(orig_score) if orig_score is not None else None,
                "instruction":       instruction,          # full, no truncation
                "system_prompt":     system_prompt,        # full, no truncation
                "baseline_response": baseline_response,    # full, no truncation
                "combined_response": combined_response,    # full, no truncation
                "worst_layers_steered": list(sv_map.keys()),
            }

            # Store each per-layer response in full
            for layer_idx, resp in per_layer_responses.items():
                result[f"layer{layer_idx}_response"] = resp  # full, no truncation

            results.append(result)
            print(f"  [{idx}] cap={cap:20s} orig={orig_score} | "
                  f"baseline_len={len(baseline_response)} "
                  f"steered_len={len(combined_response)}")

        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            import traceback; traceback.print_exc()
            if "CUDA" in str(e) and "assert" in str(e):
                print("\nCRITICAL: CUDA device is in corrupted state")
                if results:
                    cap_suffix = f"_{args.filter_capability}" if args.filter_capability else ""
                    partial_path = output_dir / f"partial_results_reverse_steering{cap_suffix}.csv"
                    pd.DataFrame(results).to_csv(partial_path, index=False)
                    print(f"Partial results saved to: {partial_path}")
                return None
            continue

        if idx % 5 == 0:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"\nCUDA error during cleanup at sample {idx}, stopping early")
                    break

    # ── 6. Save outputs ───────────────────────────────────────────────
    if not results:
        print("No results collected.")
        return None

    results_df = pd.DataFrame(results)

    cap_suffix = f"_{args.filter_capability}" if args.filter_capability else ""
    csv_path = output_dir / f"reverse_steering_results{cap_suffix}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV  → {csv_path}")

    summary = {
        "args": vars(args),
        "worst_layers_used": {k: v for k, v in worst_layers_map.items()},
        "num_samples": len(results),
        "layers_steered": sorted({l for sv in all_steering_vectors.values() for l in sv}),
        "steering_direction": "reversed (negative - positive)",
    }
    json_path = output_dir / f"reverse_steering_summary{cap_suffix}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON → {json_path}")

    print(f"\n{'='*65}")
    print("GENERATION COMPLETE")
    print(f"Samples:        {len(results)}")
    print(f"Layers steered: {summary['layers_steered']}")
    print(f"SV direction:   reversed (negative - positive)")
    print(f"{'='*65}")

    return results_df


# ==================== Alpha Search ====================

def run_worst_layer_alpha_search(args):
    """
    Search for best alpha using the identified worst layers.
    Ranks alphas by average response length change (as a proxy for
    steering effect magnitude). Saves all generated responses for
    offline analysis.
    """
    from tqdm import tqdm

    print(f"\n{'='*65}")
    print("Alpha Search — Worst Layer Steering (REVERSED SV)")
    print(f"Alpha values: {args.alpha_values}")
    print(f"{'='*65}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    worst_layers_map = load_worst_layers_auto(
        args.analysis_dir, args.filter_capability, args.condition, args.top_n_worst
    )

    df = pd.read_csv(args.data_path)

    # Validate required columns
    required_cols = [
        args.instruction_col,
        args.response_col,
        args.reference_col,
        args.system_prompt_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    if args.filter_capability and "capability" in df.columns:
        df = df[df["capability"] == args.filter_capability].copy()

    eval_df = df.sample(n=min(args.num_eval_samples, len(df)), random_state=42)

    extractor = SteeringVectorExtractor(args.model_name)

    cap_key = args.filter_capability or list(worst_layers_map.keys())[0]
    target_layers = worst_layers_map.get(cap_key, DEFAULT_WORST_LAYERS)
    target_layers = [l for l in target_layers if l < extractor.num_layers]
    print(f"Steering at worst layers: {target_layers}")

    steering_vectors = {}
    for layer_idx in target_layers:
        try:
            sv = extractor.compute_steering_vector(
                df=df,
                layer_idx=layer_idx,
                instruction_col=args.instruction_col,
                response_col=args.response_col,
                reference_col=args.reference_col,
                score_col="orig_score",
                system_prompt_col=args.system_prompt_col,
                high_score=args.high_score_threshold,
                low_score=args.low_score_threshold,
                max_samples=args.max_steering_samples
            )
            steering_vectors[layer_idx] = sv
        except Exception as e:
            print(f"  Error at layer {layer_idx}: {e}")

    alpha_results = {}
    for alpha in args.alpha_values:
        print(f"\nTesting alpha={alpha}...")
        rows = []

        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"α={alpha}"):
            try:
                system_prompt = row.get(args.system_prompt_col, "Let's think step by step:")
                prompt    = f"{system_prompt}\n\n{row[args.instruction_col]}"
                baseline  = extractor.generate_without_steering(prompt, args.max_new_tokens)
                steered   = extractor.generate_with_multi_layer_steering(
                    prompt, steering_vectors, alpha, args.max_new_tokens
                )
                rows.append({
                    "instruction":       row[args.instruction_col],
                    "system_prompt":     system_prompt,
                    "baseline_response": baseline,
                    "steered_response":  steered,
                    "baseline_len":      len(baseline),
                    "steered_len":       len(steered),
                    "len_delta":         len(steered) - len(baseline),
                })
            except Exception as e:
                print(f"  Error: {e}")

        alpha_df = pd.DataFrame(rows)
        alpha_csv = output_dir / f"reverse_alpha_search_a{alpha}.csv"
        alpha_df.to_csv(alpha_csv, index=False)

        alpha_results[str(alpha)] = {
            "mean_baseline_len": float(alpha_df["baseline_len"].mean()) if rows else 0.0,
            "mean_steered_len":  float(alpha_df["steered_len"].mean())  if rows else 0.0,
            "mean_len_delta":    float(alpha_df["len_delta"].mean())    if rows else 0.0,
            "responses_csv":     str(alpha_csv),
        }
        print(f"  α={alpha}: mean steered len={alpha_results[str(alpha)]['mean_steered_len']:.1f}  "
              f"Δlen={alpha_results[str(alpha)]['mean_len_delta']:+.1f}")

    cap_suffix = f"_{args.filter_capability}" if args.filter_capability else ""
    out_path = output_dir / f"reverse_alpha_search{cap_suffix}.json"
    with open(out_path, "w") as f:
        json.dump({
            "worst_layers": target_layers,
            "alpha_results": alpha_results,
            "steering_direction": "reversed (negative - positive)",
        }, f, indent=2)
    print(f"\nAlpha search → {out_path}")
    return alpha_results


# ==================== CLI ====================

def main():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            print(f"CUDA initialized: {torch.cuda.device_count()} device(s) available")
        except Exception as e:
            print(f"Warning: CUDA reset failed: {e}")

    parser = argparse.ArgumentParser(
        description="Worst-Layer Activation Steering (Reversed SV) using Bilingual Analysis"
    )

    # Data
    parser.add_argument("--data_path",    type=str, required=True)
    parser.add_argument("--output_dir",   type=str, default="./reverse_steering_output")

    # Analysis results
    parser.add_argument("--analysis_dir", type=str, required=True,
                        help="Directory containing bilingual_metrics.csv or bilingual_report.txt")
    parser.add_argument("--condition",    type=str, default="Multilingual",
                        choices=["English", "Multilingual"])
    parser.add_argument("--top_n_worst",  type=int, default=5)

    # Model (the one being steered)
    parser.add_argument("--model_name",   type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")

    # Column selection
    parser.add_argument("--instruction_col",   type=str, default="translated_input")
    parser.add_argument("--response_col",      type=str, default="translated_orig_response")
    parser.add_argument("--reference_col",     type=str, default="translated_reference_answer")
    parser.add_argument("--system_prompt_col", type=str, default="translated_system_prompt")

    # Mode
    parser.add_argument("--mode", type=str, default="steer",
                        choices=["steer", "alpha_search"])

    # Steering
    parser.add_argument("--alpha",        type=float, default=0.5)
    parser.add_argument("--alpha_values", type=float, nargs="+",
                        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--position",     type=str, default="last",
                        choices=["last", "mean"])

    # Contrastive pairs
    parser.add_argument("--high_score_threshold", type=float, default=5.0)
    parser.add_argument("--low_score_threshold",  type=float, default=1.0)
    parser.add_argument("--max_steering_samples", type=int,   default=50)

    # Capability filter
    parser.add_argument("--filter_capability", type=str, default=None,
                        choices=["grounding", "instruction_following", "planning",
                                 "reasoning", "safety", "theory_of_mind", None])

    # Generation
    parser.add_argument("--num_eval_samples", type=int,   default=50)
    parser.add_argument("--max_new_tokens",   type=int,   default=512)
    parser.add_argument("--temperature",      type=float, default=0.7)
    parser.add_argument("--do_sample",        action="store_true", default=False)
    parser.add_argument("--use_greedy",       action="store_true", default=True)

    args = parser.parse_args()

    if args.mode == "steer":
        run_worst_layer_steering(args)
    elif args.mode == "alpha_search":
        run_worst_layer_alpha_search(args)


if __name__ == "__main__":
    main()