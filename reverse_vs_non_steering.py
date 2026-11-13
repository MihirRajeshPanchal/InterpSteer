"""
nlp_metrics_steering.py
========================
Extends reverse_vs_non_steering.py with NLP metric evaluation:
  - ROUGE-1, ROUGE-2, ROUGE-L
  - BLEU (corpus + sentence)
  - BLEURT (if available, else skipped gracefully)
  - BERTScore (optional)
  - Lexical diversity (TTR)
  - Perplexity proxy (avg token-length entropy)

Metrics are computed:
  A) Forward-steered vs Baseline  (how much fwd changes baseline)
  B) Reverse-steered vs Baseline  (how much rev changes baseline)
  C) Forward-steered vs Reverse-steered (direct comparison)

All four experiment types are handled:
  1. per-capability steering
  2. per-capability reverse steering
  3. unified sum/average steering
  4. unified reverse sum/average

Usage:
  python nlp_metrics_steering.py \
      --model_name  sarvam-1 \
      --forward_csvs /path/to/steering/grounding/worst_layer_steering_results_grounding.csv ... \
      --reverse_csvs /path/to/reverse_steering/grounding/worst_layer_steering_results_grounding.csv ... \
      --output_dir  /path/to/analysis/nlp_metrics/

  # Cross-model mode:
  python nlp_metrics_steering.py \
      --mode cross_model \
      --summary_jsons /model_a/nlp_summary.json /model_b/nlp_summary.json \
      --output_dir  /path/to/cross_model_nlp/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Network probe ──────────────────────────────────────────────────────────────
import socket

def _hf_reachable(timeout: float = 3.0) -> bool:
    """Return True only if huggingface.co is reachable on port 443."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo("huggingface.co", 443)
        return True
    except Exception:
        return False

_HF_ONLINE = _hf_reachable()
if not _HF_ONLINE:
    print("  [info] HuggingFace is not reachable from this environment.")
    print("         BLEURT and BERTScore will be skipped automatically.")
    print("         Pass --no_network to suppress this check entirely.\n")

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("  [warn] rouge_score not installed — ROUGE disabled. pip install rouge-score")

try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    import nltk
    for _tok in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(_tok)
            break
        except LookupError:
            try:
                nltk.download(_tok.split("/")[1], quiet=True)
                break
            except Exception:
                pass
    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False
    print("  [warn] nltk not installed — BLEU disabled. pip install nltk")

# ── BERTScore ──────────────────────────────────────────────────────────────────
# bert_score's `model_type` argument uses short legacy names from its internal
# registry (e.g. "distilbert-base-uncased"), NOT namespaced Hub IDs like
# "distilbert/distilbert-base-uncased".  We store both:
#   model_type  → short name passed to bert_score API
#   hub_id      → full namespaced ID used only for pre-downloading weights
#
# Candidates are tried in priority order; first successful smoke-test wins.

HAS_BERTSCORE = False
_BERTSCORE_MODEL = None  # set to the short model_type name on success

# (label, bert_score_model_type, hub_id)
# hub_id is used to pre-fetch weights via transformers when model_type alone
# isn't in bert_score's registry; model_type is always used for scoring calls.
_BERTSCORE_HUB_CANDIDATES: List[Tuple[str, str, str]] = [
    ("distilbert-base-uncased", "distilbert-base-uncased", "distilbert/distilbert-base-uncased"),
    ("roberta-large",           "roberta-large",           "FacebookAI/roberta-large"),
    ("bert-base-uncased",       "bert-base-uncased",       "google-bert/bert-base-uncased"),
]


def _ensure_model_downloaded(hub_id: str) -> None:
    """
    Pre-download model weights from HF Hub so bert_score can find them
    in the local cache under the short name it expects.
    Silently no-ops if transformers is not installed or download fails.
    """
    try:
        import logging
        from transformers import AutoTokenizer, AutoModel, logging as hf_logging
        _prev = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        AutoTokenizer.from_pretrained(hub_id)
        AutoModel.from_pretrained(hub_id)
        hf_logging.set_verbosity(_prev)
    except Exception:
        pass  # bert_score may still work from its own cache


def _get_available_gpus() -> List[int]:
    """Return list of CUDA device indices, empty if none available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def _gpu_free_mb(device: int) -> float:
    """Return free VRAM in MB for a given CUDA device index."""
    try:
        import torch
        free, _ = torch.cuda.mem_get_info(device)
        return free / (1024 ** 2)
    except Exception:
        return 0.0


def _silence_loggers():
    """Silence transformers loggers; return (tf_logger, prev_level, prev_hf)."""
    import logging
    tf_logger = logging.getLogger("transformers")
    prev_level = tf_logger.level
    tf_logger.setLevel(logging.ERROR)
    prev_hf = None
    try:
        from transformers import logging as hf_logging
        prev_hf = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    return tf_logger, prev_level, prev_hf


def _restore_loggers(tf_logger, prev_level, prev_hf) -> None:
    tf_logger.setLevel(prev_level)
    if prev_hf is not None:
        try:
            from transformers import logging as hf_logging
            hf_logging.set_verbosity(prev_hf)
        except Exception:
            pass


def _quiet_bertscore_on_device(
    model_type: str,
    hypotheses: List[str],
    references: List[str],
    device: Optional[int],        # None → CPU
    batch_size: int = 32,
):
    """
    Run bert_score on a specific device (int → cuda:N, None → cpu).
    Verbosity is fully suppressed.
    """
    tf_logger, prev_level, prev_hf = _silence_loggers()
    try:
        from bert_score import score as _bscore
        kwargs = dict(
            model_type=model_type,
            lang="en",
            verbose=False,
            rescale_with_baseline=False,
            batch_size=batch_size,
            nthreads=1,
        )
        if device is not None:
            kwargs["device"] = f"cuda:{device}"
        else:
            kwargs["device"] = "cpu"
        result = _bscore(hypotheses, references, **kwargs)
    finally:
        _restore_loggers(tf_logger, prev_level, prev_hf)
    return result


def _quiet_bertscore(
    model_type: str,
    hypotheses: List[str],
    references: List[str],
    batch_size: int = 32,
):
    """
    Multi-GPU batched BERTScore with automatic CPU fallback.

    Strategy:
      1. Collect all available CUDA devices sorted by free VRAM (most free first).
      2. Split the input into N equal chunks (one per GPU).
      3. Score each chunk on its assigned GPU.
      4. If any GPU chunk raises OOM, retry that chunk on CPU.
      5. If no GPUs exist, run everything on CPU.

    Concatenate per-chunk P/R/F1 tensors before returning.
    """
    import torch

    gpus = _get_available_gpus()

    # Sort GPUs by free VRAM descending so the fullest GPU gets the smallest chunk
    if gpus:
        gpus = sorted(gpus, key=lambda g: _gpu_free_mb(g), reverse=True)
        print(f"  [BERTScore] {len(gpus)} GPU(s) available — "
              + ", ".join(f"cuda:{g} ({_gpu_free_mb(g):.0f} MB free)" for g in gpus))

    n = len(hypotheses)

    if not gpus:
        # CPU-only path
        print(f"  [BERTScore] No GPUs — running on CPU (n={n})")
        return _quiet_bertscore_on_device(model_type, hypotheses, references,
                                          device=None, batch_size=batch_size)

    # Split input into chunks, one per GPU
    chunk_size = math.ceil(n / len(gpus))
    chunks = [
        (
            hypotheses[i: i + chunk_size],
            references[i: i + chunk_size],
            gpus[idx],
        )
        for idx, i in enumerate(range(0, n, chunk_size))
    ]

    P_all, R_all, F1_all = [], [], []

    for hyp_chunk, ref_chunk, gpu in chunks:
        if not hyp_chunk:
            continue
        try:
            P, R, F1 = _quiet_bertscore_on_device(
                model_type, hyp_chunk, ref_chunk,
                device=gpu, batch_size=batch_size,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Free cache and retry on CPU
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                print(f"  [BERTScore] cuda:{gpu} OOM — retrying chunk on CPU")
                P, R, F1 = _quiet_bertscore_on_device(
                    model_type, hyp_chunk, ref_chunk,
                    device=None, batch_size=max(1, batch_size // 4),
                )
            else:
                raise
        P_all.append(P)
        R_all.append(R)
        F1_all.append(F1)

    return (
        torch.cat(P_all),
        torch.cat(R_all),
        torch.cat(F1_all),
    )


def _smoke_test_bertscore(model_type: str) -> bool:
    """
    Return True if bert_score can score a trivial pair using the given
    model_type (short name understood by bert_score's internal registry).
    Uses only the most-free GPU (or CPU) to avoid wasting VRAM on a smoke test.
    """
    try:
        gpus = _get_available_gpus()
        device = sorted(gpus, key=lambda g: _gpu_free_mb(g), reverse=True)[0] if gpus else None
        _quiet_bertscore_on_device(model_type, ["hello world"], ["hello world"],
                                   device=device, batch_size=1)
        return True
    except Exception as e:
        print(f"    [BERTScore] model_type='{model_type}' failed: {type(e).__name__}: {str(e)[:120]}")
        return False


def _init_bertscore(user_model: Optional[str] = None) -> bool:
    """
    Resolve and smoke-test a BERTScore model from HuggingFace Hub.
    Sets global _BERTSCORE_MODEL (short model_type name) on success.
    Priority: user_model > hub candidates (distilbert > roberta-large > bert-base).
    Requires HuggingFace to be reachable; skips gracefully otherwise.
    """
    global _BERTSCORE_MODEL

    try:
        import bert_score  # noqa: F401
    except ImportError:
        print("  [info] bert_score not installed — BERTScore disabled. pip install bert-score")
        return False

    if not _HF_ONLINE:
        print("  [info] BERTScore: HuggingFace not reachable — skipping.")
        return False

    # Build candidate list: (label, model_type, hub_id)
    candidates: List[Tuple[str, str, str]] = []
    if user_model:
        # For user-supplied models treat the value as both model_type and hub_id
        candidates.append((f"user-specified ({user_model})", user_model, user_model))
    candidates.extend(_BERTSCORE_HUB_CANDIDATES)

    for label, model_type, hub_id in candidates:
        print(f"  [info] BERTScore: trying {label} ...", end=" ", flush=True)
        # Ensure weights are in the HF cache before bert_score tries to load them
        _ensure_model_downloaded(hub_id)
        if _smoke_test_bertscore(model_type):
            _BERTSCORE_MODEL = model_type
            print(f"OK → using model_type='{model_type}'")
            return True

    print("  [info] BERTScore: all Hub candidates failed — skipping.")
    return False


HAS_BERTSCORE = _init_bertscore()


# ── BLEURT ─────────────────────────────────────────────────────────────────────
# BLEURT requires `pip install git+https://github.com/google-research/bleurt`
# which is a non-standard dependency not available on PyPI.  We attempt to load
# it via the HuggingFace evaluate library but skip gracefully if the extra
# dependency is missing — no noise, just a clean info message.

HAS_BLEURT = False
bleurt_metric = None


def _init_bleurt_evaluate(path_or_name: str) -> bool:
    """Try loading BLEURT via the HuggingFace evaluate library."""
    global bleurt_metric
    try:
        import evaluate as _evaluate
        bleurt_metric = _evaluate.load(path_or_name)
        result = bleurt_metric.compute(predictions=["hello"], references=["hello"])
        if "scores" not in result:
            raise ValueError(f"Unexpected bleurt output keys: {list(result.keys())}")
        print(f"  [info] BLEURT ready via evaluate (name: {path_or_name})")
        return True
    except ImportError as e:
        # Missing git-installed bleurt package — no point retrying other names
        print(
            f"  [info] BLEURT skipped: missing dependency — "
            f"install with: pip install git+https://github.com/google-research/bleurt"
        )
        bleurt_metric = None
        return False
    except Exception as e:
        print(f"  [info] evaluate.load('{path_or_name}') failed: {type(e).__name__}: {str(e)[:140]}")
        bleurt_metric = None
        return False


def _init_bleurt(user_name: Optional[str] = None) -> bool:
    """
    Load BLEURT from HuggingFace Hub via the evaluate library.
    BLEURT requires a non-PyPI package; skips cleanly if unavailable.
    Skips entirely if Hub is unreachable.
    """
    if not _HF_ONLINE:
        print("  [info] BLEURT: HuggingFace not reachable — skipping.")
        return False

    candidates = []
    if user_name:
        candidates.append(user_name)
    candidates.extend(["bleurt", "google/bleurt-20"])

    for name in candidates:
        result = _init_bleurt_evaluate(name)
        if result:
            return True
        # If it was a missing-dependency ImportError, no point trying other names
        if bleurt_metric is None and name == candidates[0]:
            # Check if the failure was an ImportError (already printed a clean msg)
            try:
                import bleurt  # type: ignore  # noqa: F401
            except ImportError:
                return False  # bleurt package not installed; skip remaining candidates

    print("  [info] BLEURT: all Hub strategies failed — skipping.")
    return False


HAS_BLEURT = _init_bleurt()

# ─────────────────────────────────────────────────────────────────────────────
# Constants (shared with reverse_vs_non_steering.py)
# ─────────────────────────────────────────────────────────────────────────────

ALL_CAPABILITIES = [
    "grounding", "instruction_following", "planning",
    "reasoning", "safety", "theory_of_mind",
]
UNIFIED_MODES = ["sum", "average"]

RESPONSE_COLS = {
    "baseline": "baseline_response",
    "steered":  "combined_response",
    "unified":  "unified_response",
}

PALETTE = {
    "baseline":        "#4878D0",
    "forward_steered": "#EE854A",
    "reverse_steered": "#6ACC65",
    "fwd_vs_rev":      "#956CB4",
}

METRIC_DISPLAY = {
    "rouge1_f":  "ROUGE-1 F1",
    "rouge2_f":  "ROUGE-2 F1",
    "rougeL_f":  "ROUGE-L F1",
    "bleu":      "BLEU",
    "bleurt":    "BLEURT",
    "bertscore_f1": "BERTScore F1",
    "ttr":       "Lexical Diversity (TTR)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Path inference (copied from base script)
# ─────────────────────────────────────────────────────────────────────────────

def infer_meta_from_path(csv_path: str) -> Dict[str, str]:
    p = Path(csv_path)
    parts = [x.lower() for x in p.parts]
    direction = "reverse" if "reverse_steering" in parts else "forward"
    fname = p.stem.lower()
    for mode in UNIFIED_MODES:
        if f"unified_{mode}" in fname or f"unified_{mode}" in parts:
            return {"exp_type": f"unified_{mode}", "capability": "all", "direction": direction}
    parent = p.parent.name.lower()
    capability = parent if parent in ALL_CAPABILITIES else "unknown"
    m = re.search(r"results_(.+)$", fname)
    if m:
        candidate = m.group(1)
        if candidate in ALL_CAPABILITIES:
            capability = candidate
    return {"exp_type": "per_cap", "capability": capability, "direction": direction}


def load_csv_with_meta(csv_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Warning: could not read {csv_path}: {e}")
        return None
    if df.empty:
        return None
    meta = infer_meta_from_path(csv_path)
    df["_exp_type"]   = meta["exp_type"]
    df["_capability"] = meta["capability"]
    df["_direction"]  = meta["direction"]
    df["_source_csv"] = csv_path
    return df


def discover_csvs(base_dir: str) -> Tuple[List[str], List[str]]:
    base = Path(base_dir)
    forward, reverse = [], []
    for csv in sorted(base.rglob("*.csv")):
        if any(x in csv.name for x in ("partial_", "alpha_search")):
            continue
        if "reverse_steering" in str(csv):
            reverse.append(str(csv))
        elif "steering" in str(csv):
            forward.append(str(csv))
    return forward, reverse


# ─────────────────────────────────────────────────────────────────────────────
# NLP metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_str(x) -> str:
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def type_token_ratio(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    if not HAS_ROUGE or not hypotheses:
        return {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1_f, r2_f, rl_f = [], [], []
    for hyp, ref in zip(hypotheses, references):
        if not hyp or not ref:
            continue
        scores = scorer.score(ref, hyp)
        r1_f.append(scores["rouge1"].fmeasure)
        r2_f.append(scores["rouge2"].fmeasure)
        rl_f.append(scores["rougeL"].fmeasure)
    return {
        "rouge1_f": float(np.mean(r1_f)) if r1_f else float("nan"),
        "rouge2_f": float(np.mean(r2_f)) if r2_f else float("nan"),
        "rougeL_f": float(np.mean(rl_f)) if rl_f else float("nan"),
    }


def compute_bleu(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    if not HAS_BLEU or not hypotheses:
        return {}
    smooth = SmoothingFunction().method1
    refs_tok  = [[tokenize(r)] for r in references if r]
    hyps_tok  = [tokenize(h) for h in hypotheses if h]
    if not hyps_tok or not refs_tok:
        return {}
    try:
        corpus = corpus_bleu(refs_tok, hyps_tok, smoothing_function=smooth)
    except Exception:
        corpus = float("nan")
    sent_scores = []
    for ref_list, hyp in zip(refs_tok, hyps_tok):
        try:
            s = sentence_bleu(ref_list, hyp, smoothing_function=smooth)
            sent_scores.append(s)
        except Exception:
            pass
    return {
        "bleu_corpus": float(corpus),
        "bleu_sentence": float(np.mean(sent_scores)) if sent_scores else float("nan"),
    }


def compute_bleurt(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    if not HAS_BLEURT or not hypotheses:
        return {}
    valid = [(h, r) for h, r in zip(hypotheses, references) if h and r]
    if not valid:
        return {}
    hyps, refs = zip(*valid)
    try:
        results = bleurt_metric.compute(predictions=list(hyps), references=list(refs))
        scores = results["scores"]
        return {"bleurt": float(np.mean(scores))}
    except Exception as e:
        print(f"    [BLEURT error] {e}")
        return {}


def compute_bertscore(
    hypotheses: List[str],
    references: List[str],
    batch_size: int = 32,
) -> Dict[str, float]:
    if not HAS_BERTSCORE or _BERTSCORE_MODEL is None or not hypotheses:
        return {}
    valid = [(h, r) for h, r in zip(hypotheses, references) if h and r]
    if not valid:
        return {}
    hyps, refs = zip(*valid)
    try:
        P, R, F1 = _quiet_bertscore(
            _BERTSCORE_MODEL, list(hyps), list(refs), batch_size=batch_size
        )
        return {
            "bertscore_p":  float(P.mean()),
            "bertscore_r":  float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }
    except (OSError, ConnectionError, TimeoutError) as e:
        print(f"    [BERTScore] network error — skipping: {e}")
        return {}
    except Exception as e:
        print(f"    [BERTScore error] {e}")
        return {}


def compute_ttr(texts: List[str]) -> Dict[str, float]:
    ttrs = [type_token_ratio(t) for t in texts if t]
    return {"ttr": float(np.mean(ttrs)) if ttrs else float("nan")}


def all_metrics(hypotheses: List[str], references: List[str], batch_size: int = 32) -> Dict[str, float]:
    """Compute all available NLP metrics for hypothesis vs reference."""
    out: Dict[str, float] = {}
    out.update(compute_rouge(hypotheses, references))
    out.update(compute_bleu(hypotheses, references))
    out.update(compute_bleurt(hypotheses, references))
    out.update(compute_bertscore(hypotheses, references, batch_size=batch_size))
    out.update(compute_ttr(hypotheses))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Merge forward + reverse DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def resolve_steered_col(df: pd.DataFrame) -> str:
    exp = df["_exp_type"].iloc[0] if "_exp_type" in df.columns else "per_cap"
    if "unified" in exp:
        return RESPONSE_COLS["unified"]
    return RESPONSE_COLS["steered"]


def merge_fwd_rev(fwd_dfs: List[pd.DataFrame],
                  rev_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine forward and reverse DataFrames.
    Each row gets:
      baseline_response, fwd_response, rev_response, capability, exp_type
    """
    def prep(dfs: List[pd.DataFrame], tag: str) -> pd.DataFrame:
        out = []
        for df in dfs:
            df = df.copy()
            scol = resolve_steered_col(df)
            resp_col = f"{tag}_response"
            if scol in df.columns:
                df[resp_col] = df[scol].fillna("").astype(str)
            else:
                df[resp_col] = ""
            df["baseline_response"] = df.get(
                RESPONSE_COLS["baseline"], pd.Series([""] * len(df))
            ).fillna("").astype(str)
            out.append(df)
        if not out:
            return pd.DataFrame()
        return pd.concat(out, ignore_index=True)

    fwd_all = prep(fwd_dfs, "fwd")
    rev_all = prep(rev_dfs, "rev")

    if fwd_all.empty and rev_all.empty:
        return pd.DataFrame()
    if fwd_all.empty:
        rev_all["fwd_response"] = ""
        return rev_all
    if rev_all.empty:
        fwd_all["rev_response"] = ""
        return fwd_all

    merge_keys = [k for k in ("idx", "instruction", "_capability", "_exp_type")
                  if k in fwd_all.columns and k in rev_all.columns]

    if not merge_keys:
        fwd_all["rev_response"] = ""
        return fwd_all

    rev_slim = rev_all[merge_keys + ["rev_response"]].copy()
    merged = pd.merge(fwd_all, rev_slim, on=merge_keys, how="outer")

    for col in ("fwd_response", "rev_response", "baseline_response"):
        if col in merged.columns:
            merged[col] = merged[col].fillna("")

    if "_capability" in merged.columns:
        merged["capability"] = merged["_capability"]

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Per-group metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_group_metrics(grp: pd.DataFrame, batch_size: int = 32) -> Dict[str, Dict[str, float]]:
    """
    For a sub-DataFrame, compute metrics for three comparisons:
      fwd_vs_baseline, rev_vs_baseline, fwd_vs_rev
    """
    base = [_safe_str(x) for x in grp.get("baseline_response", [])]
    fwd  = [_safe_str(x) for x in grp.get("fwd_response", [])]
    rev  = [_safe_str(x) for x in grp.get("rev_response", [])]

    results: Dict[str, Dict[str, float]] = {}
    results["fwd_vs_baseline"] = all_metrics(fwd, base, batch_size=batch_size)
    results["rev_vs_baseline"] = all_metrics(rev, base, batch_size=batch_size)
    results["fwd_vs_rev"]      = all_metrics(fwd, rev,  batch_size=batch_size)
    results["baseline_ttr"]    = compute_ttr(base)
    results["fwd_ttr"]         = compute_ttr(fwd)
    results["rev_ttr"]         = compute_ttr(rev)
    return results


def compute_all_metrics(merged: pd.DataFrame, batch_size: int = 32) -> Dict:
    """
    Compute metrics at three granularities:
      overall, per_capability, per_exp_type
    """
    output: Dict = {
        "overall":        compute_group_metrics(merged, batch_size=batch_size),
        "per_capability": {},
        "per_exp_type":   {},
    }

    cap_col = "capability" if "capability" in merged.columns else "_capability"
    if cap_col in merged.columns:
        for cap, grp in merged.groupby(cap_col):
            output["per_capability"][str(cap)] = compute_group_metrics(grp, batch_size=batch_size)

    if "_exp_type" in merged.columns:
        for et, grp in merged.groupby("_exp_type"):
            output["per_exp_type"][str(et)] = compute_group_metrics(grp, batch_size=batch_size)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Verdict / winner logic
# ─────────────────────────────────────────────────────────────────────────────

HIGHER_IS_BETTER = {
    "rouge1_f", "rouge2_f", "rougeL_f",
    "bleu_corpus", "bleu_sentence", "bleurt",
    "bertscore_f1", "bertscore_p", "bertscore_r", "ttr",
}

def winner_per_metric(metrics: Dict) -> Dict[str, str]:
    """
    For each metric, compare fwd_vs_baseline and rev_vs_baseline.
    Returns {"metric_name": "forward" | "reverse" | "tie"}.
    """
    fwd = metrics.get("fwd_vs_baseline", {})
    rev = metrics.get("rev_vs_baseline", {})
    verdicts: Dict[str, str] = {}

    all_metric_names = set(fwd.keys()) | set(rev.keys())
    for m in sorted(all_metric_names):
        fv = fwd.get(m, float("nan"))
        rv = rev.get(m, float("nan"))
        if math.isnan(fv) and math.isnan(rv):
            verdicts[m] = "n/a"
        elif math.isnan(fv):
            verdicts[m] = "reverse"
        elif math.isnan(rv):
            verdicts[m] = "forward"
        else:
            diff = abs(fv - rv)
            if diff < 0.001:
                verdicts[m] = "tie"
            elif fv > rv:
                verdicts[m] = "forward"
            else:
                verdicts[m] = "reverse"
    return verdicts


def overall_verdict(verdicts: Dict[str, str]) -> str:
    counts = Counter(verdicts.values())
    fwd_wins = counts.get("forward", 0)
    rev_wins = counts.get("reverse", 0)
    if fwd_wins > rev_wins:
        return f"FORWARD steering (wins {fwd_wins}/{len(verdicts)} metrics)"
    elif rev_wins > fwd_wins:
        return f"REVERSE steering (wins {rev_wins}/{len(verdicts)} metrics)"
    else:
        return f"TIE ({fwd_wins} each)"


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_metrics_by_capability(
    all_metrics_dict: Dict,
    model_name: str,
    output_dir: Path,
    plot_format: str = "png",
    dpi: int = 150,
) -> None:
    per_cap = all_metrics_dict.get("per_capability", {})
    if not per_cap:
        per_cap = {"all": all_metrics_dict.get("overall", {})}

    caps = sorted(per_cap.keys())
    sample = next(iter(per_cap.values()), {})
    sample_fvb = sample.get("fwd_vs_baseline", {})
    available_metrics = [m for m in METRIC_DISPLAY if m in sample_fvb]
    if not available_metrics:
        print("  Skipping capability metric plot: no metrics available.")
        return

    ncols = min(3, len(available_metrics))
    nrows = math.ceil(len(available_metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 4),
                             squeeze=False)

    x = np.arange(len(caps))
    width = 0.35

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx // ncols][idx % ncols]
        fwd_vals = [per_cap[c].get("fwd_vs_baseline", {}).get(metric, np.nan) for c in caps]
        rev_vals = [per_cap[c].get("rev_vs_baseline", {}).get(metric, np.nan) for c in caps]

        ax.bar(x - width / 2, fwd_vals, width, label="Forward vs Baseline",
               color=PALETTE["forward_steered"], alpha=0.85)
        ax.bar(x + width / 2, rev_vals, width, label="Reverse vs Baseline",
               color=PALETTE["reverse_steered"], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in caps], fontsize=8)
        ax.set_title(METRIC_DISPLAY.get(metric, metric), fontsize=10)
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, min(1.05, max(
            [v for v in fwd_vals + rev_vals if not np.isnan(v)] or [1.0]
        ) * 1.15))

    for j in range(len(available_metrics), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"{model_name} — NLP Metrics by Capability", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir / f"metrics_by_capability.{plot_format}", dpi)


def plot_metric_heatmap(
    all_metrics_dict: Dict,
    model_name: str,
    output_dir: Path,
    plot_format: str = "png",
    dpi: int = 150,
) -> None:
    per_cap = all_metrics_dict.get("per_capability", {})
    if not per_cap:
        return

    caps = sorted(per_cap.keys())
    sample = next(iter(per_cap.values()), {})
    available = [m for m in METRIC_DISPLAY if m in sample.get("fwd_vs_baseline", {})]
    if not available:
        return

    def build_matrix(comparison: str) -> np.ndarray:
        mat = np.full((len(caps), len(available)), np.nan)
        for ci, cap in enumerate(caps):
            for mi, metric in enumerate(available):
                mat[ci, mi] = per_cap[cap].get(comparison, {}).get(metric, np.nan)
        return mat

    fwd_mat = build_matrix("fwd_vs_baseline")
    rev_mat = build_matrix("rev_vs_baseline")

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(available) * 2.2), max(4, len(caps) * 0.9)))
    col_labels = [METRIC_DISPLAY.get(m, m) for m in available]
    row_labels  = caps

    for ax, mat, title in [
        (axes[0], fwd_mat, "Forward vs Baseline"),
        (axes[1], rev_mat, "Reverse vs Baseline"),
    ]:
        vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
        im = ax.imshow(mat, aspect="auto", cmap="YlGn", vmin=0, vmax=max(vmax, 0.01))
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        for ci in range(len(caps)):
            for mi in range(len(available)):
                v = mat[ci, mi]
                if not np.isnan(v):
                    ax.text(mi, ci, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black")

    fig.suptitle(f"{model_name} — NLP Metric Heatmap", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir / f"metric_heatmap.{plot_format}", dpi)


def plot_fwd_vs_rev_direct(
    all_metrics_dict: Dict,
    model_name: str,
    output_dir: Path,
    plot_format: str = "png",
    dpi: int = 150,
) -> None:
    per_cap = all_metrics_dict.get("per_capability", {})
    if not per_cap:
        return

    caps = sorted(per_cap.keys())
    sample = next(iter(per_cap.values()), {})
    available = [m for m in METRIC_DISPLAY if m in sample.get("fwd_vs_rev", {})]
    if not available:
        return

    x = np.arange(len(caps))
    width = max(0.1, 0.8 / len(available))

    fig, ax = plt.subplots(figsize=(max(10, len(caps) * 2), 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(available)))

    for mi, (metric, color) in enumerate(zip(available, colors)):
        vals = [per_cap[c].get("fwd_vs_rev", {}).get(metric, np.nan) for c in caps]
        offset = (mi - len(available) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=METRIC_DISPLAY.get(metric, metric),
               color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in caps], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — Forward vs Reverse (Direct Similarity)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, output_dir / f"fwd_vs_rev_direct.{plot_format}", dpi)


def plot_verdict_summary(
    all_metrics_dict: Dict,
    model_name: str,
    output_dir: Path,
    plot_format: str = "png",
    dpi: int = 150,
) -> None:
    per_cap = all_metrics_dict.get("per_capability", {})
    if not per_cap:
        overall = all_metrics_dict.get("overall", {})
        if not overall:
            return
        per_cap = {"overall": overall}

    caps = sorted(per_cap.keys())
    fwd_counts, rev_counts, tie_counts = [], [], []

    for cap in caps:
        v = winner_per_metric(per_cap[cap])
        c = Counter(v.values())
        fwd_counts.append(c.get("forward", 0))
        rev_counts.append(c.get("reverse", 0))
        tie_counts.append(c.get("tie", 0) + c.get("n/a", 0))

    x = np.arange(len(caps))
    fig, ax = plt.subplots(figsize=(max(8, len(caps) * 1.5), 5))
    ax.bar(x, fwd_counts, label="Forward wins",
           color=PALETTE["forward_steered"], alpha=0.85)
    ax.bar(x, rev_counts, bottom=fwd_counts, label="Reverse wins",
           color=PALETTE["reverse_steered"], alpha=0.85)
    ax.bar(x, tie_counts,
           bottom=[f + r for f, r in zip(fwd_counts, rev_counts)],
           label="Tie / N/A", color="#CCCCCC", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in caps], fontsize=9)
    ax.set_ylabel("Number of metrics")
    ax.set_title(f"{model_name} — Metric Wins: Forward vs Reverse per Capability")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, output_dir / f"verdict_summary.{plot_format}", dpi)


def plot_ttr_comparison(
    all_metrics_dict: Dict,
    model_name: str,
    output_dir: Path,
    plot_format: str = "png",
    dpi: int = 150,
) -> None:
    per_cap = all_metrics_dict.get("per_capability", {})
    if not per_cap:
        return

    caps = sorted(per_cap.keys())
    base_ttr = [per_cap[c].get("baseline_ttr", {}).get("ttr", np.nan) for c in caps]
    fwd_ttr  = [per_cap[c].get("fwd_ttr", {}).get("ttr", np.nan)      for c in caps]
    rev_ttr  = [per_cap[c].get("rev_ttr", {}).get("ttr", np.nan)       for c in caps]

    x = np.arange(len(caps))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(caps) * 1.8), 5))
    ax.bar(x - width, base_ttr, width, label="Baseline TTR",
           color=PALETTE["baseline"], alpha=0.85)
    ax.bar(x,         fwd_ttr,  width, label="Forward TTR",
           color=PALETTE["forward_steered"], alpha=0.85)
    ax.bar(x + width, rev_ttr,  width, label="Reverse TTR",
           color=PALETTE["reverse_steered"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in caps], fontsize=9)
    ax.set_ylabel("Type-Token Ratio")
    ax.set_title(f"{model_name} — Lexical Diversity (TTR)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, output_dir / f"ttr_comparison.{plot_format}", dpi)


def plot_exp_type_metrics(
    all_metrics_dict: Dict,
    model_name: str,
    output_dir: Path,
    plot_format: str = "png",
    dpi: int = 150,
) -> None:
    per_et = all_metrics_dict.get("per_exp_type", {})
    if not per_et:
        return

    exp_types = sorted(per_et.keys())
    sample = next(iter(per_et.values()), {})
    available = [m for m in ["rouge1_f", "rouge2_f", "rougeL_f", "bleu_corpus", "bleurt", "bertscore_f1"]
                 if m in sample.get("fwd_vs_baseline", {})]
    if not available:
        available = [m for m in METRIC_DISPLAY if m in sample.get("fwd_vs_baseline", {})]
    if not available:
        return

    x = np.arange(len(exp_types))
    width = 0.35
    ncols = min(3, len(available))
    nrows = math.ceil(len(available) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.5),
                             squeeze=False)

    for idx, metric in enumerate(available):
        ax = axes[idx // ncols][idx % ncols]
        fwd_vals = [per_et[et].get("fwd_vs_baseline", {}).get(metric, np.nan) for et in exp_types]
        rev_vals = [per_et[et].get("rev_vs_baseline", {}).get(metric, np.nan) for et in exp_types]

        ax.bar(x - width / 2, fwd_vals, width, label="Forward",
               color=PALETTE["forward_steered"], alpha=0.85)
        ax.bar(x + width / 2, rev_vals, width, label="Reverse",
               color=PALETTE["reverse_steered"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([e.replace("_", "\n") for e in exp_types], fontsize=8)
        ax.set_title(METRIC_DISPLAY.get(metric, metric), fontsize=9)
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for j in range(len(available), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"{model_name} — NLP Metrics by Experiment Type", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir / f"metrics_by_exp_type.{plot_format}", dpi)


def plot_cross_model_metrics(
    summaries: List[Dict],
    output_dir: Path,
    plot_format: str = "png",
    dpi: int = 150,
) -> None:
    if not summaries:
        return

    models = [s.get("model_name", f"model_{i}") for i, s in enumerate(summaries)]
    caps = ALL_CAPABILITIES
    metric = "rouge1_f"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    for ci, cap in enumerate(caps):
        ax = axes[ci // 3][ci % 3]
        fwd_vals, rev_vals = [], []
        for s in summaries:
            cap_metrics = s.get("nlp_metrics", {}).get("per_capability", {}).get(cap, {})
            fwd_vals.append(cap_metrics.get("fwd_vs_baseline", {}).get(metric, np.nan))
            rev_vals.append(cap_metrics.get("rev_vs_baseline", {}).get(metric, np.nan))
        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w / 2, fwd_vals, w, label="Forward",
               color=PALETTE["forward_steered"], alpha=0.85)
        ax.bar(x + w / 2, rev_vals, w, label="Reverse",
               color=PALETTE["reverse_steered"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([m[:10] for m in models], fontsize=8, rotation=15)
        ax.set_title(cap.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel(METRIC_DISPLAY.get(metric, metric))
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.axhline(0, color="black", linewidth=0.6)

    fig.suptitle(f"Cross-Model: {METRIC_DISPLAY.get(metric, metric)} — Forward vs Reverse",
                 fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir / f"cross_model_{metric}.{plot_format}", dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Text summary / report
# ─────────────────────────────────────────────────────────────────────────────

def build_text_report(
    model_name: str,
    all_metrics_dict: Dict,
    n_rows: int,
) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(f"NLP METRICS REPORT — {model_name}")
    lines.append("=" * 70)
    lines.append(f"Total samples analysed: {n_rows}")
    lines.append("")

    overall = all_metrics_dict.get("overall", {})
    lines.append("── OVERALL ──────────────────────────────────────────────────────")
    for comp_key, comp_label in [
        ("fwd_vs_baseline", "Forward vs Baseline"),
        ("rev_vs_baseline", "Reverse vs Baseline"),
        ("fwd_vs_rev",      "Forward vs Reverse (direct)"),
    ]:
        lines.append(f"\n  {comp_label}:")
        comp = overall.get(comp_key, {})
        if not comp:
            lines.append("    (no data)")
        for m, v in comp.items():
            lines.append(f"    {METRIC_DISPLAY.get(m, m):<30s}: {v:.4f}")

    overd = winner_per_metric(overall)
    lines.append(f"\n  ► Overall winner: {overall_verdict(overd)}")
    lines.append("")

    per_cap = all_metrics_dict.get("per_capability", {})
    if per_cap:
        lines.append("── PER CAPABILITY ───────────────────────────────────────────────")
        for cap in sorted(per_cap.keys()):
            lines.append(f"\n  Capability: {cap.upper()}")
            cap_m = per_cap[cap]
            verdicts = winner_per_metric(cap_m)
            for m, winner in sorted(verdicts.items()):
                fv = cap_m.get("fwd_vs_baseline", {}).get(m, float("nan"))
                rv = cap_m.get("rev_vs_baseline", {}).get(m, float("nan"))
                tag = "▶ FWD" if winner == "forward" else ("▶ REV" if winner == "reverse" else "  TIE")
                lines.append(
                    f"    {METRIC_DISPLAY.get(m, m):<30s}  "
                    f"fwd={fv:.3f}  rev={rv:.3f}  {tag}"
                )
            lines.append(f"    → Capability winner: {overall_verdict(verdicts)}")

    per_et = all_metrics_dict.get("per_exp_type", {})
    if per_et:
        lines.append("\n── PER EXPERIMENT TYPE ──────────────────────────────────────────")
        for et in sorted(per_et.keys()):
            lines.append(f"\n  Exp type: {et}")
            et_m = per_et[et]
            verdicts = winner_per_metric(et_m)
            lines.append(f"    → Winner: {overall_verdict(verdicts)}")
            for m, winner in sorted(verdicts.items()):
                fv = et_m.get("fwd_vs_baseline", {}).get(m, float("nan"))
                rv = et_m.get("rev_vs_baseline", {}).get(m, float("nan"))
                tag = "▶ FWD" if winner == "forward" else ("▶ REV" if winner == "reverse" else "  TIE")
                lines.append(
                    f"    {METRIC_DISPLAY.get(m, m):<30s}  "
                    f"fwd={fv:.3f}  rev={rv:.3f}  {tag}"
                )

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main per-model pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_per_model(args) -> Dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"NLP Metric Analysis — {args.model_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*65}\n")

    forward_csvs = list(args.forward_csvs) if args.forward_csvs else []
    reverse_csvs = list(args.reverse_csvs) if args.reverse_csvs else []

    if not forward_csvs and not reverse_csvs:
        if hasattr(args, "base_storage_dir") and args.base_storage_dir:
            base = Path(args.base_storage_dir) / args.model_name
            forward_csvs, reverse_csvs = discover_csvs(str(base))
        else:
            print("  Warning: no CSVs provided — nothing to analyse.")
            return {}

    print(f"Forward CSVs: {len(forward_csvs)}")
    for p in forward_csvs:
        print(f"  {p}")
    print(f"Reverse CSVs: {len(reverse_csvs)}")
    for p in reverse_csvs:
        print(f"  {p}")

    fwd_dfs = [df for p in forward_csvs if (df := load_csv_with_meta(p)) is not None]
    rev_dfs = [df for p in reverse_csvs if (df := load_csv_with_meta(p)) is not None]

    if not fwd_dfs and not rev_dfs:
        print("  Error: all CSVs failed to load.")
        return {}

    print("\nMerging DataFrames...")
    merged = merge_fwd_rev(fwd_dfs, rev_dfs)
    print(f"  Merged shape: {merged.shape}")

    merged_csv = output_dir / "merged_for_nlp.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"  Merged CSV → {merged_csv}")

    print("\nComputing NLP metrics (this may take a while)...")
    metrics_dict = compute_all_metrics(merged, batch_size=args.bertscore_batch_size)

    print("\nGenerating plots...")
    fmt = args.plot_format
    dpi = args.dpi

    plot_metrics_by_capability(metrics_dict, args.model_name, output_dir, fmt, dpi)
    plot_metric_heatmap(metrics_dict, args.model_name, output_dir, fmt, dpi)
    plot_fwd_vs_rev_direct(metrics_dict, args.model_name, output_dir, fmt, dpi)
    plot_verdict_summary(metrics_dict, args.model_name, output_dir, fmt, dpi)
    plot_ttr_comparison(metrics_dict, args.model_name, output_dir, fmt, dpi)
    plot_exp_type_metrics(metrics_dict, args.model_name, output_dir, fmt, dpi)

    report_text = build_text_report(args.model_name, metrics_dict, len(merged))
    report_path = output_dir / "nlp_metrics_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n  Report → {report_path}")
    print("\n" + report_text)

    summary = {
        "model_name":   args.model_name,
        "forward_csvs": forward_csvs,
        "reverse_csvs": reverse_csvs,
        "n_rows":       int(len(merged)),
        "nlp_metrics":  metrics_dict,
        "available_metrics": {
            "rouge": HAS_ROUGE,
            "bleu":  HAS_BLEU,
            "bleurt": HAS_BLEURT,
            "bertscore": HAS_BERTSCORE,
        },
    }
    summary_path = output_dir / "nlp_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON → {summary_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Cross-model pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_model(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for p in args.summary_jsons:
        try:
            with open(p) as f:
                summaries.append(json.load(f))
            print(f"  Loaded: {p}")
        except Exception as e:
            print(f"  Warning: {p}: {e}")

    if not summaries:
        print("  No summaries loaded.")
        return

    plot_cross_model_metrics(summaries, output_dir, args.plot_format, args.dpi)

    with open(output_dir / "cross_model_nlp_summary.json", "w") as f:
        json.dump({"summaries": summaries}, f, indent=2)
    print(f"\n  Cross-model summary → {output_dir / 'cross_model_nlp_summary.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NLP metric comparison: forward vs reverse activation steering.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mode", default="per_model",
                        choices=["per_model", "cross_model"])
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--forward_csvs", nargs="*", default=None)
    parser.add_argument("--reverse_csvs", nargs="*", default=None)
    parser.add_argument("--base_storage_dir", default=None)
    parser.add_argument("--output_dir", default="./nlp_metrics_analysis")
    parser.add_argument("--summary_jsons", nargs="*", default=None)
    parser.add_argument("--plot_format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--capabilities", nargs="*", default=ALL_CAPABILITIES)
    parser.add_argument("--unified_modes", nargs="*", default=UNIFIED_MODES)
    parser.add_argument(
        "--bertscore_batch_size", type=int, default=32,
        help=(
            "Batch size per GPU for BERTScore. Input is split evenly across all "
            "available GPUs; each GPU processes this many sentences per step. "
            "Reduce (e.g. --bertscore_batch_size 8) if you hit OOM. Default: 32."
        ),
    )
    parser.add_argument(
        "--no_network", action="store_true",
        help="Disable all HuggingFace-dependent metrics (BLEURT, BERTScore) "
             "without attempting a network probe. Use when HF is blocked.",
    )

    args = parser.parse_args()

    if args.no_network:
        import sys
        current_module = sys.modules[__name__]
        current_module._HF_ONLINE = False
        current_module.HAS_BLEURT = False
        current_module.HAS_BERTSCORE = False
        print("  [info] --no_network set: BLEURT and BERTScore disabled.")

    if args.mode == "cross_model":
        if not args.summary_jsons:
            parser.error("--summary_jsons required for cross_model mode")
        run_cross_model(args)
    else:
        run_per_model(args)


if __name__ == "__main__":
    main()