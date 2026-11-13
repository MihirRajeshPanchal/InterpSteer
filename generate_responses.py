"""
generate_responses.py
─────────────────────
Calls Ollama (gpt-oss:120b) to generate orig_response
for every row in combined_dataset.csv and writes results to --output.

Output schema: all input columns + orig_score + orig_response

Self-resuming: if the output CSV already exists, rows that already have
both orig_score and orig_response filled in are skipped. Just re-run the
same command to continue from where it left off.

No scratch/progress file — the output CSV itself is the progress tracker.
Every completed row is flushed to disk immediately (crash-safe).

Few-shot examples are loaded from --fewshot CSV, filtered by matching
(capability, task). Up to 5 examples are used; falls back to 4→3→2→1.
The reference_answer column (score=5 gold standard) is also shown in the prompt.
"""

import argparse
import os
import time

import pandas as pd
from ollama import Client
from tqdm.auto import tqdm

# ── Ollama config ──────────────────────────────────────────────────────────────
OLLAMA_MODEL = "gpt-oss:120b"

client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer " + "35da4d4604024348b407175708f93110.V8VZfkqm-F6e26N9VC84FN1_"}
)

SCORES      = [1, 2, 3, 4]
MAX_FEWSHOT = 5   # try 5, fall back to 4 → 3 → 2 → 1

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a language model being evaluated on a benchmark. "
    "You will be given a system prompt, a user input, a scoring rubric, a target score (1-4), "
    "few-shot examples showing responses at various quality levels, and a gold reference answer at score 5. "
    "Write a response that matches exactly that quality level according to the rubric. "
    "Study the few-shot examples carefully to understand the expected style and quality for each score. "
    "The gold reference answer represents the best possible response (score 5) — "
    "use it to calibrate what an ideal answer looks like. "
    "Do NOT mention the score, the rubric, or this instruction in your response. "
    "Write naturally as if you are answering the user directly."
)


# ── Few-shot formatting ────────────────────────────────────────────────────────
def _format_fewshot_block(examples: pd.DataFrame) -> str:
    """Format a DataFrame of few-shot rows into a readable prompt block."""
    if examples is None or len(examples) == 0:
        return ""

    lines = ["=== Few-Shot Examples (same capability & task) ==="]
    for i, (_, ex) in enumerate(examples.iterrows(), start=1):
        score = ex.get("orig_score", "?")
        instr = str(ex.get("orig_instruction", "")).strip()
        resp  = str(ex.get("orig_response",    "")).strip()
        lines.append(
            f"\n--- Example {i}  [Score {score}/4] ---\n"
            f"User Input:\n{instr}\n\n"
            f"Response:\n{resp}"
        )
    return "\n".join(lines)


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_user_prompt(row: dict, target_score: int, fewshot_block: str) -> str:
    reference   = str(row.get("reference_answer", "")).strip()
    ref_section = (
        f"=== Gold Reference Answer (Score 5 — ideal/perfect response) ===\n"
        f"{reference}\n\n"
    ) if reference else ""

    fewshot_section = f"{fewshot_block}\n\n" if fewshot_block else ""

    return (
        f"=== System Prompt ===\n{row['system_prompt']}\n\n"
        f"=== User Input ===\n{row['input']}\n\n"
        f"=== Score Rubric ===\n{row['score_rubric']}\n\n"
        f"{ref_section}"
        f"{fewshot_section}"
        f"=== Target Score ===\n{target_score}  "
        f"(write a response that matches this quality level)\n\n"
        f"Your response:"
    )


# ── Balanced score assignment ──────────────────────────────────────────────────
def assign_scores_balanced(n: int) -> list[int]:
    return [SCORES[i % len(SCORES)] for i in range(n)]


# ── Few-shot index ─────────────────────────────────────────────────────────────
def build_fewshot_index(df_fewshot: pd.DataFrame) -> dict:
    index    = {}
    required = {"capability", "task", "orig_instruction", "orig_response", "orig_score"}
    missing  = required - set(df_fewshot.columns)
    if missing:
        print(f"  ⚠  Few-shot CSV is missing columns: {missing}. Few-shot disabled.")
        return index

    for (cap, task), grp in df_fewshot.groupby(["capability", "task"], sort=False):
        index[(str(cap), str(task))] = grp.reset_index(drop=True)

    return index


def get_fewshot_examples(
    index:      dict,
    capability: str,
    task:       str,
    max_k:      int = MAX_FEWSHOT,
) -> pd.DataFrame | None:
    grp = index.get((str(capability), str(task)))
    if grp is None or grp.empty:
        return None

    for k in range(max_k, 0, -1):
        if len(grp) >= k:
            return grp.sample(n=k, random_state=42) if len(grp) > k else grp.copy()

    return None


# ── Ollama call ────────────────────────────────────────────────────────────────
def call_ollama(user_prompt: str, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    try:
        full_response = ""
        for part in client.chat(
            OLLAMA_MODEL,
            messages=messages,
            stream=True,
            options={"temperature": temperature},
        ):
            full_response += part["message"]["content"]
        return full_response.strip()
    except Exception as e:
        print(f"  ❌ Ollama call failed: {e}")
        return ""


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="reasoning_probe/data/combined_dataset.csv",
        help="Input CSV (never modified)",
    )
    parser.add_argument(
        "--output",
        default="reasoning_probe/data/combined_dataset_with_responses.csv",
        help="Output CSV — input columns + orig_score + orig_response. Acts as resume state.",
    )
    parser.add_argument(
        "--fewshot",
        default="/Data/deeksha/probing/combined_cleaned_xx.csv",
        help=(
            "Few-shot examples CSV. "
            "Required columns: capability, task, system_prompt, "
            "orig_instruction, orig_response, orig_score"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── Load input ─────────────────────────────────────────────────────────────
    df_input = pd.read_csv(args.input).reset_index(drop=True)

    for col in ["orig_score", "orig_response"]:
        if col in df_input.columns:
            df_input = df_input.drop(columns=[col])

    print(f"\n📄 Input   : {args.input}  ({len(df_input)} rows)")
    print(f"📄 Output  : {args.output}")
    print(f"🤖 Model   : {OLLAMA_MODEL} via Ollama")

    has_reference = "reference_answer" in df_input.columns
    if has_reference:
        non_empty = df_input["reference_answer"].notna().sum()
        print(f"📄 Gold ref: 'reference_answer' column found ({non_empty} non-null values)")
    else:
        print("  ⚠  'reference_answer' column NOT found in input — gold reference skipped.")

    # ── Load few-shot dataset ──────────────────────────────────────────────────
    fewshot_index: dict = {}
    if args.fewshot and os.path.exists(args.fewshot):
        try:
            df_fewshot    = pd.read_csv(args.fewshot)
            fewshot_index = build_fewshot_index(df_fewshot)
            print(
                f"📄 Few-shot: {args.fewshot}  "
                f"({len(df_fewshot)} rows, "
                f"{len(fewshot_index)} unique capability×task groups)"
            )
        except Exception as e:
            print(f"  ⚠  Could not load few-shot CSV: {e}. Continuing without few-shots.")
    else:
        print(f"  ⚠  Few-shot file not found: {args.fewshot}. Continuing without few-shots.")

    # ── Assign balanced scores per (capability, task) group ───────────────────
    score_map: dict[int, int] = {}
    for (_, __), group in df_input.groupby(["capability", "task"], sort=False):
        for idx, score in zip(group.index, assign_scores_balanced(len(group))):
            score_map[idx] = score

    # ── Resume: load output CSV if it exists, else start fresh ────────────────
    if os.path.exists(args.output):
        df_out = pd.read_csv(args.output)
        if len(df_out) != len(df_input):
            print(
                f"  ⚠  Output row count ({len(df_out)}) != input ({len(df_input)}). "
                "Starting fresh."
            )
            df_out = df_input.copy()
            df_out["orig_score"]    = None
            df_out["orig_response"] = None
        else:
            print("  ♻️  Resuming from existing output file.")
    else:
        df_out = df_input.copy()
        df_out["orig_score"]    = None
        df_out["orig_response"] = None

    # ── Determine which rows still need processing ─────────────────────────────
    done_mask    = df_out["orig_score"].notna() & df_out["orig_response"].notna()
    todo_indices = df_out.index[~done_mask].tolist()

    print(f"\n  ✅ Already done : {done_mask.sum()} / {len(df_out)} rows")
    print(f"  ⏳ Remaining    : {len(todo_indices)} rows\n")

    if not todo_indices:
        print("Nothing to do — all rows already completed.")
        return

    # ── Generate and flush to output CSV after every row ──────────────────────
    for row_idx in tqdm(todo_indices, desc="Generating", unit="row"):
        row   = df_input.loc[row_idx].to_dict()
        score = score_map[row_idx]

        cap  = str(row.get("capability", ""))
        task = str(row.get("task",       ""))

        examples      = get_fewshot_examples(fewshot_index, cap, task)
        fewshot_block = _format_fewshot_block(examples) if examples is not None else ""

        if examples is not None:
            tqdm.write(
                f"  📚 Row {row_idx}: {len(examples)} few-shot example(s) "
                f"for capability='{cap}' / task='{task}'"
            )
        else:
            tqdm.write(
                f"  ℹ️  Row {row_idx}: no few-shot examples for "
                f"capability='{cap}' / task='{task}'"
            )

        prompt   = build_user_prompt(row, score, fewshot_block)
        response = call_ollama(prompt, args.temperature)

        df_out.at[row_idx, "orig_score"]    = score
        df_out.at[row_idx, "orig_response"] = response

        df_out.to_csv(args.output, index=False)

    print(f"\n🎉 Done — {len(df_out)} rows written to {args.output}")
    print("\n=== Score distribution ===")
    print(df_out["orig_score"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()