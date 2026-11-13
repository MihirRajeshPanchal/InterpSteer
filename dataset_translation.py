import pandas as pd
from deep_translator import GoogleTranslator
import time
import os
from tqdm import tqdm

df = pd.read_csv("/Data/deeksha/reasoning_probe/data/combined_dataset_with_responses.csv")
print(f"Shape: {df.shape}")

LANGUAGE_DICT = {
    "bengali": "bn",
    "german":  "de",
    "french":  "fr",
    "hindi":   "hi",
    "telugu":  "te",
    "urdu":    "ur",
}

TRANSLATE_COL_MAP = {
    "system_prompt":    "translated_system_prompt",
    "input":            "translated_input",
    "reference_answer": "translated_reference_answer",
    "orig_response":    "translated_orig_response",
}

OUTPUT_PATH = "/Data/deeksha/reasoning_probe/data/biggen_bench_translated.csv"

print(f"Languages: {list(LANGUAGE_DICT.keys())}")
print(f"Column mapping: {TRANSLATE_COL_MAP}")


def translate_text(text, target_lang_code, delay=0.3):
    if not isinstance(text, str) or text.strip() == "":
        return text
    try:
        result = GoogleTranslator(source="auto", target=target_lang_code).translate(text)
        time.sleep(delay)
        return result
    except Exception as e:
        print(f"  ⚠️ Error ({target_lang_code}): {e}")
        return text


def get_already_done(output_path):
    """Returns a set of (original_index, language) tuples already saved."""
    if not os.path.exists(output_path):
        return set()
    try:
        existing = pd.read_csv(output_path)
        if "orig_index" in existing.columns and "language" in existing.columns:
            return set(zip(existing["orig_index"], existing["language"]))
    except Exception:
        pass
    return set()


def translate_dataframe(df, language_dict, col_map, output_path, num_rows=None):
    df_subset = df.iloc[:num_rows].copy() if num_rows else df.copy()
    df_subset["orig_index"] = df_subset.index  # track original row index
    total = len(df_subset)

    already_done = get_already_done(output_path)
    print(f"Processing {total} rows × {len(language_dict)} languages...")
    print(f"Already saved: {len(already_done)} row-language pairs. Skipping those.\n")

    write_header = not os.path.exists(output_path)

    for lang_name, lang_code in tqdm(language_dict.items(), desc="Languages", unit="lang"):
        rows_to_process = [
            row for _, row in df_subset.iterrows()
            if (row["orig_index"], lang_name) not in already_done
        ]

        if not rows_to_process:
            tqdm.write(f"  ✅ [{lang_name}] already fully translated, skipping.")
            continue

        tqdm.write(f"\n  🌐 [{lang_name}] Translating {len(rows_to_process)} rows...")

        for row in tqdm(rows_to_process, desc=lang_name, unit="row"):
            translated_row = row.to_dict()
            translated_row["language"] = lang_name

            for src_col, tgt_col in col_map.items():
                translated_row[tgt_col] = translate_text(row[src_col], lang_code)

            # Append this single translated row immediately to CSV
            row_df = pd.DataFrame([translated_row])
            row_df.to_csv(
                output_path,
                mode="a",
                index=False,
                header=write_header,
                encoding="utf-8-sig",
            )
            write_header = False  # only write header once

    print(f"\n✅ Done. Saved to {output_path}")

    final_df = pd.read_csv(output_path)
    print(f"Rows: {len(final_df)} | Languages: {final_df['language'].nunique()}")
    print(final_df["language"].value_counts())
    return final_df


df_translated = translate_dataframe(
    df,
    language_dict=LANGUAGE_DICT,
    col_map=TRANSLATE_COL_MAP,
    output_path=OUTPUT_PATH,
    num_rows=None,
)