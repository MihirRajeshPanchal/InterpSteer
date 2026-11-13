import pandas as pd

def analyze_file(filename):
    print(f"\n=== {filename} ===")
    df = pd.read_csv(filename)
    print("Total rows:", len(df))
    if 'capability' in df.columns:
        print("\nCapability distribution:")
        print(df['capability'].value_counts())
    if 'orig_score' in df.columns:
        print("\nScore distribution:")
        print(df['orig_score'].value_counts())
    if 'language' in df.columns:
        print("\nLanguage distribution:")
        print(df['language'].value_counts())

analyze_file('/Data/deeksha/reasoning_probe/data/combined_dataset.csv')
analyze_file('/Data/deeksha/reasoning_probe/data/combined_dataset_with_responses.csv')
analyze_file('/Data/deeksha/reasoning_probe/data/biggen_bench_translated.csv')
