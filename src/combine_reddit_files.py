from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw/reddit_posts")
OUTPUT_PATH = Path("data/processed/reddit_comments_combined.csv")


def main():
    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    dfs = []

    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = file.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Remove exact duplicates if the same post was scraped twice
    dedupe_cols = [col for col in ["post_id", "comment_id"] if col in combined.columns]

    if len(dedupe_cols) == 2:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="first")
        after = len(combined)
        print(f"Removed duplicates: {before - after:,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Files combined: {len(files):,}")
    print(f"Rows saved: {len(combined):,}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()