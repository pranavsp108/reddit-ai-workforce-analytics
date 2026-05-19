import pandas as pd
from pathlib import Path


INPUT_PATH = "data/processed/reddit_comments_combined.csv"
OUTPUT_PATH = "data/processed/reddit_comments_model_ready.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Convert timestamps safely
    for col in ["post_created_utc", "created_utc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove empty/deleted comments if needed
    if "comment_body" in df.columns:
        df["comment_body"] = df["comment_body"].astype(str)
        df = df[
            ~df["comment_body"].str.lower().isin(["[deleted]", "[removed]", "nan"])
        ].copy()

    # Add readable datetime columns
    if "post_created_utc" in df.columns:
        df["post_created_datetime"] = pd.to_datetime(
            df["post_created_utc"], unit="s", errors="coerce"
        )

    if "created_utc" in df.columns:
        df["comment_created_datetime"] = pd.to_datetime(
            df["created_utc"], unit="s", errors="coerce"
        )

    # Add comment timing feature
    if {"created_utc", "post_created_utc"}.issubset(df.columns):
        df["comment_minutes_after_post"] = (
            (df["created_utc"] - df["post_created_utc"]) / 60
        )

    # Add comment order within each post
    if {"post_id", "created_utc"}.issubset(df.columns):
        df = df.sort_values(
            by=["post_id", "created_utc"],
            ascending=[True, True]
        ).copy()

        df["comment_order_within_post"] = (
            df.groupby("post_id").cumcount() + 1
        )

    # Add actual scraped comment count per post
    if "post_id" in df.columns:
        post_comment_counts = (
            df.groupby("post_id")
            .size()
            .reset_index(name="scraped_comment_count")
        )

        df = df.merge(post_comment_counts, on="post_id", how="left")

    # Add high engagement target
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

        threshold = df["score"].quantile(0.90)
        df["high_engagement"] = (df["score"] >= threshold).astype(int)

    # Final recommended sort
    sort_cols = [
        col for col in [
            "subreddit",
            "post_created_utc",
            "post_id",
            "created_utc"
        ]
        if col in df.columns
    ]

    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved model-ready dataset to: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    if "post_id" in df.columns:
        print(f"Unique posts: {df['post_id'].nunique():,}")

    if "subreddit" in df.columns:
        print(f"Unique subreddits: {df['subreddit'].nunique():,}")


if __name__ == "__main__":
    main()