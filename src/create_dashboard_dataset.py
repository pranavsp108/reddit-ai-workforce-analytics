from pathlib import Path
import pandas as pd


INPUT_PATH = "data/processed/reddit_comments_topics_labeled.csv"
OUTPUT_PATH = "data/processed/dashboard_dataset.csv"

OUTPUT_DIR = Path("outputs/dashboard_assets")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

        # Add fallback sentiment columns if missing
    if "vader_compound" not in df.columns:
        df["vader_compound"] = 0.0

    if "vader_sentiment_label" not in df.columns:
        df["vader_sentiment_label"] = "unknown"

    # Ensure score and target
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    threshold = df["score"].quantile(0.90)
    df["high_engagement"] = (df["score"] >= threshold).astype(int)

    # Datetime cleanup
    if "created_utc" in df.columns:
        df["comment_created_datetime"] = pd.to_datetime(
            df["created_utc"], unit="s", errors="coerce"
        )

    if "post_created_utc" in df.columns:
        df["post_created_datetime"] = pd.to_datetime(
            df["post_created_utc"], unit="s", errors="coerce"
        )

    if "comment_created_datetime" in df.columns:
        df["comment_month"] = df["comment_created_datetime"].dt.to_period("M").astype(str)

    if "post_created_datetime" in df.columns:
        df["post_month"] = df["post_created_datetime"].dt.to_period("M").astype(str)

    # Keep dashboard-friendly columns
    keep_cols = [
        "comment_id",
        "post_id",
        "post_title",
        "subreddit",
        "comment_body",
        "score",
        "high_engagement",
        "topic_id",
        "topic_label",
        "topic_group",
        "use_in_dashboard",
        "primary_disruption_category",
        "comment_primary_disruption_category",
        "occupation_bucket",
        "occupation_bucket_source",
        "impact_severity",
        "vader_sentiment_label",
        "vader_compound",
        "comment_length_words",
        "comment_length_chars",
        "comment_hours_after_post",
        "comment_order_within_post",
        "depth",
        "created_time_readable",
        "post_created_time_readable",
        "comment_created_datetime",
        "post_created_datetime",
        "comment_month",
        "post_month",
        "job_displacement_flag",
        "replacement_risk_flag",
        "hiring_disruption_flag",
        "entry_level_pressure_flag",
        "career_anxiety_flag",
        "adaptation_reskilling_flag",
        "safe_jobs_flag",
        "macro_economic_concern_flag",
        "quality_trust_safety_flag",
        "corporate_cost_cutting_flag",
        "ai_hype_skepticism_flag",
        "productivity_augmentation_flag",
        "any_workforce_risk_flag",
        "any_adaptation_or_safe_jobs_flag",
    ]

    keep_cols = [col for col in keep_cols if col in df.columns]

    dashboard_df = df[keep_cols].copy()

    # Filter out topic labels marked not for dashboard if desired
    dashboard_df_with_misc = dashboard_df.copy()
    dashboard_df = dashboard_df[
        dashboard_df["use_in_dashboard"].fillna("yes") == "yes"
    ].copy()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    dashboard_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    dashboard_df_with_misc.to_csv(
        "data/processed/dashboard_dataset_with_misc.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Saved dashboard dataset: {OUTPUT_PATH}")
    print(f"Rows, dashboard only: {len(dashboard_df):,}")
    print(f"Rows, with misc: {len(dashboard_df_with_misc):,}")
    print(f"Columns: {len(dashboard_df.columns):,}")
    print(f"High engagement threshold: {threshold}")

    # Summary tables for dashboard
    topic_summary = (
        dashboard_df.groupby(["topic_id", "topic_label", "topic_group"])
        .agg(
            comments=("comment_id", "count"),
            avg_score=("score", "mean"),
            median_score=("score", "median"),
            high_engagement_rate=("high_engagement", "mean"),
            avg_severity=("impact_severity", "mean"),
            avg_sentiment=("vader_compound", "mean"),
        )
        .reset_index()
        .sort_values("comments", ascending=False)
    )

    category_summary = (
        dashboard_df.groupby("primary_disruption_category")
        .agg(
            comments=("comment_id", "count"),
            avg_score=("score", "mean"),
            high_engagement_rate=("high_engagement", "mean"),
            avg_severity=("impact_severity", "mean"),
        )
        .reset_index()
        .sort_values("comments", ascending=False)
    )

    occupation_summary = (
        dashboard_df.groupby("occupation_bucket")
        .agg(
            comments=("comment_id", "count"),
            avg_score=("score", "mean"),
            high_engagement_rate=("high_engagement", "mean"),
            avg_severity=("impact_severity", "mean"),
        )
        .reset_index()
        .sort_values("comments", ascending=False)
    )

    subreddit_summary = (
        dashboard_df.groupby("subreddit")
        .agg(
            comments=("comment_id", "count"),
            posts=("post_id", "nunique"),
            avg_score=("score", "mean"),
            high_engagement_rate=("high_engagement", "mean"),
            avg_severity=("impact_severity", "mean"),
        )
        .reset_index()
        .sort_values("comments", ascending=False)
    )

    topic_summary.to_csv(OUTPUT_DIR / "dashboard_topic_summary.csv", index=False)
    category_summary.to_csv(OUTPUT_DIR / "dashboard_category_summary.csv", index=False)
    occupation_summary.to_csv(OUTPUT_DIR / "dashboard_occupation_summary.csv", index=False)
    subreddit_summary.to_csv(OUTPUT_DIR / "dashboard_subreddit_summary.csv", index=False)

    print("\nSaved dashboard summary tables to outputs/dashboard_assets/")


if __name__ == "__main__":
    main()