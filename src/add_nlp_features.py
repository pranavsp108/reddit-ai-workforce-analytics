import re
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


INPUT_PATH = "data/processed/reddit_comments_clean_features.csv"
OUTPUT_PATH = "data/processed/reddit_comments_nlp_features.csv"


KEYWORD_GROUPS = {
    "mentions_job_loss": [
        "lost my job", "lose my job", "job loss", "laid off", "layoff",
        "fired", "terminated", "unemployed", "redundant"
    ],
    "mentions_replacement": [
        "replaced by ai", "replace my job", "replacing jobs", "take my job",
        "taking my job", "automated away", "automation", "ai replaced"
    ],
    "mentions_hiring": [
        "hiring", "recruiter", "resume", "ats", "interview", "job market",
        "application", "applicants", "ghosted"
    ],
    "mentions_entry_level": [
        "entry level", "new grad", "junior", "intern", "graduate",
        "college", "first job"
    ],
    "mentions_white_collar": [
        "white collar", "office job", "desk job", "knowledge work",
        "corporate", "analyst", "consultant"
    ],
    "mentions_freelance": [
        "freelance", "freelancer", "client", "contract", "contractor",
        "gig", "upwork", "fiverr"
    ],
    "mentions_reskilling": [
        "reskill", "upskill", "learn", "pivot", "switch careers",
        "career change", "adapt"
    ],
    "mentions_ai_skepticism": [
        "not ai", "executives", "outsourcing", "hype", "bubble",
        "overrated", "not replacing", "blaming ai"
    ],
}


OCCUPATION_BUCKETS = {
    "software_tech": [
        "software engineer", "developer", "programmer", "coding", "code",
        "web developer", "frontend", "backend", "tech worker", "it"
    ],
    "data_analytics": [
        "data scientist", "data analyst", "analytics", "machine learning",
        "ml engineer", "business analyst", "bi analyst", "sql"
    ],
    "writing_content": [
        "writer", "copywriter", "content writer", "editor", "journalist",
        "blog", "seo", "technical writer"
    ],
    "design_creative": [
        "designer", "graphic design", "artist", "illustrator", "creative",
        "ux", "ui", "photoshop"
    ],
    "voice_media": [
        "voice actor", "voiceover", "voice over", "audio", "podcast",
        "narrator", "acting"
    ],
    "customer_support": [
        "customer service", "support", "call center", "chat support",
        "help desk"
    ],
    "education": [
        "teacher", "professor", "tutor", "education", "school",
        "student", "teaching"
    ],
    "healthcare": [
        "doctor", "nurse", "healthcare", "medical", "therapist",
        "clinic", "hospital"
    ],
    "legal_finance": [
        "lawyer", "legal", "paralegal", "finance", "accountant",
        "accounting", "banking", "underwriting"
    ],
}


def contains_keyword(text, keywords):
    text = str(text).lower()
    return int(any(keyword in text for keyword in keywords))


def assign_occupation_bucket(text):
    text = str(text).lower()

    for bucket, keywords in OCCUPATION_BUCKETS.items():
        if any(keyword in text for keyword in keywords):
            return bucket

    return "unknown"


def assign_primary_disruption_category(row):
    if row["mentions_job_loss"] == 1:
        return "direct_job_loss"
    if row["mentions_replacement"] == 1:
        return "replacement_automation"
    if row["mentions_hiring"] == 1:
        return "hiring_market_disruption"
    if row["mentions_entry_level"] == 1:
        return "entry_level_pressure"
    if row["mentions_reskilling"] == 1:
        return "adaptation_reskilling"
    if row["mentions_ai_skepticism"] == 1:
        return "skepticism_counterpoint"
    if row["mentions_white_collar"] == 1:
        return "white_collar_disruption"
    if row["mentions_freelance"] == 1:
        return "freelance_client_disruption"

    return "general_ai_workforce_discussion"


def sentiment_label(compound):
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


def main():
    df = pd.read_csv(INPUT_PATH)

    df["comment_body"] = df["comment_body"].astype(str)

    # VADER sentiment
    analyzer = SentimentIntensityAnalyzer()

    sentiment_scores = df["comment_body"].apply(analyzer.polarity_scores)

    df["vader_neg"] = sentiment_scores.apply(lambda x: x["neg"])
    df["vader_neu"] = sentiment_scores.apply(lambda x: x["neu"])
    df["vader_pos"] = sentiment_scores.apply(lambda x: x["pos"])
    df["vader_compound"] = sentiment_scores.apply(lambda x: x["compound"])
    df["vader_sentiment_label"] = df["vader_compound"].apply(sentiment_label)

    # Keyword labels
    for label, keywords in KEYWORD_GROUPS.items():
        df[label] = df["comment_body"].apply(lambda x: contains_keyword(x, keywords))

    # Occupation bucket
    df["occupation_bucket"] = df["comment_body"].apply(assign_occupation_bucket)

    # Primary disruption category
    df["primary_disruption_category"] = df.apply(assign_primary_disruption_category, axis=1)

    # Engagement target
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    engagement_threshold = df["score"].quantile(0.90)

    df["high_engagement"] = (df["score"] >= engagement_threshold).astype(int)
    df["engagement_threshold_90p"] = engagement_threshold

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"High engagement threshold: {engagement_threshold}")

    print("\nSentiment distribution:")
    print(df["vader_sentiment_label"].value_counts())

    print("\nTop disruption categories:")
    print(df["primary_disruption_category"].value_counts().head(10))

    print("\nOccupation buckets:")
    print(df["occupation_bucket"].value_counts().head(10))


if __name__ == "__main__":
    main()