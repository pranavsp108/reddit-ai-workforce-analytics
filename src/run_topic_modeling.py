from pathlib import Path

import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from umap import UMAP
from hdbscan import HDBSCAN


INPUT_PATH = "data/processed/reddit_comments_nlp_features.csv"
OUTPUT_DIR = Path("outputs/topic_modeling")
MODEL_PATH = Path("models/bertopic_model.pkl")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    df["comment_body"] = df["comment_body"].fillna("").astype(str)

    # Keep comments with enough text for topic modeling
    df_topic = df[df["comment_body"].str.split().str.len() >= 8].copy()
    df_topic = df_topic.reset_index(drop=True)

    docs = df_topic["comment_body"].tolist()

    print(f"Rows available: {len(df):,}")
    print(f"Rows used for topic modeling: {len(df_topic):,}")

    # Free local embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=40,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=10,
        ngram_range=(1, 2)
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs)

    df_topic["topic_id"] = topics

    # Save comment-level topic assignments
    comment_topics_path = OUTPUT_DIR / "comment_topics.csv"
    df_topic.to_csv(comment_topics_path, index=False, encoding="utf-8-sig")

    # Save topic info
    topic_info = topic_model.get_topic_info()
    topic_info_path = OUTPUT_DIR / "topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False, encoding="utf-8-sig")

    # Build topic-level summary with your existing labels
    topic_summary = (
        df_topic.groupby("topic_id")
        .agg(
            comment_count=("comment_id", "count"),
            avg_score=("score", "mean"),
            median_score=("score", "median"),
            avg_severity=("impact_severity", "mean"),
            top_subreddit=("subreddit", lambda x: x.value_counts().index[0]),
            top_category=("primary_disruption_category", lambda x: x.value_counts().index[0]),
            top_occupation=("occupation_bucket", lambda x: x.value_counts().index[0]),
        )
        .reset_index()
        .sort_values("comment_count", ascending=False)
    )

    topic_summary_path = OUTPUT_DIR / "topic_summary.csv"
    topic_summary.to_csv(topic_summary_path, index=False, encoding="utf-8-sig")

    # Save representative docs
    rep_docs = []

    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue

        try:
            representatives = topic_model.get_representative_docs(topic_id)
            for i, doc in enumerate(representatives[:5]):
                rep_docs.append(
                    {
                        "topic_id": topic_id,
                        "representative_rank": i + 1,
                        "comment_body": doc,
                    }
                )
        except Exception:
            continue

    rep_docs_df = pd.DataFrame(rep_docs)
    rep_docs_path = OUTPUT_DIR / "representative_comments.csv"
    rep_docs_df.to_csv(rep_docs_path, index=False, encoding="utf-8-sig")

    # Save model
    topic_model.save(str(MODEL_PATH), serialization="pickle")

    print(f"\nSaved comment topics: {comment_topics_path}")
    print(f"Saved topic info: {topic_info_path}")
    print(f"Saved topic summary: {topic_summary_path}")
    print(f"Saved representative comments: {rep_docs_path}")
    print(f"Saved model: {MODEL_PATH}")

    print("\nTop topics:")
    print(topic_info.head(20))


if __name__ == "__main__":
    main()