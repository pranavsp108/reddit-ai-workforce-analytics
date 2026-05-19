from pathlib import Path

import numpy as np
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


INPUT_PATH = "data/processed/reddit_comments_nlp_features.csv"

OUTPUT_DIR = Path("outputs/topic_modeling_kmeans")
MODEL_PATH = Path("models/bertopic_kmeans_model.pkl")
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["comment_body"] = df["comment_body"].fillna("").astype(str)

    # Filter very short comments
    df_topic = df[df["comment_body"].str.split().str.len() >= 8].copy()
    df_topic = df_topic.reset_index(drop=True)

    docs = df_topic["comment_body"].tolist()

    print(f"Rows available: {len(df):,}")
    print(f"Rows used for topic modeling: {len(df_topic):,}")

    # Embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Cache embeddings so future tuning is faster
    if EMBEDDINGS_PATH.exists():
        print(f"Loading cached embeddings from {EMBEDDINGS_PATH}")
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        print("Creating embeddings...")
        embeddings = embedding_model.encode(
            docs,
            batch_size=32,
            show_progress_bar=True
        )
        np.save(EMBEDDINGS_PATH, embeddings)
        print(f"Saved embeddings to {EMBEDDINGS_PATH}")

    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    # KMeans assigns every comment to a topic
    cluster_model = MiniBatchKMeans(
        n_clusters=45,
        random_state=42,
        batch_size=2048,
        n_init="auto"
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=10,
        ngram_range=(1, 2)
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True
    )

    topics, _ = topic_model.fit_transform(docs, embeddings)

    df_topic["topic_id"] = topics

    # Save comment-level topic assignments
    comment_topics_path = OUTPUT_DIR / "comment_topics.csv"
    df_topic.to_csv(comment_topics_path, index=False, encoding="utf-8-sig")

    # Save topic info
    topic_info = topic_model.get_topic_info()
    topic_info_path = OUTPUT_DIR / "topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False, encoding="utf-8-sig")

    # Topic-level summary
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

    # Representative comments
    rep_docs = []

    for topic_id in sorted(set(topics)):
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
    print(topic_info.head(30))

    print("\nTopic count distribution:")
    print(df_topic["topic_id"].value_counts().head(20))


if __name__ == "__main__":
    main()