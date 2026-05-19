import pandas as pd

df = pd.read_csv("data/processed/reddit_comments_combined.csv")

print("Rows:", len(df))
print("Columns:", df.shape[1])
print("Unique posts:", df["post_id"].nunique())
print("Unique subreddits:", df["subreddit"].nunique())

print("\nColumns:")
print(df.columns.tolist())