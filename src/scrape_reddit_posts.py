import os
import re
import time
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import praw
from dotenv import load_dotenv


def clean_tracker_csv(input_path: str) -> pd.DataFrame:
    """
    Reads the Reddit post tracker CSV.
    Handles the current format where the first row contains real headers:
    r/, Time, Title, Description, Votes, Comments, URL.
    """
    df = pd.read_csv(input_path)

    # Your tracker currently has columns like Unnamed: 0, Unnamed: 1, ...
    # and the real header is stored in the first row.
    if "URL" not in df.columns:
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

    # Standardize column names
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    # Expected columns after cleanup:
    # r/, time, title, description, votes, comments, url
    rename_map = {
        "r/": "subreddit",
        "comments": "comments_display",
        "votes": "votes_display",
    }
    df = df.rename(columns=rename_map)

    if "url" not in df.columns:
        raise ValueError("Could not find a URL column in the tracker CSV.")

    df = df.dropna(subset=["url"])
    df["url"] = df["url"].astype(str).str.strip()

    # Keep only Reddit URLs
    df = df[df["url"].str.contains("reddit.com", case=False, na=False)].copy()

    return df


def parse_comment_count(value) -> int:
    """
    Converts comment count strings like:
    525, '1.1K', '3.3k', '1,900'
    into integers.
    """
    if pd.isna(value):
        return 0

    text = str(value).strip().replace(",", "")

    if text == "":
        return 0

    multiplier = 1
    if text.lower().endswith("k"):
        multiplier = 1000
        text = text[:-1]

    try:
        return int(float(text) * multiplier)
    except ValueError:
        return 0


def slugify(text: str, max_len: int = 70) -> str:
    """
    Makes a safe filename from a Reddit title.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if text else "reddit_post"


def get_reddit_client() -> praw.Reddit:
    """
    Loads Reddit API credentials from .env.
    """
    load_dotenv()

    required_vars = [
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
        "REDDIT_USERNAME",
        "REDDIT_PASSWORD",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
    )

    return reddit


def get_vote_counts(submission):
    """
    Approximate upvotes/downvotes from Reddit score and upvote ratio.
    Reddit does not expose exact upvote/downvote counts.
    """
    score = submission.score
    ratio = submission.upvote_ratio

    if ratio == 0.5 or (2 * ratio - 1) == 0:
        return None, None

    try:
        total_votes = score / (2 * ratio - 1)
        upvotes = total_votes * ratio
        downvotes = total_votes - upvotes
        return round(upvotes), round(downvotes)
    except Exception:
        return None, None


def scrape_submission(reddit: praw.Reddit, post_url: str) -> tuple[pd.DataFrame, dict]:
    """
    Scrapes one Reddit post and returns:
    1. comments dataframe
    2. post metadata dictionary
    """
    submission = reddit.submission(url=post_url)

    print(f"\nFetching: {submission.title}")
    print(f"Subreddit: r/{submission.subreddit}")
    print(f"Reported comments: {submission.num_comments}")

    submission.comments.replace_more(limit=None)

    post_upvotes, post_downvotes = get_vote_counts(submission)

    post_author = submission.author.name if submission.author else "[deleted]"
    post_created_readable = dt.datetime.fromtimestamp(
        submission.created_utc
    ).strftime("%Y-%m-%d %H:%M:%S")

    post_metadata = {
        "post_id": submission.id,
        "post_title": submission.title,
        "post_url": post_url,
        "subreddit": str(submission.subreddit),
        "post_author": post_author,
        "post_score": submission.score,
        "post_upvote_ratio": submission.upvote_ratio,
        "post_approx_upvotes": post_upvotes,
        "post_approx_downvotes": post_downvotes,
        "post_num_comments_reported": submission.num_comments,
        "post_created_utc": submission.created_utc,
        "post_created_time_readable": post_created_readable,
        "post_is_self": submission.is_self,
        "post_selftext": submission.selftext,
        "post_permalink": f"https://www.reddit.com{submission.permalink}",
    }

    comment_rows = []

    for comment in submission.comments.list():
        author_name = comment.author.name if comment.author else "[deleted]"

        comment_rows.append({
            # Post-level fields
            "post_id": submission.id,
            "post_title": submission.title,
            "post_url": post_url,
            "subreddit": str(submission.subreddit),
            "post_created_utc": submission.created_utc,
            "post_created_time_readable": post_created_readable,
            "post_score": submission.score,
            "post_upvote_ratio": submission.upvote_ratio,
            "post_num_comments_reported": submission.num_comments,

            # Comment-level fields
            "comment_id": comment.id,
            "comment_body": comment.body,
            "author": author_name,
            "score": comment.score,
            "created_utc": comment.created_utc,
            "created_time_readable": dt.datetime.fromtimestamp(
                comment.created_utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "parent_id": comment.parent_id,
            "link_id": comment.link_id,
            "is_submitter": comment.is_submitter,
            "edited": comment.edited,
            "depth": getattr(comment, "depth", None),
            "controversiality": getattr(comment, "controversiality", None),
        })

    comments_df = pd.DataFrame(comment_rows)

    return comments_df, post_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Reddit comments from a CSV tracker of Reddit post URLs."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to Reddit post tracker CSV."
    )

    parser.add_argument(
        "--output",
        default="data/raw/reddit_posts",
        help="Folder where individual post CSVs will be saved."
    )

    parser.add_argument(
        "--min-comments",
        type=int,
        default=100,
        help="Minimum comment count from tracker required to scrape a post."
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between posts."
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    tracker_df = clean_tracker_csv(args.input)

    if "comments_display" in tracker_df.columns:
        tracker_df["comments_numeric"] = tracker_df["comments_display"].apply(parse_comment_count)
    else:
        tracker_df["comments_numeric"] = 0

    # Filter posts based on comment count
    scrape_df = tracker_df[tracker_df["comments_numeric"] >= args.min_comments].copy()

    print(f"Total tracker rows: {len(tracker_df)}")
    print(f"Rows selected with >= {args.min_comments} comments: {len(scrape_df)}")

    reddit = get_reddit_client()

    all_comments = []
    all_metadata = []
    error_rows = []

    for idx, row in scrape_df.iterrows():
        url = row["url"]

        tracker_title = row.get("title", "reddit_post")
        tracker_subreddit = row.get("subreddit", "unknown")
        tracker_comments = row.get("comments_numeric", 0)

        try:
            comments_df, metadata = scrape_submission(reddit, url)

            title_slug = slugify(metadata["post_title"])
            subreddit_slug = slugify(metadata["subreddit"])
            file_name = f"{metadata['post_id']}_{subreddit_slug}_{title_slug}.csv"
            output_path = output_dir / file_name

            comments_df.to_csv(output_path, index=False, encoding="utf-8-sig")

            metadata["tracker_title"] = tracker_title
            metadata["tracker_subreddit"] = tracker_subreddit
            metadata["tracker_comments_numeric"] = tracker_comments
            metadata["saved_file"] = str(output_path)

            all_metadata.append(metadata)
            all_comments.append(comments_df)

            print(f"Saved {len(comments_df)} comments -> {output_path}")

        except Exception as e:
            print(f"ERROR scraping URL: {url}")
            print(f"Reason: {e}")

            error_rows.append({
                "url": url,
                "tracker_title": tracker_title,
                "tracker_subreddit": tracker_subreddit,
                "tracker_comments_numeric": tracker_comments,
                "error": str(e),
            })

        time.sleep(args.sleep)

    # Save metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = processed_dir / "reddit_posts_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved post metadata -> {metadata_path}")

    # Save combined comments
    if all_comments:
        combined_df = pd.concat(all_comments, ignore_index=True)
        combined_path = processed_dir / "reddit_comments_combined.csv"
        combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"Saved combined comments -> {combined_path}")
        print(f"Total comments scraped: {len(combined_df)}")
    else:
        print("No comments were scraped.")

    # Save scrape errors
    if error_rows:
        errors_df = pd.DataFrame(error_rows)
        error_path = processed_dir / "reddit_scrape_errors.csv"
        errors_df.to_csv(error_path, index=False, encoding="utf-8-sig")
        print(f"Saved scrape errors -> {error_path}")


if __name__ == "__main__":
    main()