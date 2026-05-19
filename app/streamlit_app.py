from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Page config
# -----------------------------

st.set_page_config(
    page_title="AI Workforce Impact Intelligence",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Paths
# -----------------------------

DATA_PATH = Path("data/processed/dashboard_dataset.csv")
DATA_WITH_MISC_PATH = Path("data/processed/dashboard_dataset_with_misc.csv")
ENGAGEMENT_FULL_IMPORTANCE_PATH = Path(
    "outputs/engagement_model/random_forest_feature_importance.csv"
)
ENGAGEMENT_CONTENT_IMPORTANCE_PATH = Path(
    "outputs/engagement_model/random_forest_feature_importance_content_only.csv"
)
FULL_MODEL_CHART_PATH = Path(
    "outputs/engagement_model/top_feature_importance_full_model.png"
)
CONTENT_MODEL_CHART_PATH = Path(
    "outputs/engagement_model/top_feature_importance_content_only.png"
)


# -----------------------------
# Helper functions
# -----------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(
            "dashboard_dataset.csv not found. Run: python src/create_dashboard_dataset.py"
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)

    # Datetime parsing
    for col in ["comment_created_datetime", "post_created_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric cleanup
    numeric_cols = [
        "score",
        "high_engagement",
        "impact_severity",
        "comment_length_words",
        "comment_length_chars",
        "comment_hours_after_post",
        "comment_order_within_post",
        "depth",
        "vader_compound",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def load_feature_importance(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def format_category_name(value):
    if pd.isna(value):
        return "Unknown"
    return str(value).replace("_", " ").title()


def metric_card(label, value):
    st.metric(label, value)


def horizontal_bar(df, x, y, title, height=450):
    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation="h",
        title=title,
        text=x,
    )
    fig.update_layout(
        height=height,
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# -----------------------------
# Load data
# -----------------------------

df = load_data()

# -----------------------------
# Sidebar filters
# -----------------------------

st.sidebar.title("Filters")

subreddits = sorted(df["subreddit"].dropna().unique().tolist())
selected_subreddits = st.sidebar.multiselect(
    "Subreddit",
    options=subreddits,
    default=subreddits,
)

topic_groups = sorted(df["topic_group"].dropna().unique().tolist())
selected_topic_groups = st.sidebar.multiselect(
    "Topic group",
    options=topic_groups,
    default=topic_groups,
)

categories = sorted(df["primary_disruption_category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Disruption category",
    options=categories,
    default=categories,
    format_func=format_category_name,
)

occupations = sorted(df["occupation_bucket"].dropna().unique().tolist())
selected_occupations = st.sidebar.multiselect(
    "Occupation bucket",
    options=occupations,
    default=occupations,
    format_func=format_category_name,
)

severity_min, severity_max = st.sidebar.slider(
    "Impact severity",
    min_value=int(df["impact_severity"].min()),
    max_value=int(df["impact_severity"].max()),
    value=(int(df["impact_severity"].min()), int(df["impact_severity"].max())),
)

high_engagement_only = st.sidebar.checkbox("Show only high-engagement comments")

filtered = df[
    df["subreddit"].isin(selected_subreddits)
    & df["topic_group"].isin(selected_topic_groups)
    & df["primary_disruption_category"].isin(selected_categories)
    & df["occupation_bucket"].isin(selected_occupations)
    & df["impact_severity"].between(severity_min, severity_max)
].copy()

if high_engagement_only:
    filtered = filtered[filtered["high_engagement"] == 1].copy()


# -----------------------------
# App title
# -----------------------------

st.title("AI Workforce Impact Intelligence Platform")

st.caption(
    "Reddit-scale NLP analytics platform for identifying AI workforce disruption themes, "
    "occupation-level concerns, engagement drivers, and labor-market narratives."
)


# -----------------------------
# Tabs
# -----------------------------

tabs = st.tabs(
    [
        "Executive Overview",
        "Topic Explorer",
        "Disruption Analysis",
        "Occupation Impact",
        "Engagement Drivers",
        "Comment Explorer",
    ]
)


# -----------------------------
# Tab 1: Executive Overview
# -----------------------------

with tabs[0]:
    st.subheader("Executive Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        metric_card("Comments", f"{len(filtered):,}")

    with col2:
        metric_card("Posts", f"{filtered['post_id'].nunique():,}")

    with col3:
        metric_card("Subreddits", f"{filtered['subreddit'].nunique():,}")

    with col4:
        high_rate = filtered["high_engagement"].mean() if len(filtered) else 0
        metric_card("High-engagement rate", f"{high_rate:.1%}")

    with col5:
        avg_severity = filtered["impact_severity"].mean() if len(filtered) else 0
        metric_card("Avg. severity", f"{avg_severity:.2f}")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        subreddit_summary = (
            filtered.groupby("subreddit")
            .agg(comments=("comment_id", "count"))
            .reset_index()
            .sort_values("comments", ascending=False)
        )

        fig = px.bar(
            subreddit_summary,
            x="subreddit",
            y="comments",
            title="Comments by Subreddit",
            text="comments",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        category_summary = (
            filtered.groupby("primary_disruption_category")
            .agg(comments=("comment_id", "count"))
            .reset_index()
            .sort_values("comments", ascending=False)
            .head(12)
        )
        category_summary["category_clean"] = category_summary[
            "primary_disruption_category"
        ].apply(format_category_name)

        fig = horizontal_bar(
            category_summary,
            x="comments",
            y="category_clean",
            title="Top Disruption Categories",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Topic Group Distribution")

    topic_group_summary = (
        filtered.groupby("topic_group")
        .agg(
            comments=("comment_id", "count"),
            avg_severity=("impact_severity", "mean"),
            high_engagement_rate=("high_engagement", "mean"),
        )
        .reset_index()
        .sort_values("comments", ascending=False)
        .head(15)
    )

    fig = px.bar(
        topic_group_summary,
        x="topic_group",
        y="comments",
        color="avg_severity",
        title="Comments by Topic Group",
        text="comments",
    )
    fig.update_layout(height=450, xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Tab 2: Topic Explorer
# -----------------------------

with tabs[1]:
    st.subheader("Topic Explorer")

    topic_summary = (
        filtered.groupby(["topic_id", "topic_label", "topic_group"])
        .agg(
            comments=("comment_id", "count"),
            avg_score=("score", "mean"),
            median_score=("score", "median"),
            high_engagement_rate=("high_engagement", "mean"),
            avg_severity=("impact_severity", "mean"),
        )
        .reset_index()
        .sort_values("comments", ascending=False)
    )

    st.dataframe(
        topic_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    selected_topic = st.selectbox(
        "Select topic",
        options=topic_summary["topic_label"].tolist(),
    )

    topic_df = filtered[filtered["topic_label"] == selected_topic].copy()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Comments", f"{len(topic_df):,}")
    with c2:
        st.metric("Avg. score", f"{topic_df['score'].mean():.2f}")
    with c3:
        st.metric("High-engagement rate", f"{topic_df['high_engagement'].mean():.1%}")
    with c4:
        st.metric("Avg. severity", f"{topic_df['impact_severity'].mean():.2f}")

    st.markdown("### Representative high-engagement comments")

    display_cols = [
        "subreddit",
        "post_title",
        "score",
        "impact_severity",
        "primary_disruption_category",
        "comment_body",
    ]

    st.dataframe(
        topic_df.sort_values("score", ascending=False)[display_cols].head(20),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------
# Tab 3: Disruption Analysis
# -----------------------------

with tabs[2]:
    st.subheader("Disruption Category Analysis")

    disruption_summary = (
        filtered.groupby("primary_disruption_category")
        .agg(
            comments=("comment_id", "count"),
            avg_score=("score", "mean"),
            high_engagement_rate=("high_engagement", "mean"),
            avg_severity=("impact_severity", "mean"),
        )
        .reset_index()
        .sort_values("comments", ascending=False)
    )

    disruption_summary["category_clean"] = disruption_summary[
        "primary_disruption_category"
    ].apply(format_category_name)

    c1, c2 = st.columns(2)

    with c1:
        fig = horizontal_bar(
            disruption_summary.head(15),
            x="comments",
            y="category_clean",
            title="Comment Volume by Disruption Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            disruption_summary.sort_values("high_engagement_rate", ascending=False).head(15),
            x="high_engagement_rate",
            y="category_clean",
            orientation="h",
            title="High-engagement Rate by Disruption Category",
            text="high_engagement_rate",
        )
        fig.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
        fig.update_traces(texttemplate="%{text:.1%}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Category Summary Table")
    st.dataframe(
        disruption_summary[
            [
                "category_clean",
                "comments",
                "avg_score",
                "high_engagement_rate",
                "avg_severity",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------
# Tab 4: Occupation Impact
# -----------------------------

with tabs[3]:
    st.subheader("Occupation Impact Map")

    occupation_summary = (
        filtered.groupby("occupation_bucket")
        .agg(
            comments=("comment_id", "count"),
            avg_score=("score", "mean"),
            high_engagement_rate=("high_engagement", "mean"),
            avg_severity=("impact_severity", "mean"),
        )
        .reset_index()
        .sort_values("comments", ascending=False)
    )

    occupation_summary["occupation_clean"] = occupation_summary[
        "occupation_bucket"
    ].apply(format_category_name)

    c1, c2 = st.columns(2)

    with c1:
        fig = horizontal_bar(
            occupation_summary.head(15),
            x="comments",
            y="occupation_clean",
            title="Comment Volume by Occupation Bucket",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            occupation_summary,
            x="avg_severity",
            y="high_engagement_rate",
            size="comments",
            color="occupation_clean",
            hover_name="occupation_clean",
            title="Occupation Buckets by Severity and Engagement",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Occupation Summary Table")
    st.dataframe(
        occupation_summary[
            [
                "occupation_clean",
                "comments",
                "avg_score",
                "high_engagement_rate",
                "avg_severity",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------
# Tab 5: Engagement Drivers
# -----------------------------

with tabs[4]:
    st.subheader("Engagement Drivers")

    st.markdown(
        """
        The project trains two engagement models:

        - **Full model:** includes timing, thread position, metadata, topics, and NLP labels.
        - **Content-only model:** excludes timing and thread-position features to isolate content-driven signal.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Full Model")
        st.markdown(
            """
            - ROC-AUC: **0.848**
            - PR-AUC: **0.425**
            - Main signal: timing, comment order, depth, and metadata.
            """
        )
        if FULL_MODEL_CHART_PATH.exists():
            st.image(str(FULL_MODEL_CHART_PATH), use_container_width=True)

    with c2:
        st.markdown("### Content-only Model")
        st.markdown(
            """
            - ROC-AUC: **0.635**
            - PR-AUC: **0.175**
            - Main signal: comment length, subreddit, severity, topic group, and disruption category.
            """
        )
        if CONTENT_MODEL_CHART_PATH.exists():
            st.image(str(CONTENT_MODEL_CHART_PATH), use_container_width=True)

    st.divider()

    st.markdown("### Content-based feature importance")

    content_importance = load_feature_importance(CONTENT_MODEL_CHART_PATH.parent / "random_forest_feature_importance_content_only.csv")

    if not content_importance.empty:
        top_features = content_importance.head(25).sort_values("importance")
        fig = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            title="Top Content-Based Drivers of High Engagement",
        )
        fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Content-only feature importance file not found.")


# -----------------------------
# Tab 6: Comment Explorer
# -----------------------------

with tabs[5]:
    st.subheader("Comment Explorer")

    search_term = st.text_input("Search comments", "")

    explorer_df = filtered.copy()

    if search_term:
        explorer_df = explorer_df[
            explorer_df["comment_body"].fillna("").str.contains(
                search_term, case=False, na=False
            )
        ]

    sort_option = st.selectbox(
        "Sort by",
        options=[
            "score",
            "impact_severity",
            "comment_length_words",
            "comment_hours_after_post",
        ],
        index=0,
    )

    ascending = st.checkbox("Ascending sort", value=False)

    explorer_df = explorer_df.sort_values(sort_option, ascending=ascending)

    display_cols = [
        "subreddit",
        "post_title",
        "score",
        "high_engagement",
        "topic_label",
        "topic_group",
        "primary_disruption_category",
        "occupation_bucket",
        "impact_severity",
        "comment_body",
    ]

    display_cols = [col for col in display_cols if col in explorer_df.columns]

    st.dataframe(
        explorer_df[display_cols].head(200),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download filtered comments",
        data=explorer_df[display_cols].to_csv(index=False),
        file_name="filtered_reddit_comments.csv",
        mime="text/csv",
    )