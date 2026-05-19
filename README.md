# AI Workforce Impact Intelligence Platform

An end-to-end NLP and machine learning platform that analyzes Reddit discussions about AI-driven workforce disruption, career anxiety, hiring-market shifts, and occupation-level impact.

The project transforms unstructured Reddit comments into structured labor-market intelligence using Reddit data collection, text preprocessing, weak-label taxonomy design, sentiment analysis, embedding-based topic modeling, engagement prediction, and an interactive Streamlit dashboard.

---

## Project Overview

AI is rapidly changing how people think about work, hiring, career planning, and job security. Public discussions on Reddit contain a large volume of firsthand experiences, concerns, skepticism, and adaptation strategies, but these conversations are difficult to analyze at scale because they are noisy, unstructured, and spread across many communities.

This project builds a full analytics pipeline to answer:

- Which AI workforce disruption themes are most common?
- Which occupations and career paths are discussed most often?
- What types of AI-related job concerns receive the most engagement?
- Are people discussing direct job loss, career anxiety, hiring-market distortion, or adaptation?
- What drives high-engagement comments: content, timing, topic, subreddit, or severity?

---

## Key Results

| Area | Result |
|---|---:|
| Clean Reddit comments analyzed | **20,696** |
| Topic-modeled comments | **18,691** |
| Reddit posts collected | **48** |
| Subreddits covered | **6** |
| Final topic clusters | **45** |
| Full engagement model ROC-AUC | **0.848** |
| Full engagement model PR-AUC | **0.425** |
| Content-only engagement model ROC-AUC | **0.635** |
| Content-only engagement model PR-AUC | **0.175** |

The full engagement model showed that Reddit engagement is heavily shaped by platform mechanics such as comment timing, thread position, and depth. A second content-only model was trained to isolate the contribution of topic, severity, disruption category, occupation bucket, and subreddit-level signals.

---

## Business Problem

Organizations, workers, educators, and policymakers need better ways to understand how AI is perceived in the labor market. Traditional labor datasets often lag behind real-time public sentiment, while social platforms contain faster but messier signals.

This project treats Reddit discussions as a public discourse dataset and converts them into a structured intelligence product for analyzing AI-related workforce concerns.

The final output is a dashboard that helps explore:

- AI job-displacement narratives
- Career anxiety and entry-level pressure
- Occupation-level risk perceptions
- Hiring-market disruption from AI-generated resumes
- Creative-work and software-work disruption
- Safe-job and reskilling discussions
- Engagement drivers behind viral workforce narratives

---

## Data Sources

The dataset was collected from Reddit posts related to AI, jobs, careers, layoffs, automation, and hiring-market disruption.

Example subreddit categories included:

- `r/AskReddit`
- `r/jobs`
- `r/careerguidance`
- `r/recruitinghell`
- `r/datascience`
- `r/ArtificialInteligence`

The repository does not include raw scraped Reddit comments or user identifiers. Public sample and feature-only datasets are provided for reproducibility while reducing redistribution and privacy risk.

---

## Methodology

### 1. Reddit Data Collection

Reddit posts were collected using the Reddit API through PRAW. The scraper extracts:

- post metadata
- subreddit
- post title
- comment text
- score
- timestamp
- parent/comment hierarchy
- thread depth
- submitter flag

Each post is saved separately and then combined into a master dataset.

### 2. Data Cleaning and Feature Engineering

The cleaning pipeline removes deleted/removed comments, duplicate comment IDs, and extremely short text. It also creates metadata and timing features such as:

- comment length
- comment order within post
- comment hours after post
- post month
- comment month
- high-engagement target
- score-based engagement threshold

### 3. Weak-Label Taxonomy Design

A rule-based weak-labeling system classifies comments into workforce disruption themes.

Final disruption categories include:

- direct displacement
- replacement risk
- hiring-market disruption
- entry-level pressure
- career anxiety
- adaptation and reskilling
- safe-jobs discussion
- macroeconomic concern
- corporate cost-cutting
- AI hype skepticism
- productivity augmentation
- quality, trust, and safety concern

To avoid label leakage, comment-level signals were separated from post-title context. Post titles were used only as fallback context when a comment had no clear workforce signal.

### 4. Occupation Bucket Classification

Comments were mapped into broad occupation buckets using regex-based keyword rules.

Occupation buckets include:

- software engineering and IT
- data, AI, and analytics
- creative content and media
- business operations and administration
- customer, sales, and support
- education and research
- healthcare and social services
- legal, finance, and accounting
- trades and manual labor
- manufacturing, agriculture, and logistics
- management and executive roles
- general white-collar work
- unknown

The `unknown` category was intentionally retained because many Reddit comments discuss AI and work at a general level without naming a specific occupation.

### 5. Sentiment and Severity Features

VADER sentiment was used as a baseline polarity signal. Because social-media sentiment can misclassify sarcasm or mixed anxiety, the project also creates workforce-specific flags and an impact severity score.

Severity levels range from:

| Severity | Meaning |
|---:|---|
| 1 | General or low-signal discussion |
| 2 | Mild concern, adaptation, or safety discussion |
| 3 | Workforce risk, replacement concern, or macro concern |
| 4 | Hiring disruption, entry-level pressure, or strong replacement risk |
| 5 | Direct job, client, business, or income loss |

### 6. Topic Modeling

Two topic-modeling approaches were evaluated:

1. **BERTopic with HDBSCAN**
   - Useful for discovering natural clusters
   - Produced a high outlier rate on noisy Reddit comments

2. **SentenceTransformers + MiniBatchKMeans + BERTopic representations**
   - Final approach
   - Assigns every comment to a topic
   - Produces dashboard-friendly topic coverage

The final model uses:

- `sentence-transformers/all-MiniLM-L6-v2`
- UMAP dimensionality reduction
- MiniBatchKMeans clustering
- BERTopic topic representation

The final model produced **45 interpretable topics**.

Example topic groups:

- Hiring Market
- Workforce Disruption
- Creative Work
- Tech Roles
- Career Anxiety
- Economic Impact
- Corporate Strategy
- AI Quality and Trust
- Automation and Robotics
- Human-Centered Work
- Legal and Governance

### 7. Engagement Prediction

The project trains models to predict whether a comment reaches top-decile engagement.

Target:

```text
high_engagement = 1 if comment score >= 90th percentile