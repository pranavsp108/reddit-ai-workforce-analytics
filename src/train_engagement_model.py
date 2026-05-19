from pathlib import Path
import json

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


INPUT_PATH = "data/processed/reddit_comments_topics_labeled.csv"

OUTPUT_DIR = Path("outputs/engagement_model")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # Ensure numeric score
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Define high engagement target using top 10% score threshold
    threshold = df["score"].quantile(0.90)
    df["high_engagement"] = (df["score"] >= threshold).astype(int)

    print(f"High engagement threshold: {threshold}")
    print("\nTarget distribution:")
    print(df["high_engagement"].value_counts(normalize=True))

    # Useful numeric cleanup
    numeric_candidates = [
        "comment_length_words",
        "comment_length_chars",
        "comment_hours_after_post",
        "comment_order_within_post",
        "depth",
        "vader_compound",
        "vader_neg",
        "vader_neu",
        "vader_pos",
        "impact_severity",
    ]

    numeric_features = [col for col in numeric_candidates if col in df.columns]

    categorical_candidates = [
        "subreddit",
        "primary_disruption_category",
        "comment_primary_disruption_category",
        "occupation_bucket",
        "occupation_bucket_source",
        "topic_id",
        "topic_label",
        "topic_group",
        "vader_sentiment_label",
        "label_source",
    ]

    categorical_features = [col for col in categorical_candidates if col in df.columns]

    binary_candidates = [
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
        "is_submitter",
    ]

    binary_features = [col for col in binary_candidates if col in df.columns]

    # Convert booleans / strings to numeric where needed
    for col in binary_features:
        df[col] = df[col].replace({True: 1, False: 0, "True": 1, "False": 0})
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    features = numeric_features + categorical_features + binary_features

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    print("\nBinary features:")
    print(binary_features)

    model_df = df[features + ["high_engagement"]].copy()

    X = model_df[features]
    y = model_df["high_engagement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )

    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", binary_transformer, binary_features),
        ],
        remainder="drop",
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print(f"Training: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion matrix:")
        print(cm)

        results[model_name] = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        model_path = MODEL_DIR / f"{model_name}_engagement_model.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved model: {model_path}")

        # Save predictions
        pred_df = X_test.copy()
        pred_df["actual_high_engagement"] = y_test.values
        pred_df["predicted_high_engagement"] = y_pred
        pred_df["predicted_probability"] = y_proba

        pred_path = OUTPUT_DIR / f"{model_name}_test_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved predictions: {pred_path}")

    # Save metrics
    metrics_path = OUTPUT_DIR / "engagement_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved metrics: {metrics_path}")

    # Save feature list
    feature_config = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "binary_features": binary_features,
        "target": "high_engagement",
        "high_engagement_threshold": float(threshold),
    }

    feature_config_path = OUTPUT_DIR / "feature_config.json"
    with open(feature_config_path, "w") as f:
        json.dump(feature_config, f, indent=2)

    print(f"Saved feature config: {feature_config_path}")


if __name__ == "__main__":
    main()