from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = "models/random_forest_engagement_model_content_only.joblib"
OUTPUT_DIR = Path("outputs/engagement_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_feature_names(preprocessor):
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue

        if name == "num":
            feature_names.extend(columns)

        elif name == "cat":
            onehot = transformer.named_steps["onehot"]
            cat_names = onehot.get_feature_names_out(columns)
            feature_names.extend(cat_names)

        elif name == "bin":
            feature_names.extend(columns)

    return feature_names


def main():
    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = get_feature_names(preprocessor)
    importances = model.feature_importances_

    importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    output_path = OUTPUT_DIR / "random_forest_feature_importance_content_only.csv"
    importance_df.to_csv(output_path, index=False)

    print(f"Saved feature importance: {output_path}")
    print("\nTop 30 features:")
    print(importance_df.head(30))


if __name__ == "__main__":
    main()