from pathlib import Path

import joblib
import pandas as pd
import numpy as np


MODEL_PATH = "models/random_forest_engagement_model.joblib"
OUTPUT_DIR = Path("outputs/engagement_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_feature_names(preprocessor):
    feature_names = []

    # Numeric
    if "num" in preprocessor.named_transformers_:
        num_features = preprocessor.transformers_[0][2]
        feature_names.extend(num_features)

    # Categorical one-hot
    if "cat" in preprocessor.named_transformers_:
        cat_pipeline = preprocessor.named_transformers_["cat"]
        cat_features = preprocessor.transformers_[1][2]

        onehot = cat_pipeline.named_steps["onehot"]
        cat_names = onehot.get_feature_names_out(cat_features)
        feature_names.extend(cat_names)

    # Binary
    if "bin" in preprocessor.named_transformers_:
        bin_features = preprocessor.transformers_[2][2]
        feature_names.extend(bin_features)

    return feature_names


def main():
    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = get_feature_names(preprocessor)
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_path = OUTPUT_DIR / "random_forest_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    print(f"Saved feature importance: {importance_path}")
    print("\nTop 30 features:")
    print(importance_df.head(30))


if __name__ == "__main__":
    main()