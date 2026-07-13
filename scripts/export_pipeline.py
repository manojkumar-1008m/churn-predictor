import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("num", StandardScaler(), numeric_columns),
        ]
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_and_save(data_path: Path, output_path: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' in dataset.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError("Unexpected values in Churn column. Expected Yes/No labels.")

    pipeline = build_pipeline(X)
    pipeline.fit(X, y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Saved model pipeline to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export a Django-ready churn pipeline artifact."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=Path,
        help="Path to teleco dataset.csv",
    )
    parser.add_argument(
        "--output-path",
        default=Path("model_artifacts") / "churn_pipeline.joblib",
        type=Path,
        help="Path where the trained pipeline will be saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(args.data_path, args.output_path)
