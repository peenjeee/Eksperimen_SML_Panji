from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "monthly_income",
    "years_at_company",
    "distance_from_home",
    "job_satisfaction",
]
CATEGORICAL_FEATURES = ["overtime", "department", "education"]
TARGET = "attrition"


def make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(raw_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(raw_path)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    clean = data.copy()
    clean.columns = [column.strip().lower() for column in clean.columns]
    clean = clean.drop_duplicates(subset=["employee_id"]).drop(columns=["employee_id"])
    for column in CATEGORICAL_FEATURES + [TARGET]:
        clean[column] = clean[column].astype("object")
        clean[column] = clean[column].where(clean[column].notna(), None)
        clean[column] = clean[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    clean[TARGET] = clean[TARGET].map({"No": 0, "Yes": 1, "no": 0, "yes": 1})
    clean = clean.dropna(subset=[TARGET])
    clean[TARGET] = clean[TARGET].astype(int)
    return clean


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = list(NUMERIC_FEATURES)
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    encoded = encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    names.extend(encoded)
    return names


def preprocess_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer, dict]:
    clean = clean_data(data)
    X = clean[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    preprocessor = build_preprocessor()
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    train_ready = pd.DataFrame(X_train_ready, columns=feature_names, index=X_train.index)
    train_ready[TARGET] = y_train.to_numpy()
    train_ready["split"] = "train"

    test_ready = pd.DataFrame(X_test_ready, columns=feature_names, index=X_test.index)
    test_ready[TARGET] = y_test.to_numpy()
    test_ready["split"] = "test"

    combined = pd.concat([train_ready, test_ready], axis=0).sort_index().reset_index(drop=True)
    metadata = {
        "rows_raw": int(len(data)),
        "rows_after_cleaning": int(len(clean)),
        "train_rows": int(len(train_ready)),
        "test_rows": int(len(test_ready)),
        "target": TARGET,
        "positive_rate": float(clean[TARGET].mean()),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "encoded_feature_count": len(feature_names),
        "feature_names": feature_names,
    }
    return combined, train_ready.reset_index(drop=True), test_ready.reset_index(drop=True), preprocessor, metadata


def save_preprocessed_data(
    combined: pd.DataFrame,
    train_ready: pd.DataFrame,
    test_ready: pd.DataFrame,
    preprocessor: ColumnTransformer,
    metadata: dict,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "employee_attrition_preprocessed.csv", index=False)
    train_ready.to_csv(output / "train.csv", index=False)
    test_ready.to_csv(output / "test.csv", index=False)
    joblib.dump(preprocessor, output / "fitted_preprocessor.joblib")
    (output / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate employee attrition preprocessing.")
    parser.add_argument(
        "--raw-path",
        default=str(Path(__file__).resolve().parents[1] / "employee_attrition_raw" / "employee_attrition_raw.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "employee_attrition_preprocessing"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data = load_data(args.raw_path)
    combined, train_ready, test_ready, preprocessor, metadata = preprocess_data(data, test_size=args.test_size)
    save_preprocessed_data(combined, train_ready, test_ready, preprocessor, metadata, args.output_dir)
    print(f"Saved preprocessed dataset to {Path(args.output_dir).resolve()}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
