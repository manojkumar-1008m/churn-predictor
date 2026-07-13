from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from bson import ObjectId
from django.conf import settings
from pymongo import MongoClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_FILE = Path(settings.BASE_DIR) / "model_artifacts" / "churn_pipeline.joblib"
MODEL_SCRIPT = Path(settings.BASE_DIR) / "backend" / "churn_predictor.py"


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    # serverSelectionTimeoutMS avoids long request blocking if MongoDB is down.
    return MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2000)


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _sanitize_mapping(
    data: dict[str, Any],
    max_items: int = 50,
    max_list_items: int = 5000,
) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for index, (key, value) in enumerate(data.items()):
        if index >= max_items:
            break
        if isinstance(value, dict):
            cleaned[str(key)] = _sanitize_mapping(
                value,
                max_items=20,
                max_list_items=max_list_items,
            )
        elif isinstance(value, list):
            cleaned[str(key)] = [_safe_scalar(item) for item in value[:max_list_items]]
        else:
            cleaned[str(key)] = _safe_scalar(value)
    return cleaned


def _build_summary_from_mapping(data: dict[str, Any], max_pairs: int = 4) -> str:
    if not data:
        return "N/A"
    pairs = []
    for index, (key, value) in enumerate(data.items()):
        if index >= max_pairs:
            break
        pairs.append(f"{key}={value}")
    suffix = " ..." if len(data) > max_pairs else ""
    return ", ".join(pairs) + suffix


def log_prediction_event(
    user,
    analysis_type: str,
    analyzed_data: dict[str, Any],
    result_data: dict[str, Any],
) -> None:
    client = get_mongo_client()
    collection = client[settings.MONGODB_DB_NAME][settings.MONGODB_PREDICTIONS_COLLECTION]
    collection.insert_one(
        {
            "timestamp_utc": datetime.now(timezone.utc),
            "django_user_id": int(user.id),
            "username": str(user.username),
            "analysis_type": str(analysis_type),
            "analyzed_data": _sanitize_mapping(analyzed_data),
            "result_data": _sanitize_mapping(result_data),
        }
    )


def get_user_analysis_history(user, limit: int | None = None) -> dict[str, Any]:
    client = get_mongo_client()
    collection = client[settings.MONGODB_DB_NAME][settings.MONGODB_PREDICTIONS_COLLECTION]

    query = {"django_user_id": int(user.id)}
    total_count = int(collection.count_documents(query))

    cursor = collection.find(query).sort("timestamp_utc", -1)
    if limit is not None:
        cursor = cursor.limit(max(1, int(limit)))

    entries = list(cursor)

    history_items: list[dict[str, Any]] = []
    for entry in entries:
        analysis_type_raw = str(entry.get("analysis_type", "analysis"))
        analyzed_data = entry.get("analyzed_data") or {}
        result_data = entry.get("result_data") or {}
        timestamp = entry.get("timestamp_utc")
        history_items.append(
            {
                "entry_id": str(entry.get("_id")),
                "analysis_type": analysis_type_raw.replace("_", " ").title(),
                "analysis_type_raw": analysis_type_raw,
                "analyzed_summary": _build_summary_from_mapping(analyzed_data),
                "result_summary": _build_summary_from_mapping(result_data),
                "identifier_column": str(result_data.get("identifier_column", "")),
                "churn_count": int(result_data.get("churn_count", 0)) if str(result_data.get("churn_count", "")).isdigit() else result_data.get("churn_count"),
                "churn_customers": list(result_data.get("churn_customers", [])) if isinstance(result_data.get("churn_customers", []), list) else [],
                "churn_details": list(result_data.get("churn_details", [])) if isinstance(result_data.get("churn_details", []), list) else [],
                "timestamp_utc": timestamp,
            }
        )

    return {
        "total_count": total_count,
        "items": history_items,
    }


def get_user_analysis_history_export_rows(user, limit: int = 5000) -> list[dict[str, Any]]:
    client = get_mongo_client()
    collection = client[settings.MONGODB_DB_NAME][settings.MONGODB_PREDICTIONS_COLLECTION]

    query = {"django_user_id": int(user.id)}
    entries = list(
        collection.find(query)
        .sort("timestamp_utc", -1)
        .limit(max(1, int(limit)))
    )

    export_rows: list[dict[str, Any]] = []
    for entry in entries:
        timestamp = entry.get("timestamp_utc")
        if isinstance(timestamp, datetime):
            timestamp_value = timestamp.isoformat()
        else:
            timestamp_value = _safe_scalar(timestamp)

        export_rows.append(
            {
                "timestamp_utc": timestamp_value,
                "analysis_type": str(entry.get("analysis_type", "analysis")),
                "analyzed_data": _sanitize_mapping(entry.get("analyzed_data") or {}),
                "result_data": _sanitize_mapping(entry.get("result_data") or {}),
            }
        )

    return export_rows


def get_user_history_entry_by_id(user, entry_id: str) -> dict[str, Any] | None:
    client = get_mongo_client()
    collection = client[settings.MONGODB_DB_NAME][settings.MONGODB_PREDICTIONS_COLLECTION]

    try:
        object_id = ObjectId(str(entry_id))
    except Exception:
        return None

    entry = collection.find_one({"_id": object_id, "django_user_id": int(user.id)})
    if not entry:
        return None

    timestamp = entry.get("timestamp_utc")
    if isinstance(timestamp, datetime):
        timestamp_value = timestamp.isoformat()
    else:
        timestamp_value = _safe_scalar(timestamp)

    return {
        "entry_id": str(entry.get("_id")),
        "timestamp_utc": timestamp_value,
        "analysis_type": str(entry.get("analysis_type", "analysis")),
        "analyzed_data": _sanitize_mapping(entry.get("analyzed_data") or {}),
        "result_data": _sanitize_mapping(entry.get("result_data") or {}),
    }


def store_registered_user_details(user, phone_number: str | None = None) -> None:
    client = get_mongo_client()
    collection = client[settings.MONGODB_DB_NAME][settings.MONGODB_USERS_COLLECTION]
    collection.update_one(
        {"django_user_id": int(user.id)},
        {
            "$set": {
                "django_user_id": int(user.id),
                "username": str(user.username),
                "email": str(user.email),
                "phone_number": str(phone_number or "").strip(),
                "is_active": bool(user.is_active),
                "is_staff": bool(user.is_staff),
                "is_superuser": bool(user.is_superuser),
                "date_joined": user.date_joined,
                "last_login": user.last_login,
                "updated_at_utc": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )


@lru_cache(maxsize=1)
def load_model():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)

    artifact_dir = MODEL_FILE.parent
    candidates = sorted(artifact_dir.glob("*.joblib"))
    if candidates:
        return joblib.load(candidates[-1])

    if MODEL_SCRIPT.exists():
        raise FileNotFoundError(
            "Found backend/churn_predictor.py, but Django needs a saved .joblib model file. "
            "Also, your script currently requires teleco dataset.csv to train. "
            "After placing the dataset, run: "
            "python scripts/export_pipeline.py --data-path \"..\\teleco dataset.csv\" "
            "Then retry prediction."
        )

    raise FileNotFoundError(
        "Model artifact missing. Save one to "
        f"{MODEL_FILE}."
    )


def save_uploaded_model(uploaded_file) -> Path:
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_FILE.open("wb") as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    load_model.cache_clear()
    return MODEL_FILE


def train_from_uploaded_dataset(uploaded_file) -> Path:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc

    required_columns = {
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    working = df.copy()
    working["TotalCharges"] = pd.to_numeric(working["TotalCharges"], errors="coerce")
    working = working.dropna(subset=["TotalCharges", "Churn"])

    if "customerID" in working.columns:
        working = working.drop(columns=["customerID"])

    X = working.drop(columns=["Churn"])
    y = working["Churn"].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Churn column must contain only Yes/No values.")

    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("num", StandardScaler(), numeric_columns),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )

    pipeline.fit(X, y)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    load_model.cache_clear()
    return MODEL_FILE


def _to_binary_label(prediction_value) -> int:
    value = str(prediction_value).strip().lower()
    if value in {"1", "yes", "true", "churn", "churns"}:
        return 1
    return 0


def _risk_label(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.55:
        return "Medium"
    return "Low"


def _get_model_feature_metadata(model) -> tuple[list[str], set[str], set[str]]:
    expected_features = list(getattr(model, "feature_names_in_", []))
    categorical_features: set[str] = set()
    numeric_features: set[str] = set()

    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        if hasattr(preprocessor, "feature_names_in_") and not expected_features:
            expected_features = list(preprocessor.feature_names_in_)

        for transformer_name, _transformer, columns in getattr(preprocessor, "transformers", []):
            if not isinstance(columns, (list, tuple)):
                continue
            if transformer_name == "cat":
                categorical_features.update(columns)
            elif transformer_name == "num":
                numeric_features.update(columns)

    return expected_features, categorical_features, numeric_features


def _align_input_to_model(model, input_df: pd.DataFrame) -> pd.DataFrame:
    expected_features, categorical_features, numeric_features = _get_model_feature_metadata(model)
    if not expected_features:
        return input_df

    aligned = input_df.copy()

    for column in expected_features:
        if column in aligned.columns:
            continue
        # Use a safe fallback so older or malformed artifacts can still score.
        if column in numeric_features:
            aligned[column] = 0
        else:
            aligned[column] = "Unknown"

    for column in numeric_features:
        if column in aligned.columns:
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce").fillna(0)

    return aligned.reindex(columns=expected_features)


def _normalize_column_name(column: str) -> str:
    return "".join(ch for ch in str(column).strip().lower() if ch.isalnum())


def _build_feature_rename_map(df_columns: list[str], feature_columns: list[str]) -> dict[str, str]:
    target_by_normalized = {_normalize_column_name(col): col for col in feature_columns}

    aliases = {
        "tenure": ["tenure", "months", "monthstenure", "subscriptionmonths"],
        "monthlycharges": ["monthlycharges", "monthlycharge", "monthly_fee", "monthlyfees"],
        "totalcharges": ["totalcharges", "totalcharge", "total_fee", "totalfees"],
        "contract": ["contract", "plantype", "subscriptiontype"],
        "internetservice": ["internetservice", "internet", "internettype"],
        "paymentmethod": ["paymentmethod", "payment", "paymenttype", "payment_mode"],
        "seniorcitizen": ["seniorcitizen", "senior", "is_senior"],
        "phoneservice": ["phoneservice", "phone", "hasphone"],
        "partner": ["partner", "haspartner"],
        "dependents": ["dependents", "dependent", "hasdependents"],
    }

    alias_to_target: dict[str, str] = {}
    for target_key, alias_values in aliases.items():
        if target_key in target_by_normalized:
            target = target_by_normalized[target_key]
            for alias in alias_values:
                alias_to_target[_normalize_column_name(alias)] = target

    rename_map: dict[str, str] = {}
    for source_column in df_columns:
        normalized = _normalize_column_name(source_column)
        if normalized in target_by_normalized:
            rename_map[source_column] = target_by_normalized[normalized]
        elif normalized in alias_to_target:
            rename_map[source_column] = alias_to_target[normalized]

    return rename_map


def _select_identifier_columns(df: pd.DataFrame, feature_columns: list[str]) -> tuple[str | None, str | None]:
    feature_set = set(feature_columns)

    non_feature_columns = [
        col
        for col in df.columns
        if col not in feature_set and col not in {"Churn", "churn", "Prediction", "prediction"}
    ]

    normalized_lookup = {_normalize_column_name(col): col for col in non_feature_columns}

    id_candidates = [
        "customerid",
        "customer_id",
        "custid",
        "clientid",
        "userid",
        "user_id",
        "subscriberid",
        "accountid",
        "memberid",
        "id",
    ]
    name_candidates = [
        "customername",
        "customer_name",
        "custname",
        "clientname",
        "fullname",
        "full_name",
        "name",
    ]

    normalized_id_candidates = {_normalize_column_name(value) for value in id_candidates}
    normalized_name_candidates = {_normalize_column_name(value) for value in name_candidates}

    id_column = None
    for candidate in normalized_id_candidates:
        if candidate in normalized_lookup:
            id_column = normalized_lookup[candidate]
            break

    name_column = None
    for candidate in normalized_name_candidates:
        if candidate in normalized_lookup:
            name_column = normalized_lookup[candidate]
            break

    if id_column is None:
        for col in non_feature_columns:
            normalized = _normalize_column_name(col)
            if "id" in normalized:
                id_column = col
                break

    if name_column is None:
        for col in non_feature_columns:
            normalized = _normalize_column_name(col)
            if "name" in normalized:
                name_column = col
                break

    if id_column is None and non_feature_columns:
        # Prefer a mostly-unique non-feature column as a likely identifier.
        uniqueness_scores = []
        for col in non_feature_columns:
            series = df[col].dropna()
            if series.empty:
                continue
            unique_ratio = float(series.astype(str).nunique()) / float(len(series))
            uniqueness_scores.append((unique_ratio, col))
        if uniqueness_scores:
            uniqueness_scores.sort(reverse=True)
            best_ratio, best_col = uniqueness_scores[0]
            if best_ratio >= 0.85:
                id_column = best_col

    return id_column, name_column


def predict_churn(payload: dict) -> tuple[int, float]:
    model = load_model()
    input_df = pd.DataFrame([payload])
    input_df = _align_input_to_model(model, input_df)

    raw_prediction = model.predict(input_df)[0]
    prediction = _to_binary_label(raw_prediction)

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(max(probabilities))

    return prediction, confidence


def predict_churn_customers_from_dataset(uploaded_file) -> dict:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    feature_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]
    rename_map = _build_feature_rename_map(df.columns.tolist(), feature_columns)
    working_df = df.rename(columns=rename_map).copy()

    available_features = [col for col in feature_columns if col in working_df.columns]
    if not available_features:
        raise ValueError(
            "Dataset does not contain recognizable feature columns for prediction. "
            "Provide at least one known column such as tenure, MonthlyCharges, TotalCharges, Contract, InternetService, or PaymentMethod."
        )

    id_column, name_column = _select_identifier_columns(working_df, feature_columns)

    scoring_df = working_df[available_features].copy()
    if "TotalCharges" in scoring_df.columns:
        scoring_df["TotalCharges"] = pd.to_numeric(scoring_df["TotalCharges"], errors="coerce")
    valid_mask = scoring_df.notna().all(axis=1)
    if not valid_mask.any():
        raise ValueError(
            "No valid rows found to score. Ensure required columns contain valid values."
        )

    scoring_df = scoring_df[valid_mask]
    source_df = working_df.loc[scoring_df.index]

    model = load_model()
    scoring_df = _align_input_to_model(model, scoring_df)
    raw_predictions = model.predict(scoring_df)
    predictions = [_to_binary_label(value) for value in raw_predictions]

    churn_probabilities = [0.0] * len(scoring_df)
    if hasattr(model, "predict_proba"):
        probability_rows = model.predict_proba(scoring_df)
        for idx, row in enumerate(probability_rows):
            if len(row) >= 2:
                churn_probabilities[idx] = float(row[1])
            else:
                churn_probabilities[idx] = float(max(row))

    churn_indexes = [idx for idx, pred in enumerate(predictions) if pred == 1]

    churn_customers = []
    churn_details = []
    for idx in churn_indexes:
        row_index = scoring_df.index[idx]
        row_data = source_df.iloc[idx]
        if id_column and name_column:
            id_value = str(row_data[id_column]).strip()
            name_value = str(row_data[name_column]).strip()
            if id_value and id_value.lower() != "nan" and name_value and name_value.lower() != "nan":
                customer_label = f"{id_value} - {name_value}"
            elif id_value and id_value.lower() != "nan":
                customer_label = id_value
            elif name_value and name_value.lower() != "nan":
                customer_label = name_value
            else:
                customer_label = f"Row {int(row_index) + 1}"
        elif id_column:
            id_value = str(row_data[id_column]).strip()
            customer_label = id_value if id_value and id_value.lower() != "nan" else f"Row {int(row_index) + 1}"
        elif name_column:
            name_value = str(row_data[name_column]).strip()
            customer_label = name_value if name_value and name_value.lower() != "nan" else f"Row {int(row_index) + 1}"
        else:
            customer_label = f"Row {int(row_index) + 1}"

        probability = float(churn_probabilities[idx])
        churn_details.append(
            {
                "customer": customer_label,
                "risk": _risk_label(probability),
                "probability_percent": round(probability * 100, 2),
            }
        )
        churn_customers.append(customer_label)

    if id_column and name_column:
        identifier_label = f"{id_column} + {name_column}"
    else:
        identifier_label = id_column or name_column or "row_number"

    return {
        "identifier_column": identifier_label,
        "total_rows": int(len(working_df)),
        "scanned_rows": int(len(scoring_df)),
        "churn_count": int(len(churn_customers)),
        "churn_customers": churn_customers,
        "churn_details": churn_details,
    }
