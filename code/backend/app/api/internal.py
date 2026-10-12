"""
Internal ML Compute Endpoints for Java Spring Boot Control Plane
"""
import os
import logging
import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional
from types import SimpleNamespace

from app.ml.pipeline import AutoMLPipeline
from app.ml.explainability import explainability_service, ExplainabilityError
from app.ml.dataset_profiling import build_dataset_profile, build_drift_baseline_snapshot, build_eda_report
from app.ml.analytics_service import AnalyticsReportService, build_predictions_from_model_bundle
from app.api.datasets import (
    build_clean_preview,
    build_quality_report,
    clean_dataframe,
    _load_dataset_file,
)
from app.core.config import settings
from app.ml.runtime import get_acceleration_state

router = APIRouter(prefix="/internal")
logger = logging.getLogger(__name__)
analytics_service = AnalyticsReportService()
_model_cache: Dict[int, Any] = {}


@router.get("/capabilities")
def internal_capabilities():
    return {"acceleration": get_acceleration_state()}


def _get_cached_model(model_id: int, file_path: str) -> dict:
    if model_id not in _model_cache:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Model file not found on disk: " + file_path)
        _model_cache[model_id] = joblib.load(file_path)
    return _model_cache[model_id]


@router.post("/train")
def internal_train(payload: Dict[str, Any]):
    """Execute AutoML training pipeline synchronously on compute node."""
    job_id = payload.get("job_id", "internal_job")
    dataset_path = payload.get("dataset_path")
    target_column = payload.get("target_column")

    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found: " + str(dataset_path))

    df = _load_dataset_file(dataset_path)

    pipeline = AutoMLPipeline(job_id=job_id)
    results = pipeline.train_all_models(
        df=df,
        target_column=target_column,
        task_type=payload.get("task_type"),
        test_size=float(payload.get("test_size", 0.2)),
        cv_folds=int(payload.get("cv_folds", 5)),
        models_to_train=payload.get("models_to_train"),
        enable_tuning=bool(payload.get("enable_tuning", False)),
        tuning_trials=int(payload.get("tuning_trials", 12)),
        tuning_time_budget_sec=int(payload.get("tuning_time_budget_sec", 180)),
        model_hyperparams=payload.get("model_hyperparams"),
    )
    return results


@router.post("/predict")
def internal_predict(payload: Dict[str, Any]):
    """Execute model prediction on compute node."""
    model_id = payload.get("model_id")
    file_path = payload.get("file_path")
    features = payload.get("features", {})

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found: " + str(file_path))

    model_data = _get_cached_model(model_id, file_path)
    pipeline = model_data["pipeline"]
    feature_names = model_data["feature_names"]
    label_encoder = model_data.get("label_encoder")
    task_type = model_data.get("task_type")

    input_data = {}
    for feat in feature_names:
        if feat in features:
            input_data[feat] = [features[feat]]
        else:
            input_data[feat] = [None]

    df_input = pd.DataFrame(input_data)
    prediction_raw = pipeline.predict(df_input)
    prediction = prediction_raw[0]

    probability = None
    confidence = None

    if task_type == "classification":
        if label_encoder:
            try:
                prediction = label_encoder.inverse_transform([int(prediction)])[0]
            except Exception:
                pass

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(df_input)[0]
            classes = label_encoder.classes_ if label_encoder else [str(i) for i in range(len(proba))]
            probability = {str(cls): float(p) for cls, p in zip(classes, proba)}
            confidence = float(max(proba))
    else:
        prediction = float(prediction)

    return {
        "model_id": model_id,
        "prediction": prediction,
        "confidence": confidence,
        "probability": probability,
    }


def _explainability_model(payload: Dict[str, Any]) -> SimpleNamespace:
    required = ("model_id", "file_path", "dataset_path", "target_column")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise HTTPException(status_code=422, detail="Missing explainability fields: " + ", ".join(missing))

    return SimpleNamespace(
        id=int(payload["model_id"]),
        file_path=str(payload["file_path"]),
        model_name=str(payload.get("model_name") or "Model " + str(payload["model_id"])),
        model_type=str(payload.get("model_type") or "estimator"),
        task_type=str(payload.get("task_type") or "unknown"),
        job_id=str(payload.get("job_id") or "unknown"),
    )


def _explainability_error(exc: ExplainabilityError) -> HTTPException:
    status = 404 if exc.code in {
        "MODEL_FILE_MISSING",
        "DATASET_FILE_MISSING",
        "TARGET_COLUMN_MISSING",
    } else 400
    return HTTPException(
        status_code=status,
        detail={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
    )


@router.post("/models/explain/shap")
def internal_shap(payload: Dict[str, Any]):
    """Generate the same SHAP payload exposed through the public control plane."""
    model = _explainability_model(payload)
    try:
        return explainability_service.build_shap(
            model=model,
            dataset_path=str(payload["dataset_path"]),
            target_column=str(payload["target_column"]),
            sample_index=int(payload.get("sample_index", 0)),
            nsamples=int(payload.get("nsamples", 200)),
        )
    except ExplainabilityError as exc:
        raise _explainability_error(exc) from exc


@router.post("/models/explain/lime")
def internal_lime(payload: Dict[str, Any]):
    """Generate the same LIME payload exposed through the public control plane."""
    model = _explainability_model(payload)
    try:
        return explainability_service.build_lime(
            model=model,
            dataset_path=str(payload["dataset_path"]),
            target_column=str(payload["target_column"]),
            sample_index=int(payload.get("sample_index", 0)),
            num_features=int(payload.get("num_features", 12)),
            custom_input=payload.get("custom_input"),
        )
    except ExplainabilityError as exc:
        raise _explainability_error(exc) from exc


@router.post("/datasets/profile")
def internal_dataset_profile(payload: Dict[str, Any]):
    file_path = payload.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = _load_dataset_file(file_path)
    return build_dataset_profile(
        df,
        dataset_id=payload.get("dataset_id", 1),
        dataset_name=payload.get("dataset_name", "dataset"),
        target_column=payload.get("target_column"),
    )


@router.post("/datasets/quality-report")
def internal_quality_report(payload: Dict[str, Any]):
    file_path = payload.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = _load_dataset_file(file_path)
    return build_quality_report(df)


@router.post("/datasets/clean-preview")
def internal_clean_preview(payload: Dict[str, Any]):
    file_path = payload.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return build_clean_preview(_load_dataset_file(file_path), preview_rows=int(payload.get("preview_rows", 10)))


@router.post("/datasets/clean-and-save")
def internal_clean_and_save(payload: Dict[str, Any]):
    source_path = payload.get("file_path")
    output_path = payload.get("output_file_path")
    if not source_path or not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not output_path:
        raise HTTPException(status_code=422, detail="output_file_path is required")

    resolved_output = os.path.abspath(output_path)
    datasets_root = os.path.abspath(settings.DATASET_STORAGE_PATH)
    if os.path.commonpath([resolved_output, datasets_root]) != datasets_root:
        raise HTTPException(status_code=400, detail="Output path must be inside dataset storage")

    cleaned, applied_fixes = clean_dataframe(_load_dataset_file(source_path))
    os.makedirs(os.path.dirname(resolved_output), exist_ok=True)
    if resolved_output.lower().endswith(".csv"):
        cleaned.to_csv(resolved_output, index=False)
    else:
        cleaned.to_excel(resolved_output, index=False)

    return {
        "file_path": resolved_output,
        "rows_after_cleaning": len(cleaned),
        "applied_fixes": applied_fixes,
    }


@router.post("/datasets/leakage-report")
def internal_leakage_report(payload: Dict[str, Any]):
    file_path = payload.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    profile = build_dataset_profile(
        _load_dataset_file(file_path),
        dataset_id=payload.get("dataset_id", 1),
        dataset_name=payload.get("dataset_name", "dataset"),
        target_column=payload.get("target_column"),
    )
    return {"leakage_risks": profile.get("leakage_risks", [])}


@router.post("/datasets/drift-baseline")
def internal_drift_baseline(payload: Dict[str, Any]):
    file_path = payload.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "drift_baseline": build_drift_baseline_snapshot(
            _load_dataset_file(file_path),
            target_column=payload.get("target_column"),
        )
    }


@router.post("/models/monitoring")
def internal_monitoring(payload: Dict[str, Any]):
    training_path = payload.get("training_file_path")
    comparison_path = payload.get("comparison_file_path")
    if not training_path or not comparison_path or not os.path.exists(training_path) or not os.path.exists(comparison_path):
        raise HTTPException(status_code=404, detail="Training or comparison dataset not found")

    target_column = payload.get("target_column")
    training = _load_dataset_file(training_path).drop(columns=[target_column], errors="ignore")
    comparison = _load_dataset_file(comparison_path).drop(columns=[target_column], errors="ignore")
    shared_numeric = [
        column for column in training.select_dtypes(include=["int64", "float64"]).columns
        if column in comparison.columns
    ]
    per_feature = []
    for column in shared_numeric[:30]:
        train_values = training[column].dropna()
        current_values = comparison[column].dropna()
        if train_values.empty or current_values.empty:
            continue
        train_mean = float(train_values.mean())
        current_mean = float(current_values.mean())
        train_std = max(float(train_values.std()), 1e-9)
        per_feature.append({
            "feature": column,
            "train_mean": train_mean,
            "current_mean": current_mean,
            "z_drift": round(abs(current_mean - train_mean) / train_std, 4),
        })

    average = float(np.mean([item["z_drift"] for item in per_feature])) if per_feature else 0.0
    return {
        "features_compared": len(per_feature),
        "avg_z_drift": round(average, 4),
        "drift_level": "low" if average < 0.5 else "medium" if average < 1.0 else "high",
        "top_drift_features": sorted(per_feature, key=lambda item: item["z_drift"], reverse=True)[:10],
    }


@router.post("/datasets/eda-report")
def internal_eda_report(payload: Dict[str, Any]):
    file_path = payload.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = _load_dataset_file(file_path)
    return build_eda_report(
        df,
        dataset_name=payload.get("dataset_name", "dataset"),
        target_column=payload.get("target_column"),
    )


@router.post("/datasets/analytics-report")
def internal_analytics_report(payload: Dict[str, Any]):
    file_path = payload.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = _load_dataset_file(file_path)
    return analytics_service.generate_report(
        dataset_id=payload.get("dataset_id", 1),
        dataset_name=payload.get("dataset_name", "dataset"),
        df=df,
        target_column=payload.get("target_column"),
        model_type=payload.get("model_type"),
    )
