"""Fit and package the map-matching classifiers for deployment or reuse."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .artifact_inference import predict_candidate_probabilities
from .candidate_models import _read_candidates


def _assemble_training_arrays(
    feature_root: Path,
    trip_ids: list[str],
    feature_columns: list[str],
    *,
    observations_per_trip: int,
    knn_observations_per_trip: int,
    calibration_fraction: float,
    random_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    columns = feature_columns + [
        "observation_id",
        "supervision_usable",
        "is_target",
    ]
    full_x_parts = []
    full_y_parts = []
    svm_x_parts = []
    svm_y_parts = []
    calibration_x_parts = []
    calibration_y_parts = []
    knn_x_parts = []
    knn_y_parts = []
    fit_observations = 0
    svm_fit_observations = 0
    calibration_observations = 0
    knn_observations = 0
    contributing_trips = []

    for trip_id in trip_ids:
        path = feature_root / "train" / trip_id / "candidates.csv.gz"
        frame = _read_candidates(path, columns)
        frame = frame.loc[frame["supervision_usable"]]
        observation_ids = frame["observation_id"].drop_duplicates().to_numpy()
        if not len(observation_ids):
            continue

        selected_ids = np.sort(
            rng.choice(
                observation_ids,
                size=min(observations_per_trip, len(observation_ids)),
                replace=False,
            )
        )
        sampled = frame.loc[frame["observation_id"].isin(selected_ids)]
        x = sampled[feature_columns].to_numpy(dtype=np.float32)
        y = sampled["is_target"].to_numpy(dtype=np.uint8)
        full_x_parts.append(x)
        full_y_parts.append(y)
        fit_observations += len(selected_ids)
        contributing_trips.append(trip_id)

        shuffled_ids = rng.permutation(selected_ids)
        calibration_count = (
            max(1, int(round(len(selected_ids) * calibration_fraction)))
            if len(selected_ids) >= 10
            else 0
        )
        calibration_ids = set(shuffled_ids[:calibration_count])
        svm_calibration_mask = (
            sampled["observation_id"].isin(calibration_ids).to_numpy()
        )
        svm_fit_mask = ~svm_calibration_mask
        svm_x_parts.append(x[svm_fit_mask])
        svm_y_parts.append(y[svm_fit_mask])
        calibration_x_parts.append(x[svm_calibration_mask])
        calibration_y_parts.append(y[svm_calibration_mask])
        svm_fit_observations += len(selected_ids) - calibration_count
        calibration_observations += calibration_count

        knn_ids = rng.choice(
            selected_ids,
            size=min(knn_observations_per_trip, len(selected_ids)),
            replace=False,
        )
        knn_mask = sampled["observation_id"].isin(knn_ids).to_numpy()
        knn_x_parts.append(x[knn_mask])
        knn_y_parts.append(y[knn_mask])
        knn_observations += len(knn_ids)
        del frame, sampled, x, y

    if not full_x_parts:
        raise ValueError(
            "No usable labeled observations found in the training partition."
        )

    arrays = {
        "full_x": np.concatenate(full_x_parts),
        "full_y": np.concatenate(full_y_parts),
        "svm_x": np.concatenate(svm_x_parts),
        "svm_y": np.concatenate(svm_y_parts),
        "calibration_x": np.concatenate(calibration_x_parts),
        "calibration_y": np.concatenate(calibration_y_parts),
        "knn_x": np.concatenate(knn_x_parts),
        "knn_y": np.concatenate(knn_y_parts),
        "fit_observations": fit_observations,
        "svm_fit_observations": svm_fit_observations,
        "calibration_observations": calibration_observations,
        "knn_observations": knn_observations,
        "contributing_trips": contributing_trips,
    }
    for key in ("full_y", "svm_y", "calibration_y", "knn_y"):
        if np.unique(arrays[key]).size != 2:
            raise ValueError(f"{key} must contain both candidate classes.")
    return arrays


def _fit_svm(
    arrays: dict[str, Any],
    random_seed: int,
) -> tuple[LinearSVC, StandardScaler, LogisticRegression, float]:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(arrays["svm_x"]).astype(np.float32, copy=False)
    x_calibration = scaler.transform(arrays["calibration_x"]).astype(
        np.float32, copy=False
    )
    estimator = LinearSVC(
        C=0.25,
        dual="auto",
        max_iter=3000,
        random_state=random_seed,
    )
    start = time.perf_counter()
    estimator.fit(x_fit, arrays["svm_y"])
    calibration_scores = estimator.decision_function(x_calibration).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=200, solver="lbfgs")
    calibrator.fit(calibration_scores, arrays["calibration_y"])
    return estimator, scaler, calibrator, time.perf_counter() - start


def _file_record(path: Path, base: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.relative_to(base)),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def export_all_models(
    project_root: str | Path,
    *,
    artifact_root: str | Path | None = None,
    observations_per_trip: int = 500,
    knn_observations_per_trip: int = 20,
    calibration_fraction: float = 0.15,
    random_seed: int = 42,
) -> tuple[Path, dict[str, Any]]:
    """Train and export all comparison models plus the Bayesian-tuned winner."""
    project_root = Path(project_root)
    feature_root = project_root / "data" / "processed" / "features"
    previous_results = project_root / "results" / "candidate_models"
    tuned_results = previous_results / "bayesian_hmm"
    artifact_root = (
        Path(artifact_root)
        if artifact_root is not None
        else project_root / "artifacts" / "map_matching_models"
    )
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise FileExistsError(
            f"Artifact destination is not empty; refusing to overwrite: {artifact_root}"
        )

    schema_path = feature_root / "feature_schema.json"
    selection_path = previous_results / "model_selection.json"
    tuned_model_path = tuned_results / "bayesian_tuned_random_forest.joblib"
    if not schema_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(
            "Prepared feature schema and five-model selection record are required."
        )
    if not tuned_model_path.is_file():
        raise FileNotFoundError(
            f"Bayesian-tuned Random Forest is missing: {tuned_model_path}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("best_model") != "Random Forest":
        raise ValueError("The recorded selected model is not Random Forest.")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_columns = schema["feature_columns"]
    audit = pd.read_csv(
        feature_root / "feature_quality_audit.csv", dtype={"trip_id": str}
    )
    training_ids = sorted(audit.loc[audit["partition"] == "train", "trip_id"].unique())
    arrays = _assemble_training_arrays(
        feature_root,
        training_ids,
        feature_columns,
        observations_per_trip=observations_per_trip,
        knn_observations_per_trip=knn_observations_per_trip,
        calibration_fraction=calibration_fraction,
        random_seed=random_seed,
    )

    models_dir = artifact_root / "models"
    reports_dir = artifact_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    smoke_values = arrays["full_x"][: min(4, len(arrays["full_x"]))].copy()

    full_scaler = StandardScaler()
    full_x = full_scaler.fit_transform(arrays["full_x"]).astype(np.float32, copy=False)
    knn_x = full_scaler.transform(arrays["knn_x"]).astype(np.float32, copy=False)
    model_specs = [
        (
            "Logistic Regression",
            "logistic_regression.joblib",
            lambda: LogisticRegression(C=1.0, max_iter=200, solver="lbfgs"),
            full_x,
            arrays["full_y"],
            full_scaler,
            None,
            {
                "C": 1.0,
                "max_iter": 200,
                "solver": "lbfgs",
            },
            arrays["fit_observations"],
        ),
        (
            "Random Forest (baseline)",
            "random_forest_baseline.joblib",
            lambda: RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_leaf=10,
                max_features="sqrt",
                n_jobs=2,
                random_state=random_seed,
            ),
            full_x,
            arrays["full_y"],
            full_scaler,
            None,
            {
                "n_estimators": 50,
                "max_depth": 10,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
            },
            arrays["fit_observations"],
        ),
        (
            "XGBoost",
            "xgboost.joblib",
            lambda: XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=2,
                random_state=random_seed,
            ),
            full_x,
            arrays["full_y"],
            full_scaler,
            None,
            {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.08,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_lambda": 2.0,
            },
            arrays["fit_observations"],
        ),
        (
            "KNN",
            "knn.joblib",
            lambda: KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                algorithm="kd_tree",
                leaf_size=40,
                n_jobs=2,
            ),
            knn_x,
            arrays["knn_y"],
            full_scaler,
            None,
            {
                "n_neighbors": 15,
                "weights": "distance",
                "algorithm": "kd_tree",
                "leaf_size": 40,
                "observations_per_trip": knn_observations_per_trip,
            },
            arrays["knn_observations"],
        ),
    ]

    artifact_records: dict[str, dict[str, Any]] = {}
    for (
        name,
        filename,
        factory,
        x,
        y,
        scaler,
        calibrator,
        params,
        fit_count,
    ) in model_specs:
        estimator = factory()
        start = time.perf_counter()
        estimator.fit(x, y)
        training_seconds = time.perf_counter() - start
        bundle = {
            "model_name": name,
            "estimator": estimator,
            "scaler": scaler,
            "calibrator": calibrator,
            "feature_columns": feature_columns,
            "class_labels": [False, True],
            "positive_class": True,
            "classification_threshold": 0.5,
            "hyperparameters": params,
            "training_observations": int(fit_count),
            "training_trips": len(arrays["contributing_trips"]),
            "training_seconds": training_seconds,
            "student_test_used_for_training": False,
        }
        path = models_dir / filename
        joblib.dump(bundle, path)
        loaded = joblib.load(path)
        smoke_frame = pd.DataFrame(smoke_values, columns=feature_columns)
        smoke_probabilities = predict_candidate_probabilities(loaded, smoke_frame)
        if (
            len(smoke_probabilities) != len(smoke_frame)
            or not np.isfinite(smoke_probabilities).all()
        ):
            raise ValueError(f"Artifact inference check failed for {name}.")
        artifact_records[name] = {
            "file": str(path.relative_to(artifact_root)),
            "variant": "baseline" if name.endswith("(baseline)") else "comparison",
            "hyperparameters": params,
            "training_observations": int(fit_count),
            "training_trips": len(arrays["contributing_trips"]),
            "training_seconds": training_seconds,
            "smoke_test_predictions": len(smoke_probabilities),
        }
        print(
            f"Exported {name}: {filename}; fit={training_seconds:.1f}s; "
            f"sample observations={fit_count}"
        )
        del estimator, bundle, loaded

    svm_estimator, svm_scaler, svm_calibrator, svm_training_seconds = _fit_svm(
        arrays, random_seed
    )
    svm_bundle = {
        "model_name": "Linear SVM (calibrated)",
        "estimator": svm_estimator,
        "scaler": svm_scaler,
        "calibrator": svm_calibrator,
        "feature_columns": feature_columns,
        "class_labels": [False, True],
        "positive_class": True,
        "classification_threshold": 0.5,
        "hyperparameters": {
            "C": 0.25,
            "max_iter": 3000,
            "calibrator": "LogisticRegression",
            "calibration_fraction": calibration_fraction,
        },
        "training_observations": int(arrays["svm_fit_observations"]),
        "calibration_observations": int(arrays["calibration_observations"]),
        "training_trips": len(arrays["contributing_trips"]),
        "training_seconds": svm_training_seconds,
        "student_test_used_for_training": False,
    }
    svm_path = models_dir / "linear_svm_calibrated.joblib"
    joblib.dump(svm_bundle, svm_path)
    loaded_svm = joblib.load(svm_path)
    smoke_frame = pd.DataFrame(smoke_values, columns=feature_columns)
    svm_probabilities = predict_candidate_probabilities(loaded_svm, smoke_frame)
    if not np.isfinite(svm_probabilities).all():
        raise ValueError("Artifact inference check failed for calibrated Linear SVM.")
    artifact_records["Linear SVM (calibrated)"] = {
        "file": str(svm_path.relative_to(artifact_root)),
        "variant": "comparison_with_training_only_calibrator",
        "hyperparameters": svm_bundle["hyperparameters"],
        "training_observations": int(arrays["svm_fit_observations"]),
        "calibration_observations": int(arrays["calibration_observations"]),
        "training_trips": len(arrays["contributing_trips"]),
        "training_seconds": svm_training_seconds,
        "smoke_test_predictions": len(svm_probabilities),
    }
    del svm_estimator, svm_scaler, svm_calibrator, svm_bundle, loaded_svm

    tuned_bundle = joblib.load(tuned_model_path)
    tuned_filename = "random_forest_bayesian_tuned.joblib"
    tuned_bundle["model_name"] = "Random Forest (Bayesian tuned)"
    tuned_bundle["student_test_used_for_training"] = False
    tuned_bundle["artifact_source"] = str(tuned_model_path.relative_to(project_root))
    tuned_path = models_dir / tuned_filename
    joblib.dump(tuned_bundle, tuned_path)
    loaded_tuned = joblib.load(tuned_path)
    smoke_frame = pd.DataFrame(smoke_values, columns=feature_columns)
    tuned_probabilities = predict_candidate_probabilities(loaded_tuned, smoke_frame)
    if not np.isfinite(tuned_probabilities).all():
        raise ValueError(
            "Artifact inference check failed for Bayesian-tuned Random Forest."
        )
    artifact_records["Random Forest (Bayesian tuned)"] = {
        "file": str(tuned_path.relative_to(artifact_root)),
        "variant": "selected_and_bayesian_tuned",
        "hyperparameters": tuned_bundle["best_params"],
        "training_observations": tuned_bundle["final_fit_observations"],
        "training_trips": tuned_bundle["final_fit_trips"],
        "student_test_used_for_training": False,
        "smoke_test_predictions": len(tuned_probabilities),
    }
    del tuned_bundle, loaded_tuned

    shutil.copy2(schema_path, artifact_root / "feature_schema.json")
    reports = [
        (
            previous_results / "model_comparison.csv",
            reports_dir / "five_model_validation_comparison.csv",
        ),
        (
            selection_path,
            reports_dir / "five_model_selection.json",
        ),
        (
            tuned_results / "bayesian_optimization_results.csv",
            reports_dir / "bayesian_optimization_results.csv",
        ),
        (
            tuned_results / "bayesian_optimization_summary.json",
            reports_dir / "bayesian_optimization_summary.json",
        ),
        (
            tuned_results / "bayesian_refit_summary.json",
            reports_dir / "bayesian_refit_summary.json",
        ),
        (
            tuned_results / "train_validation_hmm_metrics.json",
            reports_dir / "train_validation_hmm_metrics.json",
        ),
        (
            tuned_results / "train_validation_hmm_predictions.csv",
            reports_dir / "train_validation_hmm_predictions.csv",
        ),
        (
            tuned_results / "train_validation_viterbi_route_sequence.csv",
            reports_dir / "train_validation_viterbi_route_sequence.csv",
        ),
        (
            tuned_results / "student_test_hmm_metrics.json",
            reports_dir / "student_test_hmm_metrics.json",
        ),
        (
            tuned_results / "student_test_hmm_predictions.csv",
            reports_dir / "student_test_hmm_predictions.csv",
        ),
        (
            tuned_results / "student_test_viterbi_route_sequence.csv",
            reports_dir / "student_test_viterbi_route_sequence.csv",
        ),
    ]
    for source, destination in reports:
        if not source.is_file():
            raise FileNotFoundError(f"Required report is missing: {source}")
        shutil.copy2(source, destination)

    inference_source = Path(__file__).with_name("artifact_inference.py")
    shutil.copy2(inference_source, artifact_root / "inference.py")
    (artifact_root / "README.md").write_text(
        "# PathSnap model artifacts\n\n"
        "The recommended model is `models/random_forest_bayesian_tuned.joblib`. "
        "The folder also includes the five comparison model families, a baseline "
        "Random Forest, the feature schema, validation metrics, Bayesian-search "
        "results, and HMM/Viterbi route outputs.\n\n"
        "Each `.joblib` file is a bundle containing the estimator, scaler, feature "
        "column order, threshold, and (for Linear SVM) its training-only probability "
        "calibrator. Load and score candidate rows with:\n\n"
        "```python\n"
        "import pandas as pd\n"
        "from inference import load_model, predict_candidate_probabilities\n"
        "\n"
        "model = load_model('models/random_forest_bayesian_tuned.joblib')\n"
        "candidates = pd.read_csv('candidate_rows.csv')\n"
        "scores = predict_candidate_probabilities(model, candidates)\n"
        "```\n\n"
        "The `scores` array contains positive-class probabilities aligned to the "
        "input rows. Candidate feature generation must follow `feature_schema.json`. "
        "Artifacts were fit only on the training partition; student-test data was "
        "not used for fitting or model selection.\n",
        encoding="utf-8",
    )

    files = sorted(
        path
        for path in artifact_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest: dict[str, Any] = {
        "artifact_version": 1,
        "recommended_model": "models/random_forest_bayesian_tuned.joblib",
        "models": artifact_records,
        "feature_schema": "feature_schema.json",
        "training_data": {
            "partition": "train",
            "assigned_trip_count": len(training_ids),
            "contributing_trip_count": len(arrays["contributing_trips"]),
            "fit_observations_per_trip_cap": observations_per_trip,
            "fit_observations": int(arrays["fit_observations"]),
            "svm_fit_observations": int(arrays["svm_fit_observations"]),
            "svm_calibration_observations": int(arrays["calibration_observations"]),
            "knn_observations_per_trip_cap": knn_observations_per_trip,
            "knn_fit_observations": int(arrays["knn_observations"]),
            "student_test_used_for_training": False,
        },
        "inference": {
            "feature_columns": feature_columns,
            "threshold": 0.5,
            "loader": "inference.py:load_model",
            "helper": "inference.py:predict_candidate_probabilities",
        },
        "validation_comparison_note": (
            "five_model_validation_comparison.csv reports the original 64-train/16-validation "
            "comparison; exported baseline artifacts are refit on the full training partition."
        ),
        "runtime_versions": {
            "scikit_learn": version("scikit-learn"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "xgboost": version("xgboost"),
        },
        "files": [_file_record(path, artifact_root) for path in files],
    }
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(artifact_records)} model artifacts to {artifact_root}")
    return manifest_path, manifest
