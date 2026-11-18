"""Train and compare candidate-road classifiers using trip-held-out data."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def _read_candidates(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=columns)
    if "supervision_usable" in frame:
        frame["supervision_usable"] = frame["supervision_usable"].astype(bool)
    if "is_target" in frame:
        frame["is_target"] = frame["is_target"].astype(bool)
    for column in (
        "label_confident",
        "label_unambiguous",
        "candidate_label_found",
        "route_geometry_suspect",
        "route_sequence_suspect",
        "route_sequence_consistent",
    ):
        if column in frame:
            frame[column] = frame[column].astype(bool)
    return frame


def _probabilities(
    model, frame: pd.DataFrame, scaler: StandardScaler, features: list[str]
):
    x = scaler.transform(frame[features].to_numpy(dtype=np.float32, copy=True))
    if model["name"] == "Linear SVM":
        decision = model["estimator"].decision_function(x).reshape(-1, 1)
        return model["calibrator"].predict_proba(decision)[:, 1]
    return model["estimator"].predict_proba(x)[:, 1]


def _route_scores(
    predictions: pd.DataFrame,
    arc_endpoints: dict[str, dict[int, tuple[int, int]]],
    route_positions: dict[str, dict[int, int]],
) -> tuple[float, float, float]:
    continuity_numerator = 0
    continuity_denominator = 0
    order_numerator = 0
    order_denominator = 0

    for trip_id, group in predictions.groupby("trip_id", sort=False):
        group = group.sort_values("observation_id", kind="stable")
        observation_ids = group["observation_id"].to_numpy(dtype=int)
        arc_ids = group["candidate_arc_id"].to_numpy(dtype=int)
        endpoints = arc_endpoints[trip_id]
        route_index = route_positions[trip_id]
        links = [endpoints.get(int(arc_id)) for arc_id in arc_ids]
        positions = np.array(
            [route_index.get(int(arc_id), -1) for arc_id in arc_ids], dtype=int
        )

        for index in range(1, len(group)):
            previous_link = links[index - 1]
            current_link = links[index]
            if observation_ids[index] == observation_ids[index - 1] + 1:
                continuity_denominator += 1
                if (
                    previous_link
                    and current_link
                    and (
                        arc_ids[index - 1] == arc_ids[index]
                        or previous_link[1] == current_link[0]
                    )
                ):
                    continuity_numerator += 1
            if positions[index - 1] >= 0 and positions[index] >= 0:
                order_denominator += 1
                if positions[index] >= positions[index - 1]:
                    order_numerator += 1

    route_continuity = (
        continuity_numerator / continuity_denominator
        if continuity_denominator
        else float("nan")
    )
    sequence_order = (
        order_numerator / order_denominator if order_denominator else float("nan")
    )
    trip_exact = (
        predictions.groupby("trip_id", sort=False)["is_correct"].all().mean()
        if len(predictions)
        else float("nan")
    )
    return route_continuity, sequence_order, float(trip_exact)


def train_candidate_models(
    feature_root: str | Path,
    cleaned_root: str | Path,
    output_root: str | Path,
    *,
    max_train_observations_per_trip: int = 500,
    calibration_fraction: float = 0.15,
    knn_observations_per_trip: int = 20,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, str, Path, dict]:
    """Train five classifiers and compare them on the saved validation trips."""
    from xgboost import XGBClassifier

    feature_root = Path(feature_root)
    cleaned_root = Path(cleaned_root)
    output_root = Path(output_root)
    schema = json.loads(
        (feature_root / "feature_schema.json").read_text(encoding="utf-8")
    )
    features = schema["feature_columns"]
    audit = pd.read_csv(
        feature_root / "feature_quality_audit.csv", dtype={"trip_id": str}
    )
    training = audit.loc[(audit["partition"] == "train") & (audit["split"] == "train")]
    validation = audit.loc[
        (audit["partition"] == "train") & (audit["split"] == "validation")
    ]
    training_ids = set(training["trip_id"])
    validation_ids = set(validation["trip_id"])
    if not training_ids or not validation_ids:
        raise ValueError("Both trip-level training and validation splits are required.")
    if not training_ids.isdisjoint(validation_ids):
        raise ValueError("A trip appears in both training and validation splits.")
    if not 0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between 0 and 0.5.")

    rng = np.random.default_rng(random_seed)
    train_columns = features + [
        "observation_id",
        "supervision_usable",
        "is_target",
    ]
    fit_x_parts = []
    fit_y_parts = []
    calibration_x_parts = []
    calibration_y_parts = []
    knn_x_parts = []
    knn_y_parts = []
    fit_observation_count = 0
    calibration_observation_count = 0
    knn_observation_count = 0

    for trip_id in sorted(training_ids):
        path = feature_root / "train" / trip_id / "candidates.csv.gz"
        frame = _read_candidates(path, train_columns)
        frame = frame.loc[frame["supervision_usable"]]
        observation_ids = frame["observation_id"].drop_duplicates().to_numpy()
        if not len(observation_ids):
            continue
        selected_ids = np.sort(
            rng.choice(
                observation_ids,
                size=min(max_train_observations_per_trip, len(observation_ids)),
                replace=False,
            )
        )
        frame = frame.loc[frame["observation_id"].isin(selected_ids)]
        selected_ids = np.sort(frame["observation_id"].unique())
        calibration_count = (
            max(1, int(round(len(selected_ids) * calibration_fraction)))
            if len(selected_ids) >= 10
            else 0
        )
        shuffled_ids = rng.permutation(selected_ids)
        calibration_ids = set(shuffled_ids[:calibration_count])
        knn_ids = set(
            rng.choice(
                shuffled_ids[calibration_count:],
                size=min(
                    knn_observations_per_trip, len(shuffled_ids) - calibration_count
                ),
                replace=False,
            )
        )
        calibration_mask = frame["observation_id"].isin(calibration_ids).to_numpy()
        fit_mask = ~calibration_mask
        x = frame[features].to_numpy(dtype=np.float32)
        y = frame["is_target"].to_numpy(dtype=np.uint8)
        fit_x_parts.append(x[fit_mask])
        fit_y_parts.append(y[fit_mask])
        calibration_x_parts.append(x[calibration_mask])
        calibration_y_parts.append(y[calibration_mask])
        fit_observation_count += len(selected_ids) - calibration_count
        calibration_observation_count += calibration_count

        knn_mask = fit_mask & frame["observation_id"].isin(knn_ids).to_numpy()
        knn_x_parts.append(x[knn_mask])
        knn_y_parts.append(y[knn_mask])
        knn_observation_count += len(knn_ids)
        del frame, x, y

    if not fit_x_parts or not calibration_x_parts:
        raise ValueError(
            "No labeled candidate examples were available for model fitting."
        )
    x_fit = np.concatenate(fit_x_parts)
    y_fit = np.concatenate(fit_y_parts)
    x_calibration = np.concatenate(calibration_x_parts)
    y_calibration = np.concatenate(calibration_y_parts)
    x_knn = np.concatenate(knn_x_parts)
    y_knn = np.concatenate(knn_y_parts)
    if np.unique(y_fit).size != 2 or np.unique(y_calibration).size != 2:
        raise ValueError("Training and calibration data must contain both classes.")
    del fit_x_parts, fit_y_parts, calibration_x_parts, calibration_y_parts
    del knn_x_parts, knn_y_parts

    scaler = StandardScaler(copy=False)
    x_fit = scaler.fit_transform(x_fit).astype(np.float32, copy=False)
    x_calibration = scaler.transform(x_calibration).astype(np.float32, copy=False)
    x_knn = scaler.transform(x_knn).astype(np.float32, copy=False)

    validation_columns = features + [
        "observation_id",
        "candidate_arc_id",
        "supervision_usable",
        "is_target",
        "label_confident",
        "label_unambiguous",
        "candidate_label_found",
        "route_geometry_suspect",
        "route_sequence_suspect",
        "route_sequence_consistent",
    ]
    validation_frames = []
    arc_endpoints = {}
    route_positions = {}
    candidate_misses = 0
    reliable_reference_count = 0
    for trip_id in sorted(validation_ids):
        path = feature_root / "train" / trip_id / "candidates.csv.gz"
        frame = _read_candidates(path, validation_columns)
        observation_metadata = frame.drop_duplicates("observation_id")
        reliable_reference = (
            observation_metadata["label_confident"]
            & observation_metadata["label_unambiguous"]
            & observation_metadata["route_sequence_consistent"]
            & ~observation_metadata["route_geometry_suspect"]
            & ~observation_metadata["route_sequence_suspect"]
        )
        reliable_reference_count += int(reliable_reference.sum())
        candidate_misses += int(
            (reliable_reference & ~observation_metadata["candidate_label_found"]).sum()
        )

        usable = frame.loc[frame["supervision_usable"]].copy()
        if not usable.empty:
            usable["trip_id"] = trip_id
            validation_frames.append(
                usable[
                    features
                    + [
                        "trip_id",
                        "observation_id",
                        "candidate_arc_id",
                        "is_target",
                    ]
                ].reset_index(drop=True)
            )

        arc_table = pd.read_csv(
            cleaned_root / "train" / trip_id / "arcs.csv",
            usecols=["arc_id", "from_node", "to_node"],
        )
        arc_endpoints[trip_id] = {
            int(row.arc_id): (int(row.from_node), int(row.to_node))
            for row in arc_table.itertuples(index=False)
        }
        route_ids = pd.read_csv(
            cleaned_root / "train" / trip_id / "route.csv",
            usecols=["arc_id"],
        )["arc_id"].to_numpy(dtype=int)
        route_positions[trip_id] = {}
        for index, arc_id in enumerate(route_ids):
            route_positions[trip_id].setdefault(int(arc_id), index)
        del frame, observation_metadata, usable, arc_table

    if not validation_frames:
        raise ValueError("No usable labeled observations were found in validation.")
    validation_observations = sum(
        frame["observation_id"].nunique() for frame in validation_frames
    )

    models = [
        (
            "Logistic Regression",
            lambda: LogisticRegression(C=1.0, max_iter=200, solver="lbfgs"),
            False,
        ),
        (
            "Random Forest",
            lambda: RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_leaf=10,
                max_features="sqrt",
                n_jobs=2,
                random_state=random_seed,
            ),
            False,
        ),
        (
            "Linear SVM",
            lambda: LinearSVC(
                C=0.25,
                dual="auto",
                max_iter=3000,
                random_state=random_seed,
            ),
            True,
        ),
        (
            "XGBoost",
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
            False,
        ),
        (
            "KNN",
            lambda: KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                algorithm="kd_tree",
                leaf_size=40,
                n_jobs=2,
            ),
            False,
        ),
    ]

    result_rows = []
    best_key = None
    best_name = ""
    best_bundle = None
    for name, model_factory, needs_calibration in models:
        estimator = model_factory()
        x_model = x_knn if name == "KNN" else x_fit
        y_model = y_knn if name == "KNN" else y_fit
        calibrator = None
        start = time.perf_counter()
        estimator.fit(x_model, y_model)
        if needs_calibration:
            calibration_scores = estimator.decision_function(x_calibration).reshape(
                -1, 1
            )
            calibrator = LogisticRegression(max_iter=200, solver="lbfgs")
            calibrator.fit(calibration_scores, y_calibration)
        training_seconds = time.perf_counter() - start

        model = {
            "name": name,
            "estimator": estimator,
            "calibrator": calibrator,
        }
        y_true_parts = []
        probability_parts = []
        top_prediction_parts = []
        inference_seconds = 0.0
        for frame in validation_frames:
            start = time.perf_counter()
            probabilities = _probabilities(model, frame, scaler, features)
            inference_seconds += time.perf_counter() - start
            y_true_parts.append(frame["is_target"].to_numpy(dtype=np.uint8))
            probability_parts.append(probabilities.astype(np.float32, copy=False))

            score_frame = pd.DataFrame(
                {
                    "observation_id": frame["observation_id"].to_numpy(),
                    "score": probabilities,
                }
            )
            top_indices = (
                score_frame.groupby("observation_id", sort=False)["score"]
                .idxmax()
                .to_numpy(dtype=int)
            )
            top = frame.iloc[top_indices][
                ["trip_id", "observation_id", "candidate_arc_id", "is_target"]
            ].copy()
            top["is_correct"] = top["is_target"].astype(bool)
            top_prediction_parts.append(top)

        y_true = np.concatenate(y_true_parts)
        probabilities = np.concatenate(probability_parts).astype(float)
        y_pred = probabilities >= 0.5
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        top_predictions = pd.concat(top_prediction_parts, ignore_index=True)
        route_continuity, sequence_order, sequence_exact = _route_scores(
            top_predictions, arc_endpoints, route_positions
        )
        row = {
            "model": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": matrix.tolist(),
            "log_loss": log_loss(
                y_true, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1]
            ),
            "observation_top1_accuracy": float(top_predictions["is_correct"].mean()),
            "candidate_miss_rate": (
                candidate_misses / reliable_reference_count
                if reliable_reference_count
                else float("nan")
            ),
            "candidate_miss_count": candidate_misses,
            "reliable_reference_observations": reliable_reference_count,
            "route_continuity": route_continuity,
            "sequence_order_consistency": sequence_order,
            "sequence_exact_trip_rate": sequence_exact,
            "evaluated_observations": validation_observations,
            "evaluated_candidate_rows": len(y_true),
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "training_observations": fit_observation_count,
            "calibration_observations": calibration_observation_count,
            "knn_training_observations": (
                knn_observation_count if name == "KNN" else np.nan
            ),
            "classification_threshold": 0.5,
        }
        result_rows.append(row)
        selection_key = (
            row["observation_top1_accuracy"],
            row["route_continuity"] if np.isfinite(row["route_continuity"]) else -1.0,
            row["f1"],
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_name = name
            best_bundle = {
                "model_name": name,
                "estimator": estimator,
                "scaler": scaler,
                "calibrator": calibrator,
                "feature_columns": features,
                "positive_class": True,
                "classification_threshold": 0.5,
            }
        print(
            f"{name}: top-1={row['observation_top1_accuracy']:.4f}, "
            f"F1={row['f1']:.4f}, fit={training_seconds:.1f}s, "
            f"inference={inference_seconds:.1f}s"
        )
        del model, estimator, calibrator
        gc.collect()

    results = pd.DataFrame(result_rows).sort_values(
        ["observation_top1_accuracy", "route_continuity", "f1"],
        ascending=False,
        kind="stable",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_root / "model_comparison.csv", index=False)
    best_model_path = output_root / "best_model.joblib"
    joblib.dump(best_bundle, best_model_path)
    run_summary = {
        "selection_metric": "observation_top1_accuracy",
        "selection_reason": (
            "It directly measures whether the highest-scored road candidate "
            "matches the reference segment for each GPS observation."
        ),
        "best_model": best_name,
        "validation_trip_count": len(validation_ids),
        "training_trip_count": len(training_ids),
        "max_train_observations_per_trip": max_train_observations_per_trip,
        "training_observations": fit_observation_count,
        "calibration_observations": calibration_observation_count,
        "knn_observations_per_trip": knn_observations_per_trip,
        "validation_observations": validation_observations,
        "validation_candidate_rows": int(len(y_true)),
        "reliable_candidate_references": reliable_reference_count,
        "candidate_misses": candidate_misses,
        "best_model_path": str(best_model_path),
        "student_test_used_for_selection": False,
    }
    (output_root / "model_selection.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    return results, best_name, best_model_path, run_summary
