"""Bayesian tuning and HMM/Viterbi route decoding for candidate-road models."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, qmc
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from .candidate_models import _read_candidates


_SEARCH_BOUNDS = np.array(
    [
        [40.0, 80.0],
        [6.0, 18.0],
        [2.0, 25.0],
        [0.2, 1.0],
        [0.65, 1.0],
    ],
    dtype=float,
)


def _decode_search_point(point: np.ndarray) -> dict[str, int | float]:
    values = _SEARCH_BOUNDS[:, 0] + point * (
        _SEARCH_BOUNDS[:, 1] - _SEARCH_BOUNDS[:, 0]
    )
    return {
        "n_estimators": int(round(values[0])),
        "max_depth": int(round(values[1])),
        "min_samples_leaf": int(round(values[2])),
        "max_features": round(float(values[3]), 4),
        "max_samples": round(float(values[4]), 4),
    }


def _encode_search_point(params: dict[str, int | float]) -> np.ndarray:
    values = np.array(
        [
            params["n_estimators"],
            params["max_depth"],
            params["min_samples_leaf"],
            params["max_features"],
            params["max_samples"],
        ],
        dtype=float,
    )
    return (values - _SEARCH_BOUNDS[:, 0]) / (
        _SEARCH_BOUNDS[:, 1] - _SEARCH_BOUNDS[:, 0]
    )


def _search_key(params: dict[str, int | float]) -> tuple[int | float, ...]:
    return (
        int(params["n_estimators"]),
        int(params["max_depth"]),
        int(params["min_samples_leaf"]),
        round(float(params["max_features"]), 4),
        round(float(params["max_samples"]), 4),
    )


def _fit_rows(
    feature_root: Path,
    trip_ids: list[str],
    feature_columns: list[str],
    *,
    observations_per_trip: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    rng = np.random.default_rng(random_seed)
    x_parts = []
    y_parts = []
    used_trip_count = 0
    observation_count = 0
    columns = feature_columns + [
        "observation_id",
        "supervision_usable",
        "is_target",
    ]

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
        x_parts.append(sampled[feature_columns].to_numpy(dtype=np.float32))
        y_parts.append(sampled["is_target"].to_numpy(dtype=np.uint8))
        used_trip_count += 1
        observation_count += len(selected_ids)
        del frame, sampled

    if not x_parts:
        raise ValueError("No usable labeled observations were found for fitting.")
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    if np.unique(y).size != 2:
        raise ValueError("Training candidates must contain both target classes.")
    return x, y, observation_count, used_trip_count


def _validation_rows(
    feature_root: Path,
    trip_ids: list[str],
    feature_columns: list[str],
) -> list[pd.DataFrame]:
    columns = feature_columns + [
        "observation_id",
        "candidate_arc_id",
        "supervision_usable",
        "is_target",
    ]
    frames = []
    for trip_id in trip_ids:
        path = feature_root / "train" / trip_id / "candidates.csv.gz"
        frame = _read_candidates(path, columns)
        frame = frame.loc[frame["supervision_usable"]].copy()
        if frame.empty:
            continue
        frame["trip_id"] = trip_id
        frames.append(frame.reset_index(drop=True))
    if not frames:
        raise ValueError("No usable validation labels were found.")
    return frames


def _validation_metrics(
    estimator: RandomForestClassifier,
    scaler: StandardScaler,
    frames: list[pd.DataFrame],
    feature_columns: list[str],
) -> tuple[dict[str, float], float]:
    y_parts = []
    probability_parts = []
    top1_correct = []
    inference_seconds = 0.0
    for frame in frames:
        values = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
        start = time.perf_counter()
        probabilities = estimator.predict_proba(scaler.transform(values))[:, 1]
        inference_seconds += time.perf_counter() - start
        y = frame["is_target"].to_numpy(dtype=np.uint8)
        y_parts.append(y)
        probability_parts.append(probabilities.astype(np.float32, copy=False))
        score_frame = pd.DataFrame(
            {
                "observation_id": frame["observation_id"].to_numpy(),
                "score": probabilities,
            }
        )
        best_indices = (
            score_frame.groupby("observation_id", sort=False)["score"]
            .idxmax()
            .to_numpy(dtype=int)
        )
        top1_correct.extend(
            frame.iloc[best_indices]["is_target"].to_numpy(dtype=bool).tolist()
        )

    y_true = np.concatenate(y_parts)
    probabilities = np.concatenate(probability_parts).astype(float)
    predicted = probabilities >= 0.5
    return (
        {
            "accuracy": float(accuracy_score(y_true, predicted)),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "log_loss": float(
                log_loss(
                    y_true,
                    np.clip(probabilities, 1e-7, 1 - 1e-7),
                    labels=[0, 1],
                )
            ),
            "observation_top1_accuracy": float(np.mean(top1_correct)),
            "evaluated_observations": int(
                sum(frame["observation_id"].nunique() for frame in frames)
            ),
            "evaluated_candidate_rows": int(len(y_true)),
        },
        inference_seconds,
    )


def optimize_random_forest(
    feature_root: str | Path,
    output_root: str | Path,
    *,
    chosen_model_name: str,
    observations_per_trip: int = 200,
    initial_points: int = 5,
    bayesian_iterations: int = 4,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int | float], Path, dict[str, Any]]:
    """Optimize Random Forest parameters using GP expected improvement."""
    if chosen_model_name != "Random Forest":
        raise ValueError(
            "Bayesian tuning is scoped to the previously selected Random Forest."
        )
    if initial_points < 2 or bayesian_iterations < 1:
        raise ValueError("Use at least two initial points and one Bayesian iteration.")

    feature_root = Path(feature_root)
    output_root = Path(output_root)
    schema = json.loads(
        (feature_root / "feature_schema.json").read_text(encoding="utf-8")
    )
    feature_columns = schema["feature_columns"]
    audit = pd.read_csv(
        feature_root / "feature_quality_audit.csv", dtype={"trip_id": str}
    )
    training_ids = sorted(
        audit.loc[
            (audit["partition"] == "train") & (audit["split"] == "train"), "trip_id"
        ].unique()
    )
    validation_ids = sorted(
        audit.loc[
            (audit["partition"] == "train") & (audit["split"] == "validation"),
            "trip_id",
        ].unique()
    )
    if not training_ids or not validation_ids:
        raise ValueError("A trip-separated training and validation split is required.")
    if set(training_ids).intersection(validation_ids):
        raise ValueError("Training and validation trips must be disjoint.")

    x_train, y_train, sampled_observations, sampled_trips = _fit_rows(
        feature_root,
        training_ids,
        feature_columns,
        observations_per_trip=observations_per_trip,
        random_seed=random_seed,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    validation_frames = _validation_rows(feature_root, validation_ids, feature_columns)
    rng = np.random.default_rng(random_seed)
    latin_points = qmc.LatinHypercube(d=len(_SEARCH_BOUNDS), seed=random_seed).random(
        max(initial_points - 1, 0)
    )
    baseline = {
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_leaf": 10,
        "max_features": round(
            math.sqrt(len(feature_columns)) / len(feature_columns), 4
        ),
        "max_samples": 1.0,
    }
    initial_configs = [baseline]
    initial_configs.extend(_decode_search_point(point) for point in latin_points)

    configs: list[dict[str, int | float]] = []
    scores: list[float] = []
    records: list[dict[str, Any]] = []
    seen: set[tuple[int | float, ...]] = set()
    best_key: tuple[float, float] | None = None
    best_bundle: dict[str, Any] | None = None
    total_trials = initial_points + bayesian_iterations

    for trial_index in range(total_trials):
        if trial_index < len(initial_configs):
            params = initial_configs[trial_index]
            key = _search_key(params)
            while key in seen:
                params = _decode_search_point(rng.random(len(_SEARCH_BOUNDS)))
                key = _search_key(params)
        else:
            gp = GaussianProcessRegressor(
                kernel=(
                    ConstantKernel(1.0, (0.1, 10.0))
                    * Matern(length_scale=np.ones(len(_SEARCH_BOUNDS)), nu=2.5)
                    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2))
                ),
                alpha=1e-5,
                normalize_y=True,
                n_restarts_optimizer=1,
                random_state=random_seed,
            )
            x_observed = np.vstack([_encode_search_point(item) for item in configs])
            gp.fit(x_observed, np.asarray(scores))
            pool = rng.random((512, len(_SEARCH_BOUNDS)))
            pool_params = [_decode_search_point(point) for point in pool]
            unique_params = []
            unique_points = []
            pool_seen = set(seen)
            for point, candidate in zip(pool, pool_params, strict=True):
                candidate_key = _search_key(candidate)
                if candidate_key not in pool_seen:
                    pool_seen.add(candidate_key)
                    unique_params.append(candidate)
                    unique_points.append(point)
            if not unique_params:
                raise RuntimeError("Bayesian search exhausted its candidate pool.")
            candidate_points = np.vstack(
                [_encode_search_point(item) for item in unique_params]
            )
            mean, standard_deviation = gp.predict(candidate_points, return_std=True)
            standard_deviation = np.maximum(standard_deviation, 1e-9)
            improvement = mean - max(scores) - 0.001
            z_score = improvement / standard_deviation
            expected_improvement = improvement * norm.cdf(
                z_score
            ) + standard_deviation * norm.pdf(z_score)
            next_index = int(np.argmax(expected_improvement))
            params = unique_params[next_index]
            key = _search_key(params)

        seen.add(key)
        estimator = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=float(params["max_features"]),
            max_samples=float(params["max_samples"]),
            n_jobs=2,
            random_state=random_seed,
        )
        start = time.perf_counter()
        estimator.fit(x_train, y_train)
        training_seconds = time.perf_counter() - start
        metrics, inference_seconds = _validation_metrics(
            estimator, scaler, validation_frames, feature_columns
        )
        objective = metrics["observation_top1_accuracy"]
        configs.append(params)
        scores.append(objective)
        row = {
            "trial": trial_index + 1,
            **params,
            **metrics,
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "objective": objective,
        }
        records.append(row)
        trial_key = (objective, metrics["f1"])
        if best_key is None or trial_key > best_key:
            best_key = trial_key
            best_bundle = {
                "model_name": "Random Forest",
                "estimator": estimator,
                "scaler": scaler,
                "calibrator": None,
                "feature_columns": feature_columns,
                "best_params": params,
                "classification_threshold": 0.5,
                "fit_observations": sampled_observations,
                "fit_trips": sampled_trips,
            }
        print(
            f"Trial {trial_index + 1}/{total_trials}: "
            f"Top-1={objective:.4f}, F1={metrics['f1']:.4f}, "
            f"params={params}, fit={training_seconds:.1f}s"
        )
        if best_bundle is None or best_bundle["estimator"] is not estimator:
            del estimator

    if best_bundle is None:
        raise RuntimeError("Bayesian optimization did not produce a fitted model.")
    results = pd.DataFrame(records).sort_values(
        ["observation_top1_accuracy", "f1"],
        ascending=False,
        kind="stable",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_root / "bayesian_optimization_results.csv", index=False)
    validation_model_path = output_root / "bayesian_validation_model.joblib"
    joblib.dump(best_bundle, validation_model_path)
    best_params = dict(best_bundle["best_params"])
    summary: dict[str, Any] = {
        "chosen_model": "Random Forest",
        "optimization_method": "Gaussian-process Bayesian optimization with expected improvement",
        "selection_metric": "observation_top1_accuracy",
        "selection_reason": (
            "Maximize the fraction of labeled validation observations whose "
            "highest-scored candidate matches the reference road arc."
        ),
        "best_params": best_params,
        "best_validation_top1_accuracy": float(best_key[0]),
        "initial_points": initial_points,
        "bayesian_iterations": bayesian_iterations,
        "trial_count": len(records),
        "training_trip_count": len(training_ids),
        "validation_trip_count": len(validation_ids),
        "sampled_training_trip_count": sampled_trips,
        "sampled_training_observations": sampled_observations,
        "observations_per_training_trip_cap": observations_per_trip,
        "validation_observations": int(
            sum(frame["observation_id"].nunique() for frame in validation_frames)
        ),
        "student_test_used_for_optimization": False,
        "validation_model_path": str(validation_model_path),
    }
    (output_root / "bayesian_optimization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return results, best_params, validation_model_path, summary


def refit_tuned_random_forest(
    feature_root: str | Path,
    output_root: str | Path,
    best_params: dict[str, int | float],
    *,
    observations_per_trip: int = 500,
    random_seed: int = 42,
) -> tuple[Path, dict[str, Any]]:
    """Refit the tuned model on all usable training-partition trips."""
    feature_root = Path(feature_root)
    output_root = Path(output_root)
    schema = json.loads(
        (feature_root / "feature_schema.json").read_text(encoding="utf-8")
    )
    feature_columns = schema["feature_columns"]
    audit = pd.read_csv(
        feature_root / "feature_quality_audit.csv", dtype={"trip_id": str}
    )
    training_ids = sorted(audit.loc[audit["partition"] == "train", "trip_id"].unique())
    x, y, observation_count, used_trip_count = _fit_rows(
        feature_root,
        training_ids,
        feature_columns,
        observations_per_trip=observations_per_trip,
        random_seed=random_seed,
    )
    scaler = StandardScaler()
    x = scaler.fit_transform(x).astype(np.float32, copy=False)
    estimator = RandomForestClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        max_features=float(best_params["max_features"]),
        max_samples=float(best_params["max_samples"]),
        n_jobs=2,
        random_state=random_seed,
    )
    start = time.perf_counter()
    estimator.fit(x, y)
    training_seconds = time.perf_counter() - start
    bundle = {
        "model_name": "Random Forest",
        "estimator": estimator,
        "scaler": scaler,
        "calibrator": None,
        "feature_columns": feature_columns,
        "best_params": best_params,
        "classification_threshold": 0.5,
        "final_fit_trips": used_trip_count,
        "final_fit_observations": observation_count,
        "observations_per_trip_cap": observations_per_trip,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / "bayesian_tuned_random_forest.joblib"
    joblib.dump(bundle, model_path)
    summary = {
        "model_name": "Random Forest",
        "best_params": best_params,
        "training_partition_trip_count": len(training_ids),
        "contributing_training_trip_count": used_trip_count,
        "fit_observations": observation_count,
        "observations_per_trip_cap": observations_per_trip,
        "training_seconds": training_seconds,
        "student_test_used_for_training": False,
        "model_path": str(model_path),
    }
    (output_root / "bayesian_refit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return model_path, summary


def _road_graph(
    cleaned_trip_dir: Path,
    *,
    include_reference_route: bool = True,
) -> dict[str, Any]:
    nodes = pd.read_csv(
        cleaned_trip_dir / "nodes.csv",
        usecols=["node_id", "longitude", "latitude"],
    )
    arcs = pd.read_csv(
        cleaned_trip_dir / "arcs.csv",
        usecols=["arc_id", "from_node", "to_node"],
    )
    track = pd.read_csv(cleaned_trip_dir / "track.csv", usecols=["latitude"])
    node_positions = {
        int(node_id): position
        for position, node_id in enumerate(nodes["node_id"].to_numpy())
    }
    source_positions = arcs["from_node"].map(node_positions)
    target_positions = arcs["to_node"].map(node_positions)
    if source_positions.isna().any() or target_positions.isna().any():
        raise ValueError(f"Arc references a missing node in {cleaned_trip_dir}.")
    source_positions = source_positions.to_numpy(dtype=int)
    target_positions = target_positions.to_numpy(dtype=int)
    lonlat = nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    reference_latitude = float(track["latitude"].mean())
    scale = np.array([111_320.0 * np.cos(np.radians(reference_latitude)), 111_320.0])
    xy = lonlat * scale
    arc_lengths = np.linalg.norm(xy[source_positions] - xy[target_positions], axis=1)
    arc_ids = arcs["arc_id"].to_numpy(dtype=int)
    from_nodes = arcs["from_node"].to_numpy(dtype=int)
    to_nodes = arcs["to_node"].to_numpy(dtype=int)
    endpoints = {
        int(arc_id): (int(start), int(end))
        for arc_id, start, end in zip(arc_ids, from_nodes, to_nodes, strict=True)
    }
    lengths = {
        int(arc_id): float(length)
        for arc_id, length in zip(arc_ids, arc_lengths, strict=True)
    }
    outgoing: dict[int, list[tuple[int, float, int]]] = {}
    for arc_id, start, end, length in zip(
        arc_ids, from_nodes, to_nodes, arc_lengths, strict=True
    ):
        outgoing.setdefault(int(start), []).append(
            (int(end), float(length), int(arc_id))
        )
    for edges in outgoing.values():
        edges.sort(key=lambda edge: (edge[1], edge[2]))

    route_path = cleaned_trip_dir / "route.csv"
    if include_reference_route:
        route_ids = pd.read_csv(route_path, usecols=["arc_id"])["arc_id"].to_numpy(
            dtype=int
        )
    else:
        route_ids = np.empty(0, dtype=int)
    route_positions: dict[int, int] = {}
    for index, arc_id in enumerate(route_ids):
        route_positions.setdefault(int(arc_id), index)
    return {
        "endpoints": endpoints,
        "lengths": lengths,
        "outgoing": outgoing,
        "route_positions": route_positions,
        "route_ids": route_ids,
    }


def _transition_scores(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    graph: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    previous_arc_ids = previous["candidate_arc_id"].to_numpy(dtype=int)
    current_arc_ids = current["candidate_arc_id"].to_numpy(dtype=int)
    previous_to = np.array(
        [graph["endpoints"][arc_id][1] for arc_id in previous_arc_ids], dtype=int
    )
    current_from = np.array(
        [graph["endpoints"][arc_id][0] for arc_id in current_arc_ids], dtype=int
    )
    same_arc = previous_arc_ids[:, None] == current_arc_ids[None, :]
    directly_connected = previous_to[:, None] == current_from[None, :]

    current_indices_by_start: dict[int, list[int]] = {}
    for index, node_id in enumerate(current_from):
        current_indices_by_start.setdefault(int(node_id), []).append(index)
    intermediate_ids = np.full(
        (len(previous_arc_ids), len(current_arc_ids)), -1, dtype=int
    )
    intermediate_lengths = np.full(intermediate_ids.shape, np.inf, dtype=float)
    for previous_index, end_node in enumerate(previous_to):
        for next_node, length, arc_id in graph["outgoing"].get(int(end_node), []):
            for current_index in current_indices_by_start.get(next_node, []):
                if (
                    same_arc[previous_index, current_index]
                    or directly_connected[previous_index, current_index]
                ):
                    continue
                old_length = intermediate_lengths[previous_index, current_index]
                old_arc_id = intermediate_ids[previous_index, current_index]
                if (length, arc_id) < (old_length, old_arc_id):
                    intermediate_lengths[previous_index, current_index] = length
                    intermediate_ids[previous_index, current_index] = arc_id

    previous_fraction = previous["projection_fraction"].to_numpy(dtype=float)
    current_fraction = current["projection_fraction"].to_numpy(dtype=float)
    previous_lengths = np.array(
        [graph["lengths"][arc_id] for arc_id in previous_arc_ids], dtype=float
    )
    current_lengths = np.array(
        [graph["lengths"][arc_id] for arc_id in current_arc_ids], dtype=float
    )
    path_lengths = np.zeros(intermediate_ids.shape, dtype=float)
    same_distance = (
        np.abs(current_fraction[None, :] - previous_fraction[:, None])
        * previous_lengths[:, None]
    )
    direct_distance = (1 - previous_fraction[:, None]) * previous_lengths[
        :, None
    ] + current_fraction[None, :] * current_lengths[None, :]
    intermediate_distance = (
        (1 - previous_fraction[:, None]) * previous_lengths[:, None]
        + intermediate_lengths
        + current_fraction[None, :] * current_lengths[None, :]
    )
    path_lengths[same_arc] = same_distance[same_arc]
    path_lengths[directly_connected & ~same_arc] = direct_distance[
        directly_connected & ~same_arc
    ]
    has_intermediate = (intermediate_ids >= 0) & ~same_arc & ~directly_connected
    path_lengths[has_intermediate] = intermediate_distance[has_intermediate]

    observation_gap = max(
        1,
        int(current["observation_id"].iloc[0])
        - int(previous["observation_id"].iloc[0]),
    )
    gps_distance = max(float(current["gps_step_distance_m"].iloc[0]), 0.0)
    beta = max(20.0, 0.25 * gps_distance)
    transition = np.full(
        intermediate_ids.shape, -8.0 / math.sqrt(observation_gap), dtype=float
    )
    connected = same_arc | directly_connected | has_intermediate
    transition[connected] = -np.abs(path_lengths[connected] - gps_distance) / beta
    moving_backwards = same_arc & (
        current_fraction[None, :] + 0.02 < previous_fraction[:, None]
    )
    transition[moving_backwards] -= 3.0
    connector_ids = intermediate_ids.copy()
    connector_ids[same_arc | directly_connected] = -1
    return transition, connector_ids, connected


def _decode_viterbi(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    graph: dict[str, Any],
) -> pd.DataFrame:
    if len(frame) != len(probabilities):
        raise ValueError("Candidate rows and emission scores must have equal length.")
    ordered = frame.reset_index(drop=True).copy()
    ordered["emission_probability"] = np.asarray(probabilities, dtype=float)
    grouped = [
        (int(observation_id), group)
        for observation_id, group in ordered.groupby(
            "observation_id", sort=True, observed=True
        )
    ]
    if not grouped:
        raise ValueError("Viterbi decoding requires at least one observation.")

    emission_logs = []
    for _, group in grouped:
        values = np.log(
            np.clip(group["emission_probability"].to_numpy(dtype=float), 1e-9, 1.0)
        )
        emission_logs.append(values - logsumexp(values))

    backpointers: list[np.ndarray] = []
    connector_history: list[np.ndarray] = []
    connectivity_history: list[np.ndarray] = []
    scores = emission_logs[0]
    for index in range(1, len(grouped)):
        previous = grouped[index - 1][1]
        current = grouped[index][1]
        transition, connectors, connected = _transition_scores(previous, current, graph)
        joint = scores[:, None] + transition
        backpointer = np.argmax(joint, axis=0)
        scores = joint[backpointer, np.arange(len(current))] + emission_logs[index]
        backpointers.append(backpointer)
        connector_history.append(connectors)
        connectivity_history.append(connected)

    state = int(np.argmax(scores))
    states = [0] * len(grouped)
    states[-1] = state
    for index in range(len(grouped) - 1, 0, -1):
        states[index - 1] = int(backpointers[index - 1][states[index]])

    rows = []
    for index, ((observation_id, group), state_index) in enumerate(
        zip(grouped, states, strict=True)
    ):
        selected = group.iloc[state_index].to_dict()
        selected["observation_id"] = observation_id
        selected["connector_arc_id_before"] = -1
        selected["transition_connected"] = True
        if index:
            previous_state = states[index - 1]
            selected["connector_arc_id_before"] = int(
                connector_history[index - 1][previous_state, state_index]
            )
            selected["transition_connected"] = bool(
                connectivity_history[index - 1][previous_state, state_index]
            )
        rows.append(selected)
    return pd.DataFrame(rows)


def _route_sequence(predictions: pd.DataFrame) -> pd.DataFrame:
    route_rows = []
    for trip_id, group in predictions.groupby("trip_id", sort=False):
        group = group.sort_values("observation_id", kind="stable")
        sequence_index = 0
        previous_arc_id: int | None = None
        for row in group.itertuples(index=False):
            arc_id = int(row.viterbi_arc_id)
            connector_id = int(row.connector_arc_id_before)
            if previous_arc_id is None:
                route_rows.append(
                    {
                        "trip_id": trip_id,
                        "route_index": sequence_index,
                        "arc_id": arc_id,
                        "inserted_connector": False,
                        "at_observation_id": int(row.observation_id),
                    }
                )
                sequence_index += 1
                previous_arc_id = arc_id
                continue
            if connector_id >= 0 and connector_id != previous_arc_id:
                route_rows.append(
                    {
                        "trip_id": trip_id,
                        "route_index": sequence_index,
                        "arc_id": connector_id,
                        "inserted_connector": True,
                        "at_observation_id": int(row.observation_id),
                    }
                )
                sequence_index += 1
                previous_arc_id = connector_id
            if arc_id != previous_arc_id:
                route_rows.append(
                    {
                        "trip_id": trip_id,
                        "route_index": sequence_index,
                        "arc_id": arc_id,
                        "inserted_connector": False,
                        "at_observation_id": int(row.observation_id),
                    }
                )
                sequence_index += 1
                previous_arc_id = arc_id
    return pd.DataFrame(route_rows)


def reconstruct_candidate_route(
    candidates: pd.DataFrame,
    probabilities: np.ndarray,
    cleaned_trip_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode candidate scores without reading reference-route labels."""
    required = {
        "trip_id",
        "observation_id",
        "candidate_arc_id",
        "projection_fraction",
        "gps_step_distance_m",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate frame is missing columns: {sorted(missing)}")
    if candidates["trip_id"].nunique(dropna=False) != 1:
        raise ValueError("Candidate frame must contain exactly one trip.")
    scores = np.asarray(probabilities, dtype=float)
    if len(candidates) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Candidate rows and finite emission scores must align.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Emission probabilities must be between zero and one.")

    graph = _road_graph(Path(cleaned_trip_dir), include_reference_route=False)
    unknown_arcs = set(candidates["candidate_arc_id"].astype(int)).difference(
        graph["endpoints"]
    )
    if unknown_arcs:
        raise ValueError(
            f"Candidate arcs are absent from the road graph: {sorted(unknown_arcs)[:5]}"
        )
    decoded = _decode_viterbi(candidates, scores, graph)
    route_predictions = decoded.rename(columns={"candidate_arc_id": "viterbi_arc_id"})
    return decoded, _route_sequence(route_predictions)


def _continuity(
    group: pd.DataFrame,
    prediction_column: str,
    endpoints: dict[int, tuple[int, int]],
) -> tuple[int, int]:
    group = group.sort_values("observation_id", kind="stable")
    observation_ids = group["observation_id"].to_numpy(dtype=int)
    arc_ids = group[prediction_column].to_numpy(dtype=int)
    numerator = 0
    denominator = 0
    for index in range(1, len(group)):
        if observation_ids[index] != observation_ids[index - 1] + 1:
            continue
        denominator += 1
        previous = endpoints[arc_ids[index - 1]]
        current = endpoints[arc_ids[index]]
        if arc_ids[index - 1] == arc_ids[index] or previous[1] == current[0]:
            numerator += 1
    return numerator, denominator


def _sequence_order(
    group: pd.DataFrame,
    prediction_column: str,
    route_positions: dict[int, int],
) -> tuple[int, int]:
    group = group.sort_values("observation_id", kind="stable")
    arc_ids = group[prediction_column].to_numpy(dtype=int)
    positions = [route_positions.get(arc_id, -1) for arc_id in arc_ids]
    numerator = 0
    denominator = 0
    for previous, current in zip(positions, positions[1:]):
        if previous < 0 or current < 0:
            continue
        denominator += 1
        if current >= previous:
            numerator += 1
    return numerator, denominator


def evaluate_hmm_partition(
    feature_root: str | Path,
    cleaned_root: str | Path,
    model_bundle: dict[str, Any],
    output_root: str | Path,
    *,
    partition: str,
    split: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Decode candidate sequences and score labeled observations in one partition."""
    feature_root = Path(feature_root)
    cleaned_root = Path(cleaned_root)
    output_root = Path(output_root)
    if partition not in {"train", "student_test"}:
        raise ValueError("partition must be 'train' or 'student_test'.")
    if partition == "student_test" and split is not None:
        raise ValueError("A split selector is only valid for the train partition.")

    audit = pd.read_csv(
        feature_root / "feature_quality_audit.csv", dtype={"trip_id": str}
    )
    selected_audit = audit.loc[audit["partition"] == partition]
    if split is not None:
        selected_audit = selected_audit.loc[selected_audit["split"] == split]
    trip_ids = sorted(selected_audit["trip_id"].unique())
    if not trip_ids:
        raise ValueError(
            f"No trips found for partition={partition!r}, split={split!r}."
        )
    features = model_bundle["feature_columns"]
    columns = features + [
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
    prediction_frames = []
    route_graphs: dict[str, dict[str, Any]] = {}
    inference_seconds = 0.0
    decoding_seconds = 0.0
    candidate_misses = 0
    reliable_reference_count = 0

    for trip_id in trip_ids:
        data_dir = feature_root / partition / trip_id
        frame = _read_candidates(data_dir / "candidates.csv.gz", columns)
        observation_metadata = frame.drop_duplicates("observation_id")
        reliable = (
            observation_metadata["label_confident"]
            & observation_metadata["label_unambiguous"]
            & observation_metadata["route_sequence_consistent"]
            & ~observation_metadata["route_geometry_suspect"]
            & ~observation_metadata["route_sequence_suspect"]
        )
        reliable_reference_count += int(reliable.sum())
        candidate_misses += int(
            (reliable & ~observation_metadata["candidate_label_found"]).sum()
        )

        graph = _road_graph(cleaned_root / partition / trip_id)
        values = frame[features].to_numpy(dtype=np.float32, copy=True)
        start = time.perf_counter()
        probabilities = model_bundle["estimator"].predict_proba(
            model_bundle["scaler"].transform(values)
        )[:, 1]
        inference_seconds += time.perf_counter() - start
        frame = frame.reset_index(drop=True)
        frame["emission_probability"] = probabilities
        start = time.perf_counter()
        decoded = _decode_viterbi(frame, probabilities, graph)
        decoding_seconds += time.perf_counter() - start

        scores = pd.DataFrame(
            {
                "observation_id": frame["observation_id"].to_numpy(),
                "score": probabilities,
            }
        )
        top_indices = (
            scores.groupby("observation_id", sort=False)["score"]
            .idxmax()
            .to_numpy(dtype=int)
        )
        top1 = frame.iloc[top_indices][
            ["observation_id", "candidate_arc_id", "emission_probability"]
        ].rename(
            columns={
                "candidate_arc_id": "top1_arc_id",
                "emission_probability": "top1_emission_probability",
            }
        )
        usable_by_observation = (
            frame.groupby("observation_id", sort=False)["supervision_usable"]
            .first()
            .to_dict()
        )
        reference_arcs = (
            frame.loc[frame["supervision_usable"] & frame["is_target"]]
            .groupby("observation_id")["candidate_arc_id"]
            .agg(list)
            .to_dict()
        )
        if any(len(arc_ids) != 1 for arc_ids in reference_arcs.values()):
            raise ValueError(
                f"Expected exactly one target arc per labeled observation in {trip_id}."
            )
        decoded = decoded.rename(
            columns={
                "candidate_arc_id": "viterbi_arc_id",
                "emission_probability": "viterbi_emission_probability",
            }
        )
        decoded["label_usable"] = (
            decoded["observation_id"].map(usable_by_observation).astype(bool)
        )
        decoded["reference_arc_id"] = decoded["observation_id"].map(
            {key: value[0] for key, value in reference_arcs.items()}
        )
        decoded["trip_id"] = trip_id
        decoded["candidate_count"] = decoded["observation_id"].map(
            frame.groupby("observation_id", sort=False).size().to_dict()
        )
        merged = decoded[
            [
                "trip_id",
                "observation_id",
                "viterbi_arc_id",
                "connector_arc_id_before",
                "transition_connected",
                "viterbi_emission_probability",
                "label_usable",
                "reference_arc_id",
                "candidate_count",
            ]
        ].merge(top1, on="observation_id", validate="one_to_one")
        prediction_frames.append(merged)
        route_graphs[trip_id] = graph
        del frame, decoded, merged, values

    predictions = pd.concat(prediction_frames, ignore_index=True)
    route_sequences = _route_sequence(predictions)
    continuity_totals = {
        "top1": [0, 0],
        "viterbi": [0, 0],
        "order_top1": [0, 0],
        "order_viterbi": [0, 0],
        "exact_top1": [],
        "exact_viterbi": [],
        "labeled_observations": 0,
        "scored_trips": 0,
        "connected_transitions": 0,
        "transition_count": 0,
    }
    for trip_id, group in predictions.groupby("trip_id", sort=False):
        graph = route_graphs[trip_id]
        top_n, top_d = _continuity(group, "top1_arc_id", graph["endpoints"])
        vit_n, vit_d = _continuity(group, "viterbi_arc_id", graph["endpoints"])
        continuity_totals["top1"][0] += top_n
        continuity_totals["top1"][1] += top_d
        continuity_totals["viterbi"][0] += vit_n
        continuity_totals["viterbi"][1] += vit_d
        top_order = _sequence_order(group, "top1_arc_id", graph["route_positions"])
        vit_order = _sequence_order(group, "viterbi_arc_id", graph["route_positions"])
        continuity_totals["order_top1"][0] += top_order[0]
        continuity_totals["order_top1"][1] += top_order[1]
        continuity_totals["order_viterbi"][0] += vit_order[0]
        continuity_totals["order_viterbi"][1] += vit_order[1]
        scorable = group.loc[group["label_usable"]]
        if not scorable.empty:
            continuity_totals["scored_trips"] += 1
            continuity_totals["labeled_observations"] += len(scorable)
            continuity_totals["exact_top1"].append(
                float(
                    (
                        scorable["top1_arc_id"]
                        == scorable["reference_arc_id"].astype(int)
                    ).all()
                )
            )
            continuity_totals["exact_viterbi"].append(
                float(
                    (
                        scorable["viterbi_arc_id"]
                        == scorable["reference_arc_id"].astype(int)
                    ).all()
                )
            )
        transitions = group.sort_values("observation_id", kind="stable")
        transition_rows = list(transitions.itertuples(index=False))
        for previous, current in zip(
            transition_rows[:-1], transition_rows[1:], strict=True
        ):
            if current.observation_id != previous.observation_id + 1:
                continue
            continuity_totals["transition_count"] += 1
            if current.transition_connected:
                continuity_totals["connected_transitions"] += 1

    scored_predictions = predictions.loc[predictions["label_usable"]]
    if scored_predictions.empty:
        raise ValueError("No reliable labeled observations were found for evaluation.")
    top_correct = scored_predictions["top1_arc_id"] == scored_predictions[
        "reference_arc_id"
    ].astype(int)
    viterbi_correct = scored_predictions["viterbi_arc_id"] == scored_predictions[
        "reference_arc_id"
    ].astype(int)
    reconstructed_route_continuity = []
    for trip_id, group in route_sequences.groupby("trip_id", sort=False):
        endpoints = route_graphs[trip_id]["endpoints"]
        arc_ids = group.sort_values("route_index")["arc_id"].to_numpy(dtype=int)
        if len(arc_ids) < 2:
            continue
        connected = [
            endpoints[previous][1] == endpoints[current][0]
            for previous, current in zip(arc_ids, arc_ids[1:])
        ]
        reconstructed_route_continuity.extend(connected)

    metrics: dict[str, Any] = {
        "partition": partition,
        "split": split,
        "trip_count": len(trip_ids),
        "scored_trip_count": continuity_totals["scored_trips"],
        "labeled_observations": continuity_totals["labeled_observations"],
        "candidate_rows": int(predictions["candidate_count"].sum()),
        "candidate_miss_count": candidate_misses,
        "candidate_miss_rate": (
            candidate_misses / reliable_reference_count
            if reliable_reference_count
            else float("nan")
        ),
        "reliable_candidate_references": reliable_reference_count,
        "top1_observation_accuracy": float(top_correct.mean()),
        "viterbi_observation_accuracy": float(viterbi_correct.mean()),
        "top1_route_continuity": (
            continuity_totals["top1"][0] / continuity_totals["top1"][1]
            if continuity_totals["top1"][1]
            else float("nan")
        ),
        "viterbi_route_continuity": (
            continuity_totals["viterbi"][0] / continuity_totals["viterbi"][1]
            if continuity_totals["viterbi"][1]
            else float("nan")
        ),
        "viterbi_transition_feasibility": (
            continuity_totals["connected_transitions"]
            / continuity_totals["transition_count"]
            if continuity_totals["transition_count"]
            else float("nan")
        ),
        "expanded_route_continuity": (
            float(np.mean(reconstructed_route_continuity))
            if reconstructed_route_continuity
            else float("nan")
        ),
        "top1_sequence_order_consistency": (
            continuity_totals["order_top1"][0] / continuity_totals["order_top1"][1]
            if continuity_totals["order_top1"][1]
            else float("nan")
        ),
        "viterbi_sequence_order_consistency": (
            continuity_totals["order_viterbi"][0]
            / continuity_totals["order_viterbi"][1]
            if continuity_totals["order_viterbi"][1]
            else float("nan")
        ),
        "top1_exact_trip_rate": (
            float(np.mean(continuity_totals["exact_top1"]))
            if continuity_totals["exact_top1"]
            else float("nan")
        ),
        "viterbi_exact_trip_rate": (
            float(np.mean(continuity_totals["exact_viterbi"]))
            if continuity_totals["exact_viterbi"]
            else float("nan")
        ),
        "inference_seconds": inference_seconds,
        "viterbi_seconds": decoding_seconds,
        "student_test_used_for_model_selection": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = partition if split is None else f"{partition}_{split}"
    predictions.to_csv(output_root / f"{suffix}_hmm_predictions.csv", index=False)
    route_sequences.to_csv(
        output_root / f"{suffix}_viterbi_route_sequence.csv", index=False
    )
    (output_root / f"{suffix}_hmm_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return predictions, route_sequences, metrics
