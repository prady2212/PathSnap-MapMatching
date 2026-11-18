"""FastAPI service for stored-trip map matching and HMM evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.models.artifact_inference import load_model, predict_candidate_probabilities
from src.models.bayesian_hmm import evaluate_hmm_partition, reconstruct_candidate_route

LOGGER = logging.getLogger(__name__)
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MatchRequest(BaseModel):
    trip_id: str = Field(pattern=r"^\d{8}$")
    partition: Literal["train", "student_test"] = "student_test"
    model_name: str | None = None
    score_candidates_per_observation: int = Field(default=3, ge=1, le=5)


class CandidateScore(BaseModel):
    observation_id: int
    arc_id: int
    rank: int
    probability: float


class ObservationMatch(BaseModel):
    observation_id: int
    candidate_count: int
    top1_arc_id: int
    top1_probability: float
    viterbi_arc_id: int
    viterbi_probability: float
    connector_arc_id_before: int
    transition_connected: bool


class RouteStep(BaseModel):
    route_index: int
    arc_id: int
    inserted_connector: bool
    at_observation_id: int


class MatchResponse(BaseModel):
    trip_id: str
    partition: str
    model_name: str
    candidate_scores: list[CandidateScore]
    observations: list[ObservationMatch]
    route_sequence: list[RouteStep]


class EvaluationRequest(BaseModel):
    evaluation_set: Literal["validation", "student_test"] = "student_test"


class EvaluationResponse(BaseModel):
    evaluation_set: str
    model_name: str
    metrics: dict[str, int | float | str | bool | None]
    predictions_file: str
    route_sequence_file: str


def _resolve_child(root: Path, relative_path: str) -> Path:
    normalized = Path(relative_path.replace("\\", "/"))
    resolved = (root / normalized).resolve()
    if root not in resolved.parents:
        raise ValueError(f"Artifact path escapes its root: {relative_path}")
    return resolved


def _verify_artifacts(artifact_root: Path, manifest: dict[str, Any]) -> None:
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Artifact manifest has no file inventory.")
    for record in records:
        path = _resolve_child(artifact_root, record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Artifact file is missing: {path}")
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Artifact size does not match manifest: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Artifact hash does not match manifest: {path}")


def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        result[key] = value
    return result


def create_app(
    project_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> FastAPI:
    """Create the API app; roots can be overridden for deployment or tests."""
    project_path = Path(
        project_root or os.environ.get("PATHSNAP_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
    ).resolve()
    artifacts_path = Path(
        artifact_root
        or os.environ.get(
            "PATHSNAP_ARTIFACT_ROOT",
            project_path / "artifacts" / "map_matching_models",
        )
    ).resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manifest_path = artifacts_path / "manifest.json"
        feature_root = project_path / "data" / "processed" / "features"
        audit_path = feature_root / "feature_quality_audit.csv"
        schema_path = artifacts_path / "feature_schema.json"
        for required_path in (manifest_path, audit_path, schema_path):
            if not required_path.is_file():
                raise FileNotFoundError(
                    f"Required API input is missing: {required_path}"
                )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _verify_artifacts(artifacts_path, manifest)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        feature_columns = schema["feature_columns"]
        model_records = manifest.get("models", {})
        if not isinstance(model_records, dict) or not model_records:
            raise ValueError("Artifact manifest contains no models.")
        recommended_path = manifest.get("recommended_model")
        if not isinstance(recommended_path, str):
            raise ValueError("Artifact manifest has no recommended model.")
        recommended_file = _resolve_child(artifacts_path, recommended_path)
        default_model = next(
            (
                name
                for name, record in model_records.items()
                if isinstance(record, dict)
                and "file" in record
                and _resolve_child(artifacts_path, record["file"]) == recommended_file
            ),
            None,
        )
        if default_model is None:
            raise ValueError(
                "Recommended model file does not match an artifact manifest entry."
            )

        audit = pd.read_csv(audit_path, dtype={"trip_id": str})
        required_audit_columns = {"trip_id", "partition", "split"}
        missing_audit_columns = required_audit_columns.difference(audit.columns)
        if missing_audit_columns:
            raise ValueError(
                f"Feature audit is missing columns: {sorted(missing_audit_columns)}"
            )
        available_trips: dict[str, dict[str, str]] = {}
        for row in audit.itertuples(index=False):
            available_trips.setdefault(str(row.partition), {})[str(row.trip_id)] = str(
                row.split
            )

        default_record = model_records[default_model]
        if not isinstance(default_record, dict) or "file" not in default_record:
            raise ValueError("Recommended model is missing from the artifact manifest.")
        default_bundle = load_model(
            _resolve_child(artifacts_path, default_record["file"])
        )
        if default_bundle["feature_columns"] != feature_columns:
            raise ValueError("Recommended model feature order differs from its schema.")

        app.state.project_root = project_path
        app.state.artifact_root = artifacts_path
        app.state.feature_root = feature_root
        app.state.cleaned_root = project_path / "data" / "processed" / "cleaned"
        app.state.manifest = manifest
        app.state.model_records = model_records
        app.state.default_model = default_model
        app.state.model_cache = {default_model: default_bundle}
        app.state.model_lock = threading.Lock()
        app.state.feature_columns = feature_columns
        app.state.available_trips = available_trips
        LOGGER.info(
            "PathSnap API loaded %d models and %d trips",
            len(model_records),
            sum(len(trips) for trips in available_trips.values()),
        )
        yield

    app = FastAPI(
        title="PathSnap Map-Matching API",
        description=(
            "Scores stored candidate-road features and reconstructs routes using "
            "the trained classifier and HMM/Viterbi decoder."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_model(model_name: str | None) -> tuple[str, dict[str, Any]]:
        selected_name = model_name or app.state.default_model
        record = app.state.model_records.get(selected_name)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown model: {selected_name}"
            )
        with app.state.model_lock:
            bundle = app.state.model_cache.get(selected_name)
            if bundle is None:
                bundle = load_model(
                    _resolve_child(app.state.artifact_root, record["file"])
                )
                if bundle["feature_columns"] != app.state.feature_columns:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Model feature schema mismatch: {selected_name}",
                    )
                app.state.model_cache[selected_name] = bundle
        return selected_name, bundle

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "default_model": app.state.default_model,
            "available_model_count": len(app.state.model_records),
        }

    @app.get("/models")
    def models() -> dict[str, Any]:
        return {
            "recommended_model": app.state.default_model,
            "models": app.state.model_records,
        }

    @app.get("/trips")
    def trips(partition: Literal["train", "student_test"] = Query("student_test")):
        partition_trips = app.state.available_trips.get(partition, {})
        return {
            "partition": partition,
            "trips": [
                {"trip_id": trip_id, "split": split}
                for trip_id, split in sorted(partition_trips.items())
            ],
        }

    @app.post("/match", response_model=MatchResponse)
    def match_trip(request: MatchRequest) -> MatchResponse:
        trip_splits = app.state.available_trips.get(request.partition, {})
        if request.trip_id not in trip_splits:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Trip {request.trip_id} is not available in "
                    f"partition {request.partition}."
                ),
            )
        model_name, bundle = get_model(request.model_name)
        candidate_path = (
            app.state.feature_root
            / request.partition
            / request.trip_id
            / "candidates.csv.gz"
        )
        if not candidate_path.is_file():
            raise HTTPException(
                status_code=404, detail="Prepared candidate features were not found."
            )
        required_columns = app.state.feature_columns + [
            "observation_id",
            "candidate_arc_id",
            "candidate_rank",
        ]
        frame = pd.read_csv(candidate_path, usecols=required_columns)
        if frame.empty:
            raise HTTPException(
                status_code=422, detail="The trip has no candidate rows."
            )
        features = frame[app.state.feature_columns]
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise HTTPException(
                status_code=422, detail="Candidate features contain non-finite values."
            )
        frame["trip_id"] = request.trip_id
        probabilities = predict_candidate_probabilities(bundle, frame)
        decoded, route_sequence = reconstruct_candidate_route(
            frame,
            probabilities,
            app.state.cleaned_root / request.partition / request.trip_id,
        )

        scored = frame[["observation_id", "candidate_arc_id", "candidate_rank"]].copy()
        scored["probability"] = probabilities
        scored = scored.sort_values(["observation_id", "candidate_rank"], kind="stable")
        ranked_scores = scored.sort_values(
            ["observation_id", "probability", "candidate_rank"],
            ascending=[True, False, True],
            kind="stable",
        )
        candidate_scores = [
            CandidateScore(
                observation_id=int(row.observation_id),
                arc_id=int(row.candidate_arc_id),
                rank=int(row.candidate_rank),
                probability=float(row.probability),
            )
            for row in ranked_scores.groupby(
                "observation_id", sort=False, group_keys=False
            )
            .head(request.score_candidates_per_observation)
            .itertuples(index=False)
        ]
        top1 = ranked_scores.drop_duplicates("observation_id").set_index(
            "observation_id"
        )
        candidate_counts = scored.groupby("observation_id", sort=False).size()
        observation_matches = []
        for row in decoded.itertuples(index=False):
            observation_id = int(row.observation_id)
            top = top1.loc[observation_id]
            observation_matches.append(
                ObservationMatch(
                    observation_id=observation_id,
                    candidate_count=int(candidate_counts.loc[observation_id]),
                    top1_arc_id=int(top["candidate_arc_id"]),
                    top1_probability=float(top["probability"]),
                    viterbi_arc_id=int(row.candidate_arc_id),
                    viterbi_probability=float(row.emission_probability),
                    connector_arc_id_before=int(row.connector_arc_id_before),
                    transition_connected=bool(row.transition_connected),
                )
            )
        route_steps = [
            RouteStep(
                route_index=int(row.route_index),
                arc_id=int(row.arc_id),
                inserted_connector=bool(row.inserted_connector),
                at_observation_id=int(row.at_observation_id),
            )
            for row in route_sequence.itertuples(index=False)
        ]
        return MatchResponse(
            trip_id=request.trip_id,
            partition=request.partition,
            model_name=model_name,
            candidate_scores=candidate_scores,
            observations=observation_matches,
            route_sequence=route_steps,
        )

    @app.post("/evaluate", response_model=EvaluationResponse)
    def evaluate(request: EvaluationRequest) -> EvaluationResponse:
        model_name, bundle = get_model(app.state.default_model)
        output_root = (
            app.state.project_root / "results" / "api_evaluations" / uuid4().hex
        )
        if request.evaluation_set == "validation":
            partition, split = "train", "validation"
            prediction_name = "train_validation_hmm_predictions.csv"
            route_name = "train_validation_viterbi_route_sequence.csv"
        else:
            partition, split = "student_test", None
            prediction_name = "student_test_hmm_predictions.csv"
            route_name = "student_test_viterbi_route_sequence.csv"
        _, _, metrics = evaluate_hmm_partition(
            feature_root=app.state.feature_root,
            cleaned_root=app.state.cleaned_root,
            model_bundle=bundle,
            output_root=output_root,
            partition=partition,
            split=split,
        )
        return EvaluationResponse(
            evaluation_set=request.evaluation_set,
            model_name=model_name,
            metrics=_json_safe_metrics(metrics),
            predictions_file=str(
                (output_root / prediction_name)
                .relative_to(app.state.project_root)
                .as_posix()
            ),
            route_sequence_file=str(
                (output_root / route_name)
                .relative_to(app.state.project_root)
                .as_posix()
            ),
        )

    return app


app = create_app()
