"""Build audited candidate-edge features for PathSnap map matching."""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


FEATURE_COLUMNS = [
    "distance_m",
    "projection_fraction",
    "arc_length_m",
    "distance_to_start_m",
    "distance_to_end_m",
    "heading_diff_rad",
    "heading_alignment_cos",
    "heading_alignment_sin",
    "gps_heading_valid",
    "gps_speed_mps",
    "gps_step_distance_m",
    "gps_time_gap_s",
    "gps_time_gap_valid",
    "source_in_degree",
    "source_out_degree",
    "target_in_degree",
    "target_out_degree",
]

METADATA_COLUMNS = [
    "trip_id",
    "partition",
    "split",
    "observation_id",
    "candidate_arc_id",
    "candidate_rank",
    "label_distance_m",
    "label_margin_m",
    "label_confident",
    "label_unambiguous",
    "candidate_label_found",
    "route_geometry_suspect",
    "route_sequence_suspect",
    "route_sequence_consistent",
    "supervision_usable",
    "is_target",
]


def _read_clean_trip(trip_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ("track", "nodes", "arcs", "route"):
        path = trip_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing cleaned input: {path}")
        tables[name] = pd.read_csv(path)
        if tables[name].empty:
            raise ValueError(f"Cleaned input is empty: {path}")

    required = {
        "track": {"longitude", "latitude", "timestamp", "observation_id"},
        "nodes": {"longitude", "latitude", "node_id"},
        "arcs": {"from_node", "to_node", "arc_id"},
        "route": {"arc_id"},
    }
    for name, columns in required.items():
        missing = columns.difference(tables[name].columns)
        if missing:
            raise ValueError(f"{trip_dir.name}/{name}.csv missing {sorted(missing)}")
    tables["track"] = tables["track"].sort_values(
        "observation_id", kind="stable"
    ).reset_index(drop=True)
    return tables


def _project_coordinates(
    longitude_latitude: np.ndarray, origin: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (longitude_latitude - origin) * scale


def _route_labels(
    gps_xy: np.ndarray,
    arc_start: np.ndarray,
    arc_delta: np.ndarray,
    arc_length_squared: np.ndarray,
    route_arc_rows: np.ndarray,
    route_arc_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    route_rows, first_positions = np.unique(route_arc_rows, return_index=True)
    order = np.argsort(first_positions, kind="stable")
    route_rows = route_rows[order]
    unique_route_ids = route_arc_ids[first_positions[order]]
    if not len(route_rows):
        return (
            np.full(len(gps_xy), -1, dtype=int),
            np.full(len(gps_xy), np.inf),
            np.full(len(gps_xy), np.nan),
        )

    starts = arc_start[route_rows]
    deltas = arc_delta[route_rows]
    denominators = arc_length_squared[route_rows]
    best_ids = np.empty(len(gps_xy), dtype=int)
    best_distances = np.empty(len(gps_xy), dtype=float)
    margins = np.full(len(gps_xy), np.nan, dtype=float)

    for start in range(0, len(gps_xy), 512):
        stop = min(start + 512, len(gps_xy))
        offsets = gps_xy[start:stop, None, :] - starts[None, :, :]
        fractions = np.clip(
            np.einsum("ijk,jk->ij", offsets, deltas) / denominators[None, :],
            0.0,
            1.0,
        )
        distances = np.linalg.norm(
            offsets - fractions[:, :, None] * deltas[None, :, :], axis=2
        )
        nearest = np.argmin(distances, axis=1)
        rows = np.arange(stop - start)
        best_ids[start:stop] = unique_route_ids[nearest]
        best_distances[start:stop] = distances[rows, nearest]
        if len(route_rows) > 1:
            ordered = np.argsort(distances, axis=1, kind="stable")
            second = ordered[:, 1]
            margins[start:stop] = (
                distances[rows, second] - distances[rows, nearest]
            )
    return best_ids, best_distances, margins


def _gps_motion(
    gps_xy: np.ndarray, timestamps: np.ndarray, timestamp_to_seconds: float
) -> tuple[np.ndarray, ...]:
    count = len(gps_xy)
    step_distance = np.zeros(count, dtype=float)
    time_gap = np.zeros(count, dtype=float)
    speed = np.zeros(count, dtype=float)
    heading = np.zeros(count, dtype=float)
    heading_valid = np.zeros(count, dtype=bool)
    gap_valid = np.zeros(count, dtype=bool)
    if count < 2:
        return step_distance, time_gap, speed, heading, heading_valid, gap_valid

    deltas = np.diff(gps_xy, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    raw_gaps = np.diff(timestamps) * timestamp_to_seconds
    valid_gaps = np.isfinite(raw_gaps) & (raw_gaps > 0)
    speeds = np.zeros(len(raw_gaps), dtype=float)
    np.divide(distances, raw_gaps, out=speeds, where=valid_gaps)
    bearings = np.arctan2(deltas[:, 1], deltas[:, 0])

    step_distance[1:] = distances
    time_gap[1:] = np.where(valid_gaps, raw_gaps, 0.0)
    speed[1:] = speeds
    gap_valid[1:] = valid_gaps
    heading_valid[1:] = valid_gaps & (distances > 0)
    heading[1:] = bearings
    heading[0] = bearings[0]
    heading_valid[0] = heading_valid[1]
    speed[0] = speeds[0]
    step_distance[0] = distances[0]
    time_gap[0] = time_gap[1]
    gap_valid[0] = gap_valid[1]
    return step_distance, time_gap, speed, heading, heading_valid, gap_valid


def _longest_nondecreasing_mask(
    route_positions: np.ndarray, eligible: np.ndarray
) -> np.ndarray:
    observation_ids = np.flatnonzero(eligible)
    tails: list[int] = []
    tail_observations: list[int] = []
    previous = np.full(len(route_positions), -1, dtype=int)

    for observation_id in observation_ids:
        route_position = int(route_positions[observation_id])
        level = bisect_right(tails, route_position)
        if level:
            previous[observation_id] = tail_observations[level - 1]
        if level == len(tails):
            tails.append(route_position)
            tail_observations.append(int(observation_id))
        else:
            tails[level] = route_position
            tail_observations[level] = int(observation_id)

    keep = np.zeros(len(route_positions), dtype=bool)
    observation_id = tail_observations[-1] if tail_observations else -1
    while observation_id >= 0:
        keep[observation_id] = True
        observation_id = previous[observation_id]
    return keep


def _densify_arcs(
    arc_start: np.ndarray, arc_end: np.ndarray, max_spacing_m: float
) -> tuple[np.ndarray, np.ndarray]:
    deltas = arc_end - arc_start
    lengths = np.linalg.norm(deltas, axis=1)
    pieces = np.maximum(1, np.ceil(lengths / max_spacing_m).astype(int))
    sample_counts = pieces + 1
    sample_arc_rows = np.repeat(np.arange(len(pieces)), sample_counts)
    offsets = np.arange(len(sample_arc_rows)) - np.repeat(
        np.cumsum(sample_counts) - sample_counts, sample_counts
    )
    fractions = offsets / pieces[sample_arc_rows]
    sample_points = (
        arc_start[sample_arc_rows]
        + fractions[:, None] * deltas[sample_arc_rows]
    )
    return sample_points, sample_arc_rows


def _build_trip_candidates(
    trip_dir: Path,
    partition: str,
    split: str,
    max_candidates: int,
    nearest_samples: int,
    sample_spacing_m: float,
    label_radius_m: float,
    ambiguity_margin_m: float,
    timestamp_to_seconds: float,
) -> tuple[pd.DataFrame, dict]:
    tables = _read_clean_trip(trip_dir)
    track, nodes, arcs, route = (
        tables["track"],
        tables["nodes"],
        tables["arcs"],
        tables["route"],
    )
    gps_lonlat = track[["longitude", "latitude"]].to_numpy(dtype=float)
    origin = np.nanmean(gps_lonlat, axis=0)
    scale = np.array(
        [111_320.0 * np.cos(np.radians(origin[1])), 111_320.0], dtype=float
    )
    if not np.isfinite(scale).all() or scale[0] <= 0:
        raise ValueError(f"Invalid local projection for trip {trip_dir.name}")
    gps_xy = _project_coordinates(gps_lonlat, origin, scale)
    node_xy = _project_coordinates(
        nodes[["longitude", "latitude"]].to_numpy(dtype=float), origin, scale
    )

    node_positions = {
        int(node_id): position
        for position, node_id in enumerate(nodes["node_id"].to_numpy())
    }
    source_positions = arcs["from_node"].map(node_positions)
    target_positions = arcs["to_node"].map(node_positions)
    if source_positions.isna().any() or target_positions.isna().any():
        raise ValueError(f"Arc references missing node in trip {trip_dir.name}")
    source_positions = source_positions.to_numpy(dtype=int)
    target_positions = target_positions.to_numpy(dtype=int)
    arc_ids = arcs["arc_id"].to_numpy(dtype=int)
    arc_start = node_xy[source_positions]
    arc_end = node_xy[target_positions]
    arc_delta = arc_end - arc_start
    arc_length = np.linalg.norm(arc_delta, axis=1)
    arc_length_squared = np.maximum(arc_length**2, 1e-8)

    arc_id_to_row = {int(arc_id): row for row, arc_id in enumerate(arc_ids)}
    route_ids = route["arc_id"].to_numpy(dtype=int)
    unknown_route_ids = sorted(set(route_ids).difference(arc_id_to_row))
    if unknown_route_ids:
        raise ValueError(
            f"Route references unknown arcs in trip {trip_dir.name}: "
            f"{unknown_route_ids[:5]}"
        )
    route_rows = np.array([arc_id_to_row[int(arc_id)] for arc_id in route_ids])
    target_arc_ids, label_distances, label_margins = _route_labels(
        gps_xy, arc_start, arc_delta, arc_length_squared, route_rows, route_ids
    )
    confident = label_distances <= label_radius_m
    unambiguous = ~np.isfinite(label_margins) | (label_margins > ambiguity_margin_m)
    route_geometry_suspect = (
        float(np.median(label_distances)) > label_radius_m
    )
    first_route_position = {}
    for route_position, arc_id in enumerate(route_ids):
        first_route_position.setdefault(int(arc_id), route_position)
    label_route_positions = np.array(
        [first_route_position.get(int(arc_id), -1) for arc_id in target_arc_ids],
        dtype=int,
    )
    sequence_eligible = confident & unambiguous
    route_sequence_consistent = _longest_nondecreasing_mask(
        label_route_positions, sequence_eligible
    )
    sequence_labels_dropped = int(
        (sequence_eligible & ~route_sequence_consistent).sum()
    )
    route_sequence_suspect = bool(
        sequence_eligible.any()
        and sequence_labels_dropped / sequence_eligible.sum() > 0.10
    )

    step_distance, time_gap, speed, heading, heading_valid, gap_valid = _gps_motion(
        gps_xy, track["timestamp"].to_numpy(dtype=float), timestamp_to_seconds
    )
    source_in = np.bincount(target_positions, minlength=len(nodes))
    source_out = np.bincount(source_positions, minlength=len(nodes))

    sample_points, sample_arc_rows = _densify_arcs(
        arc_start, arc_end, sample_spacing_m
    )
    sample_tree = cKDTree(sample_points)
    k = min(nearest_samples, len(sample_points))
    chunk_frames = []
    candidate_hits = np.zeros(len(track), dtype=bool)
    for chunk_start in range(0, len(gps_xy), 512):
        chunk_stop = min(chunk_start + 512, len(gps_xy))
        sample_indices = np.asarray(
            sample_tree.query(gps_xy[chunk_start:chunk_stop], k=k)[1]
        )
        if sample_indices.ndim == 1:
            sample_indices = sample_indices[:, None]

        edge_rows = []
        local_rows = []
        distances_by_row = []
        fractions_by_row = []
        for local_row, nearest in enumerate(sample_indices):
            possible_rows = np.unique(sample_arc_rows[nearest])
            offsets = gps_xy[chunk_start + local_row] - arc_start[possible_rows]
            fractions = np.clip(
                np.einsum("ij,ij->i", offsets, arc_delta[possible_rows])
                / arc_length_squared[possible_rows],
                0.0,
                1.0,
            )
            distances = np.linalg.norm(
                offsets - fractions[:, None] * arc_delta[possible_rows], axis=1
            )
            order = np.argsort(distances, kind="stable")[:max_candidates]
            selected_rows = possible_rows[order]
            selected_distances = distances[order]
            selected_fractions = fractions[order]
            edge_rows.append(selected_rows)
            local_rows.append(np.full(len(order), local_row, dtype=int))
            distances_by_row.append(selected_distances)
            fractions_by_row.append(selected_fractions)
            candidate_hits[chunk_start + local_row] = bool(
                np.any(arc_ids[selected_rows] == target_arc_ids[chunk_start + local_row])
            )

        edge_rows_flat = np.concatenate(edge_rows)
        local_rows_flat = np.concatenate(local_rows)
        distances_flat = np.concatenate(distances_by_row)
        fractions_flat = np.concatenate(fractions_by_row)
        global_rows = chunk_start + local_rows_flat
        candidate_ids = arc_ids[edge_rows_flat]
        angle_delta = heading[global_rows] - np.arctan2(
            arc_delta[edge_rows_flat, 1], arc_delta[edge_rows_flat, 0]
        )
        angle_delta = (angle_delta + np.pi) % (2 * np.pi) - np.pi
        angle_delta = np.where(heading_valid[global_rows], angle_delta, 0.0)
        selected_target_ids = target_arc_ids[global_rows]
        selected_confident = confident[global_rows]
        selected_unambiguous = unambiguous[global_rows]
        selected_found = candidate_hits[global_rows]
        selected_sequence_consistent = route_sequence_consistent[global_rows]
        usable = (
            selected_confident
            & selected_unambiguous
            & selected_found
            & selected_sequence_consistent
            & (not route_geometry_suspect)
            & (not route_sequence_suspect)
        )
        rank = np.concatenate(
            [np.arange(1, len(rows) + 1) for rows in edge_rows]
        )

        chunk_frames.append(
            pd.DataFrame(
                {
                    "trip_id": trip_dir.name,
                    "partition": partition,
                    "split": split,
                    "observation_id": track["observation_id"].to_numpy(dtype=int)[
                        global_rows
                    ],
                    "candidate_arc_id": candidate_ids,
                    "candidate_rank": rank,
                    "label_distance_m": label_distances[global_rows],
                    "label_margin_m": label_margins[global_rows],
                    "label_confident": selected_confident,
                    "label_unambiguous": selected_unambiguous,
                    "candidate_label_found": selected_found,
                    "route_geometry_suspect": route_geometry_suspect,
                    "route_sequence_suspect": route_sequence_suspect,
                    "route_sequence_consistent": selected_sequence_consistent,
                    "supervision_usable": usable,
                    "is_target": usable & (candidate_ids == selected_target_ids),
                    "distance_m": distances_flat,
                    "projection_fraction": fractions_flat,
                    "arc_length_m": arc_length[edge_rows_flat],
                    "distance_to_start_m": fractions_flat * arc_length[edge_rows_flat],
                    "distance_to_end_m": (1.0 - fractions_flat)
                    * arc_length[edge_rows_flat],
                    "heading_diff_rad": np.abs(angle_delta),
                    "heading_alignment_cos": np.cos(angle_delta)
                    * heading_valid[global_rows],
                    "heading_alignment_sin": np.sin(angle_delta)
                    * heading_valid[global_rows],
                    "gps_heading_valid": heading_valid[global_rows].astype(np.int8),
                    "gps_speed_mps": speed[global_rows],
                    "gps_step_distance_m": step_distance[global_rows],
                    "gps_time_gap_s": time_gap[global_rows],
                    "gps_time_gap_valid": gap_valid[global_rows].astype(np.int8),
                    "source_in_degree": source_in[source_positions[edge_rows_flat]],
                    "source_out_degree": source_out[source_positions[edge_rows_flat]],
                    "target_in_degree": source_in[target_positions[edge_rows_flat]],
                    "target_out_degree": source_out[target_positions[edge_rows_flat]],
                }
            )
        )

    candidates = pd.concat(chunk_frames, ignore_index=True)
    for column in (
        "label_confident",
        "label_unambiguous",
        "candidate_label_found",
        "route_geometry_suspect",
        "route_sequence_suspect",
        "route_sequence_consistent",
        "supervision_usable",
        "is_target",
    ):
        candidates[column] = candidates[column].astype(bool)
    route_disconnected = int(
        np.count_nonzero(
            target_positions[route_rows[:-1]] != source_positions[route_rows[1:]]
        )
    )
    confident_hits = candidate_hits & confident
    unambiguous_high_confidence = confident & unambiguous
    audit = {
        "trip_id": trip_dir.name,
        "partition": partition,
        "split": split,
        "gps_observations": len(track),
        "road_arcs": len(arcs),
        "route_arcs": len(route_ids),
        "route_disconnected_transitions": route_disconnected,
        "route_transition_count": max(len(route_ids) - 1, 0),
        "high_confidence_labels": int(confident.sum()),
        "unambiguous_labels": int(unambiguous_high_confidence.sum()),
        "ambiguous_high_confidence_labels": int(
            (confident & ~unambiguous).sum()
        ),
        "candidate_label_hits": int(confident_hits.sum()),
        "candidate_recall_high_confidence": (
            float(confident_hits.sum() / confident.sum())
            if confident.any()
            else np.nan
        ),
        "route_geometry_suspect": route_geometry_suspect,
        "route_sequence_labels_considered": int(sequence_eligible.sum()),
        "route_sequence_labels_kept": int(
            (sequence_eligible & route_sequence_consistent).sum()
        ),
        "route_sequence_labels_dropped": sequence_labels_dropped,
        "route_sequence_suspect": route_sequence_suspect,
        "supervised_observations": int(
            (
                confident
                & unambiguous
                & candidate_hits
                & route_sequence_consistent
                & (not route_geometry_suspect)
                & (not route_sequence_suspect)
            ).sum()
        ),
        "candidate_rows": len(candidates),
        "median_label_distance_m": float(np.median(label_distances)),
        "p95_label_distance_m": float(np.percentile(label_distances, 95)),
    }
    return candidates, audit


def prepare_model_data(
    cleaned_root: str | Path,
    output_root: str | Path,
    *,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    max_candidates: int = 32,
    nearest_samples: int = 128,
    sample_spacing_m: float = 30.0,
    label_radius_m: float = 50.0,
    ambiguity_margin_m: float = 5.0,
    timestamp_to_seconds: float = 1.0,
) -> pd.DataFrame:
    """Create per-trip candidate tables and a deterministic trip-level split."""
    cleaned_root = Path(cleaned_root)
    output_root = Path(output_root)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if max_candidates < 1 or nearest_samples < 1 or sample_spacing_m <= 0:
        raise ValueError("Candidate counts and sample spacing must be positive.")
    if label_radius_m <= 0 or ambiguity_margin_m < 0:
        raise ValueError("Label radius must be positive and margin non-negative.")
    if timestamp_to_seconds <= 0:
        raise ValueError("timestamp_to_seconds must be positive.")

    train_dirs = sorted((cleaned_root / "train").glob("*"))
    test_dirs = sorted((cleaned_root / "student_test").glob("*"))
    train_dirs = [path for path in train_dirs if path.is_dir()]
    test_dirs = [path for path in test_dirs if path.is_dir()]
    if not train_dirs:
        raise FileNotFoundError(f"No training trips found under {cleaned_root / 'train'}")
    if not test_dirs:
        raise FileNotFoundError(
            f"No student-test trips found under {cleaned_root / 'student_test'}"
        )
    if len(train_dirs) < 2:
        raise ValueError("At least two training trips are required for validation.")

    rng = np.random.default_rng(random_seed)
    shuffled_ids = np.array([path.name for path in train_dirs], dtype=object)
    rng.shuffle(shuffled_ids)
    validation_count = min(
        len(train_dirs) - 1,
        max(1, int(round(len(train_dirs) * validation_fraction))),
    )
    validation_ids = set(shuffled_ids[:validation_count])
    trip_splits = {
        path.name: ("validation" if path.name in validation_ids else "train")
        for path in train_dirs
    }
    split_by_partition = [
        (path, "train", trip_splits[path.name]) for path in train_dirs
    ] + [(path, "student_test", "student_test") for path in test_dirs]

    audits = []
    for trip_dir, partition, split in split_by_partition:
        candidates, audit = _build_trip_candidates(
            trip_dir,
            partition,
            split,
            max_candidates,
            nearest_samples,
            sample_spacing_m,
            label_radius_m,
            ambiguity_margin_m,
            timestamp_to_seconds,
        )
        trip_output = output_root / partition / trip_dir.name
        trip_output.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(
            trip_output / "candidates.csv.gz",
            index=False,
            compression="gzip",
        )
        audits.append(audit)
        print(
            f"{partition}/{trip_dir.name}: "
            f"{audit['gps_observations']:,} observations, "
            f"{audit['candidate_rows']:,} candidates, "
            f"{audit['supervised_observations']:,} supervised observations"
        )

    audit_frame = pd.DataFrame(audits)
    output_root.mkdir(parents=True, exist_ok=True)
    audit_frame.to_csv(output_root / "feature_quality_audit.csv", index=False)
    with (output_root / "feature_schema.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "metadata_columns": METADATA_COLUMNS,
                "target_column": "is_target",
                "fit_filter": (
                    "supervision_usable == True; route-sequence violations and "
                    "suspect trips are quarantined"
                ),
                "candidate_table_pattern": (
                    "<partition>/<trip_id>/candidates.csv.gz"
                ),
                "validation_trip_ids": sorted(validation_ids),
                "student_test_is_separate": True,
            },
            file,
            indent=2,
        )

    return audit_frame
