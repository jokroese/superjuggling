"""Evaluation utilities for tracking experiments.

The first benchmark is intentionally candidate-level:

    debug_candidates.csv + project-native labels.csv -> recall/precision/F1

Only frames present in the label CSV are scored. This lets a trusted 72-frame
slice from a much longer clip act as a useful benchmark without penalising or
rewarding predictions on unlabelled frames.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .labels import GroundTruthLabel, read_ground_truth_csv


@dataclass(frozen=True)
class CandidatePrediction:
    """One predicted centre candidate loaded from debug_candidates.csv."""

    frame_index: int
    x: float
    y: float
    score: float
    source: str


@dataclass(frozen=True)
class CandidateMatch:
    """One one-to-one match between a visible label and a prediction."""

    frame_index: int
    ball_id: int
    label_x: float
    label_y: float
    candidate_x: float
    candidate_y: float
    distance_px: float
    held: bool | None
    source: str


@dataclass(frozen=True)
class CandidateEvaluationSummary:
    """Aggregate candidate-vs-label metrics."""

    radius_px: float
    labelled_frames: int
    visible_labels: int
    predicted_candidates: int
    matches: int
    false_negatives: int
    false_positives: int
    recall: float
    precision: float
    f1: float
    mean_error_px: float | None
    median_error_px: float | None
    held_visible_labels: int
    held_matches: int
    held_recall: float | None
    free_visible_labels: int
    free_matches: int
    free_recall: float | None


@dataclass(frozen=True)
class CandidateEvaluation:
    """Full candidate evaluation payload."""

    summary: CandidateEvaluationSummary
    matches: list[CandidateMatch]
    source_counts: dict[str, int]


def _safe_float(value: str | None, *, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def read_candidate_predictions_csv(path: Path) -> list[CandidatePrediction]:
    """Read candidate predictions from ``debug_candidates.csv``."""
    predictions: list[CandidatePrediction] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            predictions.append(
                CandidatePrediction(
                    frame_index=int(row["frame_index"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    score=_safe_float(row.get("score")),
                    source=row.get("source") or "unknown",
                )
            )
    return predictions


def _group_labels_by_frame(
    labels: list[GroundTruthLabel],
) -> dict[int, list[GroundTruthLabel]]:
    grouped: dict[int, list[GroundTruthLabel]] = defaultdict(list)
    for label in labels:
        grouped[label.frame_index].append(label)
    return dict(grouped)


def _group_predictions_by_frame(
    predictions: list[CandidatePrediction],
) -> dict[int, list[CandidatePrediction]]:
    grouped: dict[int, list[CandidatePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.frame_index].append(prediction)
    return dict(grouped)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2.0 * precision * recall / (precision + recall), 6)


def _match_frame(
    *,
    labels: list[GroundTruthLabel],
    predictions: list[CandidatePrediction],
    radius_px: float,
) -> list[CandidateMatch]:
    visible_labels = [label for label in labels if label.visible]
    if not visible_labels or not predictions:
        return []

    distances = np.zeros((len(visible_labels), len(predictions)), dtype=np.float64)
    for i, label in enumerate(visible_labels):
        for j, prediction in enumerate(predictions):
            distances[i, j] = float(
                np.hypot(label.x - prediction.x, label.y - prediction.y)
            )

    row_ind, col_ind = linear_sum_assignment(distances)

    matches: list[CandidateMatch] = []
    for row, col in zip(row_ind, col_ind, strict=False):
        distance = float(distances[row, col])
        if distance > radius_px:
            continue
        label = visible_labels[row]
        prediction = predictions[col]
        matches.append(
            CandidateMatch(
                frame_index=label.frame_index,
                ball_id=label.ball_id,
                label_x=label.x,
                label_y=label.y,
                candidate_x=prediction.x,
                candidate_y=prediction.y,
                distance_px=distance,
                held=label.held,
                source=prediction.source,
            )
        )
    return matches


def evaluate_candidate_predictions(
    *,
    labels: list[GroundTruthLabel],
    predictions: list[CandidatePrediction],
    radius_px: float = 15.0,
) -> CandidateEvaluation:
    """Evaluate candidate centres against visible ground-truth labels.

    Only frames that appear in ``labels`` are scored. This is critical for
    partial datasets such as frames 0..71 from a longer clip.
    """
    if radius_px <= 0:
        msg = "radius_px must be > 0"
        raise ValueError(msg)

    label_frames = sorted({label.frame_index for label in labels})
    label_frame_set = set(label_frames)

    labels_by_frame = _group_labels_by_frame(labels)
    predictions_by_frame = _group_predictions_by_frame(
        [
            prediction
            for prediction in predictions
            if prediction.frame_index in label_frame_set
        ]
    )

    all_matches: list[CandidateMatch] = []
    for frame_index in label_frames:
        all_matches.extend(
            _match_frame(
                labels=labels_by_frame.get(frame_index, []),
                predictions=predictions_by_frame.get(frame_index, []),
                radius_px=radius_px,
            )
        )

    visible_labels = [label for label in labels if label.visible]
    scored_predictions = [
        prediction
        for prediction in predictions
        if prediction.frame_index in label_frame_set
    ]

    held_labels = [label for label in visible_labels if label.held is True]
    free_labels = [label for label in visible_labels if label.held is False]
    held_matches = [match for match in all_matches if match.held is True]
    free_matches = [match for match in all_matches if match.held is False]

    matches = len(all_matches)
    n_visible = len(visible_labels)
    n_predictions = len(scored_predictions)

    recall = float(_rate(matches, n_visible) or 0.0)
    precision = float(_rate(matches, n_predictions) or 0.0)
    distances = np.asarray([match.distance_px for match in all_matches])

    source_counts: dict[str, int] = {}
    for prediction in scored_predictions:
        source_counts[prediction.source] = source_counts.get(prediction.source, 0) + 1

    summary = CandidateEvaluationSummary(
        radius_px=radius_px,
        labelled_frames=len(label_frames),
        visible_labels=n_visible,
        predicted_candidates=n_predictions,
        matches=matches,
        false_negatives=n_visible - matches,
        false_positives=n_predictions - matches,
        recall=recall,
        precision=precision,
        f1=_f1(precision, recall),
        mean_error_px=(
            None if not len(distances) else round(float(np.mean(distances)), 3)
        ),
        median_error_px=(
            None if not len(distances) else round(float(np.median(distances)), 3)
        ),
        held_visible_labels=len(held_labels),
        held_matches=len(held_matches),
        held_recall=_rate(len(held_matches), len(held_labels)),
        free_visible_labels=len(free_labels),
        free_matches=len(free_matches),
        free_recall=_rate(len(free_matches), len(free_labels)),
    )
    return CandidateEvaluation(
        summary=summary,
        matches=all_matches,
        source_counts=source_counts,
    )


def evaluate_candidate_csv(
    *,
    candidates_path: Path,
    labels_path: Path,
    radius_px: float = 15.0,
) -> CandidateEvaluation:
    """Evaluate a debug candidates CSV against a project-native labels CSV."""
    labels = read_ground_truth_csv(labels_path)
    predictions = read_candidate_predictions_csv(candidates_path)
    if not labels:
        msg = f"no labels found in {labels_path}"
        raise ValueError(msg)
    return evaluate_candidate_predictions(
        labels=labels,
        predictions=predictions,
        radius_px=radius_px,
    )


def candidate_evaluation_to_dict(
    evaluation: CandidateEvaluation,
) -> dict[str, object]:
    """Serialise an evaluation payload into JSON-compatible primitives."""
    return {
        "summary": asdict(evaluation.summary),
        "source_counts": evaluation.source_counts,
        "matches": [asdict(match) for match in evaluation.matches],
    }


def write_candidate_evaluation_json(
    evaluation: CandidateEvaluation,
    out_path: Path,
) -> Path:
    """Write a machine-readable candidate evaluation report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(candidate_evaluation_to_dict(evaluation), indent=2) + "\n"
    )
    return out_path
