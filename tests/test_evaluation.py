from __future__ import annotations

import json
from pathlib import Path

from superjuggling.evaluation import (
    CandidatePrediction,
    evaluate_candidate_csv,
    evaluate_candidate_predictions,
    read_candidate_predictions_csv,
    write_candidate_evaluation_json,
)
from superjuggling.labels import GroundTruthLabel, write_ground_truth_csv


def _labels() -> list[GroundTruthLabel]:
    return [
        GroundTruthLabel(
            frame_index=0,
            ball_id=1,
            x=10.0,
            y=10.0,
            w=10.0,
            h=10.0,
            visible=True,
            held=False,
            keyframe=True,
        ),
        GroundTruthLabel(
            frame_index=0,
            ball_id=2,
            x=100.0,
            y=100.0,
            w=10.0,
            h=10.0,
            visible=True,
            held=True,
            keyframe=True,
        ),
        GroundTruthLabel(
            frame_index=1,
            ball_id=1,
            x=20.0,
            y=10.0,
            w=10.0,
            h=10.0,
            visible=True,
            held=False,
            keyframe=True,
        ),
        GroundTruthLabel(
            frame_index=1,
            ball_id=2,
            x=120.0,
            y=100.0,
            w=10.0,
            h=10.0,
            visible=False,
            held=True,
            keyframe=True,
        ),
    ]


def _predictions() -> list[CandidatePrediction]:
    return [
        CandidatePrediction(
            frame_index=0,
            x=12.0,
            y=10.0,
            score=0.9,
            source="heatmap",
        ),
        CandidatePrediction(
            frame_index=0,
            x=95.0,
            y=100.0,
            score=0.8,
            source="yolo",
        ),
        CandidatePrediction(
            frame_index=0,
            x=300.0,
            y=300.0,
            score=0.7,
            source="heatmap",
        ),
        CandidatePrediction(
            frame_index=1,
            x=100.0,
            y=100.0,
            score=0.6,
            source="heatmap",
        ),
        CandidatePrediction(
            frame_index=99,
            x=20.0,
            y=10.0,
            score=1.0,
            source="ignored",
        ),
    ]


def test_evaluate_candidate_predictions_scores_only_labelled_frames() -> None:
    evaluation = evaluate_candidate_predictions(
        labels=_labels(),
        predictions=_predictions(),
        radius_px=10.0,
    )

    summary = evaluation.summary

    assert summary.labelled_frames == 2
    assert summary.visible_labels == 3
    assert summary.predicted_candidates == 4
    assert summary.matches == 2
    assert summary.false_negatives == 1
    assert summary.false_positives == 2
    assert summary.recall == 0.666667
    assert summary.precision == 0.5
    assert summary.f1 == 0.571429
    assert summary.mean_error_px == 3.5
    assert summary.median_error_px == 3.5
    assert summary.held_visible_labels == 1
    assert summary.held_matches == 1
    assert summary.held_recall == 1.0
    assert summary.free_visible_labels == 2
    assert summary.free_matches == 1
    assert summary.free_recall == 0.5
    assert evaluation.source_counts == {"heatmap": 3, "yolo": 1}


def test_read_candidate_predictions_csv(tmp_path: Path) -> None:
    path = tmp_path / "debug_candidates.csv"
    path.write_text(
        "frame_index,t,x,y,score,radius_px,source,class_id\n"
        "0,0.0,12.0,10.0,0.9,10.0,heatmap,\n"
    )

    predictions = read_candidate_predictions_csv(path)

    assert predictions == [
        CandidatePrediction(
            frame_index=0,
            x=12.0,
            y=10.0,
            score=0.9,
            source="heatmap",
        )
    ]


def test_evaluate_candidate_csv_and_write_json(tmp_path: Path) -> None:
    labels_path = write_ground_truth_csv(_labels(), tmp_path / "labels.csv")
    candidates_path = tmp_path / "debug_candidates.csv"
    candidates_path.write_text(
        "frame_index,t,x,y,score,radius_px,source,class_id\n"
        "0,0.0,12.0,10.0,0.9,10.0,heatmap,\n"
        "0,0.0,95.0,100.0,0.8,10.0,yolo,\n"
        "0,0.0,300.0,300.0,0.7,10.0,heatmap,\n"
        "1,0.016,100.0,100.0,0.6,10.0,heatmap,\n"
        "99,1.65,20.0,10.0,1.0,10.0,ignored,\n"
    )

    evaluation = evaluate_candidate_csv(
        candidates_path=candidates_path,
        labels_path=labels_path,
        radius_px=10.0,
    )
    out_path = write_candidate_evaluation_json(
        evaluation,
        tmp_path / "candidate_evaluation.json",
    )

    payload = json.loads(out_path.read_text())
    assert payload["summary"]["matches"] == 2
    assert payload["summary"]["predicted_candidates"] == 4
    assert len(payload["matches"]) == 2
