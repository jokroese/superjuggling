from __future__ import annotations

import csv
from pathlib import Path

import pytest

from superjuggling.benchmark import (
    CandidateBenchmarkRow,
    build_candidate_benchmark_specs,
    parse_float_list,
    parse_methods,
    write_candidate_benchmark_summary,
)


def test_parse_methods() -> None:
    assert parse_methods("yolo, heatmap,hybrid") == ["yolo", "heatmap", "hybrid"]


def test_parse_methods_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown candidate method"):
        parse_methods("yolo,blob")


def test_parse_float_list() -> None:
    assert parse_float_list("0.20, 0.1,0.05") == [0.2, 0.1, 0.05]


def test_parse_float_list_rejects_out_of_range_value() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_float_list("0.1,2.0")


def test_build_candidate_benchmark_specs() -> None:
    specs = build_candidate_benchmark_specs(
        methods=["yolo", "heatmap", "hybrid"],
        yolo_confidences=[0.2, 0.1],
        tracking="ballistic",
    )

    assert [spec.name for spec in specs] == [
        "yolo-conf-200",
        "yolo-conf-100",
        "heatmap",
        "hybrid",
    ]
    assert specs[0].yolo_confidence == 0.2
    assert specs[-1].candidate_source == "hybrid"


def test_write_candidate_benchmark_summary(tmp_path: Path) -> None:
    rows = [
        CandidateBenchmarkRow(
            name="yolo-conf-100",
            candidate_source="yolo",
            tracking="ballistic",
            yolo_confidence=0.1,
            run_dir="runs/bench/yolo-conf-100",
            labelled_frames=72,
            visible_labels=216,
            predicted_candidates=100,
            matches=90,
            false_negatives=126,
            false_positives=10,
            recall=0.416667,
            precision=0.9,
            f1=0.569,
            mean_error_px=3.1,
            median_error_px=2.4,
            held_recall=0.4,
            free_recall=0.43,
        )
    ]

    paths = write_candidate_benchmark_summary(rows, tmp_path)

    assert paths["csv"].exists()
    assert paths["markdown"].exists()
    with paths["csv"].open(newline="") as fh:
        loaded = list(csv.DictReader(fh))
    assert loaded[0]["name"] == "yolo-conf-100"
    assert "yolo-conf-100" in paths["markdown"].read_text()
