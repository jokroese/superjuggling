from __future__ import annotations

import csv
from pathlib import Path

from superjuggling.labels import (
    convert_cvat_video_to_csv,
    filter_labels_by_frame_range,
    read_cvat_video_labels,
    read_ground_truth_csv,
    write_ground_truth_csv,
)


CVAT_XML = """\
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label>
          <name>ball</name>
          <type>rectangle</type>
          <attributes>
            <attribute>
              <name>held</name>
              <mutable>True</mutable>
              <input_type>select</input_type>
              <default_value>unknown</default_value>
              <values>yes
no
unknown</values>
            </attribute>
          </attributes>
        </label>
      </labels>
    </task>
  </meta>
  <track id="0" label="ball" source="manual">
    <box frame="0" outside="0" occluded="0" keyframe="1"
         xtl="10.0" ytl="20.0" xbr="30.0" ybr="60.0" z_order="0">
      <attribute name="held">yes</attribute>
    </box>
    <box frame="10" outside="0" occluded="0" keyframe="1"
         xtl="20.0" ytl="30.0" xbr="40.0" ybr="70.0" z_order="0">
      <attribute name="held">no</attribute>
    </box>
  </track>
  <track id="1" label="ball" source="manual">
    <box frame="0" outside="1" occluded="0" keyframe="1"
         xtl="100.0" ytl="120.0" xbr="140.0" ybr="160.0" z_order="0">
      <attribute name="held">unknown</attribute>
    </box>
  </track>
  <track id="2" label="person" source="manual">
    <box frame="0" outside="0" occluded="0" keyframe="1"
         xtl="0.0" ytl="0.0" xbr="10.0" ybr="10.0" z_order="0" />
  </track>
  <track id="3" label="ball" source="manual" />
</annotations>
"""


def _write_cvat_xml(tmp_path: Path) -> Path:
    path = tmp_path / "annotations.xml"
    path.write_text(CVAT_XML)
    return path


def test_read_cvat_video_labels_from_xml(tmp_path: Path) -> None:
    path = _write_cvat_xml(tmp_path)
    labels = read_cvat_video_labels(path)

    assert len(labels) == 3

    first = labels[0]
    assert first.frame_index == 0
    assert first.ball_id == 0
    assert first.x == 20.0
    assert first.y == 40.0
    assert first.w == 20.0
    assert first.h == 40.0
    assert first.visible is True
    assert first.held is True
    assert first.keyframe is True

    outside = labels[1]
    assert outside.ball_id == 1
    assert outside.visible is False
    assert outside.held is None

    later = labels[2]
    assert later.frame_index == 10
    assert later.held is False


def test_convert_cvat_video_to_csv(tmp_path: Path) -> None:
    cvat_xml = _write_cvat_xml(tmp_path)
    out = tmp_path / "labels.csv"

    result = convert_cvat_video_to_csv(cvat_xml, out)

    assert result == out
    assert out.exists()

    with out.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert rows[0]["frame_index"] == "0"
    assert rows[0]["ball_id"] == "0"
    assert rows[0]["x"] == "20.0"
    assert rows[0]["y"] == "40.0"
    assert rows[0]["visible"] == "1"
    assert rows[0]["held"] == "1"
    assert rows[1]["visible"] == "0"
    assert rows[1]["held"] == ""
    assert rows[2]["held"] == "0"


def test_ground_truth_csv_round_trip(tmp_path: Path) -> None:
    cvat_xml = _write_cvat_xml(tmp_path)
    labels = read_cvat_video_labels(cvat_xml)
    csv_path = write_ground_truth_csv(labels, tmp_path / "labels.csv")

    loaded = read_ground_truth_csv(csv_path)

    assert loaded == labels


def test_filter_labels_by_frame_range_is_inclusive(tmp_path: Path) -> None:
    cvat_xml = _write_cvat_xml(tmp_path)
    labels = read_cvat_video_labels(cvat_xml)

    filtered = filter_labels_by_frame_range(
        labels,
        frame_start=0,
        frame_end=0,
    )

    assert len(filtered) == 2
    assert {label.frame_index for label in filtered} == {0}


def test_convert_cvat_video_to_csv_can_write_trusted_slice(tmp_path: Path) -> None:
    cvat_xml = _write_cvat_xml(tmp_path)
    out = tmp_path / "labels-first-frame.csv"

    convert_cvat_video_to_csv(
        cvat_xml,
        out,
        frame_start=0,
        frame_end=0,
    )

    with out.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert {row["frame_index"] for row in rows} == {"0"}
    assert {row["ball_id"] for row in rows} == {"0", "1"}
