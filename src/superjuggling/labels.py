"""Ground-truth label conversion utilities.

The project-native evaluation label format is a simple CSV:

    frame_index,ball_id,x,y,w,h,visible,held,keyframe

CVAT remains the annotation tool, but this module keeps the rest of the repo
independent from CVAT XML details.
"""

from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class GroundTruthLabel:
    """One labelled ball location in one video frame."""

    frame_index: int
    ball_id: int
    x: float
    y: float
    w: float
    h: float
    visible: bool
    held: bool | None
    keyframe: bool


def _parse_boolish(value: str | None) -> bool | None:
    if value is None:
        return None
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "y"}:
        return True
    if normalised in {"0", "false", "no", "n"}:
        return False
    if normalised in {"", "unknown", "none", "null"}:
        return None
    return None


def _read_xml_bytes(path: Path) -> bytes:
    if not path.exists():
        msg = f"CVAT annotation export not found: {path}"
        raise FileNotFoundError(msg)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            candidates = [
                name
                for name in zf.namelist()
                if name.lower().endswith(".xml")
                and Path(name).name.lower() == "annotations.xml"
            ]
            if not candidates:
                candidates = [
                    name for name in zf.namelist() if name.lower().endswith(".xml")
                ]
            if not candidates:
                msg = f"no XML annotation file found in CVAT export: {path}"
                raise ValueError(msg)
            return zf.read(candidates[0])

    return path.read_bytes()


def _attribute_value(box: ET.Element, name: str) -> str | None:
    for attr in box.findall("attribute"):
        if attr.attrib.get("name") == name:
            return attr.text
    return None


def _float_attr(element: ET.Element, name: str) -> float:
    try:
        return float(element.attrib[name])
    except KeyError as exc:
        msg = f"missing required CVAT box attribute: {name}"
        raise ValueError(msg) from exc


def _int_attr(element: ET.Element, name: str) -> int:
    try:
        return int(element.attrib[name])
    except KeyError as exc:
        msg = f"missing required CVAT box attribute: {name}"
        raise ValueError(msg) from exc


def read_cvat_video_labels(
    path: Path,
    *,
    label_name: str = "ball",
    held_attribute: str = "held",
) -> list[GroundTruthLabel]:
    """Read a CVAT for video XML/ZIP export into project-native labels.

    Empty tracks are ignored. Only tracks whose label matches ``label_name`` are
    converted. CVAT track IDs become ``ball_id`` values.
    """
    root = ET.fromstring(_read_xml_bytes(path))
    labels: list[GroundTruthLabel] = []

    for track in root.findall(".//track"):
        if track.attrib.get("label") != label_name:
            continue

        try:
            ball_id = int(track.attrib["id"])
        except KeyError as exc:
            msg = "CVAT track is missing required id attribute"
            raise ValueError(msg) from exc

        for box in track.findall("box"):
            frame_index = _int_attr(box, "frame")
            xtl = _float_attr(box, "xtl")
            ytl = _float_attr(box, "ytl")
            xbr = _float_attr(box, "xbr")
            ybr = _float_attr(box, "ybr")

            outside = box.attrib.get("outside", "0") == "1"
            keyframe = box.attrib.get("keyframe", "0") == "1"
            w = max(0.0, xbr - xtl)
            h = max(0.0, ybr - ytl)
            held = _parse_boolish(_attribute_value(box, held_attribute))

            labels.append(
                GroundTruthLabel(
                    frame_index=frame_index,
                    ball_id=ball_id,
                    x=xtl + w / 2.0,
                    y=ytl + h / 2.0,
                    w=w,
                    h=h,
                    visible=not outside,
                    held=held,
                    keyframe=keyframe,
                )
            )

    return sorted(labels, key=lambda row: (row.frame_index, row.ball_id))


def write_ground_truth_csv(
    labels: list[GroundTruthLabel],
    out_path: Path,
) -> Path:
    """Write project-native ground-truth labels."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "frame_index",
                "ball_id",
                "x",
                "y",
                "w",
                "h",
                "visible",
                "held",
                "keyframe",
            ],
        )
        writer.writeheader()
        for label in labels:
            writer.writerow(
                {
                    "frame_index": label.frame_index,
                    "ball_id": label.ball_id,
                    "x": round(label.x, 3),
                    "y": round(label.y, 3),
                    "w": round(label.w, 3),
                    "h": round(label.h, 3),
                    "visible": int(label.visible),
                    "held": "" if label.held is None else int(label.held),
                    "keyframe": int(label.keyframe),
                }
            )
    return out_path


def read_ground_truth_csv(path: Path) -> list[GroundTruthLabel]:
    """Read project-native ground-truth labels."""
    labels: list[GroundTruthLabel] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            held_raw = row.get("held", "")
            held = None if held_raw == "" else bool(int(held_raw))
            labels.append(
                GroundTruthLabel(
                    frame_index=int(row["frame_index"]),
                    ball_id=int(row["ball_id"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    w=float(row["w"]),
                    h=float(row["h"]),
                    visible=bool(int(row["visible"])),
                    held=held,
                    keyframe=bool(int(row.get("keyframe", "0"))),
                )
            )
    return labels


def convert_cvat_video_to_csv(
    cvat_path: Path,
    out_path: Path,
    *,
    label_name: str = "ball",
    held_attribute: str = "held",
) -> Path:
    """Convert a CVAT for video XML/ZIP export into project-native CSV."""
    labels = read_cvat_video_labels(
        cvat_path,
        label_name=label_name,
        held_attribute=held_attribute,
    )
    if not labels:
        msg = f"no {label_name!r} track boxes found in CVAT export: {cvat_path}"
        raise ValueError(msg)
    return write_ground_truth_csv(labels, out_path)
