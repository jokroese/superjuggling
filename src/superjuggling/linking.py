"""Centre-candidate linking backends.

The default video path still uses ByteTrack for compatibility. These pure
Python linkers operate directly on ``BallCandidate`` streams and provide the
new architecture seam:

    BallCandidate -> CandidateLinker -> Trajectory
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import LinkingConfig
from .models import BallCandidate, CandidateFrame, Trajectory


@runtime_checkable
class CandidateLinker(Protocol):
    def update(self, candidates: list[BallCandidate]) -> None: ...
    def trajectories(self) -> list[Trajectory]: ...


@dataclass
class _ActiveTrack:
    track_id: int
    samples: list[BallCandidate] = field(default_factory=list)
    missed_frames: int = 0

    @property
    def last(self) -> BallCandidate:
        return self.samples[-1]


@dataclass(frozen=True)
class LinkDebugRow:
    frame_index: int
    track_id: int
    candidate_x: float | None
    candidate_y: float | None
    candidate_score: float | None
    predicted_x: float | None
    predicted_y: float | None
    cost: float | None
    status: str


def _candidate_distance(a: BallCandidate, xy: tuple[float, float]) -> float:
    return float(np.hypot(a.x - xy[0], a.y - xy[1]))


def _to_trajectory(track: _ActiveTrack) -> Trajectory | None:
    if not track.samples:
        return None
    arr = np.asarray(
        [(c.t, c.x, c.y, c.score) for c in track.samples],
        dtype=np.float64,
    )
    return Trajectory(
        track_id=track.track_id,
        t=arr[:, 0],
        x=arr[:, 1],
        y=arr[:, 2],
        confidence=arr[:, 3],
    )


class CentreLinker:
    """Simple centre-point linker with velocity prediction."""

    def __init__(self, cfg: LinkingConfig) -> None:
        self.cfg = cfg
        self._next_id = 1
        self._active: dict[int, _ActiveTrack] = {}
        self._finished: list[_ActiveTrack] = []
        self.debug_rows: list[LinkDebugRow] = []

    def update(self, candidates: list[BallCandidate]) -> None:
        if not candidates:
            self._miss_all()
            return

        if not self._active:
            for cand in candidates:
                self._start_track(cand)
            return

        tracks = list(self._active.values())
        cost = np.full((len(tracks), len(candidates)), np.inf, dtype=np.float64)
        predictions: list[tuple[float, float]] = []

        for i, track in enumerate(tracks):
            pred = self._predict(track, candidates[0].t)
            predictions.append(pred)
            for j, cand in enumerate(candidates):
                cost[i, j] = self._association_cost(track, cand, pred)

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_tracks: set[int] = set()
        matched_candidates: set[int] = set()

        for r, c in zip(row_ind, col_ind, strict=False):
            value = float(cost[r, c])
            if not np.isfinite(value) or value > self.cfg.max_match_distance_px:
                continue
            track = tracks[r]
            cand = candidates[c]
            pred = predictions[r]
            track.samples.append(cand)
            track.missed_frames = 0
            matched_tracks.add(track.track_id)
            matched_candidates.add(c)
            self.debug_rows.append(
                LinkDebugRow(
                    frame_index=cand.frame_index,
                    track_id=track.track_id,
                    candidate_x=cand.x,
                    candidate_y=cand.y,
                    candidate_score=cand.score,
                    predicted_x=pred[0],
                    predicted_y=pred[1],
                    cost=value,
                    status="matched",
                )
            )

        for track in tracks:
            if track.track_id not in matched_tracks:
                self._miss_track(track)

        for j, cand in enumerate(candidates):
            if j not in matched_candidates:
                self._start_track(cand)

    def _start_track(self, cand: BallCandidate) -> None:
        track = _ActiveTrack(track_id=self._next_id, samples=[cand])
        self._next_id += 1
        self._active[track.track_id] = track
        self.debug_rows.append(
            LinkDebugRow(
                frame_index=cand.frame_index,
                track_id=track.track_id,
                candidate_x=cand.x,
                candidate_y=cand.y,
                candidate_score=cand.score,
                predicted_x=None,
                predicted_y=None,
                cost=None,
                status="started",
            )
        )

    def _miss_all(self) -> None:
        for track in list(self._active.values()):
            self._miss_track(track)

    def _miss_track(self, track: _ActiveTrack) -> None:
        track.missed_frames += 1
        last = track.last
        self.debug_rows.append(
            LinkDebugRow(
                frame_index=last.frame_index + track.missed_frames,
                track_id=track.track_id,
                candidate_x=None,
                candidate_y=None,
                candidate_score=None,
                predicted_x=None,
                predicted_y=None,
                cost=None,
                status="missed",
            )
        )
        if len(track.samples) > 1:
            sample_times = [s.t for s in track.samples]
            median_dt = max(float(np.median(np.diff(sample_times))), 1e-6)
            fps = 1.0 / median_dt
        else:
            fps = 30.0
        max_missed = max(1, int(round(self.cfg.max_gap_s * fps)))
        if track.missed_frames > max_missed:
            self._finish_track(track.track_id)

    def _finish_track(self, track_id: int) -> None:
        track = self._active.pop(track_id)
        if len(track.samples) >= self.cfg.min_track_points:
            self._finished.append(track)

    def _predict(self, track: _ActiveTrack, t: float) -> tuple[float, float]:
        if len(track.samples) < 2:
            return track.last.x, track.last.y
        a, b = track.samples[-2], track.samples[-1]
        dt = max(b.t - a.t, 1e-6)
        vx = (b.x - a.x) / dt
        vy = (b.y - a.y) / dt
        future = max(t - b.t, 0.0)
        return b.x + vx * future, b.y + vy * future

    def _association_cost(
        self,
        track: _ActiveTrack,
        cand: BallCandidate,
        pred: tuple[float, float],
    ) -> float:
        return (
            _candidate_distance(cand, pred) - self.cfg.confidence_bonus_px * cand.score
        )

    def trajectories(self) -> list[Trajectory]:
        tracks = [*self._finished, *self._active.values()]
        out: list[Trajectory] = []
        for track in tracks:
            if len(track.samples) < self.cfg.min_track_points:
                continue
            traj = _to_trajectory(track)
            if traj is not None:
                out.append(traj)
        return out


class PhysicsLinker(CentreLinker):
    """Centre linker with short-window ballistic consistency in the cost."""

    def _association_cost(
        self,
        track: _ActiveTrack,
        cand: BallCandidate,
        pred: tuple[float, float],
    ) -> float:
        base = super()._association_cost(track, cand, pred)
        if len(track.samples) < 5:
            return base

        residual = self._physics_residual(track, cand)
        if residual is None:
            return base
        return base + self.cfg.physics_residual_weight * residual

    def _physics_residual(
        self,
        track: _ActiveTrack,
        cand: BallCandidate,
        window: int = 8,
    ) -> float | None:
        samples = track.samples[-window:]
        if len(samples) < 5:
            return None

        t_ref = samples[-1].t
        u = np.asarray([s.t - t_ref for s in samples], dtype=np.float64)
        x = np.asarray([s.x for s in samples], dtype=np.float64)
        y = np.asarray([s.y for s in samples], dtype=np.float64)

        try:
            x_coeff, *_ = np.linalg.lstsq(
                np.column_stack((u, np.ones_like(u))), x, rcond=None
            )
            y_coeff, *_ = np.linalg.lstsq(
                np.column_stack((u * u, u, np.ones_like(u))), y, rcond=None
            )
        except np.linalg.LinAlgError:
            return None

        cand_u = cand.t - t_ref
        pred_x = float(x_coeff[0] * cand_u + x_coeff[1])
        pred_y = float(y_coeff[0] * cand_u * cand_u + y_coeff[1] * cand_u + y_coeff[2])
        return float(np.hypot(cand.x - pred_x, cand.y - pred_y))


def link_candidate_frames(
    frames: list[CandidateFrame],
    cfg: LinkingConfig,
) -> tuple[list[Trajectory], list[LinkDebugRow]]:
    """Run the selected pure-Python candidate linker."""
    if cfg.backend == "centre":
        linker: CentreLinker = CentreLinker(cfg)
    elif cfg.backend == "physics":
        linker = PhysicsLinker(cfg)
    else:
        msg = "link_candidate_frames only supports centre/physics backends"
        raise ValueError(msg)

    for frame in frames:
        linker.update(frame.candidates)
    return linker.trajectories(), linker.debug_rows


def write_links_csv(rows: list[LinkDebugRow], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "frame_index",
                "track_id",
                "candidate_x",
                "candidate_y",
                "candidate_score",
                "predicted_x",
                "predicted_y",
                "cost",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return out_path
