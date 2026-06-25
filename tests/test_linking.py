from __future__ import annotations

from superjuggling.config import LinkingConfig
from superjuggling.linking import link_candidate_frames
from superjuggling.models import BallCandidate, CandidateFrame


def _frame(i: int, *points: tuple[float, float]) -> CandidateFrame:
    return CandidateFrame(
        frame_index=i,
        t=i / 60.0,
        candidates=[
            BallCandidate(i, i / 60.0, x, y, 1.0, source="test") for x, y in points
        ],
    )


def test_centre_linker_links_smooth_motion() -> None:
    frames = [_frame(i, (100.0 + i * 2, 200.0)) for i in range(10)]

    trajectories, rows = link_candidate_frames(
        frames,
        LinkingConfig(backend="centre", min_track_points=3),
    )

    assert len(trajectories) == 1
    assert len(trajectories[0].t) == 10
    assert rows


def test_physics_linker_links_parabolic_motion() -> None:
    frames: list[CandidateFrame] = []
    for i in range(20):
        t = i / 60.0
        x = 100.0 + 30.0 * t
        y = 300.0 - 80.0 * t + 200.0 * t * t
        frames.append(
            CandidateFrame(
                frame_index=i,
                t=t,
                candidates=[BallCandidate(i, t, x, y, 1.0, source="test")],
            )
        )

    trajectories, _ = link_candidate_frames(
        frames,
        LinkingConfig(backend="physics", min_track_points=5),
    )

    assert len(trajectories) == 1
    assert len(trajectories[0].t) == 20
