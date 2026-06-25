from __future__ import annotations

import numpy as np

from superjuggling.config import SegmentConfig
from superjuggling.models import Trajectory
from superjuggling.segments import segment_apex, segment_trajectories_into_flights


def test_segment_fitting_recovers_apex() -> None:
    t = np.linspace(0.0, 1.0, 31)
    x = 700.0 + 20.0 * t
    y = 200.0 + 400.0 * ((t - 0.5) / 0.5) ** 2
    traj = Trajectory(track_id=1, t=t, x=x, y=y)

    segments = segment_trajectories_into_flights([traj], SegmentConfig())

    assert len(segments) == 1
    apex = segment_apex(segments[0])
    assert apex is not None
    t_apex, x_apex, y_apex = apex
    assert abs(t_apex - 0.5) < 0.01
    assert abs(x_apex - 710.0) < 1.0
    assert abs(y_apex - 200.0) < 1.0
