"""Walking / running recognition for tracked people.

No trained model is needed. Each person's speed is measured from their tracked
bounding box over a short sliding window and expressed in body-heights per
second, so it works without camera calibration. Thresholds are approximate
(a walking adult covers ~0.8 body heights/s, a run is usually >1.3) and can be
tuned per camera.

Assumes a fixed camera (the same assumption the zone polygons already make).
A single image has no motion, so activity cannot be measured from one frame.
"""

import math
import statistics
from collections import deque

WALKING = "walking"
RUNNING = "running"
STANDING = "standing"
UNKNOWN = "unknown"


class ActivityRecognizer:
    """Classify each tracked person as standing, walking or running."""

    def __init__(
        self,
        window_seconds=1.0,
        min_seconds=0.5,
        stationary_speed=0.1,
        run_enter_speed=0.8,
        run_exit_speed=0.8,
        track_timeout_seconds=3.0,
    ):
        if min_seconds <= 0 or window_seconds < min_seconds:
            raise ValueError("Need 0 < min_seconds <= window_seconds.")
        if not stationary_speed < run_exit_speed <= run_enter_speed:
            raise ValueError(
                "Need stationary_speed < run_exit_speed <= run_enter_speed."
            )
        self.window_seconds = window_seconds
        self.min_seconds = min_seconds
        self.stationary_speed = stationary_speed
        self.run_enter_speed = run_enter_speed
        # Separate enter/exit thresholds stop the label flickering near the limit.
        self.run_exit_speed = run_exit_speed
        self.track_timeout_seconds = track_timeout_seconds
        self.tracks = {}

    @staticmethod
    def _measure(samples):
        """Return speed in body-heights/second from a list of (t, x, y, h)."""
        n = len(samples)
        span = samples[-1][0] - samples[0][0]
        k = max(1, n // 4)
        # Average the ends of the window to damp box jitter and gait bobbing.
        start, end = samples[:k], samples[-k:]
        sx = statistics.fmean(s[1] for s in start)
        sy = statistics.fmean(s[2] for s in start)
        ex = statistics.fmean(s[1] for s in end)
        ey = statistics.fmean(s[2] for s in end)
        t0 = statistics.fmean(s[0] for s in start)
        t1 = statistics.fmean(s[0] for s in end)
        height = statistics.median(s[3] for s in samples)
        dt = t1 - t0
        if dt <= 0 or height <= 0 or span <= 0:
            return None
        return math.hypot(ex - sx, ey - sy) / height / dt

    def update(self, workers, video_time):
        """Set worker['activity'] and worker['activity_speed'] in place."""
        now = float(video_time)

        # Forget people who vanished so reused IDs start with fresh history.
        for key in list(self.tracks):
            if now - self.tracks[key]["last"] >= self.track_timeout_seconds:
                del self.tracks[key]

        for key, worker in workers.items():
            worker["activity"] = UNKNOWN
            worker["activity_speed"] = None

            # Untracked people have no identity to build a motion history on.
            if worker.get("track_id") is None:
                continue

            x1, y1, x2, y2 = worker["bbox"]
            state = self.tracks.setdefault(
                key, {"samples": deque(), "label": UNKNOWN, "last": now}
            )
            samples = state["samples"]

            if samples and now <= samples[-1][0]:
                continue  # Ignore repeated or out-of-order timestamps.

            samples.append((now, (x1 + x2) / 2, (y1 + y2) / 2, y2 - y1))
            state["last"] = now
            while samples and now - samples[0][0] > self.window_seconds:
                samples.popleft()

            if samples[-1][0] - samples[0][0] < self.min_seconds:
                continue  # Not enough history yet.

            speed = self._measure(list(samples))
            if speed is None:
                continue

            previous = state["label"]
            run_limit = (
                self.run_exit_speed
                if previous == RUNNING
                else self.run_enter_speed
            )
            if speed >= run_limit:
                label = RUNNING
            elif speed >= self.stationary_speed:
                label = WALKING
            else:
                label = STANDING

            state["label"] = label
            worker["activity"] = label
            worker["activity_speed"] = speed

        return workers

    @staticmethod
    def label_still(workers):
        """A single image has no motion, so activity cannot be determined."""
        for worker in workers.values():
            worker["activity"] = "n/a (single image)"
            worker["activity_speed"] = None
        return workers
