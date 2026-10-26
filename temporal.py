"""Experimental event confirmation from consecutive, valid observations."""
import math


class EventTracker:
    def __init__(self, closed_seconds=2.0, yawn_seconds=1.5, max_gap=0.5):
        if any(not math.isfinite(v) or v <= 0 for v in (closed_seconds, yawn_seconds, max_gap)):
            raise ValueError('Event durations and max gap must be finite and positive')
        self.thresholds = {'eyes': closed_seconds, 'mouth': yawn_seconds}
        self.max_gap = max_gap
        self.previous_time = None
        self.starts = {'eyes': None, 'mouth': None}
        self.active = {'eyes': False, 'mouth': False}
        self.counts = {'eyes': 0, 'mouth': 0}

    def reset(self):
        self.starts = {'eyes': None, 'mouth': None}
        self.active = {'eyes': False, 'mouth': False}

    def update(self, timestamp, eye_state, mouth_state):
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Invalid observation timestamp')
        if self.previous_time is not None:
            gap = timestamp - self.previous_time
            if gap < 0 or gap > self.max_gap:
                self.reset()
        self.previous_time = timestamp
        if eye_state not in {'OPEN', 'CLOSED'} or mouth_state not in {'NO_YAWN', 'YAWN'}:
            self.reset()
            return self._result({'eyes': 0.0, 'mouth': 0.0}, [])
        durations, events = {}, []
        for name, positive in [('eyes', eye_state == 'CLOSED'), ('mouth', mouth_state == 'YAWN')]:
            if not positive:
                self.starts[name] = None
                self.active[name] = False
                durations[name] = 0.0
                continue
            if self.starts[name] is None:
                self.starts[name] = timestamp
            durations[name] = timestamp - self.starts[name]
            if durations[name] >= self.thresholds[name] and not self.active[name]:
                self.active[name] = True
                self.counts[name] += 1
                events.append(name)
        return self._result(durations, events)

    def _result(self, durations, events):
        return {'closed_duration_s': durations['eyes'], 'yawn_duration_s': durations['mouth'],
                'closed_alert': self.active['eyes'], 'yawn_alert': self.active['mouth'],
                'new_closed_event': 'eyes' in events, 'new_yawn_event': 'mouth' in events}


class VideoClock:
    """Use decoded presentation times, falling back to FPS when unavailable."""
    def __init__(self, fps):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video FPS is invalid; specify --fps')
        self.step = 1.0 / fps
        self.origin = None
        self.previous = None

    def next(self, position_ms):
        valid = math.isfinite(position_ms) and position_ms >= 0
        if self.previous is None:
            self.origin = position_ms / 1000 if valid else 0.0
            self.previous = 0.0
            return 0.0, 'video_start'
        candidate = position_ms / 1000 - self.origin if valid else -1
        if candidate > self.previous:
            self.previous = candidate
            return candidate, 'video_pts'
        self.previous += self.step
        return self.previous, 'fps_fallback'
