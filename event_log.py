"""Summarize predicted positive intervals, including those below alert thresholds."""

EVENT_FIELDS = ['event_id', 'task', 'predicted_label', 'start_s', 'last_positive_s',
                'ended_at_s', 'duration_s', 'start_frame', 'last_positive_frame',
                'confirmed', 'confirmed_at_s', 'end_reason']


class EventJournal:
    def __init__(self, max_gap):
        self.max_gap = max_gap
        self.previous_time = None
        self.next_id = 1
        self.active = {}

    def _close(self, task, ended_at, reason):
        event = self.active.pop(task)
        return {**event, 'ended_at_s': ended_at,
                'duration_s': event['last_positive_s'] - event['start_s'],
                'end_reason': reason}

    def update(self, row):
        timestamp = row['timestamp_s']
        closed = []
        gap = timestamp - self.previous_time if self.previous_time is not None else 0
        if gap < 0 or gap > self.max_gap:
            closed.extend(self.finish('observation_gap', timestamp))
        self.previous_time = timestamp
        valid = row['status'] == 'OK'
        for task, key, label, alert in [('eyes', 'eye_state', 'CLOSED', 'closed_alert'),
                                         ('mouth', 'mouth_state', 'YAWN', 'yawn_alert')]:
            positive = valid and row[key] == label
            if task in self.active and not positive:
                reason = 'prediction_changed' if valid else row['status']
                closed.append(self._close(task, timestamp, reason))
            if not positive:
                continue
            if task not in self.active:
                self.active[task] = {'event_id': self.next_id, 'task': task, 'predicted_label': label,
                    'start_s': timestamp, 'last_positive_s': timestamp,
                    'start_frame': row['frame_id'], 'last_positive_frame': row['frame_id'],
                    'confirmed': False, 'confirmed_at_s': None}
                self.next_id += 1
            event = self.active[task]
            event.update(last_positive_s=timestamp, last_positive_frame=row['frame_id'])
            if row[alert] and not event['confirmed']:
                event.update(confirmed=True, confirmed_at_s=timestamp)
        return closed

    def finish(self, reason='session_end', timestamp=None):
        # Without a subsequent observation, end at the last positive timestamp.
        return [self._close(task, self.active[task]['last_positive_s'] if timestamp is None else timestamp, reason)
                for task in list(self.active)]
