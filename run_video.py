"""Run one saved SVM/DT/RF pair on a camera or local video; log every frame."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from threading import Condition, Thread
from time import perf_counter
from uuid import uuid4

from features import LEFT_EYE, RIGHT_EYE, ratios
from temporal import EventTracker, VideoClock
from event_log import EVENT_FIELDS, EventJournal
from stream_input import ResultPublisher, SnapshotCapture

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {'svm': 'svm', 'dt': 'decision_tree', 'rf': 'random_forest'}
FEATURES = ['ear_l', 'ear_r', 'mar']
FIELDS = ['frame_id', 'timestamp_s', 'clock_source', 'status', 'face_count',
          *FEATURES, 'eye_state', 'mouth_state', 'closed_duration_s', 'yawn_duration_s',
          'closed_alert', 'yawn_alert', 'new_closed_event', 'new_yawn_event',
          'capture_ms', 'landmark_ms', 'classifier_ms', 'process_ms',
          'source_frame_id', 'stream_request_ms', 'sender_frame_age_ms', 'preview_ms', 'error']


def load_pair(directory, algorithm):
    import joblib
    import sklearn
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['versions']['sklearn'] != sklearn.__version__:
        raise ValueError(f"Models use scikit-learn {metadata['versions']['sklearn']}; current environment uses {sklearn.__version__}")
    pair = {}
    for task in ('eyes', 'mouth'):
        config = metadata['tasks'][task]
        if config['features'] != FEATURES:
            raise ValueError(f'{task}: model feature order must be {FEATURES}')
        info = config['models'][ALGORITHMS[algorithm]]
        path = (directory / info['file']).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Model path outside model directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Model checksum mismatch: {path}')
        model = joblib.load(path)
        if model.n_features_in_ != 3 or set(model.classes_) != set(config['labels']):
            raise ValueError(f'Model feature/class mismatch: {task}')
        pair[task] = model
    return pair, metadata


def draw_overlay(frame, landmarks, row, algorithm, processing_fps):
    import cv2
    if landmarks:
        height, width = frame.shape[:2]
        for indices, color in [(LEFT_EYE, (0, 255, 0)), (RIGHT_EYE, (255, 120, 0)),
                               ((13, 14, 78, 308), (0, 0, 255))]:
            for i in indices:
                cv2.circle(frame, (int(landmarks[i].x * width), int(landmarks[i].y * height)), 2, color, -1)
    lines = [f"{algorithm.upper()} | {row['status']} | processing {processing_fps:.1f} FPS",
             f"Time {row['timestamp_s']:.2f}s | Frame {row['frame_id']}",
             f"Eyes: {row['eye_state']} | Mouth: {row['mouth_state']}",
             f"Closed {row['closed_duration_s']:.2f}s | Yawn {row['yawn_duration_s']:.2f}s"]
    if all(row[k] is not None and math.isfinite(row[k]) for k in FEATURES):
        lines.append(f"EAR L/R {row['ear_l']:.3f}/{row['ear_r']:.3f} | MAR {row['mar']:.3f}")
    if row['closed_alert']:
        lines.append('ALERT: EYES CLOSED LONG')
    if row['yawn_alert']:
        lines.append('EVENT: SUSTAINED YAWN')
    # Draw each line once. Different stroke weights can change glyph spacing
    # in OpenCV 5, so a thick shadow pass may not align with the foreground.
    height, width = frame.shape[:2]
    padding = max(1, min(10, width // 30, height // 30))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    sizes = [cv2.getTextSize(line, font, scale, 1) for line in lines]
    natural_width = max(size[0][0] for size in sizes)
    natural_height = len(lines) * (max(size[0][1] + size[1] for size in sizes) + 6)
    scale *= min(1.0, max(1, width - 2 * padding) / natural_width,
                 max(1, height - 2 * padding) / natural_height)
    sizes = [cv2.getTextSize(line, font, scale, 1) for line in lines]
    text_height = max(size[0][1] for size in sizes)
    baseline = max(size[1] for size in sizes)
    line_height = text_height + baseline + 6
    panel_width = min(width, max(size[0][0] for size in sizes) + 2 * padding)
    panel_height = min(height, len(lines) * line_height + 2 * padding)
    cv2.rectangle(frame, (0, 0), (panel_width - 1, panel_height - 1), (28, 28, 28), -1)
    for i, line in enumerate(lines):
        y = padding + text_height + i * line_height
        color = (80, 100, 255) if line.startswith(('ALERT', 'EVENT')) else (245, 245, 245)
        cv2.putText(frame, line, (padding, y), fontFace=font, fontScale=scale,
                    color=color, thickness=1, lineType=cv2.LINE_AA)


class LatestPreviewSender:
    """Publish the newest annotated frame without blocking the inference loop."""

    def __init__(self, publisher, algorithm):
        self.publisher = publisher
        self.algorithm = algorithm
        self.condition = Condition()
        self.pending = None
        self.closed = False
        self.sent = 0
        self.errors = 0
        self.dropped = 0
        self.total_ms = 0.0
        self.worker = Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, frame, landmarks, row, processing_fps, event_counts):
        with self.condition:
            if self.pending is not None:
                self.dropped += 1
            self.pending = (frame.copy(), landmarks, row.copy(), processing_fps, event_counts.copy())
            self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.closed:
                    return
                frame, landmarks, row, processing_fps, event_counts = self.pending
                self.pending = None
            started = perf_counter()
            try:
                draw_overlay(frame, landmarks, row, self.algorithm, processing_fps)
                self.publisher.send_preview(frame, row, processing_fps, event_counts)
            except Exception as exc:
                # Keep the preview worker alive if drawing or sending a frame fails.
                with self.condition:
                    self.errors += 1
                    errors = self.errors
                if errors == 1 or errors % 30 == 0:
                    print(f'[PREVIEW WARNING] {exc}', flush=True)
            else:
                with self.condition:
                    self.sent += 1
            finally:
                with self.condition:
                    self.total_ms += (perf_counter() - started) * 1000

    def finish(self):
        with self.condition:
            self.closed = True
            if self.pending is not None:
                self.dropped += 1
                self.pending = None
            self.condition.notify()
        self.worker.join()
        return self.sent, self.errors, self.dropped, self.total_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--video', type=Path, help='Local video; omit to use webcam')
    source.add_argument('--camera', type=int, default=0)
    source.add_argument('--stream-url', help='Laptop frame endpoint, e.g. http://127.0.0.1:8765/frame.jpg via SSH')
    parser.add_argument('--stream-timeout', type=float, default=7.0, help='HTTP read timeout in seconds')
    parser.add_argument('--return-preview', action='store_true', help='Return annotated stream frames to laptop browser')
    parser.add_argument('--result-url', help='Receiver endpoint for annotated camera frames, e.g. http://127.0.0.1:8766/result.jpg')
    parser.add_argument('--preview-fps', type=float, default=5.0, help='Maximum published preview FPS (default: 5)')
    parser.add_argument('--inference-width', type=int, default=0,
                        help='Resize wide frames before Face Landmarker; 0 keeps source size')
    parser.add_argument('--algorithm', choices=ALGORITHMS, default='svm')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models/classifiers')
    parser.add_argument('--landmarker', type=Path, default=ROOT / 'models/face_landmarker.task')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/video_runs')
    parser.add_argument('--closed-seconds', type=float, default=2.0)
    parser.add_argument('--yawn-seconds', type=float, default=1.5)
    parser.add_argument('--max-gap-seconds', type=float, default=0.5)
    parser.add_argument('--fps', type=float, help='FPS for timestamp fallback and saved video; required if metadata FPS is invalid')
    parser.add_argument('--headless', action='store_true', help='Disable preview window')
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument('--save-video', dest='save_video', action='store_true', help='Save annotated MP4 (default for webcam)')
    recording.add_argument('--no-save-video', dest='save_video', action='store_false', help='Disable video recording; CSV logs are still saved')
    parser.set_defaults(save_video=None)
    parser.add_argument('--max-frames', type=int, default=0, help='Stop after N frames; 0 means no limit')
    args = parser.parse_args()
    if not math.isfinite(args.stream_timeout) or args.stream_timeout <= 0:
        parser.error('stream-timeout must be finite and positive')
    if args.return_preview and not args.stream_url:
        parser.error('return-preview requires stream-url')
    if args.return_preview and args.result_url:
        parser.error('use either --return-preview or --result-url')
    if not math.isfinite(args.preview_fps) or not 0 < args.preview_fps <= 60:
        parser.error('preview-fps must be finite and in (0, 60]')
    if args.inference_width < 0:
        parser.error('inference-width must be >= 0')
    try:
        result_publisher = ResultPublisher(args.result_url) if args.result_url else None
    except ValueError as exc:
        parser.error(str(exc))
    if args.stream_url:
        try:
            remote_capture = SnapshotCapture(args.stream_url, args.stream_timeout)
        except ValueError as exc:
            parser.error(str(exc))
    if args.save_video is None:
        args.save_video = args.video is None
    if args.max_frames < 0 or (args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0)):
        parser.error('max-frames must be >= 0; fps must be finite and > 0')
    tracker = EventTracker(args.closed_seconds, args.yawn_seconds, args.max_gap_seconds)
    journal = EventJournal(args.max_gap_seconds)
    if not args.landmarker.is_file() or (args.video and not args.video.is_file()):
        parser.error('Landmarker model or input video does not exist')
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    pair, model_metadata = load_pair(args.models_dir, args.algorithm)
    capture = remote_capture if args.stream_url else cv2.VideoCapture(str(args.video) if args.video else args.camera)
    video_writer = None
    started = perf_counter()
    count = 0
    counts = Counter()
    totals = Counter()
    outcome = 'error'
    run_dir = None
    segment_counts = Counter()
    events_stream = events_writer = None
    preview_sent = preview_errors = 0
    preview_dropped = 0
    preview_total_ms = 0.0
    last_preview = float('-inf')
    preview_sender = None

    def log_events(events):
        for event in events:
            events_writer.writerow(event)
            segment_counts[event['task']] += 1
            print(f"[SEGMENT #{event['event_id']}] {event['task']} {event['predicted_label']}: "
                  f"{event['start_s']:.2f}s - {event['last_positive_s']:.2f}s "
                  f"({event['duration_s']:.2f}s), confirmed={event['confirmed']}, "
                  f"end={event['end_reason']}", flush=True)
        if events:
            events_stream.flush()

    try:
        if not capture.isOpened():
            raise RuntimeError('Could not open video/camera')
        fps = args.fps if args.fps is not None else (15.0 if args.stream_url else capture.get(cv2.CAP_PROP_FPS))
        clock = VideoClock(fps) if args.video else None
        save_fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        run_dir = args.output_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{args.algorithm}_{uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(args.landmarker.resolve())),
            # Same mode as training feature extraction; no landmark tracking/smoothing.
            running_mode=vision.RunningMode.IMAGE, num_faces=2)
        print(f"Running {args.algorithm}: press Q or Esc to stop. Logs: {run_dir}", flush=True)
        print(f"Annotated recording: {'ON' if args.save_video else 'OFF'}. "
              'Optional manual markers: C=close eyes, O=open eyes, Y=yawn start, N=yawn end.', flush=True)
        if args.stream_url:
            print('Laptop stream: event time uses sender capture timestamps; capture_ms includes network/JPEG.', flush=True)
        if args.return_preview:
            print(f'Returning annotated preview to laptop (up to {args.preview_fps:g} FPS). Open laptop monitor page.', flush=True)
        if args.result_url:
            print(f'Publishing annotated preview to {args.result_url} (up to {args.preview_fps:g} FPS).', flush=True)
            preview_sender = LatestPreviewSender(result_publisher, args.algorithm)
        events_stream = (run_dir / 'events.csv').open('w', newline='', encoding='utf-8')
        events_writer = csv.DictWriter(events_stream, fieldnames=EVENT_FIELDS)
        events_writer.writeheader()
        events_stream.flush()
        with ExitStack() as stack:
            stream = stack.enter_context((run_dir / 'frames.csv').open('w', newline='', encoding='utf-8'))
            markers_stream = stack.enter_context((run_dir / 'manual_markers.csv').open('w', newline='', encoding='utf-8'))
            markers_writer = csv.DictWriter(markers_stream, fieldnames=['frame_id', 'timestamp_s', 'marker'])
            markers_writer.writeheader()
            detector = stack.enter_context(vision.FaceLandmarker.create_from_options(options))
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            while not args.max_frames or count < args.max_frames:
                capture_start = perf_counter()
                ok, frame = capture.read()
                captured = perf_counter()
                if not ok:
                    if not args.video or count == 0:
                        raise RuntimeError('No frame received from source')
                    outcome = 'end_of_video'
                    break
                if args.stream_url:
                    timestamp, basis = capture.timestamp, 'sender_monotonic'
                else:
                    timestamp, basis = clock.next(capture.get(cv2.CAP_PROP_POS_MSEC)) if clock else (captured - started, 'monotonic')
                row = dict.fromkeys(FIELDS)
                row.update(frame_id=count, timestamp_s=timestamp, clock_source=basis,
                           status='OK', face_count=0, eye_state='UNKNOWN', mouth_state='UNKNOWN',
                           capture_ms=(captured - capture_start) * 1000,
                           landmark_ms=0.0, classifier_ms=0.0,
                           source_frame_id=capture.frame_id if args.stream_url else None,
                           stream_request_ms=capture.request_ms if args.stream_url else None,
                           sender_frame_age_ms=capture.sender_frame_age_ms if args.stream_url else None,
                           preview_ms=0.0, error='')
                landmarks = None
                try:
                    inference_frame = frame
                    if args.inference_width and frame.shape[1] > args.inference_width:
                        height = max(1, round(frame.shape[0] * args.inference_width / frame.shape[1]))
                        inference_frame = cv2.resize(frame, (args.inference_width, height),
                                                     interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    landmark_start = perf_counter()
                    try:
                        faces = detector.detect(image).face_landmarks
                    finally:
                        row['landmark_ms'] = (perf_counter() - landmark_start) * 1000
                    row['face_count'] = len(faces)
                    if len(faces) != 1:
                        row['status'] = 'NO_FACE' if not faces else 'MULTIPLE_FACES'
                    else:
                        landmarks = faces[0]
                        row.update(ratios(landmarks, image.width, image.height))
                        values = [row[k] for k in FEATURES]
                        if any(v is None or not math.isfinite(v) or v < 0 for v in values):
                            row['status'] = 'INVALID_GEOMETRY'
                        else:
                            predict_start = perf_counter()
                            try:
                                row['eye_state'] = str(pair['eyes'].predict([values])[0])
                                row['mouth_state'] = str(pair['mouth'].predict([values])[0])
                            finally:
                                row['classifier_ms'] = (perf_counter() - predict_start) * 1000
                except (ValueError, RuntimeError, OSError) as exc:
                    row.update(status='INFERENCE_ERROR', error=str(exc))
                if row['status'] != 'OK':
                    row.update(eye_state='UNKNOWN', mouth_state='UNKNOWN')
                row.update(tracker.update(timestamp, row['eye_state'], row['mouth_state']))
                row['process_ms'] = (perf_counter() - capture_start) * 1000
                count += 1
                counts[row['status']] += 1
                for key in ('capture_ms', 'landmark_ms', 'classifier_ms', 'process_ms'):
                    totals[key] += row[key]
                if row['status'] == 'OK':
                    totals['valid_classifier_ms'] += row['classifier_ms']
                publish_preview = (args.return_preview or result_publisher is not None) and perf_counter() - last_preview >= 1 / args.preview_fps
                processing_fps = 1000 * count / totals['process_ms']
                if publish_preview and preview_sender is not None:
                    last_preview = perf_counter()
                    preview_sender.submit(frame, landmarks, row, processing_fps, tracker.counts)
                if not args.headless or args.save_video or (publish_preview and preview_sender is None):
                    draw_overlay(frame, landmarks, row, args.algorithm, processing_fps)
                if publish_preview and preview_sender is None:
                    last_preview = perf_counter()
                    preview_start = perf_counter()
                    try:
                        capture.send_preview(frame)
                        preview_sent += 1
                    except RuntimeError as exc:
                        preview_errors += 1
                        if preview_errors == 1 or preview_errors % 30 == 0:
                            print(f'[PREVIEW WARNING] {exc}', flush=True)
                    row['preview_ms'] = (perf_counter() - preview_start) * 1000
                    preview_total_ms += row['preview_ms']
                writer.writerow(row)
                log_events(journal.update(row))
                if row['new_closed_event'] or row['new_yawn_event']:
                    stream.flush()
                    active = [name for name, flag in [('eyes', row['new_closed_event']), ('mouth', row['new_yawn_event'])] if flag]
                    print(f"[CONFIRMED at {timestamp:.2f}s] {', '.join(active)}", flush=True)
                if count % 30 == 0:
                    stream.flush()
                    states = f" | Eyes {row['eye_state']} | Mouth {row['mouth_state']}" if args.stream_url else ''
                    print(f"Frames {count} | {row['status']}{states} | process {totals['process_ms']/count:.1f} ms/frame", flush=True)
                if args.save_video:
                    if video_writer is None:
                        height, width = frame.shape[:2]
                        video_writer = cv2.VideoWriter(str(run_dir / 'annotated.mp4'),
                            cv2.VideoWriter_fourcc(*'mp4v'), save_fps, (width, height))
                        if not video_writer.isOpened():
                            raise RuntimeError('Could not create annotated video')
                    video_writer.write(frame)
                if not args.headless:
                    cv2.imshow('Driver facial events - ' + args.algorithm, frame)
                    key = cv2.waitKey(1) & 0xFF
                    letter = key + 32 if ord('A') <= key <= ord('Z') else key
                    marker = {ord('c'): 'USER_EYES_CLOSE', ord('o'): 'USER_EYES_OPEN',
                              ord('y'): 'USER_YAWN_START', ord('n'): 'USER_YAWN_END'}.get(letter)
                    if marker:
                        marker_time = row['timestamp_s'] if args.video or args.stream_url else perf_counter() - started
                        markers_writer.writerow({'frame_id': row['frame_id'], 'timestamp_s': marker_time, 'marker': marker})
                        markers_stream.flush()
                        print(f'[MANUAL at {marker_time:.2f}s] {marker}', flush=True)
                    if key in (27, ord('q'), ord('Q')):
                        outcome = 'user_stopped'
                        break
                    if cv2.getWindowProperty('Driver facial events - ' + args.algorithm, cv2.WND_PROP_VISIBLE) < 1:
                        outcome = 'user_stopped'
                        break
            else:
                outcome = 'frame_limit'
    except KeyboardInterrupt:
        outcome = 'interrupted'
    finally:
        if events_stream is not None:
            try:
                log_events(journal.finish(outcome))
            finally:
                events_stream.close()
        capture.release()
        if preview_sender is not None:
            preview_sent, preview_errors, preview_dropped, preview_total_ms = preview_sender.finish()
        if video_writer is not None:
            video_writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if run_dir is not None:
            wall_seconds = perf_counter() - started
            summary = {'outcome': outcome, 'source': args.stream_url or (str(args.video.resolve()) if args.video else f'camera:{args.camera}'),
                'source_type': 'laptop_stream' if args.stream_url else ('video' if args.video else 'camera'),
                'stream_skipped_frames': capture.skipped_frames if args.stream_url else None,
                'algorithm': args.algorithm, 'features': FEATURES, 'frames': count, 'status_counts': dict(counts),
                'wall_seconds': wall_seconds, 'wall_fps': count / wall_seconds if wall_seconds else None,
                'mean_ms': {key: totals[key] / count if count else None for key in ('capture_ms', 'landmark_ms', 'classifier_ms', 'process_ms')},
                'valid_mean_classifier_ms': totals['valid_classifier_ms'] / counts['OK'] if counts['OK'] else None,
                'processing_fps': 1000 * count / totals['process_ms'] if totals['process_ms'] else None,
                'event_counts': tracker.counts,
                'positive_segment_counts': dict(segment_counts),
                'event_rules': {'closed_seconds': args.closed_seconds, 'yawn_seconds': args.yawn_seconds, 'max_gap_seconds': args.max_gap_seconds},
                'landmark_mode': 'IMAGE', 'inference_width': args.inference_width,
                'source_fps': None if args.stream_url else save_fps,
                'recording_fps': save_fps if args.save_video else None,
                'versions': {'opencv': cv2.__version__, 'mediapipe': mp.__version__,
                             'sklearn': version('scikit-learn'), 'numpy': version('numpy'),
                             'joblib': version('joblib'), 'model_training': model_metadata['versions']},
                'model_directory': str(args.models_dir.resolve()),
                'models': {task: model_metadata['tasks'][task]['models'][ALGORITHMS[args.algorithm]] for task in ('eyes', 'mouth')},
                'headless': args.headless, 'save_video': args.save_video,
                'return_preview': args.return_preview, 'result_url': args.result_url,
                'preview_fps_limit': args.preview_fps,
                'preview_frames_sent': preview_sent, 'preview_errors': preview_errors,
                'preview_frames_dropped': preview_dropped,
                'preview_ms_total': preview_total_ms,
                'notes': ['Experimental thresholds, not validated drowsiness labels.',
                          'process_ms includes capture, landmarks, classifiers and temporal logic; excludes overlay, file writes and display.',
                          'classifier_ms measures the pair of classifiers together, not MediaPipe.',
                          'No ground-truth labels loaded: these are predictions/timings, not accuracy or F1.',
                          'events.csv records all predicted CLOSED/YAWN intervals; confirmed means the experimental duration threshold was reached.',
                          'Event durations end at the last positive observation; ended_at_s is when the interval was terminated.',
                          'Manual keyboard markers are approximate self-reports, not independently verified ground truth.',
                          'Video uses source presentation times with FPS fallback; webcam uses monotonic capture times.',
                          'Laptop stream uses sender monotonic capture times relative to its first received frame; skipped frames are not inferred.',
                          'For laptop stream, capture_ms/process_ms include HTTP waiting, network transfer and JPEG decoding; they are not pure Pi compute latency.',
                          'For result-url, preview is published by a background thread with a single latest-frame slot. Per-frame preview_ms is zero; preview_ms_total includes drawing, JPEG encoding and HTTP return.',
                          'wall_fps counts completed frames over elapsed runtime; source_fps for a camera is reported by the camera driver and may differ.',
                          'Annotated output uses constant FPS; webcam event timing is defined by frames.csv.']}
            (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(f'Finished: {outcome}, {count} frames. Results: {run_dir}', flush=True)
            print(f"Confirmed events: eyes={tracker.counts['eyes']}, mouth={tracker.counts['mouth']}. "
                  'Review events.csv alongside frames.csv and annotated.mp4.', flush=True)


if __name__ == '__main__':
    main()
