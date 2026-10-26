"""Send laptop webcam frames to Pi through an SSH tunnel; no ML runs here."""
import argparse
import math
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Event, Thread
from time import perf_counter
from urllib.parse import parse_qs, urlsplit

from monitor_page import PAGE
from stream_input import MAX_JPEG_BYTES


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: int
    captured_at: float
    jpeg: bytes


class LatestFrame:
    """Single-slot cache: a slow receiver skips old frames instead of buffering."""

    def __init__(self):
        self.condition = Condition()
        self.frame = None
        self.error = None

    def publish(self, jpeg, captured_at):
        with self.condition:
            frame_id = 0 if self.frame is None else self.frame.frame_id + 1
            self.frame = CapturedFrame(frame_id, captured_at, jpeg)
            self.condition.notify_all()

    def fail(self, message):
        with self.condition:
            self.error = message
            self.condition.notify_all()

    def get_after(self, after, timeout=5.0):
        deadline = perf_counter() + timeout
        with self.condition:
            while True:
                if self.error:
                    raise RuntimeError(self.error)
                if self.frame is not None and self.frame.frame_id > after:
                    if perf_counter() - self.frame.captured_at > 2.0:
                        raise RuntimeError('Webcam frame is stale')
                    return self.frame
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise TimeoutError('No new webcam frame')
                self.condition.wait(remaining)


@dataclass(frozen=True)
class ReturnedFrame:
    frame_id: int
    captured_at: float
    received_at: float
    jpeg: bytes


class PreviewFrames:
    def __init__(self):
        self.condition = Condition()
        self.frame = None

    def publish(self, jpeg, frame_id, captured_at):
        with self.condition:
            if self.frame is not None and frame_id <= self.frame.frame_id:
                raise ValueError('Preview frame IDs must increase')
            self.frame = ReturnedFrame(frame_id, captured_at, perf_counter(), jpeg)

    def get_after(self, after):
        with self.condition:
            return self.frame if self.frame is not None and self.frame.frame_id > after else None


class FrameServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, frames):
        self.frames = frames
        self.previews = PreviewFrames()
        super().__init__(address, FrameHandler)


class FrameHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parts = urlsplit(self.path)
        if parts.path == '/':
            payload = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
            return
        if parts.path not in {'/frame.jpg', '/result.jpg'}:
            self.send_error(404, 'Use /, /frame.jpg or /result.jpg')
            return
        try:
            after = int(parse_qs(parts.query).get('after', ['-1'])[0])
            if after < -1:
                raise ValueError
        except ValueError:
            self.send_error(400, 'after must be an integer >= -1')
            return
        is_preview = parts.path == '/result.jpg'
        if is_preview:
            frame = self.server.previews.get_after(after)
            if frame is None:
                self.send_response(204)
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                return
        else:
            try:
                frame = self.server.frames.get_after(after)
            except (RuntimeError, TimeoutError) as exc:
                self.send_error(503, str(exc))
                return
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(frame.jpeg)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Frame-Id', str(frame.frame_id))
            self.send_header('X-Capture-Time', f'{frame.captured_at:.9f}')
            self.send_header('X-Frame-Age-Ms', f'{(perf_counter() - frame.captured_at) * 1000:.3f}')
            if is_preview:
                self.send_header('X-Preview-Age-Ms', f'{(perf_counter() - frame.received_at) * 1000:.3f}')
            self.end_headers()
            self.wfile.write(frame.jpeg)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Receiver may stop while a frame is being sent.

    def do_POST(self):
        if urlsplit(self.path).path != '/result.jpg':
            self.send_error(404, 'Use /result.jpg')
            return
        try:
            size = int(self.headers['Content-Length'])
            frame_id = int(self.headers['X-Source-Frame-Id'])
            captured_at = float(self.headers['X-Source-Capture-Time'])
            if (self.headers.get_content_type() != 'image/jpeg' or not 0 < size <= MAX_JPEG_BYTES
                    or frame_id < 0 or not math.isfinite(captured_at) or captured_at < 0):
                raise ValueError('Invalid preview headers')
            self.connection.settimeout(3)
            payload = self.rfile.read(size)
            if len(payload) != size or not payload.startswith(b'\xff\xd8') or not payload.endswith(b'\xff\xd9'):
                raise ValueError('Invalid preview JPEG')
            self.server.previews.publish(payload, frame_id, captured_at)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.send_error(400, str(exc))
            return
        self.send_response(204)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def capture_frames(capture, frames, stopped, fps, quality):
    import cv2

    next_publish = 0.0
    try:
        while not stopped.is_set():
            ok, frame = capture.read()
            captured_at = perf_counter()
            if not ok:
                raise RuntimeError('Could not read laptop webcam')
            # Keep draining the camera even if it ignores CAP_PROP_FPS.
            # Limit encoding/publication, not camera reads that could queue up.
            if captured_at < next_publish:
                continue
            ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError('Could not encode webcam JPEG')
            frames.publish(jpeg.tobytes(), captured_at)
            next_publish = captured_at + 1 / fps
    except Exception as exc:
        frames.fail(str(exc))
        print(f'Webcam stopped: {exc}', file=sys.stderr, flush=True)
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--quality', type=int, default=85)
    args = parser.parse_args()
    if (args.camera < 0 or not 1 <= args.port <= 65535 or min(args.width, args.height) <= 0
            or not math.isfinite(args.fps) or not 0 < args.fps <= 120 or not 1 <= args.quality <= 100):
        parser.error('Invalid camera/port/dimensions/fps/quality')
    import cv2

    frames = LatestFrame()
    # Bind first so a port collision never opens the webcam unnecessarily.
    with FrameServer(('127.0.0.1', args.port), frames) as server:
        backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
        capture = cv2.VideoCapture(args.camera, backend)
        if not capture.isOpened():
            capture.release()
            parser.error('Could not open webcam. Close other camera apps or try --camera 1.')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        stopped = Event()
        worker = Thread(target=capture_frames,
                        args=(capture, frames, stopped, args.fps, args.quality), daemon=True)
        worker.start()
        try:
            frames.get_after(-1)
            print(f'Laptop webcam ready: http://127.0.0.1:{args.port}/frame.jpg', flush=True)
            print(f'Open browser monitor: http://127.0.0.1:{args.port}/', flush=True)
            print('Camera/JPEG only; inference runs on Pi. Keep this terminal open. Ctrl+C stops.', flush=True)
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            stopped.set()
            frames.fail('Sender stopped')
            worker.join(timeout=3)


if __name__ == '__main__':
    main()
