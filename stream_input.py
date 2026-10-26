"""Read the latest laptop JPEG over HTTP, preserving sender capture timestamps."""
import math
from http.client import HTTPException
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

MAX_JPEG_BYTES = 5 * 1024 * 1024


class SnapshotCapture:
    """Pull one fresh frame per request; never consume a growing video queue."""

    def __init__(self, url, timeout=7.0):
        parts = urlsplit(url)
        if parts.scheme not in {'http', 'https'} or not parts.hostname or parts.fragment:
            raise ValueError('stream-url must be an HTTP(S) frame URL without a fragment')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('stream-timeout must be finite and positive')
        self.url = url
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))
        self.closed = False
        self.frame_id = -1
        self.capture_time = None
        self.origin = None
        self.timestamp = None
        self.request_ms = 0.0
        self.sender_frame_age_ms = None
        self.skipped_frames = 0

    def isOpened(self):
        return not self.closed

    def release(self):
        self.closed = True

    def read(self):
        if self.closed:
            return False, None
        import cv2
        import numpy as np

        parts = urlsplit(self.url)
        query = [(k, v) for k, v in parse_qsl(parts.query) if k != 'after']
        query.append(('after', str(self.frame_id)))
        url = urlunsplit(parts._replace(query=urlencode(query)))
        started = perf_counter()
        try:
            with self.opener.open(url, timeout=self.timeout) as response:
                if response.headers.get_content_type() != 'image/jpeg':
                    raise ValueError('Expected image/jpeg from webcam_stream.py')
                frame_id = int(response.headers['X-Frame-Id'])
                capture_time = float(response.headers['X-Capture-Time'])
                frame_age = float(response.headers['X-Frame-Age-Ms'])
                size = int(response.headers['Content-Length'])
                if not 0 < size <= MAX_JPEG_BYTES:
                    raise ValueError('Invalid JPEG size')
                payload = response.read(size + 1)
                if len(payload) != size:
                    raise ValueError('Incomplete JPEG response')
            if frame_id <= self.frame_id:
                raise ValueError('Repeated or out-of-order frame; restart run_video after restarting sender')
            if not math.isfinite(capture_time) or capture_time < 0:
                raise ValueError('Invalid capture timestamp')
            if self.capture_time is not None and capture_time <= self.capture_time:
                raise ValueError('Sender timestamps must increase')
            if not math.isfinite(frame_age) or not 0 <= frame_age <= 2000:
                raise ValueError('Sender frame is stale or has invalid age')
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Could not decode JPEG')
        except (URLError, HTTPException, OSError, ValueError, TypeError, KeyError) as exc:
            if isinstance(exc, HTTPError):
                exc.close()
            raise RuntimeError(f'Laptop stream failed: {exc}') from exc
        self.request_ms = (perf_counter() - started) * 1000
        if self.frame_id >= 0:
            self.skipped_frames += frame_id - self.frame_id - 1
        self.frame_id = frame_id
        self.capture_time = capture_time
        if self.origin is None:
            self.origin = capture_time
        self.timestamp = capture_time - self.origin
        self.sender_frame_age_ms = frame_age
        return True, frame

    def send_preview(self, frame):
        """Return an annotated frame over the same tunnel; inference stays on Pi."""
        import cv2

        if self.capture_time is None or self.closed:
            raise RuntimeError('No source frame available for preview')
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok or encoded.nbytes > MAX_JPEG_BYTES:
            raise RuntimeError('Could not encode preview within size limit')
        parts = urlsplit(self.url)
        url = urlunsplit(parts._replace(path='/result.jpg', query='', fragment=''))
        request = Request(url, data=encoded.tobytes(), method='POST', headers={
            'Content-Type': 'image/jpeg', 'X-Source-Frame-Id': str(self.frame_id),
            'X-Source-Capture-Time': f'{self.capture_time:.9f}'})
        try:
            with self.opener.open(request, timeout=min(2.0, self.timeout)) as response:
                if response.status != 204:
                    raise ValueError('Unexpected preview response')
        except (URLError, HTTPException, OSError, ValueError) as exc:
            if isinstance(exc, HTTPError):
                exc.close()
            raise RuntimeError(f'Could not return preview: {exc}') from exc


class ResultPublisher:
    """Send annotated camera frames to result_receiver.py through an SSH tunnel."""

    def __init__(self, url):
        parts = urlsplit(url)
        if parts.scheme not in {'http', 'https'} or not parts.hostname or parts.fragment or parts.path != '/result.jpg':
            raise ValueError('result-url must be an HTTP(S) /result.jpg URL without a fragment')
        self.url = url
        self.opener = build_opener(ProxyHandler({}))

    def send_preview(self, frame, row=None, processing_fps=None):
        import cv2

        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok or encoded.nbytes > MAX_JPEG_BYTES:
            raise RuntimeError('Could not encode preview within size limit')
        headers = {'Content-Type': 'image/jpeg'}
        if row is not None:
            headers.update({
                'X-Eye-State': str(row['eye_state']),
                'X-Mouth-State': str(row['mouth_state']),
                'X-Closed-Duration': f"{float(row['closed_duration_s'] or 0):.3f}",
                'X-Yawn-Duration': f"{float(row['yawn_duration_s'] or 0):.3f}",
            })
        if processing_fps is not None and math.isfinite(processing_fps):
            headers['X-Processing-Fps'] = f'{processing_fps:.2f}'
        request = Request(self.url, data=encoded.tobytes(), method='POST',
                          headers=headers)
        try:
            with self.opener.open(request, timeout=2.0) as response:
                if response.status != 204:
                    raise ValueError('Unexpected preview response')
        except (URLError, HTTPException, OSError, ValueError) as exc:
            if isinstance(exc, HTTPError):
                exc.close()
            raise RuntimeError(f'Could not publish preview: {exc}') from exc
