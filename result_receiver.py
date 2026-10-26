import argparse
import json
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

MAX_JPEG_BYTES = 5 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pi detection result</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; background: #0b1119; color: #e9f0f7 }
* { box-sizing: border-box }
body { margin: 0; padding: 24px }
.shell { max-width: 1280px; margin: auto }
.top { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 20px }
.top-actions { display: flex; align-items: center; gap: 12px }
h1 { font-size: clamp(24px, 3vw, 34px); margin: 0 }
.sound-toggle { border: 1px solid #41576a; border-radius: 999px; padding: 8px 13px; background: #192431;
  color: #e9f0f7; font: inherit; font-size: 13px; font-weight: 700; cursor: pointer }
.sound-toggle.enabled { border-color: #42dc8f; color: #73e8ae }
.live { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px;
  color: #8d9ba9; background: #192431; font-weight: 700; font-size: 13px; letter-spacing: .08em }
.dot { width: 9px; height: 9px; border-radius: 50%; background: #697888 }
.live.connected { color: #73e8ae }
.live.connected .dot { background: #42dc8f; box-shadow: 0 0 12px #42dc8f }
.layout { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(290px, 1fr); gap: 20px }
.card { background: #121d29; border: 1px solid #263746; border-radius: 18px; overflow: hidden }
.video { background: #05090e; min-height: 280px; display: grid; place-items: center }
.video img { width: 100%; max-height: 78vh; object-fit: contain; display: block }
.panel { padding: 24px }
h2 { margin: 0 0 6px; font-size: 22px }
.muted { color: #91a2b3; font-size: 14px; line-height: 1.5 }
#summary { min-height: 50px; margin: 26px 0; font-size: 19px; font-weight: 650 }
.alert-banner { padding: 12px 14px; border: 1px solid #34475a; border-radius: 10px; color: #aebdcd }
.alert-banner.active { border-color: #ff735f; background: #4a2526; color: #ffe1db; font-weight: 700 }
.meter { margin: 24px 0 }
.meter-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 9px }
.meter-name { font-size: 16px; font-weight: 650 }
.meter-state { color: #91a2b3; font-size: 13px; text-align: right }
.track { height: 16px; overflow: hidden; border-radius: 999px; background: #263746 }
.fill { width: 0; height: 100%; border-radius: inherit; transition: width 160ms ease-out }
#open .fill { background: #3cdb9b }
#closed .fill { background: #ffbd59 }
#yawn .fill { background: #aa8dff }
.foot { border-top: 1px solid #263746; padding-top: 18px; margin-top: 28px }
@media (max-width: 800px) { body { padding: 14px } .top { flex-wrap: wrap } .layout { grid-template-columns: 1fr } .panel { padding: 18px } }
</style></head><body>
<main class="shell">
  <div class="top"><h1>Pi detection result</h1><div class="top-actions"><button id="sound-toggle" class="sound-toggle" type="button" aria-pressed="false">Bật âm báo</button><div id="live" class="live"><span class="dot"></span><span id="live-text">ĐANG CHỜ</span></div></div></div>
  <div class="layout">
    <div class="card video"><img id="result" src="/stream.mjpg" alt="Đang chờ ảnh từ Pi"></div>
    <section class="card panel" aria-label="Dự đoán trực tiếp">
      <h2>Dự đoán trực tiếp</h2>
      <div class="muted">Cập nhật theo từng khung hình Pi gửi tới.</div>
      <div id="summary" aria-live="polite">Đang chờ dữ liệu dự đoán...</div>
      <div id="alert" class="alert-banner" role="status">Chưa có cảnh báo đã xác nhận.</div>
      <div class="meter" id="open"><div class="meter-head"><span class="meter-name">Mắt mở · OPEN</span><span class="meter-state">Chưa có dữ liệu</span></div><div class="track"><div class="fill"></div></div></div>
      <div class="meter" id="closed"><div class="meter-head"><span class="meter-name">Mắt nhắm · CLOSED</span><span class="meter-state">Chưa có dữ liệu</span></div><div class="track"><div class="fill"></div></div></div>
      <div class="meter" id="yawn"><div class="meter-head"><span class="meter-name">Ngáp · YAWN</span><span class="meter-state">Chưa có dữ liệu</span></div><div class="track"><div class="fill"></div></div></div>
      <div class="muted foot" id="details">Thanh sáng biểu thị nhãn đang được phát hiện, không phải xác suất dự đoán.</div>
    </section>
  </div>
</main>
<script>
const image = document.getElementById('result');
image.onerror = () => setTimeout(() => { image.src = '/stream.mjpg?t=' + Date.now(); }, 1000);
const live = document.getElementById('live');
const liveText = document.getElementById('live-text');
const summary = document.getElementById('summary');
const details = document.getElementById('details');
const alertBanner = document.getElementById('alert');
const soundButton = document.getElementById('sound-toggle');
const meters = Object.fromEntries(['open', 'closed', 'yawn'].map(name => [name, document.getElementById(name)]));
let lastUpdate = 0;
let audioContext = null;
let soundEnabled = false;
let lastAlarmAt = 0;
let activeAlerts = { closed: false, yawn: false };
let previousCounts = null;
function playAlarm(force = false) {
  if (!soundEnabled || !audioContext || audioContext.state !== 'running') return;
  if (!force && Date.now() - lastAlarmAt < 4000) return;
  lastAlarmAt = Date.now();
  const start = audioContext.currentTime;
  for (let i = 0; i < 3; i++) {
    const at = start + i * 0.24;
    const oscillator = audioContext.createOscillator();
    const volume = audioContext.createGain();
    oscillator.type = 'sine';
    oscillator.frequency.value = i === 1 ? 700 : 880;
    volume.gain.setValueAtTime(0, at);
    volume.gain.linearRampToValueAtTime(0.08, at + 0.02);
    volume.gain.exponentialRampToValueAtTime(0.001, at + 0.17);
    oscillator.connect(volume).connect(audioContext.destination);
    oscillator.start(at);
    oscillator.stop(at + 0.18);
    oscillator.onended = () => { oscillator.disconnect(); volume.disconnect(); };
  }
}
soundButton.addEventListener('click', async () => {
  if (soundEnabled) {
    soundEnabled = false;
    soundButton.textContent = 'Bật âm báo';
    soundButton.classList.remove('enabled');
    soundButton.setAttribute('aria-pressed', 'false');
    return;
  }
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error('Web Audio is unavailable');
    if (!audioContext) audioContext = new AudioContextClass();
    if (audioContext.state !== 'running') await audioContext.resume();
    if (audioContext.state !== 'running') throw new Error('Audio is blocked');
    soundEnabled = true;
    soundButton.textContent = 'Tắt âm báo';
    soundButton.classList.add('enabled');
    soundButton.setAttribute('aria-pressed', 'true');
    playAlarm(true); // Audible check when enabling sound.
  } catch (error) {
    soundButton.textContent = 'Không bật được âm báo';
  }
});
function setMeter(name, active, description) {
  const meter = meters[name];
  meter.querySelector('.fill').style.width = active ? '100%' : '0%';
  meter.querySelector('.meter-state').textContent = description;
}
function showWaiting(message) {
  live.classList.remove('connected');
  liveText.textContent = 'ĐANG CHỜ';
  summary.textContent = message;
  alertBanner.textContent = 'Chờ dữ liệu cảnh báo từ Pi.';
  alertBanner.classList.remove('active');
  activeAlerts = { closed: false, yawn: false };
  for (const name of Object.keys(meters)) setMeter(name, false, 'Chưa có dữ liệu');
}
const events = new EventSource('/events');
events.onmessage = event => {
  const data = JSON.parse(event.data);
  lastUpdate = Date.now();
  live.classList.add('connected');
  liveText.textContent = 'LIVE';
  if (!data.available) {
    showWaiting('Pi đang gửi ảnh; cần cập nhật run_video.py và stream_input.py để gửi nhãn.');
    return;
  }
  const open = data.eye_state === 'OPEN';
  const closed = data.eye_state === 'CLOSED';
  const yawn = data.mouth_state === 'YAWN';
  const closedAlert = data.closed_alert === true;
  const yawnAlert = data.yawn_alert === true;
  const confirmed = [closedAlert && 'Mắt nhắm lâu', yawnAlert && 'Ngáp kéo dài'].filter(Boolean);
  alertBanner.textContent = confirmed.length ? 'CẢNH BÁO: ' + confirmed.join(' · ') :
    data.closed_event_count === null ? 'Cần cập nhật code trên Pi để nhận cảnh báo.' : 'Chưa có cảnh báo đã xác nhận.';
  alertBanner.classList.toggle('active', confirmed.length > 0);
  const counts = Number.isInteger(data.closed_event_count) && Number.isInteger(data.yawn_event_count) ?
    { closed: data.closed_event_count, yawn: data.yawn_event_count } : null;
  const newCount = counts && previousCounts &&
    (counts.closed > previousCounts.closed || counts.yawn > previousCounts.yawn);
  const newAlert = (closedAlert && !activeAlerts.closed) || (yawnAlert && !activeAlerts.yawn);
  if (counts) previousCounts = counts;
  activeAlerts = { closed: closedAlert, yawn: yawnAlert };
  if (newCount || newAlert) playAlarm(true);
  setMeter('open', open, open ? 'Đang phát hiện' : 'Không phát hiện');
  setMeter('closed', closed, closed ? data.closed_duration_s.toFixed(1) + ' giây' : 'Không phát hiện');
  setMeter('yawn', yawn, yawn ? data.yawn_duration_s.toFixed(1) + ' giây' : 'Không phát hiện');
  const eye = open ? 'Mắt mở' : closed ? 'Mắt nhắm' : 'Chưa nhận diện mắt';
  const mouth = yawn ? 'Đang ngáp' : data.mouth_state === 'NO_YAWN' ? 'Không ngáp' : 'Chưa nhận diện miệng';
  summary.textContent = eye + ' · ' + mouth;
  details.textContent = 'Frame ' + data.frame_id + (data.processing_fps ? ' · Pi ' + data.processing_fps.toFixed(1) + ' FPS' : '') +
    ' · Thanh sáng là nhãn đang được phát hiện, không phải xác suất.';
};
events.onerror = () => { if (Date.now() - lastUpdate > 3000) showWaiting('Đang kết nối lại với receiver...'); };
setInterval(() => {
  if (lastUpdate && Date.now() - lastUpdate > 3000) showWaiting('Chưa có khung hình mới từ Pi.');
  else if (activeAlerts.closed || activeAlerts.yawn) playAlarm();
}, 1000);
</script></body></html>""".encode("utf-8")


class ResultServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, Handler)
        self.frame_condition = threading.Condition()
        self.latest_frame = None
        self.latest_prediction = None
        self.frame_id = 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def respond(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.respond(HTTPStatus.OK, PAGE, "text/html; charset=utf-8")
        elif path == "/result.jpg":
            with self.server.frame_condition:
                frame = self.server.latest_frame
            if frame is None:
                self.respond(HTTPStatus.NOT_FOUND, b"Waiting for Pi", "text/plain")
            else:
                self.respond(HTTPStatus.OK, frame, "image/jpeg")
        elif path == "/stream.mjpg":
            self.stream_frames()
        elif path == "/events":
            self.stream_events()
        else:
            self.respond(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")

    def stream_frames(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.connection.settimeout(10)
        last_id = 0
        while True:
            with self.server.frame_condition:
                self.server.frame_condition.wait_for(lambda: self.server.frame_id != last_id)
                last_id = self.server.frame_id
                frame = self.server.latest_frame
            part = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode("ascii") + b"\r\n\r\n" + frame + b"\r\n")
            try:
                self.wfile.write(part)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                return

    def stream_events(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.connection.settimeout(10)
        last_id = 0
        while True:
            with self.server.frame_condition:
                changed = self.server.frame_condition.wait_for(
                    lambda: self.server.frame_id != last_id, timeout=10
                )
                if changed:
                    last_id = self.server.frame_id
                    prediction = self.server.latest_prediction
            payload = (b"data: " + json.dumps(prediction, separators=(",", ":")).encode("utf-8")
                       + b"\n\n") if changed else b": keepalive\n\n"
            try:
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                return

    def do_POST(self):
        if urlsplit(self.path).path != "/result.jpg":
            self.respond(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")
            return

        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = 0

        content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
        if content_type != "image/jpeg" or not 0 < length <= MAX_JPEG_BYTES:
            self.respond(HTTPStatus.BAD_REQUEST, b"Expected JPEG up to 5 MB", "text/plain")
            return

        self.connection.settimeout(5)
        try:
            frame = self.rfile.read(length)
        except TimeoutError:
            return

        if len(frame) != length or not frame.startswith(b"\xff\xd8") or not frame.endswith(b"\xff\xd9"):
            self.respond(HTTPStatus.BAD_REQUEST, b"Incomplete JPEG", "text/plain")
            return

        eye_state = self.headers.get("X-Eye-State", "UNKNOWN")
        mouth_state = self.headers.get("X-Mouth-State", "UNKNOWN")
        available = eye_state in {"OPEN", "CLOSED", "UNKNOWN"} and mouth_state in {"YAWN", "NO_YAWN", "UNKNOWN"}
        available = available and "X-Eye-State" in self.headers and "X-Mouth-State" in self.headers
        if not available:
            eye_state = mouth_state = "UNKNOWN"

        def number(name):
            try:
                value = float(self.headers.get(name, "0"))
            except ValueError:
                return 0.0
            return min(value, 3600.0) if math.isfinite(value) and value > 0 else 0.0

        def event_count(name):
            try:
                value = int(self.headers[name])
            except (KeyError, ValueError):
                return None
            return value if 0 <= value <= 1_000_000_000 else None

        with self.server.frame_condition:
            self.server.latest_frame = frame
            self.server.frame_id += 1
            self.server.latest_prediction = {
                "frame_id": self.server.frame_id,
                "available": available,
                "eye_state": eye_state,
                "mouth_state": mouth_state,
                "closed_duration_s": number("X-Closed-Duration"),
                "yawn_duration_s": number("X-Yawn-Duration"),
                "processing_fps": number("X-Processing-Fps"),
                "closed_alert": available and self.headers.get("X-Closed-Alert") == "1",
                "yawn_alert": available and self.headers.get("X-Yawn-Alert") == "1",
                "closed_event_count": event_count("X-Closed-Event-Count") if available else None,
                "yawn_event_count": event_count("X-Yawn-Event-Count") if available else None,
            }
            self.server.frame_condition.notify_all()

        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Receive Pi detection result frames")
    parser.add_argument("--port", type=int, default=8766, help="HTTP port (default: 8766)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    server = ResultServer(("127.0.0.1", args.port))
    print(f"Open http://127.0.0.1:{args.port}/ on this PC", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
