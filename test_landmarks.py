from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import LEFT_EYE, RIGHT_EYE, ratios

ROOT = Path(__file__).resolve().parents[1]

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(ROOT / "models/face_landmarker.task")
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=2,
)

image = mp.Image.create_from_file(str(ROOT / "data/sample4.jpg"))

with vision.FaceLandmarker.create_from_options(options) as detector:
    result = detector.detect(image)

print("Số khuôn mặt:", len(result.face_landmarks))

if len(result.face_landmarks) != 1:
    raise SystemExit("Hãy thử ảnh có đúng một khuôn mặt rõ.")

landmarks = result.face_landmarks[0]
values = ratios(landmarks, image.width, image.height)
print(values)

# MediaPipe dùng RGB; OpenCV dùng BGR.
output = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)

# Xanh lá: mắt trái; xanh dương: mắt phải; đỏ: miệng.
groups = [
    (LEFT_EYE, (0, 255, 0)),
    (RIGHT_EYE, (255, 0, 0)),
    ((13, 14, 78, 308), (0, 0, 255)),
]

for indices, color in groups:
    for index in indices:
        point = landmarks[index]
        position = (
            int(point.x * image.width),
            int(point.y * image.height),
        )
        cv2.circle(output, position, 2, color, -1)

output_path = ROOT / "reports/sample_landmarks.jpg"
output_path.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(output_path), output):
    raise RuntimeError("Không lưu được ảnh kết quả.")

print("Ảnh kiểm tra:", output_path)