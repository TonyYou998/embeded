# Driver Drowsiness Detection — local baseline

Project đặt tại `D:\courses\embeded`: chuẩn hóa nhãn, trích EAR/MAR, đánh giá và lưu SVM/DT/RF, chạy inference và ghi log.

## Bắt đầu nhanh: camera Pi, xem kết quả trên PC

Cần có project và model trong `~/driver-drowsiness` trên Pi, môi trường `.venv-pi312` trên Pi và `.venv` trên PC. Chạy theo thứ tự sau; giữ hai terminal mở trong suốt buổi demo. Nếu vừa sửa code trên PC, cập nhật hai file chạy inference trên Pi:

```powershell
cd D:\courses\embeded
scp .\src\run_video.py .\src\stream_input.py tanvuu@pi:~/driver-drowsiness/src/
```

**Terminal 1 — PC, PowerShell:** chạy receiver tại cổng 8767.

```powershell
cd D:\courses\embeded
.\.venv\Scripts\python.exe src\result_receiver.py --port 8767
```

**Terminal 2 — PC, PowerShell:** mở SSH tunnel. Chạy từ PC, không chạy khi đã ở trong phiên SSH trên Pi.

```powershell
ssh -o ExitOnForwardFailure=yes -R 127.0.0.1:8766:127.0.0.1:8767 tanvuu@pi
```

**Trong terminal SSH vừa mở (trên Pi):** chạy nhận diện.

```bash
cd ~/driver-drowsiness
source .venv-pi312/bin/activate
python src/run_video.py --camera 0 --algorithm svm --headless --no-save-video \
  --result-url http://127.0.0.1:8766/result.jpg --preview-fps 30
```

Mở **http://127.0.0.1:8767/** trên PC. Bấm **Bật âm báo** trên trang để nghe âm thử và nhận tiếng cảnh báo khi Pi xác nhận sự kiện; âm nhắc lại mỗi 4 giây nếu cảnh báo còn hiệu lực. Trang phải đang mở và loa PC phải bật. `--preview-fps 30` là giới hạn gửi ảnh, không đảm bảo Pi xử lý 30 FPS. Đổi `--algorithm svm` thành `rf` hoặc `dt` nếu muốn dùng model khác. Nhấn **Ctrl+C** trong terminal Pi để dừng và lưu log vào `reports/video_runs/` trên Pi.

Nếu SSH báo `remote port forwarding failed`, đóng tunnel cũ đang dùng cổng 8766 trên Pi rồi kết nối lại. Nếu trang PC không có hình, kiểm tra receiver, tunnel và tiến trình `run_video.py` vẫn đang chạy.

Nếu muốn dùng **camera laptop** và xử lý trên Pi, xem [hướng dẫn riêng](README-pi-webcam.md).

## Cấu trúc

```text
embeded/
  Annotated Drowsiness Detection Dataset Captured Us/  # dữ liệu gốc, giữ nguyên
  src/
    prepare_labels.py       # parse + đối chiếu bounding boxes + audit
    features.py             # công thức EAR-L, EAR-R, MAR
    extract_features.py     # scaffold MediaPipe Tasks, batch IMAGE mode
  tests/test_labels.py
  data/processed/labels.csv
  data/features/            # output extraction sau khi review
  models/                   # đặt face_landmarker.task tại đây
  reports/
    label_audit.json         # distribution, sequence, lỗi từng ảnh
    manifest.csv             # tên gốc và category export
    sequences.csv            # thứ tự frame và delta theo subject
    REVIEW.md                # kết quả lần chạy ban đầu
  requirements.txt
```

## Chạy trên Windows PowerShell

Dùng Python 3.11 hoặc 3.12, 64-bit. Nếu `python` mở Microsoft Store, cài Python trước và kiểm tra `py -3.12 --version`. Parser/audit chỉ dùng thư viện chuẩn, không cần GPU hay cài MediaPipe.

```powershell
Set-Location D:\courses\embeded
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe src\prepare_labels.py --info ".\Annotated Drowsiness Detection Dataset Captured Us\modelling\info.labels" --boxes ".\Annotated Drowsiness Detection Dataset Captured Us\modelling\training\bounding_boxes.labels" ".\Annotated Drowsiness Detection Dataset Captured Us\modelling\testing\bounding_boxes.labels"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Không cần activate venv nên không cần đổi PowerShell execution policy. Có thể dùng Python có sẵn bằng đường dẫn đầy đủ thay `py -3.12`.

Lệnh trên dùng `info.labels` tổng: 700 records, bao gồm training và testing. Không cộng thêm hai `info.labels` con vì sẽ đếm trùng. `bounding_boxes.labels` chỉ dùng đối chiếu label và tọa độ; không tự điền nhãn còn thiếu từ nguồn khác. Đường dẫn `/mnt/data/...` thuộc phiên ChatGPT cũ; project dùng bản dataset tìm thấy trên ổ D.

Để xử lý một export khác, truyền `--info`, tùy chọn `--boxes`, `--output`, `--report-dir`. `--boxes` nhận nhiều file. Ảnh được kiểm tra tương đối với thư mục chứa `info.labels`.

## Quy tắc labels.csv

CSV có đúng 5 cột:

| Cột | Quy tắc |
|---|---|
| filename | `path` thật trong export, tương đối với thư mục chứa info.labels; giữ hậu tố ingestion để mở đúng ảnh |
| subject_id | lấy từ `name`, ví dụ `frame_0010_uzan.jpg` → `uzan`; không parse được → None, xuất ô rỗng |
| frame_id | số nguyên từ tên gốc; không parse được → None, xuất ô rỗng |
| eye_state | đúng 2 box mắt cùng nhãn → OPEN/CLOSED; thiếu, xung đột hoặc sai số box → UNKNOWN |
| mouth_state | đúng 1 box miệng → YAWN/NO_YAWN; thiếu, xung đột hoặc sai số box → UNKNOWN |

Mapping: `mata_terbuka` → OPEN; `mata_terpejam` → CLOSED; `menguap` → YAWN; `tidak_menguap` → NO_YAWN. Không gộp thành 4 class, không map CLOSED thành SLEEP. Không suy luận trạng thái mắt trái/phải từ vị trí box annotation. Nhãn không biết được ghi trong audit; đường dẫn ảnh trùng làm script dừng thay vì âm thầm loại bỏ. CSV được sort theo subject rồi frame số.

## Review trước bước tiếp theo

1. Mở `reports/REVIEW.md`, `reports/label_audit.json`, và `data/processed/labels.csv`.
2. Kiểm tra trực quan 9 ảnh được liệt kê trong `issues`, cùng vài mẫu OPEN/CLOSED và YAWN/NO_YAWN. UNKNOWN vẫn giữ trong CSV để truy vết.
3. Xem `reports/sequences.csv`: `delta_frame` là chênh lệch frame_id trong cùng subject, `step_anomaly` so với `--expected-step` (mặc định 5, giả định kiểm tra có thể đổi). Khoảng cách lớn không tự có nghĩa là ảnh bị mất. Không có FPS/timestamp nên không tính thời gian nhắm mắt, PERCLOS hoặc thời lượng ngáp từ các ảnh này.
4. Cả 3 subject xuất hiện ở cả training và testing export. Giai đoạn train sau phải chia lại theo subject; không chia ngẫu nhiên các frame gần nhau. Nếu một subject có nhiều video, cần bổ sung recording_id trước khi phân tích chuỗi thời gian.

## Scaffold MediaPipe sau khi nhãn được xác nhận

Chưa chạy extraction trong giai đoạn này. Dependency tùy chọn chưa được cài/kiểm chứng trên máy. Cài vào venv khi cần:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Tải Face Landmarker model `.task` từ [tài liệu chính thức](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index#models), lưu `models\face_landmarker.task`. Scaffold theo [MediaPipe Tasks Python API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarker); không dùng legacy `mp.solutions`.

Sau khi đã review nhãn:

```powershell
.\.venv\Scripts\python.exe src\extract_features.py --image-root ".\Annotated Drowsiness Detection Dataset Captured Us\modelling" --model models\face_landmarker.task --labels-reviewed
```

Output giữ 5 cột nhãn và thêm `ear_l`, `ear_r`, `mar`, `status`, `error`. Mắt trái/phải theo giải phẫu của người. Công thức dùng tọa độ pixel 2D, hiệu chỉnh x theo chiều rộng và y theo chiều cao ảnh:

- EAR: `(distance(p2,p6) + distance(p3,p5)) / (2 * distance(p1,p4))`.
- Mắt trái: `[362,385,387,263,373,380]`; mắt phải: `[33,160,158,133,153,144]`.
- MAR được định nghĩa ở project này: `distance(13,14) / distance(78,308)`, chiều cao trong môi / chiều rộng trong môi. Không dùng chung threshold với biến thể MAR khác.

Không tìm được mặt, nhiều mặt, ảnh thiếu, ảnh lỗi hoặc hình học suy biến đều có status rõ ràng; feature thiếu xuất ô rỗng. Model nhận tối đa hai mặt để phát hiện trường hợp nhiều người thay vì tùy ý chọn một người. Cần kiểm tra landmark overlay trước khi dùng feature cho nghiên cứu.

## Train, đánh giá và lưu cả ba thuật toán

Sau khi có `features.csv`, cài thư viện train bằng cùng Python dùng chạy script:

```bash
python -m pip install -r requirements-train.txt
python src/train_evaluate.py
```

`train_evaluate.py` xuất đánh giá nested CV độc lập theo subject tại `reports/paper_all_features`. Cả hai bài toán dùng đủ ba feature theo thứ tự `ear_l, ear_r, mar`. SVM, Decision Tree và Random Forest cùng grid được khai báo trong `src/model_config.py` để dùng chung cho đánh giá và train bản triển khai.

Để lưu model sau khi đánh giá:

```bash
python src/train_final.py
```

Script dùng GridSearchCV với LeaveOneGroupOut trên toàn bộ dữ liệu phát triển để chọn tham số riêng cho từng thuật toán/bài toán, rồi refit trên toàn bộ mẫu hợp lệ. Không chọn một outer fold tốt nhất. Điểm CV trong bước này dùng chọn tham số, không phải kết quả test độc lập; giữ báo cáo nested CV trước đó để báo cáo hiệu năng.

Output tại `models/classifiers/`:

- `svm_eyes.joblib`, `svm_mouth.joblib`: lưu cả Pipeline chuẩn hóa và classifier.
- `dt_eyes.joblib`, `dt_mouth.joblib`.
- `rf_eyes.joblib`, `rf_mouth.joblib`.
- `metadata.json`: thứ tự feature, nhãn, subject, số mẫu, tham số được chọn, phiên bản thư viện và checksum dữ liệu/model. Chỉ được ghi khi cả sáu model hoàn thành.
- `training_summary.csv`, sáu file `*_search.csv`, hai file `*_excluded.csv` để truy vết quá trình train.

Sau khi lưu, script mở lại từng model và kiểm tra dự đoán trước/sau lưu giống nhau. Nếu thư mục output đã có dữ liệu, script yêu cầu chọn thư mục mới, ví dụ `--output-dir models/classifiers_v2`, để giữ bộ model cũ.

Ví dụ nạp model trong code inference sau này (chỉ nạp file do bạn tạo hoặc tin cậy, dùng môi trường thư viện tương ứng trong metadata):

```python
import joblib

eyes = joblib.load('models/classifiers/svm_eyes.joblib')
mouth = joblib.load('models/classifiers/svm_mouth.joblib')
features = [[ear_l, ear_r, mar]]  # ba giá trị đo từ frame hiện tại
eye_state = eyes.predict(features)[0]
mouth_state = mouth.predict(features)[0]
```

Chạy lần lượt cặp SVM, DT, RF để so sánh trong cùng điều kiện. Chưa triển khai webcam, quy tắc temporal hoặc cảnh báo trong bước lưu model này.

## Chạy webcam hoặc video với model đã lưu

Cài dependency vào cùng Python đang dùng model (phiên bản scikit-learn phải khớp metadata model):

```bash
python -m pip install -r requirements-run.txt
```

Tại `D:\courses\embeded`, chạy một trong ba cặp classifier:

```bash
python src/run_video.py --camera 0 --algorithm svm
python src/run_video.py --camera 0 --algorithm dt
python src/run_video.py --camera 0 --algorithm rf
```

Chạy lần lượt, nhấn **Q** hoặc **Esc** để dừng. Nếu không mở được camera, kiểm tra quyền camera, đóng app khác đang dùng camera hoặc thử `--camera 1`. Không bật webcam tự động khi chỉ cài dependency hay chạy test.

Webcam mặc định tự ghi `annotated.mp4` để bạn xem lại sau khi thử nhắm mắt; thêm `--no-save-video` nếu chỉ muốn CSV. Khi chạy video có sẵn, vẫn truyền `--save-video` để lưu bản vẽ kết quả.

Để tự ghi mốc thử nghiệm, giữ cửa sổ preview đang được chọn: nhấn **C** rồi nhắm mắt, mở mắt trở lại và nhấn **O**. Có thể nhấn **Y** lúc bắt đầu ngáp, **N** khi kết thúc. Các phím này chỉ ghi `manual_markers.csv`, không thay đổi dự đoán hoặc bộ đếm. Đây là mốc do bạn tự đánh dấu, có độ trễ thao tác, không phải nhãn đã được xác minh. Nếu không nhấn phím, chương trình vẫn tự ghi dự đoán và sự kiện bình thường.

Với video đã có trên máy:

```bash
python src/run_video.py --video "data/videos/test.mp4" --algorithm svm --save-video
```

Đổi `svm` thành `dt` và `rf` để so sánh cùng video. Có thể dùng `--headless` để không mở cửa sổ; `--max-frames 300` để giới hạn số frame. File được xử lý nhanh theo khả năng CPU, không cố phát đúng tốc độ xem. Bộ đếm sự kiện sử dụng timestamp của video, không dùng thời gian chạy CPU. Khi timestamp video không khả dụng hoặc lặp lại, script dùng FPS dự phòng; dùng `--fps 30` nếu file không cung cấp FPS hợp lệ. Video biến thiên FPS cần kiểm tra chất lượng timestamp trước khi đánh giá sự kiện.

Preview hiển thị landmark, feature, OPEN/CLOSED và NO_YAWN/YAWN. Quy tắc temporal thử nghiệm mặc định:

- CLOSED liên tiếp ít nhất 2 giây: `ALERT: EYES CLOSED LONG`.
- YAWN liên tiếp ít nhất 1,5 giây: `EVENT: SUSTAINED YAWN`.
- Mất mặt, nhiều mặt, lỗi inference hoặc khoảng cách giữa hai quan sát lớn hơn 0,5 giây: reset bộ đếm liên tục; không suy diễn trạng thái bình thường khi không quan sát được.
- Mỗi đợt liên tục chỉ đếm một sự kiện; hai loại sự kiện có thể đồng thời xảy ra.

Các ngưỡng này là giá trị khởi đầu cho demo, chưa được xác nhận bằng ground truth, không phải tiêu chí kết luận ngủ. Điều chỉnh bằng `--closed-seconds`, `--yawn-seconds`, `--max-gap-seconds`. Ở FPS thấp cần cân nhắc độ tin cậy của chuỗi quan sát khi thay đổi max-gap. Cảnh báo hiện tại hiển thị bằng chữ trên preview; chưa có âm báo.

Mỗi lần chạy có thư mục riêng trong `reports/video_runs/`:

- `frames.csv`: timestamp, trạng thái xử lý, feature, dự đoán, thời gian duy trì, sự kiện mới và thời gian xử lý mỗi frame.
- `events.csv`: mỗi đợt dự đoán CLOSED/YAWN, gồm `start_s`, `last_positive_s`, `ended_at_s`, `duration_s`, `confirmed`, `confirmed_at_s`, frame bắt đầu/kết thúc và lý do kết thúc. Ghi cả những đợt ngắn chưa đủ ngưỡng cảnh báo; đợt đang dở được ghi khi kết thúc phiên bằng Q/Esc/Ctrl+C.
- `manual_markers.csv`: các mốc tự đánh dấu bằng C/O/Y/N để đối chiếu với dự đoán.
- `summary.json`: tổng số frame, status, sự kiện và tốc độ; tham số temporal, nguồn và model sử dụng.
- `annotated.mp4`: tự tạo khi chạy webcam, hoặc khi truyền `--save-video` cho video local. Trên ảnh có timestamp và frame ID để đối chiếu CSV dù tốc độ playback của webcam ghi lại có thể khác thời gian thực.

Sau buổi thử, mở `events.csv`, tìm dòng `task=eyes`: `confirmed=True` nghĩa là đợt dự đoán nhắm mắt đủ ngưỡng thử nghiệm, `confirmed_at_s` là lúc cảnh báo được bật. Nó không tự chứng minh mắt thật đang nhắm. Đối chiếu với video và `manual_markers.csv`. Thời lượng chỉ tính từ quan sát dương tính đầu tiên đến cuối cùng; `ended_at_s` là mốc nhận thấy đợt đã kết thúc hoặc mất quan sát. Hai loại mốc có thể khác nhau. Terminal cũng in tóm tắt mỗi đợt khi bạn mở mắt/miệng đổi trạng thái.

`classifier_ms` đo tổng hai classifier; `landmark_ms` đo riêng MediaPipe. `process_ms` gồm đọc frame, xử lý ảnh, landmark, classifier và temporal, không gồm vẽ, ghi file hay chờ hiển thị. `processing_fps` là tốc độ xử lý tương ứng, không phải FPS camera hay FPS đầy đủ của ứng dụng; `wall_seconds` bao gồm chi phí khởi tạo và xử lý thực tế. Không so sánh hiệu năng từ các lần chạy đồng thời hoặc khác điều kiện. Chưa đo CPU/RAM/nhiệt độ trong script này.

Chế độ landmark giữ IMAGE cho từng frame giống extraction ban đầu, không dùng tracking/smoothing. Tham khảo [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python) và [OpenCV video properties](https://docs.opencv.org/4.12.0/d4/d15/group__videoio__flags__base.html).

Log là dự đoán và thời gian chạy, chưa có ground truth nên không phải báo cáo accuracy/F1 video. Bước đánh giá sự kiện có nhãn video được thực hiện riêng sau khi chương trình inference chạy được.
