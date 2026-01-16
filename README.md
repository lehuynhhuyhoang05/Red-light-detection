# 🚦 Red Light Violation Detection System

> Hệ thống phát hiện vi phạm vượt đèn đỏ tự động sử dụng YOLOv11 + ByteTrack + PySide6

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Tổng quan

Hệ thống giám sát giao thông thông minh phát hiện vi phạm vượt đèn đỏ với độ chính xác cao (~88.5% mAP@50). Sử dụng:
- **YOLOv11** cho Object Detection
- **ByteTrack** cho Multi-Object Tracking
- **Logic 2 tầng** với 8 điều kiện xác nhận vi phạm
- **PySide6 GUI** với 5 tab chức năng

### Các đối tượng được phát hiện

| Class | Mô tả | Vai trò |
|-------|-------|---------|
| `car` | Ô tô | Phương tiện giám sát |
| `motorbike` | Xe máy | Phương tiện giám sát |
| `red_light` | Đèn đỏ | Điều kiện vi phạm |
| `green_light` | Đèn xanh | Trạng thái đèn |
| `yellow_light` | Đèn vàng | Trạng thái đèn |
| `stop_line` | Vạch dừng | Ranh giới vi phạm |

---

## ✨ Tính năng

### 🎯 Core Features

- ✅ **Object Detection**: YOLOv11 phát hiện 6 classes với mAP@50 ~88.5%
- ✅ **Multi-Object Tracking**: ByteTrack gán Track ID duy nhất cho mỗi phương tiện
- ✅ **Violation Detection**: Logic 2 tầng với 8 điều kiện AND
- ✅ **Smart Mechanisms**: ROI, Voting, Snapshot, Grace Period, Sideways Detection
- ✅ **Evidence Collection**: 3 ảnh bằng chứng (before/at/after) + metadata JSON
- ✅ **PDF Reports**: Biên bản vi phạm tự động với ReportLab
- ✅ **Session Management**: Lưu trữ theo phiên, dễ quản lý và tra cứu

### 🖥️ GUI Features (PySide6)

| Tab | Chức năng |
|-----|-----------|
| 📹 **Video** | Live preview với bounding box, Track ID, ROI overlay |
| ⚠️ **Vi Phạm** | Danh sách vi phạm, preview ảnh bằng chứng, tạo PDF |
| 📊 **Thống Kê** | Số liệu tổng hợp: tổng vi phạm, phân loại theo xe, FPS |
| 🔄 **So Sánh** | Benchmark YOLOv11 vs RF-DETR (mAP, FPS, Memory) |
| ⚙️ **Cài Đặt** | Config model, ROI, violation params, location info |

### 🧠 Smart Mechanisms

| Cơ chế | Mục đích |
|--------|----------|
| **ROI** | Xác định lane từ vị trí đèn đỏ → tránh bắt xe ngược chiều |
| **Voting (3/5 frames)** | Xác định trạng thái đèn ổn định → tránh flicker |
| **Snapshot** | Lưu vị trí xe khi đèn chuyển đỏ → không phạt xe đang đi hợp lệ |
| **Grace Period (1.5s)** | Thời gian ân xá → không phạt xe không kịp dừng |
| **Sideways Detection** | Phát hiện xe đi ngang → loại trừ xe từ lane khác |
| **Multi-frame (3 frames)** | Xác nhận liên tiếp → giảm detection noise |

---

## 🛠️ Cài đặt

### Yêu cầu Hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| **OS** | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| **Python** | 3.11+ | 3.11.x |
| **RAM** | 8GB | 16GB+ |
| **GPU** | - | NVIDIA GPU với CUDA 11.8+ |
| **VRAM** | - | 4GB+ (để chạy real-time) |
| **Storage** | 10GB | 50GB+ (cho dataset) |

### Bước 1: Clone Repository

```bash
git clone https://github.com/yourusername/red_light_detection.git
cd red_light_detection
```

### Bước 2: Tạo Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt PyTorch (CUDA)

**Nếu có GPU NVIDIA:**
```bash
# CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

**Nếu chỉ dùng CPU:**
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
```

### Bước 4: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

**Danh sách packages chính:**
- `ultralytics==8.1.0` - YOLOv11
- `opencv-python==4.8.1.78` - Computer Vision
- `supervision==0.16.0` - ByteTrack tracking
- `PySide6==6.6.0` - GUI
- `reportlab==4.0.7` - PDF generation
- `loguru==0.7.2` - Logging

### Bước 5: Download Model Weights

**YOLOv11 (Trained):**
```bash
# Download từ link (thay YOUR_LINK)
# Đặt file vào models/yolov11.pt
```

**Hoặc dùng pre-trained YOLOv11:**
```bash
# Ultralytics sẽ tự động download khi chạy lần đầu
```

### Bước 6: Cấu hình

Sao chép và chỉnh sửa `config.yaml`:

```yaml
# config.yaml
model:
  type: "yolov11"  # Model chính
  yolov11:
    variant: "yolov11s"
    weights: "models/yolov11.pt"  # Đường dẫn model
    img_size: 640
    conf_threshold: 0.25
    iou_threshold: 0.45

tracking:
  tracker: "bytetrack"
  track_thresh: 0.3
  track_buffer: 60
  match_thresh: 0.7

violation:
  min_frames: 3
  grace_period: 0.5  # 0.5 giây
  stop_line_threshold: 30  # pixels
  min_vehicle_confidence: 0.5
  
  # Cho phép xe máy rẽ phải khi đèn đỏ (nếu có biển P.131b)
  allow_motorbike_right_turn: false
  
  roi:
    enabled: true
    x_min: 0.25
    x_max: 0.85
    y_min: 0.20
    y_max: 0.95

location:
  intersection: "Ngã tư Test"
  city: "Đà Nẵng"
  camera_id: "CAM-001"
```

### Bước 7: Test Installation

```bash
python main.py --help
```

Nếu thành công, sẽ hiển thị help message.

---

## 🚀 Sử dụng

### 1. Chạy GUI (Khuyến nghị)

```bash
python main.py --gui
```

**Các thao tác trong GUI:**

| Tab | Thao tác |
|-----|----------|
| **Video** | Chọn video → Play/Pause → Xem vi phạm real-time |
| **Vi Phạm** | Xem danh sách → Click để preview ảnh → Tạo PDF |
| **Thống Kê** | Xem số liệu tự động cập nhật |
| **So Sánh** | Chọn video → Start Benchmark → So sánh models |
| **Cài Đặt** | Thay đổi model, ROI, thông tin location |

### 2. CLI Mode - Xử lý Video

```bash
# Xử lý video và lưu kết quả
python main.py --video path/to/video.mp4

# Xử lý với config tùy chỉnh
python main.py --video path/to/video.mp4 --config custom_config.yaml

# Xử lý và lưu video output
python main.py --video path/to/video.mp4 --save-video
```

### 3. Xử lý Webcam

```bash
python main.py --source 0
```

### 4. Chế độ Debug

```bash
python main.py --gui --debug
```

---

## 📁 Cấu trúc Thư mục

```
red_light_detection/
│
├── 📄 main.py                      # Entry point
├── 📄 config.yaml                  # Cấu hình hệ thống
├── 📄 requirements.txt             # Dependencies
├── 📄 README.md                    # Tài liệu này
├── 📄 .gitignore                   # Git ignore rules
│
├── 📂 src/                         # Source code
│   ├── detector.py                # YOLOv11/RF-DETR detection
│   ├── tracker.py                 # ByteTrack wrapper
│   ├── violation_logic.py         # Logic xác nhận vi phạm (1221 lines)
│   ├── report_generator.py        # PDF generation
│   ├── gui.py                     # PySide6 GUI (5 tabs)
│   └── utils.py                   # Utilities
│
├── 📂 scripts/                     # Utility scripts
│   ├── analyze_dataset.py         # Phân tích dataset
│   ├── download_and_extract.py    # Download video + extract frames
│   ├── download_kaggle.py         # Download từ Kaggle
│   ├── download_model.py          # Download model weights
│   ├── extract_frames.py          # Extract frames từ video
│   ├── train.py                   # Training script
│   └── test_video_demo.py         # Test demo
│
├── 📂 models/                      # Model weights
│   ├── yolov11.pt                 # YOLOv11 trained
│   └── rf-detr-base.pth           # RF-DETR (optional)
│
├── 📂 data/
│   ├── 📂 videos/                 # Input videos
│   ├── 📂 frames/                 # Extracted frames (for training)
│   ├── 📂 sessions/               # Processing sessions
│   │   └── highway_test_20260115_143000/
│   │       ├── violations/        # Ảnh bằng chứng
│   │       ├── session_data.json  # Metadata
│   │       └── report.pdf         # Biên bản PDF
│   │
│   ├── 📂 train/                  # Training dataset
│   │   ├── images/
│   │   └── labels/
│   ├── 📂 valid/                  # Validation dataset
│   └── 📂 test/                   # Test dataset
│
├── 📂 docs/                        # Documentation
│   ├── CHUONG_4_5_BAO_CAO.md      # Chương 4-5 báo cáo
│   ├── HUONG_DAN_VIET_BAO_CAO_GAN_NHAN.md
│   └── PREPROCESSING_GUIDE.md
│
└── 📂 logs/                        # Application logs
    └── app.log
```

---

## 🔬 Logic Phát hiện Vi phạm

### Điều kiện Xác nhận Vi phạm (8 điều kiện AND)

| STT | Điều kiện | Mô tả |
|-----|-----------|-------|
| 1 | Đèn đang ĐỎ | Xác nhận qua Voting 3/5 frames |
| 2 | Xe ở TRƯỚC vạch khi đèn đỏ | Snapshot vị trí khi đèn chuyển đỏ |
| 3 | Xe VƯỢT QUA vạch | Có crossing motion (không chỉ vị trí tĩnh) |
| 4 | Không trong Grace Period | Sau 1.5 giây từ khi đèn đỏ |
| 5 | Chưa ghi nhận trước đó | track_id NOT IN recorded_violations |
| 6 | Đủ số frame xác nhận | min_frames = 3 frames liên tiếp |
| 7 | Confidence đủ cao | vehicle.confidence >= 0.5 |
| 8 | Xe nằm trong ROI | Trong vùng lane do đèn đỏ kiểm soát |

### Các trường hợp KHÔNG vi phạm

| Trường hợp | Lý do |
|------------|-------|
| Xe đã ở SAU vạch khi đèn đỏ | Đang đi hợp lệ trước đó |
| Trong grace period (1.5s đầu) | Không kịp dừng |
| Xe đi ngang (sideways) | Di chuyển theo phương X |
| Xe ngoài ROI | Không thuộc lane bị kiểm soát |
| Xe máy rẽ phải (nếu enable) | Theo luật VN có biển P.131b |

### Flow Chart

```
Video Frame
    ↓
[YOLOv11 Detection] → 6 classes
    ↓
[ByteTrack] → Track ID
    ↓
[Traffic Light Voting] → RED? → NO → Skip
    ↓ YES
[Snapshot Position] → Xe ở trước vạch?
    ↓ YES
[Check Crossing] → Vượt vạch?
    ↓ YES
[Grace Period] → Sau 1.5s?
    ↓ YES
[ROI Check] → Trong vùng?
    ↓ YES
[Multi-frame] → 3 frames liên tiếp?
    ↓ YES
✅ VIOLATION CONFIRMED
```

---

## 🎓 Training (Tùy chọn)

### Bước 1: Thu thập Dataset

**Option 1: Download từ YouTube**
```bash
python scripts/download_and_extract.py --url "https://youtube.com/watch?v=..." --interval 30
```

**Option 2: Extract từ video có sẵn**
```bash
python scripts/download_and_extract.py --video path/to/video.mp4 --interval 30 --max-frames 500
```

**Tham số:**
- `--interval 30`: Trích xuất mỗi 30 frame (1 FPS với video 30 FPS)
- `--max-frames 500`: Giới hạn số frame
- `--output data/frames`: Thư mục đầu ra

### Bước 2: Annotation trên Roboflow

1. Tạo project tại [roboflow.com](https://roboflow.com)
2. Upload ảnh từ `data/frames/`
3. Gán nhãn với 6 classes:
   - `car`, `motorbike`
   - `red_light`, `green_light`, `yellow_light`
   - `stop_line`
4. Augmentation (optional):
   - Brightness: ±20%
   - Blur: up to 1.5px
   - Cutout: 5% of bounding boxes
5. Export → YOLO v8/v11 format
6. Download và giải nén vào `data/`

### Bước 3: Chuẩn bị data.yaml

```yaml
# data/data.yaml
path: D:/Training Model/red_light_detection/data
train: train/images
val: valid/images
test: test/images

nc: 6
names:
  0: car
  1: green_light
  2: motorbike
  3: red_light
  4: stop_line
  5: yellow_light
```

### Bước 4: Train Model

```bash
python scripts/train.py --model yolov11s --data data/data.yaml --epochs 100
```

**Tham số training:**
- `--model`: yolov11n/s/m/l/x (s = khuyến nghị)
- `--epochs`: 100-300 epochs
- `--batch`: 16 (điều chỉnh theo VRAM)
- `--img-size`: 640 (default)
- `--device`: 0 (GPU index) hoặc cpu

### Bước 5: Evaluate

Model được lưu tại `runs/detect/train/weights/best.pt`

```bash
# Copy vào models/
cp runs/detect/train/weights/best.pt models/yolov11_custom.pt

# Update config.yaml
# model:
#   yolov11:
#     weights: "models/yolov11_custom.pt"
```

---

## 📊 Performance Benchmark

### So sánh Model (trên dataset của project)

| Model | mAP@50 | Precision | Recall | FPS (GPU) | FPS (CPU) | VRAM | Kết luận |
|-------|--------|-----------|--------|-----------|-----------|------|----------|
| **YOLOv11s** | ~88.5% | - | - | 25-30 | 2-3 | ~2GB | ✅ **Production** |
| **RF-DETR** | ~89.3% | - | - | 2-5 | <1 | ~4GB | Offline analysis |

**Hardware test:**
- GPU: NVIDIA RTX 3060 (12GB)
- CPU: Intel i7-12700
- RAM: 32GB

---

## 🐛 Troubleshooting

### 1. CUDA out of memory

**Giải pháp:**
```yaml
# config.yaml
model:
  yolov11:
    img_size: 480  # Giảm từ 640
    half: true     # Bật FP16
```

### 2. Import Error: No module named 'ultralytics'

```bash
pip install ultralytics==8.1.0
```

### 3. GUI không hiển thị

```bash
pip uninstall PySide6
pip install PySide6==6.6.0
```

### 4. "DLL load failed" trên Windows

Cài Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### 5. Tracking không ổn định

```yaml
tracking:
  track_thresh: 0.25  # Giảm threshold
  track_buffer: 90    # Tăng buffer
```

### 6. False positive nhiều

```yaml
violation:
  min_frames: 5  # Tăng từ 3 lên 5
  grace_period: 1.0  # Tăng lên 1 giây
  min_vehicle_confidence: 0.6  # Tăng confidence
```

---

## 📖 Documentation

### Tài liệu Kỹ thuật

- [Chương 4-5: Thiết kế Hệ thống](docs/CHUONG_4_5_BAO_CAO.md)
- [Hướng dẫn Viết Báo cáo](docs/HUONG_DAN_VIET_BAO_CAO_GAN_NHAN.md)
- [Preprocessing Guide](docs/PREPROCESSING_GUIDE.md)

### External Resources

- [YOLOv11 Docs](https://docs.ultralytics.com/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [Supervision Docs](https://supervision.roboflow.com/)
- [PySide6 Docs](https://doc.qt.io/qtforpython/)

---

## 🗺️ Roadmap

### ✅ Completed (v1.0)
- [x] YOLOv11 Detection + ByteTrack
- [x] Logic 2 tầng với 8 điều kiện
- [x] GUI 5 tabs với PySide6
- [x] PDF Report generation
- [x] Session management
- [x] ROI + Voting + Grace Period + Sideways detection
- [x] Motorbike right turn support

### 🔄 In Progress (v1.1)
- [ ] OCR biển số tự động (ALPR)
- [ ] Export statistics to Excel
- [ ] Web dashboard (Flask/FastAPI)

### 📅 Planned (v2.0)
- [ ] TensorRT optimization (60+ FPS)
- [ ] Multi-camera support (4-8 cameras)
- [ ] Edge deployment (Jetson Orin Nano)
- [ ] Cloud sync + centralized database
- [ ] Real-time alert system

---

## 👥 Contributors

Developed by Lê Huỳnh Huy Hoàng



---

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv11 framework
- **Roboflow** - Dataset management & annotation
- **ByteTrack authors** - Tracking algorithm
- **Qt Company** - PySide6 framework
- **ReportLab** - PDF generation library

---

## 📞 Contact & Support

- 📧 Email: lehuynhhuyhoang05@gmail.com

---

## ⚠️ Disclaimer

Đây là hệ thống nghiên cứu cho mục đích học tập. Để triển khai thực tế cần:

1. ✅ Dataset lớn hơn (5,000+ ảnh đa dạng)
2. ✅ Testing kỹ lưỡng trong nhiều điều kiện
3. ✅ Tuân thủ quy định pháp luật về giám sát giao thông
4. ✅ Approval từ cơ quan chức năng
5. ✅ GDPR/Privacy compliance

**Không sử dụng trực tiếp cho mục đích thương mại hoặc phạt nguội mà chưa có giấy phép.**

---
