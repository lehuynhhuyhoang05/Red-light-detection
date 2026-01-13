# Red Light Violation Detection System 🚦

Hệ thống phát hiện vi phạm vượt đèn đỏ sử dụng Deep Learning (YOLOv11, YOLO-NAS, RT-DETR)

## 📋 Tổng quan

Dự án nghiên cứu và phát triển hệ thống giám sát giao thông thông minh để:
- ✅ Phát hiện phương tiện (xe máy, ô tô, xe tải)
- ✅ Nhận diện trạng thái đèn tín hiệu (đỏ, vàng, xanh)
- ✅ Xác định vạch dừng
- ✅ Phát hiện hành vi vi phạm vượt đèn đỏ
- ✅ Lưu bằng chứng và tạo biên bản tự động

## 🎯 Tính năng

### Core Features
- **Multi-Model Support**: YOLOv11, YOLO-NAS, RT-DETR
- **Object Tracking**: ByteTrack để theo dõi xe qua nhiều frame
- **Violation Logic**: Thuật toán thông minh phát hiện vi phạm
- **GUI Application**: Giao diện đồ họa với PySide6
- **CLI Mode**: Xử lý video qua command line
- **Evidence Storage**: Lưu ảnh bằng chứng và metadata
- **PDF Reports**: Tạo biên bản vi phạm tự động

### GUI Features
- 📹 **Video Tab**: Xem video real-time với annotations
- ⚠️ **Violations Tab**: Danh sách vi phạm, xem bằng chứng
- 📊 **Statistics Tab**: Thống kê chi tiết
- ⚙️ **Settings Tab**: Cấu hình hệ thống

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- CUDA 11.8+ (nếu dùng GPU)
- 16GB RAM
- 50GB dung lượng trống

### Bước 1: Clone repository

```bash
cd "c:\Study\ITS\Training Model\red_light_detection"
```

### Bước 2: Tạo môi trường ảo

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình

Chỉnh sửa `config.yaml` theo nhu cầu:

```yaml
model:
  type: "yolov11"  # hoặc yolo-nas, rt-detr
  
location:
  intersection: "Ngã tư Lê Duẩn - Điện Biên Phủ"
  city: "Đà Nẵng"
  camera_id: "CAM-001"
```

## 🚀 Sử dụng

### 1. GUI Mode (Khuyến nghị)

```bash
python main.py --gui
```

### 2. CLI Mode - Xử lý video

```bash
python main.py --video path/to/video.mp4
```

### 3. Sử dụng model cụ thể

```bash
python main.py --gui --model yolov11
python main.py --gui --model yolo-nas
python main.py --gui --model rt-detr
```

## 📊 Thu thập dữ liệu

### Download video từ YouTube

```bash
python scripts/download_and_extract.py --url "https://youtube.com/watch?v=..." --interval 30
```

### Trích xuất frames từ video có sẵn

```bash
python scripts/download_and_extract.py --video path/to/video.mp4 --interval 30 --max-frames 500
```

**Tham số:**
- `--interval 30`: Trích xuất mỗi 30 frame (1 FPS nếu video 30 FPS)
- `--max-frames`: Giới hạn số frame
- `--output`: Thư mục đầu ra (mặc định: `data/frames`)

## 🎓 Huấn luyện mô hình

### Chuẩn bị dataset

1. Upload ảnh lên [Roboflow](https://roboflow.com)
2. Gán nhãn với các class:
   - `vehicle` / `motorcycle` / `car` / `truck`
   - `red_light` / `yellow_light` / `green_light`
   - `stop_line`
3. Export dataset (YOLO format)
4. Download và giải nén vào `data/`

### Train YOLOv11

```bash
python scripts/train.py --model yolov11 --data data/data.yaml
```

### Train YOLO-NAS

```bash
python scripts/train.py --model yolo-nas --data data/data.yaml
```

### Train RT-DETR

```bash
python scripts/train.py --model rt-detr --data data/data.yaml
```

### Sau khi train

Model được lưu tại `runs/train/*/weights/best.pt`. Copy vào thư mục `models/`:

```bash
cp runs/train/yolov11_yolov11s/weights/best.pt models/yolov11s_best.pt
```

Cập nhật `config.yaml`:

```yaml
model:
  yolov11:
    weights: "models/yolov11s_best.pt"
```

## 📁 Cấu trúc thư mục

```
red_light_detection/
├── main.py                    # Entry point
├── config.yaml                # Cấu hình
├── requirements.txt           # Dependencies
│
├── src/                       # Source code
│   ├── detector.py           # Object detection (YOLOv11/NAS/RT-DETR)
│   ├── tracker.py            # ByteTrack tracking
│   ├── violation_logic.py    # Violation detection logic
│   ├── gui.py                # PySide6 GUI
│   ├── report_generator.py   # PDF generation
│   └── utils.py              # Utilities
│
├── scripts/                   # Utility scripts
│   ├── download_and_extract.py   # Download video, extract frames
│   └── train.py              # Training script
│
├── models/                    # Trained models (.pt, .pth)
├── data/
│   ├── videos/               # Input videos
│   ├── frames/               # Extracted frames
│   ├── violations/           # Evidence images
│   └── sessions/             # Processing sessions
│
└── logs/                      # Application logs
```

## 🔬 Logic phát hiện vi phạm

Hệ thống xác định vi phạm khi:

1. ✅ Đèn tín hiệu đang **ĐỎ**
2. ✅ Xe **vượt qua** vạch dừng
3. ✅ Xe **chưa** ở phía sau vạch khi đèn chuyển đỏ

### Các trường hợp không vi phạm:

- ❌ Xe đã qua vạch **trước khi** đèn chuyển đỏ
- ❌ Xe dừng đúng trước vạch
- ❌ Trong thời gian grace period (1 giây sau đèn đỏ)

## 📊 So sánh mô hình

| Mô hình    | mAP@50 | Precision | Recall | F1-Score | FPS  | Nhận xét |
|-----------|--------|-----------|--------|----------|------|----------|
| YOLOv11s  | ?      | ?         | ?      | ?        | ~60  | Cân bằng tốc độ & độ chính xác |
| YOLO-NAS  | ?      | ?         | ?      | ?        | ~45  | Độ chính xác cao |
| RT-DETR   | ?      | ?         | ?      | ?        | ~30  | Transformer-based |

*(Điền số liệu sau khi huấn luyện)*

## 🐛 Xử lý lỗi thường gặp

### 1. CUDA out of memory

```yaml
# config.yaml
performance:
  half_precision: true  # Bật FP16
  batch_size: 1         # Giảm batch size
```

Hoặc giảm kích thước ảnh:

```yaml
model:
  yolov11:
    img_size: 480  # Thay vì 640
```

### 2. Import error

```bash
pip install --upgrade ultralytics super-gradients supervision
```

### 3. GUI không hiển thị

```bash
pip uninstall PySide6
pip install PySide6==6.6.0
```

## 📝 Roadmap

- [ ] Tích hợp nhận diện biển số xe
- [ ] Deploy lên edge device (Jetson Nano)
- [ ] API REST cho tích hợp hệ thống
- [ ] Dashboard web real-time
- [ ] Multi-camera support
- [ ] Database integration (PostgreSQL)

## 📖 Tài liệu tham khảo

- [YOLOv11 Documentation](https://docs.ultralytics.com)
- [YOLO-NAS Paper](https://arxiv.org/abs/2305.15808)
- [RT-DETR Paper](https://arxiv.org/abs/2304.08069)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [Roboflow Universe](https://universe.roboflow.com)

## 👥 Đóng góp

Dự án nghiên cứu cho khóa luận tốt nghiệp - ITS Research Team

## 📄 License

MIT License - Tự do sử dụng cho mục đích học tập và nghiên cứu

---

**Lưu ý**: Đây là hệ thống nghiên cứu. Để triển khai thực tế cần:
- Dataset lớn hơn (5000+ ảnh)
- Testing kỹ lưỡng
- Tuân thủ quy định pháp luật về giám sát giao thông
