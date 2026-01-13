# KẾ HOẠCH HOÀN THIỆN HỆ THỐNG PHÁT HIỆN VI PHẠM VƯỢT ĐÈN ĐỎ

## 📊 HIỆN TRẠNG

### ✅ Đã Hoàn Thành
- [x] Logic xác nhận vi phạm (Violation Logic Core) - đã test thành công
- [x] Tài liệu lý thuyết chi tiết (Chương 4 + LY_THUYET_LOGIC_XU_LY_VI_PHAM.md)
- [x] Model RT-DETR đã download (rf-detr.pt - 127MB)
- [x] Detector module với support YOLOv11, RT-DETR, YOLO-NAS
- [x] Tracker module (ByteTrack)
- [x] ViolationDetector class hoàn chỉnh
- [x] Test script validation (test_violation_logic.py)

### ❌ Chưa Hoàn Thành
- [ ] Model YOLOv11 (cần download từ Roboflow)
- [ ] GUI hoàn chỉnh với đầy đủ tính năng
- [ ] Script so sánh 2 models
- [ ] Report Generator (PDF tickets)
- [ ] Video demo để test

---

## 🎯 KẾ HOẠCH THỰC HIỆN

### **BƯỚC 1: Download Model YOLOv11** (5-10 phút)
**Mục tiêu**: Có đủ 2 models để so sánh

**Thông tin model từ Chương 4:**
- Model ID: `red-light-violation-detect-hecrg/3`
- Model Type: YOLOv11 Object Detection
- Workspace: huyhoang
- Project: red-light-violation-detect-hecrg
- Version: 3
- mAP@50: 87.9%

**Action:**
```powershell
# Option 1: Sử dụng Roboflow API (nếu có API key)
python scripts/download_model.py --model yolov11

# Option 2: Download thủ công từ Roboflow web
# Vào: https://app.roboflow.com/huyhoang/red-light-violation-detect-hecrg/3
# Download weights -> Lưu vào models/yolov11_best.pt
```

**Kết quả mong đợi:**
```
models/
├── rf-detr.pt       (127MB) ✅
└── yolov11_best.pt  (22MB)  ← CẦN
```

---

### **BƯỚC 2: Chuẩn bị Video Demo** (10-15 phút)
**Mục tiêu**: Có video thực tế để test hệ thống

**Option A: Download video demo có sẵn**
```powershell
# Tìm video trên YouTube hoặc sử dụng video có sẵn
python scripts/collect_youtube_videos.py --url "https://youtube.com/watch?v=VIDEO_ID" --output data/videos/demo.mp4
```

**Option B: Sử dụng video ngắn để test nhanh**
- Video 30-60 giây
- Có đèn tín hiệu rõ ràng
- Có xe vượt đèn đỏ

**Kết quả mong đợi:**
```
data/videos/
└── demo.mp4  (hoặc test_video.mp4)
```

---

### **BƯỚC 3: Hoàn thiện GUI Application** (30-45 phút)

**3.1. Các tính năng cần implement:**

#### **Tab 1: Live Detection**
- [x] Video player with controls (play/pause/stop)
- [x] Real-time detection overlay
- [x] Statistics panel (vehicles, violations, light state)
- [ ] Model selector (YOLOv11 / RT-DETR)
- [ ] Confidence threshold slider
- [ ] Stop line manual setup (click to set)

#### **Tab 2: Violation Records**
- [ ] Table hiển thị danh sách vi phạm
- [ ] Thumbnail preview ảnh bằng chứng
- [ ] Filter theo thời gian, loại xe
- [ ] Export violations to CSV/JSON
- [ ] View evidence (3 ảnh + video clip)

#### **Tab 3: Model Comparison**
- [ ] Side-by-side video comparison
- [ ] Performance metrics table (FPS, mAP, Latency)
- [ ] Confusion matrix visualization
- [ ] Detection quality comparison
- [ ] Export comparison report

#### **Tab 4: Statistics & Reports**
- [ ] Session summary
- [ ] Violations by hour/type chart
- [ ] Detection heatmap
- [ ] Generate PDF report button

**3.2. File cần chỉnh sửa:**
- `src/gui.py` (main GUI class)
- `src/report_generator.py` (PDF generation)
- Tạo `src/model_comparison.py` (so sánh 2 models)

---

### **BƯỚC 4: Implement Model Comparison Module** (20-30 phút)

**File mới: `src/model_comparison.py`**

```python
class ModelComparator:
    def __init__(self, model1_config, model2_config):
        self.model1 = create_detector(model1_config)
        self.model2 = create_detector(model2_config)
    
    def compare_on_video(self, video_path):
        """So sánh 2 models trên cùng video"""
        # Return metrics: FPS, mAP, violations_detected, etc.
    
    def generate_comparison_report(self, results):
        """Tạo báo cáo so sánh"""
        # Export to PDF/JSON
```

**Metrics cần so sánh:**
- Inference time (ms/frame)
- Average FPS
- Detection accuracy (so với ground truth nếu có)
- Số vi phạm phát hiện được
- False positives/negatives
- Memory usage
- Model size

---

### **BƯỚC 5: Implement Report Generator** (20-30 phút)

**File: `src/report_generator.py`**

**Chức năng:**
1. **Violation Ticket (Biên bản phạt)** - PDF format
   - Thông tin vi phạm
   - 3 ảnh bằng chứng
   - Metadata (thời gian, địa điểm, loại xe)
   - Mã QR để tra cứu online

2. **Session Report (Báo cáo ca)** - PDF format
   - Tổng hợp vi phạm theo ca
   - Thống kê theo loại xe
   - Biểu đồ phân bố theo giờ
   - So sánh với ca trước

3. **Model Comparison Report** - PDF format
   - Bảng so sánh metrics
   - Screenshots demo
   - Kết luận và khuyến nghị

**Library sử dụng:**
```python
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
```

---

### **BƯỚC 6: Testing & Demo** (30-45 phút)

**6.1. Test từng module:**
```powershell
# Test detection với YOLOv11
python scripts/quick_test.py --model yolov11

# Test detection với RT-DETR
python scripts/quick_test.py --model rt-detr

# Test violation logic
python scripts/test_violation_logic.py  # ✅ Đã pass

# Test GUI
python main.py --gui
```

**6.2. Test so sánh models:**
```powershell
# Run comparison
python scripts/compare_models.py --video data/videos/demo.mp4 --output results/comparison.pdf
```

**6.3. Test full pipeline:**
```powershell
# Process video và generate report
python main.py --video data/videos/demo.mp4 --model yolov11 --output session_001
```

---

### **BƯỚC 7: Chuẩn bị Demo & Tài liệu Chứng minh** (30-60 phút)

**7.1. Demo Video/Screenshots:**
- Screen recording GUI đang chạy
- Highlight các tính năng chính
- Show vi phạm được phát hiện real-time
- Show report generation

**7.2. Tài liệu so sánh:**
Tạo file: `docs/SO_SANH_MODELS_RESULTS.md`
```markdown
# KẾT QUẢ SO SÁNH YOLOv11 vs RT-DETR

## Điều kiện Test
- Video: demo.mp4 (60 giây, 1080p, 30fps)
- Hardware: [CPU/GPU info]
- Dataset: 1800 frames

## Kết quả

| Metric | YOLOv11-Small | RT-DETR-Small | Winner |
|--------|---------------|---------------|--------|
| FPS | 32.5 | 28.3 | YOLOv11 ⭐ |
| Latency (ms) | 30.7 | 35.4 | YOLOv11 ⭐ |
| Model Size (MB) | 22 | 127 | YOLOv11 ⭐ |
| Violations Detected | 5 | 5 | Tie ✅ |
| False Positives | 0 | 0 | Tie ✅ |
| Memory (MB) | 850 | 1200 | YOLOv11 ⭐ |

## Kết luận
YOLOv11 vượt trội về tốc độ và hiệu quả tài nguyên.
RT-DETR có độ chính xác tương đương nhưng nặng hơn.

→ **Khuyến nghị: Sử dụng YOLOv11 cho production**
```

**7.3. Update Chương 4:**
Thêm section "4.6 Kết quả Thử nghiệm và Đánh giá"

---

## 📋 CHECKLIST HOÀN THÀNH

### Core System
- [x] Violation Logic Implementation
- [x] Detector Module (Multi-model support)
- [x] Tracker Module
- [ ] Report Generator
- [ ] Model Comparison Module

### Models
- [ ] YOLOv11 weights
- [x] RT-DETR weights

### Data
- [ ] Demo video (30-60s)
- [ ] Test cases for validation

### GUI
- [x] Basic structure (VideoProcessor, main window)
- [ ] Model selector & configuration
- [ ] Violation records viewer
- [ ] Comparison tab
- [ ] Statistics & charts
- [ ] PDF export

### Documentation
- [x] Lý thuyết logic (Chi tiết)
- [x] Chương 4 (Thiết kế hệ thống)
- [ ] So sánh models (Kết quả thực nghiệm)
- [ ] Hướng dẫn sử dụng GUI

### Testing & Demo
- [ ] Test full pipeline
- [ ] Record demo video
- [ ] Generate sample reports
- [ ] Comparison results

---

## ⏱️ TIMELINE DỰ KIẾN

**Tổng thời gian: 3-4 giờ**

| Bước | Thời gian | Ưu tiên |
|------|-----------|---------|
| Download YOLOv11 | 10 phút | 🔴 Cao |
| Chuẩn bị video demo | 15 phút | 🔴 Cao |
| Hoàn thiện GUI | 45 phút | 🔴 Cao |
| Model Comparison | 30 phút | 🟠 Trung bình |
| Report Generator | 30 phút | 🟠 Trung bình |
| Testing | 45 phút | 🔴 Cao |
| Demo & Documentation | 60 phút | 🟡 Thấp |

---

## 🚀 BƯỚC TIẾP THEO NGAY BÂY GIỜ

**Bạn muốn bắt đầu từ đâu?**

1. **Download YOLOv11 model** (quan trọng nhất)
2. **Tìm video demo** để test
3. **Hoàn thiện GUI** (Tab Model Comparison)
4. **Implement Model Comparison script**
5. **Khác (chỉ định)**

👉 **Gợi ý: Bắt đầu với #1 (Download YOLOv11) để có đủ 2 models**
