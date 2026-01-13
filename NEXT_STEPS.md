# 🚀 HƯỚNG DẪN THU THẬP DỮ LIỆU - BƯỚC TIẾP THEO

## ✅ BẠN ĐANG Ở: GIAI ĐOẠN THU THẬP DỮ LIỆU

---

## 📋 KẾ HOẠCH HÀNH ĐỘNG CỤ THỂ

### **BƯỚC 1: CÀI ĐẶT CÔNG CỤ (10 phút)**

```powershell
# Chạy trong terminal VS Code:
python scripts/setup_tools.py
```

**Kết quả mong đợi:**
- ✓ yt-dlp installed
- ✓ Kaggle CLI installed  
- ✓ OpenCV ready

---

### **BƯỚC 2A: TÌM VIDEO YOUTUBE (30-60 phút)**

#### 2A.1: Xem hướng dẫn tìm kiếm

```powershell
python scripts/collect_youtube_videos.py --guide
```

#### 2A.2: Tìm video trên YouTube

**Mở YouTube và tìm kiếm:**
```
Tiếng Việt:
- "camera giao thông ngã tư việt nam"
- "đèn tín hiệu giao thông tp hcm"
- "camera hành trình ngã tư"

Tiếng Anh:
- "traffic light intersection camera"
- "red light violation camera"
- "intersection traffic surveillance"
```

**Tiêu chí chọn video:**
- ✅ Thời lượng: >5 phút
- ✅ Chất lượng: Tối thiểu 720p
- ✅ Góc quay: Nhìn rõ đèn tín hiệu + đường
- ✅ Nhiều phương tiện đi qua
- ✅ Có đèn đỏ/vàng/xanh rõ ràng

**Mục tiêu: Tìm 15-20 video tốt**

#### 2A.3: Thêm URL vào script

Sau khi tìm được video, mở file:
```
scripts/collect_youtube_videos.py
```

Tìm dòng:
```python
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=example1",  # Replace
    "https://www.youtube.com/watch?v=example2",
]
```

Thay bằng URL thực tế:
```python
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=YOUR_VIDEO_ID_1",
    "https://www.youtube.com/watch?v=YOUR_VIDEO_ID_2",
    "https://www.youtube.com/watch?v=YOUR_VIDEO_ID_3",
    # ... thêm URLs
]
```

#### 2A.4: Tải video về

```powershell
python scripts/collect_youtube_videos.py
```

**Video sẽ được lưu vào:** `data/videos/`

---

### **BƯỚC 2B: QUAY VIDEO TỰ TẠO (Tùy chọn - 1-2 giờ)**

**Nếu bạn có thời gian, nên quay thêm video tại ngã tư gần nhà/trường:**

**Thiết bị:**
- 📱 Smartphone (iPhone/Android bất kỳ)
- Quay ở 1080p/30fps

**Địa điểm đề xuất (TP.HCM):**
- Ngã tư Hàng Xanh
- Ngã tư Bảy Hiền
- Ngã tư gần trường/nhà bạn (có đèn tín hiệu)

**Cách quay:**
1. Đứng ở góc ngã tư, cao hơn mặt đường
2. Giữ máy ổn định (dùng tripod nếu có)
3. Hướng camera nhìn rõ:
   - Đèn tín hiệu
   - Vạch dừng trên đường
   - Phương tiện đi qua
4. Quay liên tục 10-15 phút mỗi ngã tư

**Sau khi quay xong:**
- Copy video vào: `data/videos/`
- Đặt tên: `my_traffic_01.mp4`, `my_traffic_02.mp4`, ...

---

### **BƯỚC 3: TRÍCH XUẤT FRAMES (15-30 phút)**

**Sau khi có video trong `data/videos/`, chạy:**

```powershell
# Trích xuất 1 frame/giây, lọc ảnh mờ
python scripts/extract_frames.py
```

**Tham số tùy chỉnh:**
```powershell
# Lấy 2 frame/giây (nhiều hơn)
python scripts/extract_frames.py --fps 2.0

# Giảm ngưỡng lọc mờ (lấy nhiều ảnh hơn)
python scripts/extract_frames.py --blur-threshold 80

# Giới hạn 100 frames mỗi video
python scripts/extract_frames.py --max-frames 100

# Xử lý 1 video cụ thể
python scripts/extract_frames.py --video "data/videos/traffic_video_001.mp4"
```

**Kết quả:**
- Frames được lưu vào: `data/frames/`
- Tên file: `video01_frame_00000.jpg`, `video01_frame_00001.jpg`, ...

**Mục tiêu: 400-600 frames từ bước này**

---

### **BƯỚC 4: KIỂM TRA CHẤT LƯỢNG FRAMES (15 phút)**

```powershell
# Mở thư mục frames
explorer data\frames
```

**Kiểm tra:**
- ✅ Ảnh rõ nét, không mờ
- ✅ Nhìn thấy đèn tín hiệu
- ✅ Nhìn thấy phương tiện
- ✅ Đa dạng: Đông xe/vắng xe, sáng/tối

**Xóa ảnh kém chất lượng:**
- Ảnh quá tối
- Ảnh mờ
- Góc quay xấu

---

### **BƯỚC 5: BỔ SUNG TỪ DATASET CÔNG KHAI (Tùy chọn)**

**Nếu cần thêm dữ liệu:**

#### Option A: Roboflow Universe

1. Vào: https://universe.roboflow.com
2. Tìm kiếm: "traffic light detection"
3. Chọn dataset phù hợp
4. Export → YOLO Format
5. Tải về và giải nén vào `data/roboflow/`

#### Option B: Kaggle

```powershell
# Đã có script hỗ trợ
python scripts/download_kaggle.py
```

---

## 📊 KIỂM TRA TIẾN ĐỘ

**Sau khi hoàn thành bước 3, kiểm tra:**

```powershell
# Đếm số frame
Get-ChildItem data\frames\*.jpg | Measure-Object
```

**Mục tiêu:**
- ✅ Tối thiểu: 400-500 frames
- ✅ Lý tưởng: 800-1000 frames
- ✅ Đa dạng điều kiện

---

## 🎯 BƯỚC TIẾP THEO SAU KHI CÓ FRAMES

### **BƯỚC 6: GÁN NHÃN TRÊN ROBOFLOW (Tuần 3)**

**Chuẩn bị:**
1. Tạo tài khoản: https://roboflow.com
2. Tạo project mới: "Red Light Violation Detection"
3. Chọn: Object Detection

**5 classes cần gán nhãn:**
- `vehicle` (hoặc: `motorcycle`, `car`, `truck`)
- `red_light`
- `yellow_light`
- `green_light`
- `stop_line`

**Upload frames:**
- Upload tất cả ảnh trong `data/frames/`
- Bắt đầu gán nhãn

**Quy trình gán nhãn:**
1. Vẽ bounding box cho từng đối tượng
2. Đèn: Chỉ gán nhãn cho đèn đang sáng
3. Vạch dừng: Vẽ box ngang qua làn đường
4. Xe: Bao toàn bộ xe

**Mục tiêu: 50-70 ảnh/ngày**

---

## 📁 CẤU TRÚC THƯ MỤC SAU BƯỚC 3

```
data/
├── videos/           # Video gốc từ YouTube/tự quay
│   ├── traffic_video_001.mp4
│   ├── traffic_video_002.mp4
│   └── ...
│
├── frames/           # Frames đã trích xuất
│   ├── video01_frame_00000.jpg
│   ├── video01_frame_00001.jpg
│   └── ... (400-1000 ảnh)
│
├── sessions/         # (Sẽ dùng sau khi có model)
└── violations/       # (Sẽ dùng sau khi có model)
```

---

## 🆘 GẶP VẤN ĐỀ?

### Lỗi: "yt-dlp not found"
```powershell
pip install yt-dlp
```

### Lỗi: "Cannot open video"
- Kiểm tra video có hỏng không
- Thử convert bằng VLC hoặc FFmpeg

### Không tìm được video tốt trên YouTube
- Thử từ khóa khác
- Tìm trên các nền tảng khác: Vimeo, Dailymotion
- Ưu tiên quay video tự tạo

### Trích xuất quá ít frames
- Giảm `--blur-threshold` xuống 50-80
- Tăng `--fps` lên 2.0
- Kiểm tra chất lượng video gốc

---

## ✅ CHECKLIST HOÀN THÀNH GIAI ĐOẠN 2

- [ ] Cài đặt công cụ (yt-dlp, kaggle)
- [ ] Tìm được 15-20 video YouTube
- [ ] Tải được video về `data/videos/`
- [ ] Trích xuất được 400-600 frames
- [ ] Kiểm tra chất lượng frames
- [ ] Có ảnh đa dạng (sáng/tối, đông/vắng)
- [ ] Sẵn sàng upload lên Roboflow

**Khi hoàn thành → Chuyển sang GIAI ĐOẠN 3: GÁN NHÃN**

---

## 💡 LỜI KHUYÊN

1. **Đừng vội:** Chất lượng dữ liệu quan trọng hơn số lượng
2. **Đa dạng hóa:** Thu thập nhiều điều kiện khác nhau
3. **Kiểm tra kỹ:** Review frames trước khi gán nhãn
4. **Backup:** Copy dữ liệu sang ổ cứng khác/cloud

---

**BẮT ĐẦU NGAY:** `python scripts/setup_tools.py`
