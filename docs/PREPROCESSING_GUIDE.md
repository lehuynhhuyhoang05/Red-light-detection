# HƯỚNG DẪN TIỀN XỬ LÝ & TĂNG CƯỜNG DỮ LIỆU

## Áp dụng cho Bài toán Phát hiện Vi phạm Vượt Đèn Đỏ

---

## 1. TIỀN XỬ LÝ (Preprocessing)

### 1.1. Chuẩn hóa Kích thước
**Mục đích:** Đảm bảo tất cả ảnh có cùng kích thước để tối ưu hiệu suất huấn luyện.

**Cài đặt trên Roboflow:**
```
- YOLOv11 Fast: 640x640 pixels
- Giữ nguyên tỷ lệ khung hình (maintain aspect ratio)
- Padding màu đen nếu cần
```

**Trong báo cáo viết:**
> "Tất cả hình ảnh được thay đổi kích thước về tiêu chuẩn 640x640 pixel 
> cho YOLOv11 để đảm bảo tối ưu hóa hiệu suất huấn luyện."

---

### 1.2. Tự động Định hướng (Auto-Orient)
**Mục đích:** Đảm bảo ảnh được xoay đúng hướng dựa trên EXIF metadata.

**Cài đặt:**
```
✓ Enable Auto-Orient trong Roboflow
```

**Trong báo cáo viết:**
> "Các hình ảnh được tự động xoay dựa trên metadata EXIF để đảm bảo 
> tất cả đều có hướng đúng chuẩn."

---

### 1.3. Chuẩn hóa Pixel (Normalization)
**Mục đích:** Scale giá trị pixel về khoảng [0,1] để mô hình học tốt hơn.

**Cài đặt:**
```
- Normalization: Auto (Roboflow tự động xử lý)
- Pixel values: 0-255 → 0.0-1.0
```

---

## 2. TĂNG CƯỜNG DỮ LIỆU (Data Augmentation)

### ⚠️ KHUYẾN NGHỊ CHO BÀI TOÁN NÀY:

**Option A: KHÔNG DÙNG Augmentation** ⭐ (Như báo cáo mẫu)

**Lý do:**
```
✓ Dataset đã lớn (3,986 images)
✓ Đơn giản hóa bài toán
✓ Dễ phân tích kết quả
✓ Phù hợp báo cáo sinh viên
```

**Trong báo cáo viết:**
> "Ở báo cáo này, chúng tôi không áp dụng các kỹ thuật tăng cường 
> dữ liệu nhằm đơn giản hóa bài toán và tập trung vào so sánh hiệu 
> năng thuần túy của các kiến trúc mô hình."

---

**Option B: DÙNG Augmentation Nhẹ** (Nếu kết quả không tốt)

**Các kỹ thuật phù hợp:**

#### 2.1. Flip (Lật ảnh)
```
- Horizontal Flip: 50% probability
- Vertical Flip: KHÔNG dùng (đèn tín hiệu không bao giờ ngược)
```

**Lý do:** Xe có thể đi từ trái sang phải hoặc ngược lại.

#### 2.2. Brightness (Độ sáng)
```
- Range: -15% to +15%
```

**Lý do:** Mô phỏng điều kiện ánh sáng khác nhau (sáng/tối, nắng/râm).

#### 2.3. Exposure (Phơi sáng)
```
- Range: -10% to +10%
```

**Lý do:** Camera có thể tự động điều chỉnh exposure.

#### 2.4. Blur (Làm mờ)
```
- Up to 1.0px
```

**Lý do:** Mô phỏng camera chuyển động hoặc mất nét.

---

### ❌ CÁC KỸ THUẬT KHÔNG NÊN DÙNG:

```
✗ Rotation (Xoay): Đèn tín hiệu luôn thẳng đứng
✗ Crop: Có thể cắt mất đèn/xe
✗ Cutout: Có thể che mất thông tin quan trọng
✗ Mosaic: Quá phức tạp, làm mất cấu trúc ngã tư
```

---

## 3. CÀI ĐẶT TRÊN ROBOFLOW

### Bước 1: Vào Version Settings
```
1. Vào project → Dataset version
2. Click "Edit" hoặc "Preprocessing"
```

### Bước 2: Cấu hình Preprocessing
```
☑ Auto-Orient: ON
☑ Resize: 640x640
  ├─ Stretch to: 640x640
  └─ Format: Fill (with black edges)
☑ Auto-Contrast: OFF (giữ nguyên màu đèn)
```

### Bước 3: Cấu hình Augmentation (Nếu dùng Option B)
```
☑ Flip: Horizontal only
☑ Brightness: ±15%
☑ Exposure: ±10%
☑ Blur: Up to 1.0px
☐ Rotation: OFF
☐ Crop: OFF
☐ Cutout: OFF
```

### Bước 4: Apply & Generate
```
Click "Generate" → Chờ Roboflow xử lý
```

---

## 4. VIẾT TRONG BÁO CÁO

### Phần Tiền xử lý:

```markdown
#### 3.2.3. Tiền xử lý và Tăng cường Dữ liệu

**Tiền xử lý:**

Để đảm bảo tính đồng nhất của dữ liệu đầu vào, các bước tiền xử lý 
sau được áp dụng trên nền tảng Roboflow:

1. **Chuẩn hóa kích thước:** Tất cả hình ảnh được thay đổi kích thước 
   về 640x640 pixel cho YOLOv11, đảm bảo tối ưu hóa hiệu suất huấn luyện.

2. **Tự động định hướng:** Các hình ảnh được tự động xoay dựa trên 
   metadata EXIF để đảm bảo tất cả đều có hướng đúng chuẩn.

3. **Chuẩn hóa giá trị pixel:** Giá trị pixel được chuẩn hóa từ 
   khoảng [0, 255] về [0.0, 1.0] để tăng tốc độ hội tụ của mô hình.

![Hình 3.X: Giao diện cấu hình tiền xử lý trên Roboflow]

**Tăng cường dữ liệu:**

Ở báo cáo này, chúng tôi không áp dụng các kỹ thuật tăng cường dữ liệu 
nhằm đơn giản hóa bài toán và tập trung vào việc so sánh hiệu năng 
thuần túy của các kiến trúc mô hình trên tập dữ liệu gốc. Điều này 
giúp loại bỏ các biến số không kiểm soát được từ quá trình augmentation, 
đảm bảo rằng sự khác biệt về kết quả chỉ đến từ bản chất của kiến trúc.
```

---

## 5. SO SÁNH VỚI BÁO CÁO MẪU

### Giống:
- ✓ Không dùng augmentation
- ✓ Resize về kích thước chuẩn (640x640)
- ✓ Auto-orient

### Khác:
- Dataset lớn hơn (3,986 vs ~1,000)
- 6 classes vs 2 classes (helmet/no_helmet)
- Bài toán phức tạp hơn (logic vi phạm, tracking)

---

## 6. CHECKLIST TRƯỚC KHI TRAIN

```
☑ Resize: 640x640 ✓
☑ Auto-Orient: ON ✓
☑ Normalization: Auto ✓
☐ Augmentation: OFF (theo Option A) ✓
☑ Train/Valid/Test split: 70/20/10 (đã có sẵn)
☑ Model size: Fast (YOLOv11 Small) ✓
☑ Epochs: Auto (Roboflow tự chọn ~200-300)
```

---

## 7. KẾT LUẬN

**Cho bài toán của bạn:**
1. ✅ Chọn **Fast model**
2. ✅ Preprocessing: Resize + Auto-Orient + Normalize
3. ✅ Augmentation: **KHÔNG DÙNG** (Option A)
4. ✅ Lý do: Dataset đủ lớn, đơn giản hóa phân tích

**Nếu kết quả không đạt yêu cầu:**
→ Thử lại với Augmentation nhẹ (Option B)
→ Hoặc train model lớn hơn (Accurate)

---

**GHI CHÚ CHO BÁO CÁO:**
Copy đoạn "4. VIẾT TRONG BÁO CÁO" vào phần 3.2.3 của báo cáo,
thay thế các placeholder [Hình 3.X] bằng screenshot từ Roboflow.
