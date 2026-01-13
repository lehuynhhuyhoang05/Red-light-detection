# CHI TIẾT THUẬT TOÁN XỬ LÝ VI PHẠM (LOGIC CORE)

Tài liệu này trình bày lý thuyết chuyên sâu và cơ chế hoạt động của thuật toán xác nhận vi phạm vượt đèn đỏ ("Violation Logic Core") được cài đặt trong hệ thống. Đây là thành phần quan trọng nhất quyết định độ chính xác và tính pháp lý của bằng chứng.

## 1. Đặt vấn đề & Thách thức

Trong các hệ thống giám sát giao thông dựa trên thị giác máy tính truyền thống, việc xác định vi phạm thường gặp các lỗi sai (False Positives) do các nguyên nhân sau:

1.  **Vấn đề độ trễ (Latency/Lag):** Xe đã đi vào giao lộ khi đèn Vàng/Xanh, nhưng do tắc đường hoặc di chuyển chậm, khi đèn chuyển Đỏ xe vẫn còn trong khung hình. Hệ thống ngây thơ sẽ bắt lỗi xe này.
2.  **Nhiễu tín hiệu đèn (Flickering):** Việc nhận diện màu đèn (Classify) có thể không ổn định (ví dụ: đang Đỏ nhận diện nhầm thành Vàng trong 1-2 frame), gây gián đoạn logic kiểm tra.
3.  **Rung lắc hộp bao (Bounding Box Jitter):** Hộp bao quanh xe (Bounding Box) có thể co giãn, khiến mép dưới của xe "nhảy" qua lại vạch dừng dù xe đang đứng yên.

Do đó, một thuật toán dựa trên "Trạng thái tĩnh" (kiểm tra `if car inside box and red`) là không đủ. Hệ thống cần một **Thuật toán dựa trên Lịch sử và Ảnh chụp trạng thái (State Snapshot & History-based Algorithm)**.

## 2. Mô hình Toán học

Chúng tôi xây dựng mô hình vi phạm dựa trên định nghĩa pháp lý chặt chẽ:

> "Một phương tiện bị coi là vượt đèn đỏ nếu tại thời điểm tín hiệu đèn chuyển sang ĐỎ, phương tiện đó vẫn nằm HOÀN TOÀN phía trước vạch dừng, và sau đó tiếp tục di chuyển vượt qua vạch trong thời gian đèn đỏ."

Gọi $t$ là thời gian hiện tại.
Gọi $t_{red\_start}$ là thời điểm đèn tín hiệu bắt đầu chuyển sang ĐỎ.
Gọi $Y_{vehicle}(t)$ là toạ độ trọng tâm hoặc mép dưới của phương tiện tại thời điểm $t$.
Gọi $Y_{line}$ là toạ độ vạch dừng (giả sử trục Y tăng dần từ trên xuống dưới).

Điều kiện vi phạm ($V_{violation}$) được xác định bởi tổ hợp logic AND:

$$ V_{violation} = C_1 \land C_2 \land C_3 \land C_4 $$

Trong đó:
1.  **Điều kiện khởi tạo ($C_1$):** Trạng thái đèn là Đỏ ổn định.
    $$ S_{light}(t) = \text{RED\_STABLE} $$
2.  **Điều kiện vị trí ban đầu ($C_2$ - Snapshot Condition):** Tại thời điểm $t_{red\_start}$, xe chưa vượt qua vạch.
    $$ Y_{vehicle}(t_{red\_start}) < Y_{line} $$
3.  **Điều kiện hành vi ($C_3$ - Crossing Condition):** Tại thời điểm hiện tại $t > t_{red\_start}$, xe đã vượt qua vạch.
    $$ Y_{vehicle}(t) > Y_{line} + \epsilon $$
    (Với $\epsilon$ là ngưỡng dung sai an toàn)
4.  **Điều kiện vector chuyển động ($C_4$):** Xe có vector vận tốc hướng về phía giao lộ.

## 3. Kiến trúc Thuật toán

Hệ thống được thiết kế theo mô hình Máy trạng thái (State Machine) với 3 thành phần cốt lõi:

### 3.1. Bộ lọc tín hiệu đèn (Traffic Light Voter)
Để xử lý nhiễu (Flickering), tín hiệu đèn không được sử dụng trực tiếp từ model Detection ($D_t$) mà đi qua bộ lọc Voting:

- **Input:** Chuỗi trạng thái nhận diện $D_{t}, D_{t-1}, ..., D_{t-n}$.
- **Logic:**
  Trạng thái $S_{light}$ tại thời điểm $t$ được quyết định bởi đa số (Majority Voting) trong cửa sổ trượt $W$ (Window Size $\approx$ 5-10 frames).
  $$ S_{light}(t) = \text{mode}(\{D_k \mid k \in [t-W, t]\}) $$
- **Transition:** Hệ thống chỉ kích hoạt sự kiện `ON_RED_START` khi trạng thái $S_{light}$ chuyển từ `NON_RED` sang `RED` một cách ổn định.

### 3.2. Cơ chế Snapshot (State Snapshot)
Đây là "trái tim" của thuật toán để loại bỏ Oan sai.

- **Sự kiện kích hoạt:** Ngay khi `ON_RED_START` xảy ra.
- **Hành động:** Hệ thống tạo một bản sao (Snapshot) vị trí của tất cả các phương tiện đang được theo dõi trong khung hình.
- **Phân loại xe:**
  - **Nhóm SAFE (An toàn):** Các xe có $Y_{vehicle} > Y_{line}$ (đã qua vạch). Các xe này được đánh dấu whitelist vĩnh viễn trong suốt chu kỳ đèn đỏ này. Dù chúng vẫn di chuyển trong giao lộ, hệ thống sẽ **KHÔNG** bắt lỗi.
  - **Nhóm CANDIDATE (Đối tượng cần theo dõi):** Các xe có $Y_{vehicle} < Y_{line}$ (chưa qua vạch). Đây là những xe có *nguy cơ* vi phạm nếu tiếp tục di chuyển.

### 3.3. Xác thực hành vi cắt ngang (Crossing Verification)
Với các xe thuộc nhóm **CANDIDATE**, hệ thống liên tục kiểm tra vector di chuyển của chúng.

Một vi phạm chỉ được xác lập (Confirmed) khi thoả mãn đồng thời:
1.  **Vị trí hiện tại:** Đã vượt qua vạch dừng ($Y_{curr} > Y_{line}$).
2.  **Thời gian ân hạn (Grace Period):** Thời gian hiện tại $t > t_{red\_start} + T_{grace}$.
    - $T_{grace}$ (thường là 0.3s - 0.5s) để bù trừ cho độ trễ giữa tín hiệu thực tế và xử lý camera, cũng như quán tính phanh.
3.  **Quỹ đạo (Trajectory):** Phân tích 5 toạ độ gần nhất toạ thành một vector có độ lớn và hướng đi xuống rõ ràng. Điều này loại bỏ các trường hợp xe rung lắc tại chỗ nhưng toạ độ nhảy qua vạch do nhiễu detection.

## 4. Sơ đồ Luồng xử lý (Flowchart)

```mermaid
graph TD
    A[Start Frame] --> B[Detect Objects & Tracks]
    B --> C[Update Traffic Light State]
    C -->|Green/Yellow| D[Reset Violation Cache]
    C -->|Red Logic| E{Is First Red Frame?}
    
    E -- Yes --> F[SNAPSHOT Logic]
    F --> F1[Check all Tracks]
    F1 -->|Track below Line| F2[Mark as SAFE]
    F1 -->|Track above Line| F3[Mark as CANDIDATE]
    
    E -- No --> G[Process CANDIDATES Only]
    G --> H{Vehicle Cross Line?}
    H -- No --> I[Continue Tracking]
    H -- Yes --> J{Grace Period Left?}
    J -- Yes --> K[Ignore (Late Crossing)]
    J -- No --> L{Moving Vector Valid?}
    L -- Yes --> M[CONFIRM VIOLATION]
    M --> N[Save Evidence]
```

## 5. Cải tiến so với phiên bản trước

| Đặc điểm | Logic cũ (Naive) | Logic mới (Snapshot Core) |
| :--- | :--- | :--- |
| **Tiêu chí bắt lỗi** | Xe xuất hiện trong vùng cấm khi đèn đỏ | Xe CHƯA qua vạch lúc đèn đỏ bật, SAU ĐÓ mới vượt qua |
| **Xử lý xe giữa đường** | Thường bị bắt lỗi oan (False Positive) | Bỏ qua hoàn toàn nhờ cơ chế Snapshot (Nhóm Safe) |
| **Độ nhạy đèn** | Dễ bị nhiễu nếu model detect nhấp nháy đỏ/vàng | Ổn định nhờ Voting Mechanism |
| **Xử lý rung lắc** | Có thể bắt lỗi xe đứng yên sát vạch | Loại bỏ nhờ phân tích vector quỹ đạo (Trajectory) |

## 6. Kết luận
Logic xử lý vi phạm mới đảm bảo sự công bằng và chính xác cao. Việc tách biệt rạch ròi giữa "xe đã vào giao lộ" và "xe chuẩn bị vượt" thông qua sự kiện `ON_RED_START` giải quyết triệt để vấn đề tranh cãi lớn nhất trong phạt nguội tự động.
