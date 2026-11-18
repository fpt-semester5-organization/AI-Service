# Kết quả đo nhận diện tủ lạnh

Mỗi lần thay đổi prompt/cấu hình, thêm một mục mới **ở trên cùng**, giữ nguyên các mục cũ để so sánh.
Chạy: `PYTHONPATH=. .venv/bin/python scripts/eval_vision.py --runs 3` (xem `README.md`).

---

## Bước 4 + gộp khung trùng — 13/11/2026

**Thay đổi chung (giữ lại):** `normalize` gộp các khung **cùng tên** chồng lấn mạnh (IoU ≥ 0.5 hoặc khung nhỏ nằm ≥ 80% trong khung lớn) — sửa lỗi thật "cùng cuộn phô mai dê khoanh 3 lần".
**Thí nghiệm Bước 4:** thêm cấu hình `VISION_THINKING_BUDGET` (token suy nghĩ) và `VISION_BOX_KEY` (`box2d` | `box_2d`); mỗi cấu hình 3 lần × 2 ảnh, timeout thật 55s.

| Cấu hình | thinking | Trường khung | Recall | Recall đúng tên | Precision tên | IoU TB | Giây TB / max | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| Baseline (Bước 0) | 0 | box2d | 58% | 53% | 71% | 0.63 | 16.4 / 23.7 | chưa gộp khung trùng |
| **A — giữ** | 0 | box2d | **62%** | **53%** | **73%** | **0.67** | **15.1 / 17.4** | có gộp khung trùng |
| B | 1024 | box2d | 65% | 57% | 71% | 0.57 ↓ | 20.7 / 23.9 | khung kém khít hơn rõ |
| C | 0 | box_2d | 46% | 41% | 79% | 0.68 | 19.1 / 25.7 | 2/6 lần chỉ trả 5 món |
| D | 1024 | box_2d | 53% | 48% | 59% | 0.49 ↓ | 23.3 / 27.4 | 1 lần recall 0% (khung hỏng) |

**Lỗi mục tiêu (ảnh 1, cộng 3 lần):**

| Cấu hình | Ớt chuông mini | Bắp cải tím | Phô mai (số mục / lần) |
|---|---|---|---|
| A | sai 4 (→ Gừng ×3, Ớt sừng), sót 1 | sai 3 (→ Ớt chuông đỏ/xanh, Hành tây tím) | 1, 2, 2 — không còn lặp 3 lần cùng cuộn |
| B | sai 1, sót 4 | sai 2, sót 1 | 1, 1, 1 |
| C | sót 6 | sai 1, sót 2 | 0, 2, 2 |
| D | sai 2, sót 4 | sai 1, sót 2 | 1, 5, 2 |

**Kết luận:**
1. **Giữ mặc định A** (`thinking=0`, `box2d`) + gộp khung trùng. Trên trung bình: recall +4 điểm, precision +2, IoU +0.04, nhanh hơn baseline.
2. **Bật thinking không đáng:** recall tăng nhẹ (+3) nhưng IoU giảm 0.10 và chậm hơn ~5s. **`box_2d` làm kết quả bất ổn** (có lần chỉ 5 món hoặc khung hỏng). Bước 4 trong kế hoạch: **không áp dụng**; cấu hình vẫn có để thử lại khi đổi mô hình.
3. **Ớt chuông mini và bắp cải tím (bọc nilon) vẫn sai/sót ở mọi cấu hình** — prompt và cấu hình đã chạm giới hạn của `gemini-2.5-flash` với ảnh nhiều món. Hướng tiếp theo cần cách khác: lượt 2 kiểm tra lại riêng các khung thuộc nhóm dễ nhầm (cắt phóng to rồi hỏi lại), hoặc thử mô hình mạnh hơn.
4. Đã kiểm lại đáp án: vật màu be cạnh ớt mini đúng là nấm, ớt mini nhìn rõ → lỗi là của mô hình, không phải của nhãn.

---

## Baseline — 13/11/2026 (Bước 0)

**Cấu hình:** code chưa commit sau đợt sửa 12–13/11 (`visualCues` trước `name`, quy tắc ớt chuông/cà chua/bắp cải tím, gộp trùng theo tên + hộp, quét từng kệ) · `gemini-2.5-flash` qua Vertex · `thinking_budget=0` · temperature 0.2 · `vision_max_output_tokens=8192` · `vision_timeout_seconds=55` · trường khung `box2d`.

**Phạm vi:** đầu ra ai-service, **chưa** qua bước khớp tên danh mục của backend.

### Từng lần chạy

| Ảnh | Lần | Giây | Token ra | Số món (dự đoán/đáp án) | Recall | Recall đúng tên | Precision tên | IoU TB | Độ tươi |
|---|---|---|---|---|---|---|---|---|---|
| tu_lanh_doi_hong_tuoi | 1 | 20.7 | 4777 | 33/39 | 72% | 72% | 100% | 0.79 | 88% |
| tu_lanh_doi_hong_tuoi | 2 | 23.7 | **8181** | **49**/39 | 77% | 67% | 65% | 0.67 | 88% |
| tu_lanh_doi_hong_tuoi | 3 | 14.5 | 4596 | 32/39 | 72% | 67% | 81% | 0.74 | 88% |
| tu_lanh_nhieu_ngan | 1 | 12.8 | 4476 | 30/46 | 33% | 30% | 52% | 0.57 | — |
| tu_lanh_nhieu_ngan | 2 | 12.6 | 4369 | 30/46 | 52% | 44% | 71% | 0.50 | — |
| tu_lanh_nhieu_ngan | 3 | 14.4 | 4517 | 31/46 | 44% | 39% | 58% | 0.52 | — |

### Tổng hợp theo thang đo

| Chỉ số | Ảnh 1 (tủ đôi) | Ảnh 2 (nhiều ngăn) | Chung |
|---|---|---|---|
| Recall | 74% · Trung bình | 43% · Yếu | 58% · **Yếu** |
| Recall đúng tên | 69% · Trung bình | 38% · Kém | 53% · **Yếu** |
| Precision tên | 82% · Khá | 60% · Trung bình | 71% · **Trung bình** |
| IoU TB | 0.73 · Khá | 0.53 · Trung bình | 0.63 · **Khá** |
| Độ tươi | 88% · Khá | — | 88% · **Khá** |
| Thời gian | 14.5–23.7s | 12.6–14.4s | TB 16.4s, max 23.7s (dưới timeout 55s) |

### Lỗi lặp lại (cộng dồn 3 lần chạy)

**Ảnh 1 — sai tên:** Bắp cải tím → Gừng / Ớt chuông xanh · Ớt chuông mini → Ớt sừng / Cà chua · Húng quế → Ớt chuông vàng (×2; khung húng quế nằm sát dưới quả ớt nên phần ghép có thể đã gán nhầm cặp; xem lại nhãn này).
**Ảnh 1 — sót:** Chanh (4), Cam (3), Rau thơm (3), Ngô (3), Nấm (3), Ớt chuông mini (2), Phô mai (2), Xà lách (2), Ớt chuông vàng (2), Xúc xích (2).

**Ảnh 2 — sai tên:** Chanh → Trứng gà (×2, chanh nằm cạnh trứng ở ngăn cửa) · Cần tây → Cà rốt / Đậu que · Dưa chuột → Khoai tây · Sữa → Dưa chuột · Chanh → Táo vàng.
**Ảnh 2 — sót:** Chanh (13), **Nước ép (11)**, Cà chua bi (6), Trứng (4), Bắp cải (4), **Sữa (4)**, Cam, Cà chua, Bơ, Bông cải xanh, Rau thơm, Thịt, Táo, Xà lách (mỗi món 3).
**Ảnh 2 — dương tính giả:** Dưa chuột (6), Chanh vàng (4), Quả bơ (3): đúng loại nhưng khung lệch nên IoU < 0.3 (bị tính cả sót lẫn dương tính giả).

### Nhận xét

1. **Ảnh đông đồ là điểm yếu chính:** ảnh 2 lần nào cũng dừng ở ~30 món trong khi đáp án có 46 (prompt giới hạn 30) → recall Yếu/Kém. Đồ uống (sữa, nước ép đóng chai) gần như luôn bị bỏ qua.
2. **Có nguy cơ cắt cụt JSON:** lần 2 ảnh 1 sinh 49 món, 8181/8192 token, sát giới hạn `vision_max_output_tokens`. Mô hình không tuân giới hạn 30 món; `normalize` chỉ chặn ở 50. Cần giới hạn cứng hơn hoặc tăng token/giảm trường thừa.
3. **Kết quả dao động:** cùng ảnh, recall dao động 33–52%, precision 52–71% → luôn so sánh trung bình ≥ 3 lần.
4. **Khung không lệch có hệ thống:** độ lệch tâm trung vị ≤ 8/1000 và tỉ lệ kích thước ≈ 1 ở cả hai ảnh, nên IoU thấp ở ảnh 2 là nhiễu từng món (đồ chồng nhau, ảnh 1200×676), không phải lỗi quy đổi tọa độ.
5. **Độ tin cậy vẫn kém phân biệt:** mỗi lần chạy chỉ có 2–5 giá trị khác nhau.
6. **Độ tươi tốt:** 88% đúng trên các món có ghi độ tươi (mốc/héo).
7. **Thời gian lần này 12–24s**, trong khi đợt đo 12/11 có lần 40–106s → độ trễ phụ thuộc tải Vertex; timeout 55s vẫn cần thiết.

### Mục tiêu cho các bước sau (so với baseline)

| Chỉ số | Baseline | Mục tiêu sau Bước 4–6 |
|---|---|---|
| Recall chung | 58% | ≥ 75% (Khá) |
| Recall đúng tên chung | 53% | ≥ 70% |
| Precision tên | 71% | ≥ 80% |
| IoU TB | 0.63 | ≥ 0.65, ảnh 2 ≥ 0.6 |
| Thời gian TB / max | 16.4s / 23.7s | ≤ 40s / ≤ 50s |
