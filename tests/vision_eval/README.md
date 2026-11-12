# Bộ ảnh mẫu chấm điểm nhận diện tủ lạnh (FN-VIS-01)

Bước 0 của `documents/KE_HOACH_CAI_TIEN_NHAN_DIEN_TU_LANH.md`: đo độ phủ, độ đúng tên và độ khít khung bằng số thay vì nhìn bằng mắt.

## Cấu trúc

```
tests/vision_eval/
├── images/                 ảnh gốc (không vẽ khung)
├── labels/<ảnh>.json       đáp án do người gán
├── BASELINE.md             kết quả các lần đo (thêm mục mới mỗi lần thay đổi prompt/cấu hình)
└── README.md
```

Định dạng đáp án:

```json
{
  "image": "tu_lanh_doi_hong_tuoi.png",
  "note": "mô tả ngắn",
  "items": [
    {"name": "Ớt chuông vàng", "aliases": ["Ớt chuông"], "box_2d": [25, 660, 115, 755], "freshness": "FRESH"}
  ]
}
```

- `box_2d` = `[ymin, xmin, ymax, xmax]`, chuẩn hóa 0..1000 (giống đầu ra Gemini).
- `aliases`: các tên khác vẫn được tính là đúng. Món khó xác định (ví dụ quả vàng cam có múi: cà chua beefsteak hay ớt chuông) thì ghi cả hai, kèm `note`.
- `freshness` chỉ ghi khi nhìn rõ (mốc, thâm, héo → `NON-FRESH`).
- Bỏ qua vật không phải nguyên liệu (nồi, chai nước lọc).

## Chạy

Gọi Gemini **thật** (tốn quota), nên không nằm trong `pytest`. Chạy từ thư mục `veggielife-ai-service`:

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_vision.py --runs 3
```

Tùy chọn: `--only <chuỗi trong tên ảnh>` để chạy một ảnh, `--out <file.json>` để lưu toàn bộ kết quả (gồm từng dự đoán, tên sai, món sót).

Unit test cho phần chấm điểm (không gọi Gemini): `tests/unit/test_eval_vision_scoring.py`.

## Cách chấm

| Chỉ số | Ý nghĩa |
|---|---|
| Recall | Món trong đáp án được khoanh trúng chỗ (IoU ≥ 0.3) / tổng món đáp án |
| Recall đúng tên | Khoanh trúng chỗ **và** đúng tên / tổng món đáp án |
| Precision tên | Khoanh trúng chỗ và đúng tên / số món dự đoán (không tính các khung "tách nhóm") |
| IoU TB | Độ khít trung bình của các khung đã ghép |
| Độ tươi | Tỉ lệ đúng FRESH / NON-FRESH trên các món đáp án có ghi độ tươi |

- Tên đúng khi mọi từ của tên đáp án (hoặc một alias) đều có trong tên dự đoán; không phân biệt hoa thường, phần trong ngoặc được coi là tên phụ.
- "Tách nhóm": mô hình khoanh riêng từng quả trong một chùm mà đáp án gán một khung → không tính lỗi, không tính điểm.
- Thang đo tỉ lệ: Tốt ≥ 90% · Khá 75–89% · Trung bình 60–74% · Yếu 40–59% · Kém < 40%.
- Thang IoU: Tốt ≥ 0.75 · Khá 0.6–0.74 · Trung bình 0.45–0.59 · Yếu 0.3–0.44 · Kém < 0.3.

## Lưu ý

- Script chấm **đầu ra của ai-service**, chưa qua bước khớp tên danh mục của backend. Lỗi đổi tên ở backend (Bước 1 của kế hoạch) không hiện ở đây; vì quy tắc "chứa đủ từ" nên tên quá cụ thể như "Hoa chuối thái mỏng" vẫn được tính là đúng loại "Chuối".
- Kết quả Gemini dao động giữa các lần (temperature 0.2): luôn so sánh trung bình ≥ 3 lần chạy.
- Đáp án được gán bằng mắt trên lưới tọa độ; sai số khung khoảng ±20 đơn vị, nên ngưỡng ghép IoU để ở 0.3.
- Nên bổ sung 3–5 ảnh chụp thật bằng điện thoại (ít đồ, nhiều đồ, thiếu sáng); hai ảnh hiện có đều là ảnh minh họa.
