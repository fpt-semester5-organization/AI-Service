# knowledge/nutrition — tài liệu dinh dưỡng đã biên soạn

Mỗi tệp Markdown ở đây được nạp vào Qdrant với `sourceType = NUTRITION_DOC` qua `POST /rag/reindex/nutrition`. `README.md` và `_TEMPLATE.md` không được nạp.

## Quy tắc

- Tên tệp: `<id 2 chữ số>-<chu-de-khong-dau>.md`, ví dụ `01-dam-thuc-vat.md`.
- Front-matter bắt buộc:
  - `id`: số nguyên duy nhất, trùng số đầu tên tệp. Đây chính là `sourceId` trả về trong `sources` (D119).
  - `title`, `topic`, `sources` (ít nhất 1 nguồn), `updated` (YYYY-MM-DD).
- Dài 300–800 từ. Mọi số liệu phải có nguồn trong `sources`.
- Cuối tệp có lời miễn trừ y khoa.
- Duyệt: Nguyễn Văn Minh Tâm biên soạn, Nguyễn Thị Hồng Anh duyệt chéo.

## Danh mục dự kiến (khớp `tests/rag/golden.json`)

| id | Tệp | Chủ đề |
|---|---|---|
| 1 | 01-dam-thuc-vat.md | Đạm thực vật |
| 2 | 02-vitamin-b12.md | Vitamin B12 |
| 3 | 03-sat-thuc-vat.md | Sắt từ thực vật |
| 4 | 04-canxi-vitamin-d.md | Canxi và vitamin D |
| 5 | 05-omega-3.md | Omega-3 |
| 6 | 06-kem-iot.md | Kẽm và i-ốt |
| 7 | 07-tre-em-an-chay.md | Trẻ em ăn chay |
| 8 | 08-thai-ky-cho-con-bu.md | Thai kỳ và cho con bú |
| 9 | 09-nguoi-cao-tuoi.md | Người cao tuổi |
| 10 | 10-the-thao-an-chay.md | Thể thao và ăn chay |
| 11 | 11-quan-ly-can-nang.md | Quản lý cân nặng |
| 12 | 12-doc-nhan-thuc-pham-e-number.md | Đọc nhãn thực phẩm, E-number |
| 13 | 13-thay-the-thit-trung-sua.md | Thay thế thịt, trứng, sữa |
| 14 | 14-cac-truong-phai-an-chay.md | Các trường phái ăn chay |
| 15 | 15-di-ung-thuc-pham-thuc-vat.md | Dị ứng thực phẩm thực vật |
