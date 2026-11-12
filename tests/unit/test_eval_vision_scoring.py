"""Kiểm phần chấm điểm của scripts/eval_vision.py (thuần, không gọi Gemini) và tính hợp lệ của bộ đáp án."""
import json
from pathlib import Path

import pytest

from scripts.eval_vision import EVAL_DIR, grade, iou, load_cases, name_ok, score_image

LAB = lambda name, box, *aliases, **kw: {"name": name, "aliases": list(aliases), "box_2d": box, **kw}  # noqa: E731


@pytest.mark.parametrize("pred,label,ok", [
    ("Ớt chuông vàng", LAB("Ớt chuông vàng", [0, 0, 1, 1], "Ớt chuông"), True),
    ("Ớt chuông", LAB("Ớt chuông vàng", [0, 0, 1, 1], "Ớt chuông"), True),
    ("Cà chua", LAB("Ớt chuông vàng", [0, 0, 1, 1], "Ớt chuông"), False),
    ("Bông cải xanh (Broccoli)", LAB("Bông cải xanh", [0, 0, 1, 1]), True),
    ("Bí đỏ (Bí ngô)", LAB("Bí ngô", [0, 0, 1, 1]), True),
    ("Hoa chuối thái mỏng", LAB("Chuối", [0, 0, 1, 1]), True),   # chứa trọn "chuối" → tên quá cụ thể vẫn tính đúng loại
    ("Nước ép cam", LAB("Cam", [0, 0, 1, 1]), True),
    ("Thịt bò", LAB("Thịt", [0, 0, 1, 1]), True),
    ("Cà chua", LAB("Cà chua", [0, 0, 1, 1]), True),     # Unicode NFD
    ("Ớt hiểm tươi", LAB("Ớt chuông mini", [0, 0, 1, 1], "Ớt chuông", "Ớt ngọt"), False),
])
def test_name_ok(pred, label, ok):
    assert name_ok(pred, label) is ok


def test_iou_basic():
    assert iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
    assert iou([0, 0, 100, 100], [200, 200, 300, 300]) == 0.0
    assert iou([0, 0, 100, 100], [0, 50, 100, 150]) == pytest.approx(1 / 3)


def test_score_image_match_wrong_name_missed_grouped_and_false_positive():
    labels = [
        LAB("Ớt chuông vàng", [0, 0, 100, 100], "Ớt chuông", freshness="FRESH"),
        LAB("Cà chua", [0, 200, 100, 400]),
        LAB("Bắp cải tím", [500, 500, 600, 600]),
    ]
    preds = [
        {"name": "Cà chua", "box2d": [0, 0, 100, 100], "freshnessStatus": "FRESH"},      # đúng chỗ, sai tên
        {"name": "Cà chua", "box2d": [0, 200, 100, 300]},                                   # ghép (IoU 0.5)
        {"name": "Cà chua", "box2d": [0, 300, 100, 400]},                                   # quả khác trong cùng chùm → tách nhóm
        {"name": "Dưa hấu", "box2d": [800, 800, 900, 900]},                                 # dương tính giả
        {"name": "Nấm", "box2d": None},                                                     # không có khung
    ]
    s = score_image(preds, labels)
    assert (s.matched, s.named_ok, s.grouped) == (2, 1, 1)
    assert s.wrong_names == ["Ớt chuông vàng → Cà chua"]
    assert s.missed == ["Bắp cải tím"]
    assert s.false_pos == ["Dưa hấu", "Nấm"]
    assert s.recall == pytest.approx(2 / 3)
    assert s.recall_named == pytest.approx(1 / 3)
    assert s.precision == pytest.approx(1 / 4)      # 1 đúng / (5 dự đoán − 1 tách nhóm)
    assert s.fresh_acc == 1.0


def test_grade_scale():
    assert [grade(x) for x in (0.95, 0.8, 0.65, 0.5, 0.2)] == ["Tốt", "Khá", "Trung bình", "Yếu", "Kém"]


def test_label_files_are_valid():
    cases = load_cases(None)
    assert len(cases) >= 2
    for img, data in cases:
        assert Path(img).is_file(), img
        assert data["items"]
        for it in data["items"]:
            y0, x0, y1, x1 = it["box_2d"]
            assert 0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000, it
            assert it["name"].strip() and isinstance(it.get("aliases", []), list)
            assert it.get("freshness") in (None, "FRESH", "NON-FRESH")
    assert json.loads((EVAL_DIR / "labels" / "tu_lanh_doi_hong_tuoi.json").read_text(encoding="utf-8"))["items"]
