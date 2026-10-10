from app.services.rag.types import SOURCE_ID_RE, SOURCE_TYPES, point_id


def test_point_id_is_deterministic_and_distinct():
    a = point_id("RECIPE", "512", 0)
    assert a == point_id("RECIPE", "512", 0)
    assert a != point_id("RECIPE", "512", 1)
    assert a != point_id("POST", "512", 0)


def test_known_value_is_stable():
    # khóa cứng giá trị để phát hiện đổi namespace/định dạng khóa (id điểm phải ổn định giữa các lần chạy)
    assert point_id("NUTRITION_DOC", "1", 0) == "2b1a02b9-3f34-5199-8bd5-acd00e8b7c39"


def test_eight_source_types():
    assert len(SOURCE_TYPES) == 8 and "CHAT" not in SOURCE_TYPES


def test_source_id_is_digit_string():
    assert SOURCE_ID_RE.match("512") and not SOURCE_ID_RE.match("01-dam.md") and not SOURCE_ID_RE.match("")
