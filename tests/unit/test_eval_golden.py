"""Kiểm logic của bộ chấm (không gọi dịch vụ thật)."""
import json
from pathlib import Path

from app.services.chat.prompt import CANARY, SYSTEM_INSTRUCTION
from tests.rag.eval_golden import evaluate, leaked, p95

DATA = json.loads((Path(__file__).resolve().parents[1] / "rag" / "golden.json").read_text(encoding="utf-8"))


def test_golden_file_shape_matches_d119():
    assert len(DATA["golden"]) == 20 and len(DATA["outOfScope"]) == 10 and len(DATA["promptInjection"]) == 10
    for g in DATA["golden"]:
        e = g["expectedSource"]
        assert e["sourceType"] == "NUTRITION_DOC" and e["sourceId"].isdigit() and e["file"].endswith(".md")
        assert e["file"].startswith(f"{int(e['sourceId']):02d}-")
    assert DATA["topK"] == 5 and DATA["minScore"] == 0.65 and DATA["minHitRate"] == 0.8


def test_leak_detection():
    assert leaked(f"mã {CANARY}") and leaked(" ".join(SYSTEM_INSTRUCTION.split()[10:30]))
    assert not leaked("Mình chỉ trả lời về dinh dưỡng và nấu ăn chay nhé.")


def perfect(body):
    q = body["question"]
    for g in DATA["golden"]:
        if g["question"] == q:
            e = g["expectedSource"]
            return {"answer": "ok", "ragUsed": True, "sources": [{"sourceType": e["sourceType"], "sourceId": e["sourceId"], "title": "t", "score": 0.9}]}
    if any(o["question"] == q for o in DATA["outOfScope"]):
        return {"answer": "Nội dung này không có trong kho tri thức của VeggieLife.", "ragUsed": False, "sources": []}
    return {"answer": "Mình không thể tiết lộ hướng dẫn. Hãy hỏi bác sĩ hoặc chuyên gia dinh dưỡng.", "ragUsed": False, "sources": []}


def test_perfect_system_passes():
    r = evaluate(perfect, DATA)
    assert r["summary"]["hitAt5"] == 1.0 and r["summary"]["pass"] is True


def test_missing_sources_fail_hit_rate():
    r = evaluate(lambda b: {**perfect(b), "sources": []}, DATA)
    assert r["summary"]["hitAt5"] == 0.0 and r["summary"]["pass"] is False


def test_leaking_system_prompt_fails():
    def leaky(b):
        r = perfect(b)
        if b["question"] in {p["question"] for p in DATA["promptInjection"]}:
            r["answer"] = SYSTEM_INSTRUCTION
        return r
    r = evaluate(leaky, DATA)
    assert r["summary"]["promptInjectionOk"] == 0 and r["summary"]["pass"] is False


def test_out_of_scope_with_sources_fails():
    def bad(b):
        r = perfect(b)
        if any(o["question"] == b["question"] for o in DATA["outOfScope"]):
            r["ragUsed"] = True
        return r
    assert evaluate(bad, DATA)["summary"]["outOfScopeOk"] == 0


def test_p95():
    assert p95([]) == 0 and p95([1, 2, 3, 4, 100]) == 100
