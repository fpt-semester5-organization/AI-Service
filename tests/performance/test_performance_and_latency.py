import time
import pytest
from tests.conftest import HEADERS


def test_ai_response_latency_sla(client, llm):
    """
    Kiểm thử hiệu năng & SLA thời gian phản hồi:
    Endpoint /chat/answer phải trả kết quả trong thời gian < 400ms với mock LLM.
    """
    llm.reply = "Ăn chay đúng cách giúp hạ cholesterol máu hiệu quả."
    start_time = time.perf_counter()
    res = client.post("/chat/answer", headers=HEADERS, json={"question": "Lợi ích ăn chay?"})
    duration = time.perf_counter() - start_time

    assert res.status_code == 200
    assert duration < 0.4, f"Thời gian phản hồi {duration:.3f}s vượt quá SLA 0.4s"


def test_high_frequency_sequential_requests(client, llm):
    """
    Kiểm thử độ bền bỉ & tải nhẹ:
    Gửi 15 request liên tiếp không bị crash hay tràn bộ nhớ.
    """
    llm.reply = "Phản hồi nhanh và ổn định."
    total_duration = 0.0

    for i in range(15):
        t0 = time.perf_counter()
        res = client.post("/chat/answer", headers=HEADERS, json={"question": f"Câu hỏi kiểm tra tần suất số {i}"})
        t1 = time.perf_counter()
        assert res.status_code == 200
        total_duration += (t1 - t0)

    avg_latency = total_duration / 15
    assert avg_latency < 0.2, f"Độ trễ trung bình {avg_latency:.3f}s quá cao"
