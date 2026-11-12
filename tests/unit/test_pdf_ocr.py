"""OCR PDF scan cho kho tri thức: tách lô trang, gửi PDF cho Gemini, ghép đúng thứ tự, cộng usage."""
import io

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import create_app
from tests.conftest import HEADERS
from tests.fakes import FakeLLM


def blank_pdf(pages: int) -> bytes:
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def post(c, data: bytes, name="sach.pdf"):
    return c.post("/rag/ocr-pdf", headers=HEADERS, files={"file": (name, data, "application/pdf")})


def test_ocr_pdf_splits_pages_into_batches_and_sums_usage(settings, store):
    llm = FakeLLM("Canh nấm đậu phụ\n\nNguyên liệu: 200 g nấm rơm")
    settings.ocr_pages_per_batch = 6
    with TestClient(create_app(settings, llm=llm, store=store)) as c:
        r = post(c, blank_pdf(14))
        body = r.json()
        assert r.status_code == 200
        assert body["pages"] == 14
        assert len(llm.generate_calls) == 3  # 6 + 6 + 2 trang
        assert body["text"].count("Canh nấm đậu phụ") == 3 and "200 g nấm rơm" in body["text"]
        assert body["usage"]["promptTokens"] == 30 and body["usage"]["completionTokens"] == 15
        part = llm.generate_calls[0][1][0].images[0]
        assert part.mime_type == "application/pdf" and part.data.startswith(b"%PDF")
        assert "NGUYÊN VĂN" in llm.generate_calls[0][0]
        assert llm.generate_kwargs[0]["temperature"] == 0.0


def test_ocr_pdf_rejects_non_pdf_and_too_many_pages(settings, store):
    settings.ocr_max_pages = 5
    with TestClient(create_app(settings, llm=FakeLLM("x"), store=store)) as c:
        assert post(c, b"not a pdf").json()["errorCode"] == "MEDIA_INVALID"
        r = post(c, blank_pdf(6))
        assert r.status_code == 400 and "vượt giới hạn" in r.json()["message"]
        assert c.post("/rag/ocr-pdf", headers=HEADERS).status_code == 400


def test_ocr_pdf_requires_internal_key(settings, store):
    with TestClient(create_app(settings, llm=FakeLLM("x"), store=store)) as c:
        assert c.post("/rag/ocr-pdf", files={"file": ("a.pdf", blank_pdf(1), "application/pdf")}).status_code == 401
