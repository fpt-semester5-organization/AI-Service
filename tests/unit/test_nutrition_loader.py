from pathlib import Path

from app.services.rag.chunker import Chunker
from app.services.rag.embedder import Embedder
from app.services.rag.indexer import Indexer
from app.services.rag.nutrition_loader import load_directory, reindex_nutrition
from tests.fakes import FakeLLM, InMemoryStore

GOOD = """---
id: {id}
title: "{title}"
topic: "{topic}"
sources:
  - name: "WHO 2023"
    url: "https://www.who.int"
updated: "2026-10-10"
---

# {title}

Nội dung về {topic}. Đậu phụ và đậu lăng là nguồn đạm thực vật tốt.
"""


def write(d: Path, name: str, **kw):
    kw.setdefault("id", int(name.split("-")[0]))
    kw.setdefault("title", "Đạm thực vật")
    kw.setdefault("topic", "dam")
    (d / name).write_text(GOOD.format(**kw), encoding="utf-8")


def test_loads_valid_and_skips_readme_and_template(tmp_path):
    write(tmp_path, "01-dam-thuc-vat.md")
    (tmp_path / "README.md").write_text("# readme", encoding="utf-8")
    (tmp_path / "_TEMPLATE.md").write_text("---\nid: 0\n---\nx", encoding="utf-8")
    r = load_directory(tmp_path)
    assert [d.id for d in r.docs] == [1] and not r.failed


def test_reports_each_bad_file_with_reason(tmp_path):
    write(tmp_path, "01-a.md")
    write(tmp_path, "02-b.md", id=9)  # id không trùng số tên tệp
    (tmp_path / "03-c.md").write_text("---\nid: 3\ntitle: C\nsources: []\n---\nnội dung", encoding="utf-8")
    (tmp_path / "x-d.md").write_text("---\nid: 4\n---\nnội dung", encoding="utf-8")
    r = load_directory(tmp_path)
    reasons = {f["file"]: f["reason"] for f in r.failed}
    assert "không trùng" in reasons["02-b.md"]
    assert "sources" in reasons["03-c.md"]
    assert "tên tệp" in reasons["x-d.md"]
    assert [d.id for d in r.docs] == [1]


def test_duplicate_id_is_rejected(tmp_path):
    write(tmp_path, "01-a.md")
    write(tmp_path, "01-b.md")
    r = load_directory(tmp_path)
    assert len(r.docs) == 1 and "trùng" in r.failed[0]["reason"]


async def test_reindex_is_idempotent_and_removes_deleted_files(tmp_path):
    llm, store = FakeLLM(), InMemoryStore()
    ix = Indexer(Chunker(llm.count_tokens), Embedder(llm), store)
    write(tmp_path, "01-a.md")
    write(tmp_path, "02-b.md")
    r1 = await reindex_nutrition(tmp_path, ix, store)
    n1 = (await store.stats())[0]
    r2 = await reindex_nutrition(tmp_path, ix, store)
    assert (r1.files, r2.files) == (2, 2) and (await store.stats())[0] == n1
    (tmp_path / "02-b.md").unlink()
    r3 = await reindex_nutrition(tmp_path, ix, store)
    assert r3.files == 1 and await store.list_source_ids("NUTRITION_DOC") == {"1"}


async def test_stale_not_deleted_when_some_file_failed(tmp_path):
    llm, store = FakeLLM(), InMemoryStore()
    ix = Indexer(Chunker(llm.count_tokens), Embedder(llm), store)
    write(tmp_path, "01-a.md")
    write(tmp_path, "02-b.md")
    await reindex_nutrition(tmp_path, ix, store)
    (tmp_path / "02-b.md").write_text("hỏng", encoding="utf-8")  # không có front-matter
    r = await reindex_nutrition(tmp_path, ix, store)
    assert r.failed and await store.list_source_ids("NUTRITION_DOC") == {"1", "2"}
