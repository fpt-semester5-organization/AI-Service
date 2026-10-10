import pytest

from app.services.rag.chunker import Chunker, split_units
from tests.fakes import FakeLLM, words


def make(size=20, overlap=4):
    llm = FakeLLM()
    return Chunker(llm.count_tokens, chunk_tokens=size, overlap_tokens=overlap), llm


def sentences(n, per=6):
    return " ".join(" ".join(f"w{i}x{j}" for j in range(per)) + "." for i in range(n))


async def test_short_source_is_single_chunk_with_title():
    c, llm = make(size=500)
    chunks = await c.chunk("Canh nấm", "Nấu nấm với đậu phụ.")
    assert len(chunks) == 1 and chunks[0].index == 0
    assert chunks[0].text.startswith("Canh nấm\n\n")
    assert llm.count_calls == 1


async def test_empty_text_has_no_chunks():
    c, _ = make()
    assert await c.chunk("T", "   ") == []


async def test_long_text_splits_with_overlap_and_title_in_each_chunk():
    c, _ = make(size=20, overlap=6)
    text = sentences(10)  # mỗi câu 6 từ
    chunks = await c.chunk("Tiêu đề", text)
    assert len(chunks) > 1
    assert [x.index for x in chunks] == list(range(len(chunks)))
    for ch in chunks:
        assert ch.text.startswith("Tiêu đề\n\n")
        assert ch.tokens <= 20
    # chồng lấn: câu cuối của đoạn trước xuất hiện ở đầu đoạn sau
    first_body = chunks[0].text.split("\n\n", 1)[1]
    second_body = chunks[1].text.split("\n\n", 1)[1]
    last_sentence = split_units(first_body)[-1]
    assert last_sentence in second_body


async def test_every_sentence_is_covered():
    c, _ = make(size=20, overlap=6)
    text = sentences(12)
    chunks = await c.chunk("T", text)
    joined = " ".join(ch.text for ch in chunks)
    for u in split_units(text):
        assert u in joined


async def test_overlong_sentence_is_split_by_words():
    c, _ = make(size=15, overlap=3)
    long = " ".join(f"t{i}" for i in range(80)) + "."
    chunks = await c.chunk("T", long)
    assert len(chunks) > 1
    assert all(ch.tokens <= 15 for ch in chunks)


async def test_progress_guaranteed_when_overlap_large():
    c, _ = make(size=12, overlap=11)
    chunks = await c.chunk("T", sentences(20, per=3))
    assert 1 < len(chunks) < 200


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        Chunker(FakeLLM().count_tokens, chunk_tokens=10, overlap_tokens=10)


def test_words_helper():
    assert words("Đạm thực vật") == ["đạm", "thực", "vật"]
