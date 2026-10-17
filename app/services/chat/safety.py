"""Lọc an toàn câu hỏi (FN-CHAT-02): thô tục, tự hại, hướng dẫn nguy hiểm về y tế.

Danh sách điểm khởi đầu, chỉnh trong quá trình chạy golden set. So khớp không dấu, theo ranh giới từ.
Lưu ý: câu chèn lệnh (prompt injection) KHÔNG bị chặn ở đây — System Instruction xử lý, vì chặn nhầm sẽ làm hỏng
câu hỏi hợp lệ; câu bị chặn vẫn tính lượt (D119).
"""
import re
import unicodedata


def normalize(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()


_PROFANITY = ["dit me", "dit con", "du ma", "dm may", "con cac", "vl may", "thang cho", "con cho", "dcm", "dmm", "fuck", "shit"]
_SELF_HARM = ["tu tu", "tu sat", "tu huy hoai", "muon chet", "ket thuc cuoc song", "cat tay", "suicide"]
_DANGEROUS = [
    "nhin an \\d+ ngay", "nhin an het", "bo thuoc", "ngung thuoc", "thay the thuoc", "thay thuoc bang",
    "tu ngung insulin", "khong can di kham", "lieu thuoc", "uong bao nhieu thuoc",
]


def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(patterns) + r")(?![a-z0-9])")


_RULES = {
    "PROFANITY": _compile([re.escape(p) for p in _PROFANITY]),
    "SELF_HARM": _compile([re.escape(p) for p in _SELF_HARM]),
    "DANGEROUS_MEDICAL": _compile(_DANGEROUS),
}


def check_question(question: str) -> str | None:
    """Trả mã nhóm vi phạm, hoặc None nếu an toàn."""
    norm = normalize(question)
    for name, rx in _RULES.items():
        if rx.search(norm):
            return name
    return None


_GREETING_TOKENS = {
    "hi", "hello", "hey", "alo", "chao", "xin chao", "chao ban", "chao bot",
    "chao em", "chao anh", "chao chi", "chao tro ly", "good morning",
    "good afternoon", "good evening", "hi ban", "hello ban", "bonjour",
    "chao ad", "admin", "chao admin", "hai", "helo",
}


def is_greeting(question: str) -> bool:
    """Kiểm tra câu hỏi có phải thuần túy lời chào hỏi/mở đầu hội thoại hay không."""
    norm = normalize(question).strip()
    norm = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", norm).strip()
    if norm in _GREETING_TOKENS:
        return True
    words = norm.split()
    if len(words) <= 3 and words:
        if words[0] in {"hi", "hello", "hey", "chao", "alo", "xin"} or norm.startswith(("xin chao", "chao ")):
            if not any(w in words for w in ("an", "chay", "dam", "calo", "mon", "nau", "thit", "ca", "trung", "sua")):
                return True
    return False


