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
