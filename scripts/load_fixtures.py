"""Nạp tệp mẫu JSON (tests/rag/fixtures) qua POST /rag/upsert; tùy chọn nạp lại tài liệu dinh dưỡng.

Chạy:  python -m scripts.load_fixtures --base-url http://localhost:8081 --key $INTERNAL_API_KEY [--nutrition]
"""
import argparse
import json
import sys
from pathlib import Path

import httpx

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "rag" / "fixtures"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--key", required=True)
    ap.add_argument("--nutrition", action="store_true", help="gọi thêm /rag/reindex/nutrition")
    a = ap.parse_args()
    headers = {"X-Internal-Api-Key": a.key}
    failed = chunks = tokens = n = 0
    with httpx.Client(base_url=a.base_url, headers=headers, timeout=60) as c:
        for f in sorted(FIXTURES.glob("*.json")):
            for item in json.loads(f.read_text(encoding="utf-8")):
                r = c.post("/rag/upsert", json=item)
                if r.status_code != 200:
                    failed += 1
                    print(f"LỖI {f.name} {item['sourceType']}:{item['sourceId']} -> {r.status_code} {r.text[:200]}")
                    continue
                n += 1
                chunks += r.json()["chunks"]
                tokens += r.json()["embeddingTokens"]
        print(f"Đã nạp {n} nguồn, {chunks} đoạn, {tokens} token embedding; lỗi {failed}")
        if a.nutrition:
            r = c.post("/rag/reindex/nutrition")
            print("reindex/nutrition:", r.status_code, r.text[:500])
        print("status:", c.get("/rag/status").text)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
