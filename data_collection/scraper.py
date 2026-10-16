import csv
import time
from datetime import datetime

import requests


def openalex(query, api_key, n=500):
    """Lấy tối đa n bản ghi OpenAlex, phân trang bằng cursor."""
    url, cursor, out, retry = "https://api.openalex.org/works", "*", [], 0
    while cursor and len(out) < n:
        r = requests.get(url, timeout=20, params={
            "search": query, "per-page": 200, "cursor": cursor, "api_key": api_key})
        if r.status_code == 429:          # quá tốc độ: chờ rồi thử lại, có giới hạn
            retry += 1
            if retry > 5:
                raise RuntimeError("429 quá 5 lần, dừng và ghi vào search-log")
            time.sleep(10)
            continue
        r.raise_for_status()              # lỗi khác: dừng, không bỏ qua im lặng
        data = r.json()
        out += data["results"]
        cursor = data["meta"].get("next_cursor")
    return out[:n]


def abstract(inv):
    """OpenAlex trả abstract dạng {từ: [vị trí]}; dựng lại thành câu."""
    if not inv:
        return ""
    pos = {i: w for w, idx in inv.items() for i in idx}
    return " ".join(pos[i] for i in sorted(pos))


def row(w, query):                        # chuẩn hoá về cột của 01_all_records.csv
    doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
    src = (w.get("primary_location") or {}).get("source") or {}
    return {
        "source": "OpenAlex",
        "title": w.get("title") or "",
        "authors": "; ".join(a["author"]["display_name"] for a in w.get("authorships", [])),
        "year": w.get("publication_year"),
        "venue": src.get("display_name", ""),
        "doi": doi,
        "abstract": abstract(w.get("abstract_inverted_index")),
        "query": query,
    }


def export_csv(records, query, filename="01_all_records.csv"):
    if not records:
        return
    date_str = datetime.now().strftime("%Y%m%d")
    safe_query = "".join(c if c.isalnum() else "_" for c in query)[:20]
    final_filename = f"{date_str}_{safe_query}_{filename}"

    # UTF-8 with BOM (utf-8-sig) cho Excel
    with open(final_filename, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Đã xuất file: {final_filename}")
