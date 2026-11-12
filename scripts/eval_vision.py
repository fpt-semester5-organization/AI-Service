"""Chấm điểm nhận diện nguyên liệu tủ lạnh (FN-VIS-01) trên bộ ảnh mẫu có đáp án.

Gọi Gemini THẬT (tốn quota) — không chạy trong pytest. Cách dùng (từ thư mục veggielife-ai-service):

    PYTHONPATH=. .venv/bin/python scripts/eval_vision.py --runs 3
    PYTHONPATH=. .venv/bin/python scripts/eval_vision.py --runs 1 --only tu_lanh_nhieu_ngan --out /tmp/kq.json

Đáp án: tests/vision_eval/labels/<tên ảnh>.json — mỗi món {name, aliases, box_2d [ymin,xmin,ymax,xmax] 0..1000, freshness?}.

Cách chấm (mỗi ảnh, mỗi lần chạy):
- Ghép vị trí: cặp (dự đoán, đáp án) có IoU ≥ 0.3, ghép tham lam theo IoU giảm dần, mỗi bên dùng một lần.
- Tên đúng: mọi từ của `name` hoặc một `alias` đều có trong tên dự đoán (không phân biệt hoa thường, bỏ phần ngoặc).
- Dự đoán không ghép được nhưng nằm ≥ 50% trong một khung đáp án cùng tên → "tách nhóm" (không tính lỗi, không tính điểm).
- Recall = số món đáp án được ghép / tổng món đáp án; Recall đúng tên = ghép + đúng tên / tổng món đáp án.
- Precision tên = ghép + đúng tên / (số dự đoán − số tách nhóm).
- IoU TB = trung bình IoU của các cặp đã ghép.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "tests" / "vision_eval"
IOU_MATCH = 0.3
IOA_GROUP = 0.5


# ---------------------------------------------------------------- chấm điểm (thuần, có unit test)

def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().strip()


def name_variants(name: str) -> list[set[str]]:
    """"Bí đỏ (Bí ngô)" → [{bí, đỏ}, {bí, ngô}]; tách alias trong ngoặc theo '/' hoặc ','."""
    n = _norm(name)
    parts = [re.sub(r"\(.*?\)", " ", n)]
    for inner in re.findall(r"\((.*?)\)", n):
        parts += re.split(r"[/,]", inner)
    out = []
    for p in parts:
        toks = set(re.findall(r"\w+", p))
        if toks:
            out.append(toks)
    return out


def name_ok(pred_name: str, label: dict) -> bool:
    accepted = [label["name"], *label.get("aliases", [])]
    pred_vars = name_variants(pred_name)
    for acc in accepted:
        for acc_toks in name_variants(acc):
            if any(acc_toks <= pv for pv in pred_vars):
                return True
    return False


def _area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a, b) -> float:
    y0, x0, y1, x1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return max(0, y1 - y0) * max(0, x1 - x0)


def iou(a, b) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def ioa(pred, label) -> float:
    """Tỉ lệ diện tích khung dự đoán nằm trong khung đáp án."""
    a = _area(pred)
    return _inter(pred, label) / a if a > 0 else 0.0


@dataclass
class ImageScore:
    n_labels: int
    n_preds: int
    matched: int = 0
    named_ok: int = 0
    grouped: int = 0
    ious: list[float] = field(default_factory=list)
    fresh_total: int = 0
    fresh_ok: int = 0
    wrong_names: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    false_pos: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.matched / self.n_labels if self.n_labels else 0.0

    @property
    def recall_named(self) -> float:
        return self.named_ok / self.n_labels if self.n_labels else 0.0

    @property
    def precision(self) -> float:
        d = self.n_preds - self.grouped
        return self.named_ok / d if d > 0 else 0.0

    @property
    def mean_iou(self) -> float:
        return statistics.fmean(self.ious) if self.ious else 0.0

    @property
    def fresh_acc(self) -> float | None:
        return self.fresh_ok / self.fresh_total if self.fresh_total else None


def score_image(preds: list[dict], labels: list[dict]) -> ImageScore:
    """preds: [{name, box2d, freshnessStatus?}] (dự đoán không có khung bị coi là dương tính giả)."""
    s = ImageScore(n_labels=len(labels), n_preds=len(preds))
    pairs = sorted(
        ((iou(p["box2d"], lab["box_2d"]), pi, li)
         for pi, p in enumerate(preds) if p.get("box2d")
         for li, lab in enumerate(labels)),
        reverse=True)
    used_p: set[int] = set()
    used_l: set[int] = set()
    for v, pi, li in pairs:
        if v < IOU_MATCH:
            break
        if pi in used_p or li in used_l:
            continue
        used_p.add(pi)
        used_l.add(li)
        p, lab = preds[pi], labels[li]
        s.matched += 1
        s.ious.append(v)
        if name_ok(p["name"], lab):
            s.named_ok += 1
        else:
            s.wrong_names.append(f"{lab['name']} → {p['name']}")
        if lab.get("freshness"):
            s.fresh_total += 1
            s.fresh_ok += int(p.get("freshnessStatus") == lab["freshness"])
    for li, lab in enumerate(labels):
        if li not in used_l:
            s.missed.append(lab["name"])
    for pi, p in enumerate(preds):
        if pi in used_p:
            continue
        box = p.get("box2d")
        if box and any(ioa(box, lab["box_2d"]) >= IOA_GROUP and name_ok(p["name"], lab) for lab in labels):
            s.grouped += 1
        else:
            s.false_pos.append(p["name"])
    return s


# ---------------------------------------------------------------- thang đo

def grade(x: float) -> str:
    return ("Tốt" if x >= 0.9 else "Khá" if x >= 0.75 else "Trung bình" if x >= 0.6
            else "Yếu" if x >= 0.4 else "Kém")


def grade_iou(x: float) -> str:
    """IoU khắt khe hơn tỉ lệ: ≥0.75 đã là khung rất khít."""
    return ("Tốt" if x >= 0.75 else "Khá" if x >= 0.6 else "Trung bình" if x >= 0.45
            else "Yếu" if x >= 0.3 else "Kém")


# ---------------------------------------------------------------- chạy thật

def load_cases(only: str | None) -> list[tuple[Path, dict]]:
    cases = []
    for lf in sorted((EVAL_DIR / "labels").glob("*.json")):
        if only and only not in lf.stem:
            continue
        data = json.loads(lf.read_text(encoding="utf-8"))
        cases.append((EVAL_DIR / "images" / data["image"], data))
    return cases


async def run(runs: int, only: str | None) -> dict:
    sys.path.insert(0, str(ROOT))
    from app.core.config import Settings
    from app.core.gemini_client import GeminiClient
    from app.services.vision.ingredient_detector import IngredientDetector

    settings = Settings()
    det = IngredientDetector(settings, GeminiClient(settings))
    report: dict = {"model": settings.gemini_model, "timeout_s": settings.vision_timeout_seconds, "images": {}}
    for img_path, data in load_cases(only):
        rows = []
        for r in range(runs):
            t0 = time.perf_counter()
            try:
                res = await det.detect(img_path.read_bytes())
            except Exception as exc:  # noqa: BLE001 — ghi lỗi, chạy tiếp
                rows.append({"run": r + 1, "error": f"{type(exc).__name__}: {exc}",
                             "seconds": round(time.perf_counter() - t0, 1)})
                print(f"[{img_path.stem}] lần {r + 1}: LỖI {exc}", flush=True)
                continue
            secs = round(time.perf_counter() - t0, 1)
            preds = [{"name": i.name, "box2d": i.box2d, "freshnessStatus": i.freshness_status,
                      "confidence": i.confidence} for i in res.items]
            s = score_image(preds, data["items"])
            row = {"run": r + 1, "seconds": secs, "completion_tokens": res.usage.completion_tokens,
                   "model": res.usage.model, "n_preds": s.n_preds, "n_labels": s.n_labels,
                   "recall": round(s.recall, 3), "recall_named": round(s.recall_named, 3),
                   "precision": round(s.precision, 3), "mean_iou": round(s.mean_iou, 3),
                   "fresh_acc": None if s.fresh_acc is None else round(s.fresh_acc, 3),
                   "grouped": s.grouped, "wrong_names": s.wrong_names, "missed": s.missed,
                   "false_pos": s.false_pos,
                   "distinct_confidences": len({p["confidence"] for p in preds}),
                   "preds": preds}
            rows.append(row)
            print(f"[{img_path.stem}] lần {r + 1}: {secs}s, {s.n_preds} món, recall {s.recall:.0%}, "
                  f"đúng tên {s.recall_named:.0%}, precision {s.precision:.0%}, IoU {s.mean_iou:.2f}", flush=True)
        report["images"][img_path.stem] = rows
    return report


def summarize(report: dict) -> str:
    lines = [f"Mô hình: `{report['model']}` · timeout Vision {report['timeout_s']}s", "",
             "| Ảnh | Lần | Giây | Token ra | Số món | Recall | Recall đúng tên | Precision tên | IoU TB | Độ tươi |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    agg: dict[str, list[float]] = {"recall": [], "recall_named": [], "precision": [], "mean_iou": [], "seconds": []}
    for img, rows in report["images"].items():
        for r in rows:
            if "error" in r:
                lines.append(f"| {img} | {r['run']} | {r['seconds']} | — | — | LỖI: {r['error'][:60]} | | | | |")
                continue
            fa = "—" if r["fresh_acc"] is None else f"{r['fresh_acc']:.0%}"
            lines.append(f"| {img} | {r['run']} | {r['seconds']} | {r['completion_tokens']} | {r['n_preds']}/{r['n_labels']} "
                         f"| {r['recall']:.0%} | {r['recall_named']:.0%} | {r['precision']:.0%} | {r['mean_iou']:.2f} | {fa} |")
            for k in agg:
                agg[k].append(r[k])
    if agg["recall"]:
        m = {k: statistics.fmean(v) for k, v in agg.items()}
        lines += ["", "**Trung bình tất cả lần chạy:** "
                  f"Recall {m['recall']:.0%} ({grade(m['recall'])}) · "
                  f"Recall đúng tên {m['recall_named']:.0%} ({grade(m['recall_named'])}) · "
                  f"Precision tên {m['precision']:.0%} ({grade(m['precision'])}) · "
                  f"IoU {m['mean_iou']:.2f} ({grade_iou(m['mean_iou'])}) · "
                  f"thời gian {m['seconds']:.1f}s (max {max(agg['seconds']):.1f}s)"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", help="chỉ chạy ảnh có tên chứa chuỗi này")
    ap.add_argument("--out", help="ghi toàn bộ kết quả (JSON) ra file")
    args = ap.parse_args()
    report = asyncio.run(run(args.runs, args.only))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(summarize(report))


if __name__ == "__main__":
    main()
