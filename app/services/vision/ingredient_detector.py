"""Nhận diện nguyên liệu từ ảnh tủ lạnh (FN-VIS-01, BR-AI-CV-01, BR-USR-VIS-FRESH-01): kiểm ảnh → Gemini Vision → chuẩn hóa."""
import logging
import re
from typing import Any

from app.core.config import Settings
from app.core.errors import file_too_large, gemini_error, image_invalid
from app.core.gemini_client import ImagePart, LLMClient, Message, parse_json_object
from app.core.images import ImageInfo, sniff_image
from app.schemas.chat import Audit, Usage
from app.schemas.vision import DetectedIngredient, DetectIngredientsResponse

log = logging.getLogger("veggielife.vision")

FRESHNESS = ("FRESH", "NON-FRESH")
MAX_ITEMS = 50

SYSTEM_INSTRUCTION = """Bạn là chuyên gia thị giác máy tính nhận diện tủ lạnh và nguyên liệu thực phẩm của VeggieLife.
NHIỆM VỤ:
1. XÁC THỰC ẢNH TỦ LẠNH VÀ THỰC PHẨM:
   - Ảnh PHẢI là ảnh chụp thực tế tủ lạnh (mở cửa, bên trong ngăn mát/đông), rổ/kệ bảo quản thực phẩm hoặc bàn bếp chứa các nguyên liệu thực phẩm cụ thể.
   - NẾU ảnh là người, chân dung, phong cảnh, xe cộ, thú cưng, phòng khách/phòng ngủ, đồ gia dụng không phải thực phẩm, giấy tờ/văn bản, ảnh vẽ đồ họa/anime, ảnh chụp màn hình ứng dụng, HOẶC ảnh tủ lạnh trống trơn không có bất kỳ nguyên liệu thực phẩm nào:
     -> BẮT BUỘC trả về isValidFridge = false
     -> rejectionReason = "Hình ảnh không phải là tủ lạnh có chứa nguyên liệu thực phẩm. Vui lòng chụp hoặc chọn lại ảnh khác."
     -> items = []
   - NẾU ảnh HỢP LỆ (ảnh tủ lạnh hoặc thực phẩm có nguyên liệu):
     -> BẮT BUỘC trả về isValidFridge = true
     -> rejectionReason = null
     -> Nhận diện danh sách nguyên liệu vào items.

2. QUY TẮC NHẬN DIỆN NGUYÊN LIỆU (khi isValidFridge = true):
   - name: tên nguyên liệu bằng tiếng Việt ngắn gọn, dạng chung chuẩn phổ thông (ví dụ: "Bí đỏ", "Thịt heo", "Nấm rơm", "Cà chua", "Đậu phụ", "Rau muống").
   - quantity: ước tính định lượng chi tiết nhìn thấy trong ảnh bằng tiếng Việt (ví dụ: "nửa trái", "3 quả", "3 ký", "500g", "1 bó", "2 củ", "2 miếng", "300ml").
   - amount: giá trị số học tương ứng của định lượng (ví dụ: "nửa trái" -> 0.5, "3 ký" -> 3.0, "500g" -> 500.0, "3 quả" -> 3.0, "1 bó" -> 1.0, "2 củ" -> 2.0).
   - unit: đơn vị đo lường tương ứng (ví dụ: "trái", "quả", "kg", "g", "bó", "củ", "miếng", "hộp", "ml").
   - confidence: số thực từ 0.0 đến 1.0, mức độ chắc chắn.
   - freshnessStatus: "FRESH" (còn tươi ngon) hoặc "NON-FRESH" (héo, thâm, dập, mốc, cần dùng sớm).
   - foodGroup: nhóm thực phẩm ("Rau củ", "Trái cây", "Nấm", "Đậu hạt", "Gia vị", "Thịt đỏ", "Gia cầm", "Thủy hải sản", "Trứng", "Sữa & Chế phẩm", "Khác").
   - isAnimalProduct: true nếu là thịt, cá, hải sản, trứng, sữa..., ngược lại false.
   - isRedMeat: true nếu là thịt đỏ (thịt bò, thịt heo/lợn, thịt cừu, thịt dê...), ngược lại false.
   - box2d: mảng 4 số nguyên [ymin, xmin, ymax, xmax] từ 0 đến 1000 định vị hộp bao quanh nguyên liệu. Bắt buộc có nếu nhìn rõ.
   - Giới hạn: Nhận diện tối đa 30 nguyên liệu thực phẩm chính nhìn rõ nhất. Không bịa đặt nguyên liệu bị che khuất.

ĐỊNH DẠNG ĐẦU RA: JSON đúng schema đã định nghĩa."""

USER_TEXT = "Hãy kiểm tra xem ảnh có phải là tủ lạnh có chứa nguyên liệu thực phẩm hay không, và nhận diện nguyên liệu cùng định lượng trong ảnh."

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "isValidFridge": {
            "type": "BOOLEAN",
            "description": "True nếu ảnh chụp tủ lạnh/thực phẩm chứa nguyên liệu; False nếu không phải ảnh tủ lạnh/thực phẩm hoặc tủ trống",
        },
        "rejectionReason": {
            "type": "STRING",
            "description": "Lý do từ chối và hướng dẫn chụp lại khi isValidFridge là false",
        },
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "quantity": {"type": "STRING", "description": "Ước tính định lượng (ví dụ: 'nửa trái', '3 ký', '500g', '3 quả')"},
                    "amount": {"type": "NUMBER", "description": "Số lượng số học tương ứng (ví dụ: 0.5, 3.0, 500.0)"},
                    "unit": {"type": "STRING", "description": "Đơn vị đo lường (ví dụ: 'trái', 'quả', 'kg', 'g', 'bó', 'củ')"},
                    "confidence": {"type": "NUMBER"},
                    "freshnessStatus": {"type": "STRING", "enum": list(FRESHNESS)},
                    "foodGroup": {"type": "STRING"},
                    "isAnimalProduct": {"type": "BOOLEAN"},
                    "isRedMeat": {"type": "BOOLEAN"},
                    "box2d": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                        "description": "[ymin, xmin, ymax, xmax] normalized to 0-1000",
                    },
                },
                "required": ["name", "confidence", "freshnessStatus", "isAnimalProduct", "isRedMeat"],
            },
        },
    },
    "required": ["isValidFridge", "items"],
}


def validate_image(data: bytes, settings: Settings) -> ImageInfo:
    """Dung lượng ≤ MAX_UPLOAD_MB (413), JPG/PNG/WEBP theo chữ ký tệp và ≥ 640×480 (400 IMAGE_INVALID)."""
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise file_too_large(f"Ảnh vượt quá {settings.max_upload_mb} MB.")
    if not data:
        raise image_invalid("Tệp ảnh rỗng.")
    info = sniff_image(data)
    if info is None:
        raise image_invalid("Ảnh phải là JPG, PNG hoặc WEBP hợp lệ.")
    # Ảnh chụp dọc (ví dụ 480×640) vẫn đạt: so cạnh ngắn với cạnh ngắn, cạnh dài với cạnh dài.
    short_min, long_min = sorted((settings.image_min_width, settings.image_min_height))
    short, long_ = sorted((info.width, info.height))
    if short < short_min or long_ < long_min:
        raise image_invalid(f"Ảnh tối thiểu {settings.image_min_width}×{settings.image_min_height} px, "
                            f"ảnh gửi lên là {info.width}×{info.height} px.")
    return info


def parse_quantity_and_unit(
    raw_quantity: str | None,
    raw_amount: Any,
    raw_unit: Any,
) -> tuple[str | None, float | None, str | None]:
    """Chuẩn hóa chuỗi định lượng, giá trị số lượng và đơn vị đo."""
    amount: float | None = None
    if isinstance(raw_amount, (int, float)) and not isinstance(raw_amount, bool):
        amount = round(float(raw_amount), 2)

    unit: str | None = None
    if isinstance(raw_unit, str) and raw_unit.strip():
        unit = raw_unit.strip().lower()

    quantity: str | None = None
    if isinstance(raw_quantity, str) and raw_quantity.strip():
        quantity = raw_quantity.strip()[:50]

    # Nếu chưa có amount hoặc unit nhưng có chuỗi quantity, tiến hành suy luận tự động
    if quantity:
        q_lower = quantity.lower()
        if amount is None:
            if "nửa" in q_lower or "1/2" in q_lower:
                amount = 0.5
            elif "1/4" in q_lower:
                amount = 0.25
            elif "1/3" in q_lower:
                amount = 0.33
            else:
                num_match = re.search(r"(\d+(?:[.,]\d+)?)", q_lower)
                if num_match:
                    try:
                        amount = round(float(num_match.group(1).replace(",", ".")), 2)
                    except ValueError:
                        pass

        if unit is None:
            if any(k in q_lower for k in ["kg", "ký", "kí", "kilo"]):
                unit = "kg"
            elif any(k in q_lower for k in ["gam", "gram", "g"]):
                unit = "g"
            elif any(k in q_lower for k in ["trái", "trai"]):
                unit = "trái"
            elif any(k in q_lower for k in ["quả", "qua"]):
                unit = "quả"
            elif any(k in q_lower for k in ["củ", "cu"]):
                unit = "củ"
            elif any(k in q_lower for k in ["bó", "bo"]):
                unit = "bó"
            elif any(k in q_lower for k in ["cây", "cay"]):
                unit = "cây"
            elif any(k in q_lower for k in ["miếng", "mieng", "lát", "lat"]):
                unit = "miếng"
            elif any(k in q_lower for k in ["hộp", "hop"]):
                unit = "hộp"
            elif any(k in q_lower for k in ["ml", "mililit"]):
                unit = "ml"
            elif any(k in q_lower for k in ["lít", "lit", "l"]):
                unit = "lít"

    return quantity, amount, unit


def normalize(data: dict[str, Any]) -> list[DetectedIngredient]:
    """Kiểm lại đầu ra mô hình: bỏ mục sai kiểu/thiếu trường, kẹp confidence về 0..1, gộp trùng tên (giữ điểm cao nhất)."""
    best: dict[str, DetectedIngredient] = {}
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        conf = item.get("confidence")
        fresh = item.get("freshnessStatus")
        animal = item.get("isAnimalProduct")
        red_meat = item.get("isRedMeat", False)
        food_group = item.get("foodGroup")
        if (not isinstance(name, str) or not name.strip() or isinstance(conf, bool)
                or not isinstance(conf, (int, float)) or fresh not in FRESHNESS or not isinstance(animal, bool)):
            continue
        if not isinstance(red_meat, bool):
            red_meat = False
        if not isinstance(food_group, str) or not food_group.strip():
            food_group = None
        else:
            food_group = food_group.strip()[:50]

        # Xử lý toạ độ box2d [ymin, xmin, ymax, xmax] (chuẩn hóa 0..1000)
        raw_box = item.get("box2d") or item.get("box_2d") or item.get("box2D")
        box2d: list[int] | None = None
        if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            try:
                coords = [int(round(float(v))) for v in raw_box]
                ymin, xmin, ymax, xmax = coords
                # Toạ độ hợp lệ: nằm trong 0..1000 và ymin < ymax, xmin < xmax
                if 0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000:
                    box2d = [ymin, xmin, ymax, xmax]
            except (ValueError, TypeError):
                box2d = None

        name = name.strip()[:100]
        raw_quantity = item.get("quantity")
        raw_amount = item.get("amount")
        raw_unit = item.get("unit")
        quantity, amount, unit = parse_quantity_and_unit(raw_quantity, raw_amount, raw_unit)

        det = DetectedIngredient(
            name=name,
            quantity=quantity,
            amount=amount,
            unit=unit,
            confidence=round(min(1.0, max(0.0, float(conf))), 4),
            freshness_status=fresh,
            food_group=food_group,
            is_animal_product=animal or red_meat,
            is_red_meat=red_meat,
            box2d=box2d,
        )
        key = name.casefold()
        if key not in best or det.confidence > best[key].confidence:
            best[key] = det
    return list(best.values())[:MAX_ITEMS]


class IngredientDetector:
    def __init__(self, settings: Settings, llm: LLMClient):
        self._s = settings
        self._llm = llm

    async def detect(self, data: bytes) -> DetectIngredientsResponse:
        info = validate_image(data, self._s)
        msg = Message(role="user", text=USER_TEXT, images=(ImagePart(data=data, mime_type=info.mime_type),))
        gen = await self._llm.generate(SYSTEM_INSTRUCTION, [msg], temperature=min(0.4, self._s.vision_temperature),
                                       max_output_tokens=self._s.vision_max_output_tokens, json_schema=RESPONSE_SCHEMA)
        parsed = parse_json_object(gen.text)
        if parsed is None:
            log.error("Gemini vision JSON parse failed. completion_tokens=%s, raw text: %r", gen.completion_tokens, gen.text[:500])
            raise gemini_error("Gemini không trả JSON nhận diện hợp lệ.")

        raw_is_valid = parsed.get("isValidFridge")
        raw_reason = parsed.get("rejectionReason")

        # Kiểm tra tính hợp lệ của ảnh tủ lạnh
        if raw_is_valid is False:
            is_valid_fridge = False
            rejection_reason = (
                raw_reason.strip()
                if isinstance(raw_reason, str) and raw_reason.strip()
                else "Hình ảnh không phải là tủ lạnh có chứa nguyên liệu thực phẩm. Vui lòng chụp hoặc chọn lại ảnh khác."
            )
            items: list[DetectedIngredient] = []
        else:
            items = normalize(parsed)
            if len(items) == 0:
                is_valid_fridge = False
                rejection_reason = "Không tìm thấy nguyên liệu thực phẩm nào trong ảnh tủ lạnh. Vui lòng kiểm tra và chụp lại ảnh khác."
            else:
                is_valid_fridge = True
                rejection_reason = None

        prompt = (f"[SYSTEM]\n{SYSTEM_INSTRUCTION}\n\n[USER]\n{USER_TEXT}\n"
                  f"[IMAGE {info.mime_type} {info.width}x{info.height} {len(data)} bytes]")
        return DetectIngredientsResponse(
            is_valid_fridge=is_valid_fridge,
            rejection_reason=rejection_reason,
            items=items,
            usage=Usage(model=gen.model, prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens,
                        latency_ms=gen.latency_ms, temperature=gen.temperature),
            audit=Audit(prompt=prompt, raw_response=gen.text))
