"""Sinh tệp mẫu JSON (dữ liệu GIẢ LẬP để kiểm thử, không phải nội dung thật) cho các nguồn nằm ở SQL của Backend.

Định dạng mỗi phần tử đúng body của POST /rag/upsert: sourceType, sourceId (chuỗi chữ số), title, text.
Chạy:  python -m scripts.gen_fixtures
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "rag" / "fixtures"

RECIPES = [
    ("Canh nấm đậu phụ", "VEGAN", "nấm rơm, đậu phụ non, hành lá, gừng", "Phi gừng, cho nấm vào xào sơ, thêm nước sôi, cho đậu phụ cắt khối vào nấu 5 phút, nêm muối và rắc hành lá.", 25),
    ("Đậu hũ sốt cà chua", "VEGAN", "đậu phụ, cà chua, hành tây, rau mùi", "Chiên vàng đậu phụ, xào cà chua với hành tây thành sốt rồi rim đậu phụ trong sốt 8 phút.", 20),
    ("Cơm chiên rau củ", "LACTO", "cơm nguội, cà rốt, đậu Hà Lan, bắp, nước tương", "Xào rau củ, cho cơm nguội vào đảo đều, nêm nước tương và hạt nêm nấm.", 15),
    ("Bún riêu chay", "VEGAN", "bún, cà chua, đậu phụ, nấm, me", "Nấu nước dùng cà chua và me, thêm đậu phụ vò nát và nấm, chan lên bún.", 40),
    ("Phở chay", "VEGAN", "bánh phở, nấm hương, cà rốt, củ cải, quế, hồi", "Ninh củ cải, cà rốt, nấm với quế hồi 40 phút lấy nước dùng, trụng bánh phở và chan nước.", 60),
    ("Gỏi cuốn chay", "VEGAN", "bánh tráng, bún, rau thơm, dưa leo, đậu phụ chiên", "Cuốn bún, rau thơm, dưa leo và đậu phụ chiên trong bánh tráng; chấm tương đậu phộng.", 25),
    ("Đậu lăng hầm cà rốt", "VEGAN", "đậu lăng đỏ, cà rốt, hành tây, nghệ", "Xào hành tây với nghệ, thêm đậu lăng và cà rốt, hầm với nước 25 phút đến khi mềm.", 35),
    ("Salad quinoa đậu gà", "VEGAN", "quinoa, đậu gà, cà chua bi, dưa leo, chanh", "Luộc quinoa, trộn với đậu gà đã nấu chín, rau củ cắt nhỏ, chanh và dầu ô liu.", 20),
    ("Cháo yến mạch chuối", "LACTO_OVO", "yến mạch, chuối, sữa hạt, quế", "Nấu yến mạch với sữa hạt 8 phút, thêm chuối cắt lát và một chút quế.", 12),
    ("Mì Ý sốt nấm", "LACTO", "mì Ý, nấm mỡ, tỏi, kem tươi, phô mai", "Luộc mì, xào nấm với tỏi, thêm kem tươi, trộn mì vào và rắc phô mai.", 25),
    ("Chè đậu xanh", "VEGAN", "đậu xanh, đường phèn, nước cốt dừa", "Nấu đậu xanh với đường phèn đến nhừ, rưới nước cốt dừa khi dùng.", 45),
    ("Canh bí đỏ hạt sen", "VEGAN", "bí đỏ, hạt sen, rau ngò", "Nấu hạt sen mềm, thêm bí đỏ cắt miếng nấu thêm 10 phút, nêm nhạt.", 30),
    ("Tempeh áp chảo", "VEGAN", "tempeh, nước tương, tỏi, mè", "Ướp tempeh với nước tương và tỏi, áp chảo vàng hai mặt, rắc mè.", 15),
    ("Rau muống xào tỏi", "VEGAN", "rau muống, tỏi, dầu ăn", "Phi tỏi, xào rau muống lửa lớn 2 phút, nêm muối.", 10),
    ("Đậu hũ non hấp nấm", "VEGAN", "đậu hũ non, nấm đông cô, gừng, nước tương", "Xếp nấm lên đậu hũ, hấp 10 phút, rưới nước tương gừng.", 20),
    ("Bánh mì chay", "LACTO_OVO", "bánh mì, trứng, dưa leo, rau mùi, tương ớt", "Kẹp trứng chiên, dưa leo, rau mùi vào bánh mì, thêm tương ớt.", 10),
    ("Cà ri đậu gà", "VEGAN", "đậu gà, khoai tây, nước cốt dừa, bột cà ri", "Xào bột cà ri, thêm khoai tây và đậu gà, đổ nước cốt dừa nấu 20 phút.", 40),
    ("Xôi đậu xanh", "VEGAN", "gạo nếp, đậu xanh, hành phi", "Hấp đậu xanh, đồ xôi, nén đậu lên mặt và rắc hành phi.", 50),
    ("Súp bắp non", "LACTO_OVO", "bắp non, trứng, nấm, nước dùng rau", "Nấu nước dùng rau với bắp và nấm, đánh trứng đổ vào khuấy nhẹ.", 20),
    ("Lẩu nấm chay", "VEGAN", "nấm các loại, rau cải, đậu hũ, nước dùng củ quả", "Ninh nước dùng củ quả, nhúng nấm, đậu hũ và rau khi ăn.", 45),
    ("Bánh xèo chay", "VEGAN", "bột gạo, nghệ, nấm, giá, đậu phụ", "Đổ bột loãng pha nghệ vào chảo nóng, thêm nhân nấm giá đậu phụ, gập đôi khi giòn.", 35),
    ("Đậu phụ nhồi nấm", "VEGAN", "đậu phụ, nấm hương, hành lá", "Khoét đậu phụ, nhồi nấm xào, hấp rồi chiên nhẹ.", 30),
    ("Cơm cuộn rong biển", "VEGAN", "cơm, rong biển nori, cà rốt, dưa leo, bơ", "Trải cơm lên nori, xếp rau củ và bơ, cuộn chặt và cắt khoanh.", 25),
    ("Sinh tố chuối yến mạch", "LACTO", "chuối, yến mạch, sữa, mật ong", "Xay chuối, yến mạch, sữa và mật ong đến mịn.", 5),
    ("Mướp nấu đậu hũ", "VEGAN", "mướp, đậu hũ, nấm rơm", "Nấu nấm rơm trước, thêm mướp và đậu hũ, nấu 5 phút.", 20),
    ("Khoai lang nướng", "VEGAN", "khoai lang, dầu ô liu", "Nướng khoai lang 200 độ C khoảng 40 phút đến khi mềm.", 45),
    ("Hủ tiếu chay", "VEGAN", "hủ tiếu, nấm, cải ngọt, đậu phụ", "Nấu nước dùng củ quả, trụng hủ tiếu, thêm nấm và rau.", 40),
    ("Bông cải xanh xào nấm", "VEGAN", "bông cải xanh, nấm bào ngư, tỏi", "Chần sơ bông cải, xào cùng nấm và tỏi lửa lớn.", 15),
    ("Đậu đen nấu chè", "VEGAN", "đậu đen, đường thốt nốt, nước cốt dừa", "Ninh đậu đen mềm, thêm đường thốt nốt, rưới nước cốt dừa.", 60),
    ("Trứng cuộn rau củ", "LACTO_OVO", "trứng, cà rốt, hành lá, nấm", "Đánh trứng với rau củ băm nhỏ, tráng mỏng và cuộn.", 15),
]

POSTS = [
    ("Mình ăn chay trường 2 năm và bí quyết đủ đạm", "Mình chia sẻ cách bổ sung đạm từ đậu phụ, tempeh, đậu lăng và các loại hạt. Mỗi bữa mình luôn có một nguồn đạm thực vật và uống đủ nước."),
    ("Kinh nghiệm bổ sung vitamin B12 khi ăn thuần chay", "Vì B12 hầu như không có trong thực vật, mình dùng thực phẩm tăng cường B12 và kiểm tra định kỳ. Mọi người nên hỏi bác sĩ về liều dùng."),
    ("Thực đơn tuần cho người mới ăn chay", "Tuần đầu mình chuyển dần từ ăn mặn: ngày 1 chay ít bữa, tăng dần. Có nhiều món đơn giản như canh nấm, đậu hũ sốt cà chua."),
    ("Đi chợ chay ở đâu để tiết kiệm", "Mình thường mua đậu, nấm, rau củ theo mùa ở chợ gần nhà, mua đậu khô số lượng lớn rẻ hơn nhiều."),
    ("Ăn chay khi tập gym có đủ cơ không", "Mình tập 4 buổi mỗi tuần, ăn đậu, tempeh, sữa đậu nành và yến mạch. Quan trọng là tổng lượng đạm và calo trong ngày."),
    ("Mẹo đọc nhãn thực phẩm để tránh gelatin", "Mình kiểm tra mã E441 và các chất làm đặc có nguồn gốc động vật. Nhiều kẹo dẻo có gelatin nên không thuần chay."),
    ("Trẻ nhỏ ăn chay cần lưu ý gì", "Mình hỏi bác sĩ nhi để bổ sung đủ sắt, kẽm và B12 cho bé. Không tự ý cho bé ăn chay nghiêm ngặt nếu chưa tham khảo chuyên gia."),
    ("Làm sữa hạt tại nhà", "Ngâm hạnh nhân qua đêm, xay với nước lọc, lọc bã. Sữa hạt tự làm không có thêm đường và phụ gia."),
    ("Nấu ăn chay cho gia đình bận rộn", "Mình nấu sẵn nước dùng rau củ cuối tuần, chia hộp cấp đông để nấu nhanh các món canh và nấu phở."),
    ("Dị ứng đậu phộng khi ăn chay", "Bạn mình dị ứng đậu phộng nên thay bằng hạt hướng dương, kiểm tra kỹ nhãn vì nhiều sốt chay có chứa đậu phộng."),
]

E_NUMBERS = [
    ("E100", "Curcumin", "Chất tạo màu vàng chiết từ củ nghệ, có nguồn gốc thực vật; thường phù hợp với người ăn chay."),
    ("E120", "Carmine (cochineal)", "Chất tạo màu đỏ làm từ côn trùng cochineal, có nguồn gốc động vật; KHÔNG phù hợp với người ăn chay."),
    ("E129", "Allura Red AC", "Phẩm màu đỏ tổng hợp, không có nguồn gốc động vật; thường phù hợp với người ăn chay (một số người nhạy cảm với phẩm màu azo)."),
    ("E150a", "Caramel thường", "Chất tạo màu nâu từ đường nấu cháy; thường phù hợp với người ăn chay."),
    ("E153", "Than thực vật", "Chất tạo màu đen từ than thực vật; phù hợp với người ăn chay."),
    ("E160a", "Beta-caroten", "Chất tạo màu cam từ thực vật như cà rốt, cũng có thể tổng hợp; thường phù hợp với người ăn chay."),
    ("E162", "Betanin (đỏ củ dền)", "Chất tạo màu đỏ từ củ dền; có nguồn gốc thực vật, phù hợp với người ăn chay."),
    ("E270", "Axit lactic", "Chất điều chỉnh độ chua; phần lớn sản xuất bằng lên men carbohydrate nên thường phù hợp với người ăn chay, nhưng cần xem nhà sản xuất vì có thể liên quan đến sữa."),
    ("E322", "Lecithin", "Chất nhũ hóa, thường chiết từ đậu nành hoặc hướng dương nên phù hợp với người ăn chay; lecithin từ trứng ít gặp hơn. Người dị ứng đậu nành cần chú ý."),
    ("E330", "Axit citric", "Chất điều chỉnh độ chua, sản xuất bằng lên men nấm; phù hợp với người ăn chay."),
    ("E407", "Carrageenan", "Chất làm đặc chiết từ rong biển đỏ; có nguồn gốc thực vật, phù hợp với người ăn chay."),
    ("E410", "Gôm đậu châu chấu", "Chất làm đặc từ hạt cây carob; có nguồn gốc thực vật, phù hợp với người ăn chay."),
    ("E415", "Gôm xanthan", "Chất làm đặc tạo ra bằng lên men vi sinh; phù hợp với người ăn chay."),
    ("E422", "Glycerol", "Chất giữ ẩm; có thể có nguồn gốc thực vật hoặc mỡ động vật, cần xem thông tin nhà sản xuất nên không chắc chắn là thuần chay."),
    ("E441", "Gelatin", "Chất tạo đông được làm từ collagen của da, xương động vật; KHÔNG phù hợp với người ăn chay."),
    ("E471", "Mono và diglyceride của acid béo", "Chất nhũ hóa có thể có nguồn gốc thực vật hoặc động vật; cần kiểm tra nguồn nên không chắc chắn là thuần chay."),
    ("E901", "Sáp ong", "Chất tạo bóng từ sáp ong, có nguồn gốc động vật; KHÔNG phù hợp với người ăn thuần chay."),
    ("E904", "Shellac (cánh kiến)", "Chất tạo bóng từ chất tiết của côn trùng lac; có nguồn gốc động vật, KHÔNG phù hợp với người ăn thuần chay."),
    ("E920", "L-cysteine", "Chất cải thiện bột, có thể làm từ lông vũ hoặc tóc động vật hoặc lên men vi sinh; cần kiểm tra nguồn, không chắc chắn là thuần chay."),
    ("E966", "Lactitol", "Chất tạo ngọt sản xuất từ lactose của sữa; không phù hợp với người ăn thuần chay và người dị ứng sữa."),
]

ALLERGENS = [
    ("Đậu nành", "Đậu nành và các sản phẩm như đậu hũ, tempeh, sữa đậu nành, tương, nước tương có thể gây dị ứng. Cần đọc nhãn vì lecithin và protein đậu nành xuất hiện trong nhiều thực phẩm chế biến."),
    ("Gluten (lúa mì)", "Gluten có trong lúa mì, lúa mạch, mạch đen; nhiều sản phẩm thịt chay seitan làm từ gluten. Người bệnh celiac và dị ứng lúa mì cần tránh."),
    ("Đậu phộng", "Đậu phộng gây dị ứng nặng ở một số người; xuất hiện trong sốt, bánh kẹo và nhiều món chay. Phải đọc nhãn cảnh báo."),
    ("Quả hạch", "Quả hạch như hạnh nhân, óc chó, hạt điều gây dị ứng ở một số người; thường dùng làm sữa hạt và phô mai thuần chay."),
    ("Mè (vừng)", "Hạt mè có thể gây dị ứng, thường có trong dầu mè, tahini và bánh mì; cần đọc nhãn."),
]

INGREDIENTS = [
    ("Đậu phụ", "Thực phẩm làm từ đậu nành, giàu đạm thực vật và canxi nếu được đông tụ bằng muối canxi. Nhóm dị ứng: đậu nành."),
    ("Tempeh", "Thực phẩm lên men từ đậu nành, giàu đạm và chất xơ. Nhóm dị ứng: đậu nành."),
    ("Đậu lăng", "Họ đậu giàu đạm, sắt và chất xơ, nấu nhanh, không cần ngâm lâu."),
    ("Nấm hương", "Nấm có hương vị đậm, dùng nấu nước dùng chay; cung cấp vitamin D khi phơi nắng."),
    ("Hạnh nhân", "Quả hạch giàu vitamin E và chất béo tốt. Nhóm dị ứng: quả hạch."),
    ("Yến mạch", "Ngũ cốc giàu chất xơ beta-glucan; người cần tránh gluten nên chọn loại ghi nhãn không gluten."),
    ("Đậu gà", "Họ đậu giàu đạm và chất xơ, dùng cho cà ri, salad và hummus."),
    ("Rong biển nori", "Rong biển dạng lá dùng cuộn cơm; chứa i-ốt, cần dùng vừa phải."),
    ("Nước tương", "Gia vị lên men từ đậu nành và lúa mì. Nhóm dị ứng: đậu nành, gluten."),
    ("Nước cốt dừa", "Chất béo từ cùi dừa, dùng trong cà ri và chè; nhiều chất béo bão hòa nên dùng vừa phải."),
]


def dump(name, items):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    recipes = [
        {"sourceType": "RECIPE", "sourceId": str(500 + i), "title": t,
         "text": f"Trường phái: {school}. Nguyên liệu: {ing}. Cách làm: {steps} Thời gian nấu khoảng {mins} phút."}
        for i, (t, school, ing, steps, mins) in enumerate(RECIPES, start=1)]
    posts = [{"sourceType": "POST", "sourceId": str(100 + i), "title": t, "text": b} for i, (t, b) in enumerate(POSTS, start=1)]
    enums = [{"sourceType": "E_NUMBER", "sourceId": str(int("".join(c for c in code if c.isdigit()))),
              "title": f"{code} — {name}", "text": f"{code} ({name}): {desc}"}
             for code, name, desc in E_NUMBERS]
    allergens = [{"sourceType": "ALLERGEN", "sourceId": str(i), "title": f"Dị ứng {n}", "text": d}
                 for i, (n, d) in enumerate(ALLERGENS, start=1)]
    ingredients = [{"sourceType": "INGREDIENT", "sourceId": str(i), "title": n, "text": d}
                   for i, (n, d) in enumerate(INGREDIENTS, start=1)]
    for name, items in (("recipes", recipes), ("posts", posts), ("e_numbers", enums), ("allergens", allergens),
                        ("ingredients", ingredients)):
        dump(name, items)
        print(name, len(items))


if __name__ == "__main__":
    main()
