# BÁO CÁO KIỂM TRA & CẬP NHẬT TRẠNG THÁI LƯU TRỮ CƠ SỞ DỮ LIỆU (DATABASE AUDIT & UPDATES)
## Các trường: Mục tiêu sức khỏe, Khẩu vị vùng miền, Trường phái, Mức ngân sách, Tối ưu nguyên liệu, Lưu ý

> **Ngày cập nhật:** 13/11/2026  
> **Trạng thái thực thi:** ✅ **ĐÃ BỔ SUNG CỘT TRỰC TIẾP VÀO BẢNG HIỆN HỮU (`dbo.MealPlans` & `dbo.MealSlots`) - KHÔNG TẠO BẢNG MỚI**  
> **Mã nguồn đồng bộ:** Database Migration SQL ⇄ Backend `.NET C# (DAL, BLL)` ⇄ AI-Service `FastAPI (Schemas, Generator)`  
> **Script Migration mới:** `Backend/database/schema/47_MealPlans_Add_Personalization_Columns.sql`

---

## 1. BẢNG TỔNG HỢP CÁC TRƯỜNG DỮ LIỆU ĐÃ ĐƯỢC BỔ SUNG CỘT

Tuân thủ nguyên tắc: **Bổ sung cột trực tiếp vào bảng hiện hữu `MealPlans`, tuyệt đối không tạo thêm bảng mới**.

| STT | Trường thông tin (Thuộc tính) | Tên cột trong Database | Bảng lưu trữ | Kiểu dữ liệu & Ràng buộc | Trạng thái cập nhật |
|:---:|:---|:---|:---:|:---|:---:|
| **1** | **Mục tiêu sức khỏe** (`health_goal`) | `Health_Goal` | `dbo.MealPlans` | `VARCHAR(30) NULL`<br>CHECK: `IN ('LOSE_WEIGHT', 'MAINTAIN', 'GAIN_MUSCLE')` | ✅ **Đã thêm cột** |
| **2** | **Khẩu vị vùng miền** (`region_preference`) | `Region_Preference` | `dbo.MealPlans` | `VARCHAR(20) NULL`<br>CHECK: `IN ('NORTH', 'CENTRAL', 'SOUTH')` | ✅ **Đã thêm cột** |
| **3** | **Trường phái ăn chay & Ngũ vị tân** (`dietary_school`) | `Dietary_School_ID`<br>`Requires_Allium_Free` | `dbo.MealPlans` | `INT NOT NULL` (FK `DietarySchools`)<br>`BIT NOT NULL DEFAULT (0)` | ✅ **Đã hoàn thiện** |
| **4** | **Mức ngân sách** (`budget_tier`) | `Budget_Tier` | `dbo.MealPlans` | `VARCHAR(12) NOT NULL DEFAULT ('STANDARD')`<br>CHECK: `IN ('ECONOMICAL', 'STANDARD', 'PREMIUM')` | ✅ **Có sẵn & đã map đầy đủ** |
| **5** | **Tối ưu nguyên liệu** (`use_fridge` / Tủ lạnh) | `Use_Fridge` | `dbo.MealPlans` | `BIT NOT NULL DEFAULT (0)` | ✅ **Đã thêm cột** |
| **6** | **Lưu ý / Ghi chú thực đơn & bữa ăn** (`custom_instructions`) | `Custom_Instructions`<br>`Notes` (từng bữa ăn) | `dbo.MealPlans`<br>`dbo.MealSlots` | `NVARCHAR(500) NULL` (Mở rộng)<br>`NVARCHAR(200) NULL` (Lưu `matchReason` từ AI) | ✅ **Đã hoàn thiện & map dữ liệu** |

---

## 2. NỘI DUNG SCRIPT MIGRATION SQL (`47_MealPlans_Add_Personalization_Columns.sql`)

Tệp script đã được đặt tại: `Backend/database/schema/47_MealPlans_Add_Personalization_Columns.sql`:

```sql
-- 1. Bổ sung cột Health_Goal vào bảng MealPlans
IF NOT EXISTS (
    SELECT 1 FROM sys.columns
    WHERE object_id = OBJECT_ID(N'[dbo].[MealPlans]') AND name = 'Health_Goal'
)
BEGIN
    ALTER TABLE [dbo].[MealPlans] ADD [Health_Goal] VARCHAR(30) NULL;
    ALTER TABLE [dbo].[MealPlans] ADD CONSTRAINT [CK_MealPlans_Health_Goal]
        CHECK ([Health_Goal] IS NULL OR [Health_Goal] IN ('LOSE_WEIGHT', 'MAINTAIN', 'GAIN_MUSCLE'));
    PRINT N'>> Da bo sung cot [Health_Goal] vao bang dbo.MealPlans.';
END;
GO

-- 2. Bổ sung cột Region_Preference vào bảng MealPlans
IF NOT EXISTS (
    SELECT 1 FROM sys.columns
    WHERE object_id = OBJECT_ID(N'[dbo].[MealPlans]') AND name = 'Region_Preference'
)
BEGIN
    ALTER TABLE [dbo].[MealPlans] ADD [Region_Preference] VARCHAR(20) NULL;
    ALTER TABLE [dbo].[MealPlans] ADD CONSTRAINT [CK_MealPlans_Region_Preference]
        CHECK ([Region_Preference] IS NULL OR [Region_Preference] IN ('NORTH', 'CENTRAL', 'SOUTH'));
    PRINT N'>> Da bo sung cot [Region_Preference] vao bang dbo.MealPlans.';
END;
GO

-- 3. Bổ sung cột Use_Fridge vào bảng MealPlans
IF NOT EXISTS (
    SELECT 1 FROM sys.columns
    WHERE object_id = OBJECT_ID(N'[dbo].[MealPlans]') AND name = 'Use_Fridge'
)
BEGIN
    ALTER TABLE [dbo].[MealPlans] ADD [Use_Fridge] BIT NOT NULL CONSTRAINT [DF_MealPlans_Use_Fridge] DEFAULT (0);
    PRINT N'>> Da bo sung cot [Use_Fridge] vao bang dbo.MealPlans.';
END;
GO

-- 4. Bổ sung cột Requires_Allium_Free vào bảng MealPlans
IF NOT EXISTS (
    SELECT 1 FROM sys.columns
    WHERE object_id = OBJECT_ID(N'[dbo].[MealPlans]') AND name = 'Requires_Allium_Free'
)
BEGIN
    ALTER TABLE [dbo].[MealPlans] ADD [Requires_Allium_Free] BIT NOT NULL CONSTRAINT [DF_MealPlans_Requires_Allium_Free] DEFAULT (0);
    PRINT N'>> Da bo sung cot [Requires_Allium_Free] vao bang dbo.MealPlans.';
END;
GO

-- 5. Mở rộng Custom_Instructions lên NVARCHAR(500)
IF EXISTS (
    SELECT 1 FROM sys.columns
    WHERE object_id = OBJECT_ID(N'[dbo].[MealPlans]') AND name = 'Custom_Instructions' AND max_length < 1000
)
BEGIN
    ALTER TABLE [dbo].[MealPlans] ALTER COLUMN [Custom_Instructions] NVARCHAR(500) NULL;
    PRINT N'>> Da mo rong cot [Custom_Instructions] len NVARCHAR(500) trong dbo.MealPlans.';
END;
GO
```

---

## 3. CÁC ĐIỂM ĐÃ SỬA VÀ ĐỒNG BỘ TRÊN MÃ NGUỒN C# VÀ PYTHON

### 3.1. Entity & DbContext (`VeggieLife.DAL`)
1. **`VeggieLife.DAL/Entities/MealPlan.cs`**:
   - Thêm `public string? HealthGoal { get; set; }`
   - Thêm `public string? RegionPreference { get; set; }`
   - Thêm `public bool UseFridge { get; set; }`
   - Thêm `public bool RequiresAlliumFree { get; set; }`
2. **`VeggieLife.DAL/Context/AppDbContext.cs`**:
   - Cấu hình mapping `Health_Goal` (MaxLength 30, IsUnicode false).
   - Cấu hình mapping `Region_Preference` (MaxLength 20, IsUnicode false).
   - Cấu hình mapping `Use_Fridge` (Default false).
   - Cấu hình mapping `Requires_Allium_Free` (Default false).
   - Mở rộng `Custom_Instructions` thành `HasMaxLength(500)`.

### 3.2. DTO & Service (`VeggieLife.BLL`)
1. **`CreateMealPlanResponseData.cs`** & **`MealPlanDetailDto.cs`**:
   - Bổ sung `HealthGoal`, `RegionPreference`, `UseFridge`, `RequiresAlliumFree`.
2. **`MealPlanSummaryDto.cs`**:
   - Bổ sung `UseFridge`.
3. **`MealPlanService.cs`**:
   - Khi tạo thực đơn: Truyền `profile.HealthGoal`, `profile.RegionPreference`, `request.UseFridge`, `requiresAlliumFree` vào `SaveMealPlanTransactionAsync` và lưu xuống database.
   - Khi lưu từng ô bữa ăn (`MealSlots`): Lưu `s.MatchReason` do Gemini AI sinh vào cột `MealSlot.Notes` (ví dụ: *"Giảm cân thanh đạm, ít béo"*, *"Tận dụng nấm tươi trong tủ lạnh"*).
   - Đã biên dịch `dotnet build VeggieLife.sln`: **0 Error(s)**.

### 3.3. AI-Service (`AI-Service/app/schemas/meal.py`)
1. **Khắc phục lỗi crash `AttributeError: 'MealGenerateRequest' object has no attribute 'health_goal'`**:
   - Bổ sung `health_goal`, `region_preference`, `budget_tier`, `dietary_school_code`, `dietary_school_name`, `requires_allium_free`, `target_weight_kg` vào `MealGenerateRequest`.
   - Bổ sung các trường trên vào `SwapMealByNoteRequest`.
2. **Kiểm thử AI-Service**:
   - Đã chạy kiểm thử với `pytest`: Tất cả các bài test luồng thực đơn (`test_meal_api.py`, `test_meal_stress.py`) đạt **100% PASS**.

---
*Tài liệu tự động cập nhật và đồng bộ theo đúng chỉ thị kỹ thuật của người dùng.*
