# OCR Certificate Verification API

Production-oriented FastAPI service for extracting and validating structured fields from academic certificates (PDF or image uploads), with emphasis on noisy Arabic OCR inputs.

## What This Service Does
- Accepts a certificate file and expected student fields.
- Extracts:
  - `full_name`
  - `university`
  - `major`
  - `gpa`
  - `national_id`
  - `degree` (`بكالوريوس` / `ماجستير`)
- Compares extracted values against submitted values.
- Returns:
  - match status (`MATCH`, `PARTIAL_MATCH`, `MISMATCH`)
  - confidence score
  - field-level comparison details
  - debug block for OCR/extraction diagnostics

## Architecture
- `main.py`
  - API contract (`/upload`, `/verify`, `/health`)
  - request validation and response mapping
- `model.py`
  - orchestration layer: OCR -> extraction -> validation
  - confidence/status assembly
- `ocr_engine.py`
  - PDF text extraction (`pdfplumber` + `PyMuPDF`)
  - OCR fallback using PaddleOCR
  - preprocessing (DPI normalization, denoise, contrast)
  - layout-aware line ordering and boxed-ID heuristics
- `extraction_engine.py`
  - regex and rule-based field extraction
  - Arabic OCR noise handling and post-processing
- `validation_engine.py`
  - field scoring and final confidence logic

## Supported Inputs
- Documents:
  - `.pdf`
  - `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`
- Transport:
  - `multipart/form-data`

## API
### `POST /upload`
Compatibility endpoint used by the frontend. Returns:
- `status`
- `confidence`
- `extracted_data`
- `comparison_details`
- `debug`

### `POST /verify`
Core endpoint (same verification logic, raw backend-oriented format).

### `GET /health`
Health probe:
```json
{"status":"ok"}
```

## Request Fields
- `name` (string)
- `university` (string)
- `major` (string)
- `gpa` (string or numeric text)
- `national_id` (string)
- `degree` (string, optional)
- `file` (required)

## Local Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Server default:
- `http://127.0.0.1:8000`

Swagger:
- `http://127.0.0.1:8000/docs`

## Docker Setup
```bash
docker-compose up --build
```

## Frontend (Vite)
From `frontend/`:
```bash
npm install
npm run dev
```

Make sure frontend proxy points to `http://127.0.0.1:8000`.

## Example cURL
```powershell
curl -X POST "http://127.0.0.1:8000/upload" `
  -H "accept: application/json" `
  -F "name= ......" `
  -F "university=......." `
  -F "major=......" `
  -F "gpa=......" `
  -F "national_id=......." `
  -F "degree=...." `
  -F "file=@...pdf;type=application/pdf"
```


## Troubleshooting
- If PaddleOCR initialization fails:
  - verify virtual environment is active
  - pin compatible `numpy`/`paddlepaddle`/`paddleocr` versions from `requirements.txt`

------

## الدليل بالعربية

### نظرة عامة
هذا المشروع عبارة عن خدمة `FastAPI` للتحقق من شهادات أكاديمية (PDF أو صور) عبر:
- استخراج البيانات من المستند باستخدام OCR.
- مقارنة البيانات المستخرجة مع بيانات الطالب المرسلة في الطلب.
- إرجاع نتيجة نهائية بدرجة ثقة وتفاصيل لكل حقل.

الحقول المستهدفة:
- `full_name`
- `university`
- `major`
- `gpa`
- `national_id`
- `degree` (بكالوريوس / ماجستير)

### هيكل المشروع
- `main.py`: تعريف الـAPI endpoints والتحقق من المدخلات.
- `model.py`: طبقة orchestration بين OCR والاستخراج والتحقق.
- `ocr_engine.py`: استخراج النص من PDF + OCR fallback + تحسينات المعالجة.
- `extraction_engine.py`: قواعد الاستخراج (Regex + قواعد لاحقة).
- `validation_engine.py`: حساب درجات المطابقة والـconfidence.

### أنواع الملفات المدعومة
- `.pdf`
- `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`

### تشغيل المشروع محليًا
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

بعد التشغيل:
- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

### التشغيل عبر Docker
```bash
docker-compose up --build
```

### مثال طلب `curl`
```powershell
curl -X POST "http://127.0.0.1:8000/upload" `
  -H "accept: application/json" `
  -F "name=........" `
  -F "university=......." `
  -F "major=........" `
  -F "gpa=......." `
  -F "national_id=........" `
  -F "degree=........" `
  -F "file=@........pdf;type=application/pdf"
```

### شكل الاستجابة
الاستجابة تتضمن:
- `status`: (`MATCH` / `PARTIAL_MATCH` / `MISMATCH`)
- `confidence`: درجة الثقة النهائية
- `extracted_data`: البيانات التي تم استخراجها من الشهادة
- `comparison_details`: درجة المطابقة لكل حقل
- `debug`: تفاصيل OCR والاستخراج لتسهيل التشخيص


### استكشاف الأخطاء
- إذا فشل PaddleOCR في الإقلاع:
  - تأكد أن البيئة الافتراضية `.venv` مفعلة.
  - راجع توافق إصدارات `numpy` و`paddlepaddle` و`paddleocr`.
