"""
FastAPI endpoints for PDF verification.
"""
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import verify_pdf

app = FastAPI(title="PDF Verification API", version="1.0")
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/verify")
async def verify(
    name: str = Form(""),
    university: str = Form(""),
    major: str = Form(""),
    gpa: str = Form(""),
    national_id: str = Form(""),
    degree: str = Form(""),
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(400, "File required")
    ext = "." + file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type. Use PDF or image files.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    student_data = {
        "name": name,
        "university": university,
        "major": major,
        "gpa": gpa,
        "national_id": national_id,
        "degree": degree,
    }
    return verify_pdf(raw, student_data, file.filename)

@app.post("/upload")
async def upload(
    name: str = Form(""),
    university: str = Form(""),
    major: str = Form(""),
    gpa: str = Form(""),
    national_id: str = Form(""),
    degree: str = Form(""),
    file: UploadFile = File(...),
):
    """Compatibility endpoint for frontend (returns old format)."""
    if not file.filename:
        raise HTTPException(400, "File required")
    ext = "." + file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type. Use PDF or image files.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    student_data = {
        "name": name,
        "university": university,
        "major": major,
        "gpa": gpa,
        "national_id": national_id,
        "degree": degree,
    }
    result = verify_pdf(raw, student_data, file.filename)
    if "error" in result:
        return {
            "status": "MISMATCH",
            "confidence": 0.0,
            "extracted_data": {},
            "comparison_details": {},
            "error": result["error"],
        }
    fv = result.get("field_validation") or {}
    # Backward-compatible mapping: nested field result or scalar score.
    def detail(key):
        v = fv.get(key)
        if isinstance(v, dict):
            return {"match": v.get("match", False), "score": v.get("score", 0)}
        if isinstance(v, (int, float)):
            return {"match": v >= 0.75, "score": float(v)}
        return {"match": False, "score": 0}
    comparison_details = {
        "name": detail("name"),
        "university": detail("university"),
        "major": detail("major"),
        "gpa": detail("gpa"),
        "national_id": detail("national_id"),
        "degree": detail("degree"),
    }
    status = "MATCH" if result["is_valid"] else ("PARTIAL_MATCH" if result["confidence"] >= 0.60 else "MISMATCH")
    return {
        "status": status,
        "confidence": result["confidence"],
        "extracted_data": result["extracted_data"],
        "comparison_details": comparison_details,
        "debug": result.get("debug", {}),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
