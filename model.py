"""
PDF verification: orchestration only. Uses ocr_engine, extraction_engine, validation_engine.
"""
import logging
import tempfile
from pathlib import Path
import re

from ocr_engine import get_document_text_debug, fix_reversed_arabic
from extraction_engine import extract_all
from validation_engine import validate_fields

logging.basicConfig(level=logging.INFO)
MIN_TEXT_LEN = 40


def _extraction_score(extracted: dict) -> int:
    # Prioritize core academic fields, not only national_id.
    score = 0
    for key in ["full_name", "university", "major", "gpa", "degree"]:
        if extracted.get(key) is not None:
            score += 2
    if extracted.get("national_id") is not None:
        score += 1
    return score


def verify_pdf(file_bytes: bytes, student_data: dict, filename: str = "document.pdf") -> dict:
    ext = Path(filename).suffix.lower() or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        path = Path(tmp.name)
    try:
        ocr_payload = get_document_text_debug(path)
        text = ocr_payload["text"]
        ocr_debug = ocr_payload.get("debug", {})
        logging.info("Extracted text length: %d", len(text))
    except Exception as e:
        return {"is_valid": False, "confidence": 0.0, "extracted_data": {}, "field_validation": {}, "error": str(e)}
    finally:
        path.unlink(missing_ok=True)
    extracted_base = extract_all(text)
    fixed_text = " ".join(fix_reversed_arabic(text).split())
    extracted_fixed = extract_all(fixed_text) if fixed_text and fixed_text != text else extracted_base

    base_score = _extraction_score(extracted_base)
    fixed_score = _extraction_score(extracted_fixed)
    use_fixed = fixed_score > base_score
    extracted = extracted_fixed if use_fixed else extracted_base

    if len(text.strip()) < MIN_TEXT_LEN and _extraction_score(extracted) == 0:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "extracted_data": {},
            "field_validation": {},
            "error": "Insufficient text extracted from document (OCR unavailable or failed)",
            "debug": {
                "engine_version": "2026-02-16.12",
                "text_length": len(text),
                "ocr": ocr_debug,
                "text_preview": text[:220],
            },
        }

    is_valid, confidence, field_validation = validate_fields(student_data, extracted)
    lower_text = text.lower()
    text_debug = {
        "engine_version": "2026-02-16.12",
        "text_length": len(text),
        "has_university_hint": ("\u0627\u0644\u062c\u0627\u0645\u0639\u0629" in text) or ("\u062c\u0627\u0645\u0639\u0629" in text) or ("university" in lower_text),
        "has_major_hint": ("\u0627\u0644\u062a\u062e\u0635\u0635" in text) or ("\u062a\u062e\u0635\u0635" in text) or ("major" in lower_text),
        "has_gpa_hint": ("\u0627\u0644\u0645\u0639\u062f\u0644" in text) or ("gpa" in lower_text) or bool(re.search(r"\d(?:[.,]\d{1,2})?\s*/\s*[45]", text)),
        "ocr": ocr_debug,
        "text_preview": text[:220],
        "extraction_base_score": base_score,
        "extraction_fixed_score": fixed_score,
        "used_fixed_text_for_extraction": use_fixed,
        "fixed_text_preview": fixed_text[:220] if use_fixed else "",
    }
    return {
        "is_valid": is_valid,
        "confidence": confidence,
        "extracted_data": extracted,
        "field_validation": field_validation,
        "debug": text_debug,
    }
