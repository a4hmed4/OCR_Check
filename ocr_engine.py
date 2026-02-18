"""
OCR engine: PDF text detection + PaddleOCR fallback.
"""
import os
import re
import io
from pathlib import Path
from typing import Tuple, Dict, Any

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")

import fitz
import pdfplumber
from PIL import Image
import numpy as np
import cv2

MIN_TEXT_LEN = 50
TARGET_DPI = 220
FIELD_HINTS = (
    "\u0627\u0644\u062c\u0627\u0645\u0639\u0629",
    "\u062c\u0627\u0645\u0639\u0629",
    "\u0627\u0644\u062a\u062e\u0635\u0635",
    "\u062a\u062e\u0635\u0635",
    "\u0627\u0644\u0645\u0639\u062f\u0644",
    "gpa",
)
_OCR_INSTANCE = None
_OCR_INIT_ERROR = None
_OCR_DIGIT_INSTANCE = None
_OCR_DIGIT_INIT_ERROR = None


def _norm(s: str) -> str:
    return " ".join(s.split()).strip() if s else ""


def _normalize_dpi(dpi: int) -> int:
    return max(170, min(260, int(dpi)))


def _is_arabic_letter(c: str) -> bool:
    return "\u0600" <= c <= "\u06FF" and not (c.isdigit() or "\u0660" <= c <= "\u0669")


def fix_reversed_arabic(s: str) -> str:
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if _is_arabic_letter(c):
            j = i
            while j < n and _is_arabic_letter(s[j]):
                j += 1
            out.append(s[i:j][::-1])
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _maybe_fix_reversed_arabic(s: str) -> str:
    if not s:
        return s
    fixed = fix_reversed_arabic(s)
    # Apply reversal fix only when reversed-shape hints dominate.
    normal_hints = (
        "\u0627\u0644\u062c\u0627\u0645\u0639\u0629",
        "\u062c\u0627\u0645\u0639\u0629",
        "\u0627\u0644\u062a\u062e\u0635\u0635",
        "\u062a\u062e\u0635\u0635",
        "\u0627\u0644\u0645\u0639\u062f\u0644",
        "\u0627\u0644\u0631\u0642\u0645",
    )
    reversed_hints = (
        "\u0629\u0639\u0645\u0627\u062c\u0644\u0627",  # reversed form of "university"
        "\u0629\u0639\u0645\u0627\u062c",            # reversed form of "college/university"
        "\u0635\u0635\u062e\u062a\u0644\u0627",      # reversed form of "major/specialization"
        "\u0635\u0635\u062e\u062a",                  # reversed form of "specialization"
        "\u0644\u062f\u0639\u0645\u0644\u0627",      # reversed form of "gpa/rate"
        "\u0645\u0642\u0631\u0644\u0627",            # reversed form of "number"
    )
    normal_score = sum(s.count(h) for h in normal_hints) + sum(fixed.count(h) for h in normal_hints)
    reversed_score = sum(s.count(h) for h in reversed_hints)
    if reversed_score > normal_score:
        return fixed
    return s


def pdf_has_text(path: Path, min_len: int = MIN_TEXT_LEN) -> Tuple[bool, str]:
    text_pdfplumber = ""
    text_pymupdf = ""
    try:
        with pdfplumber.open(path) as pdf:
            text_pdfplumber = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        pass
    if len(text_pdfplumber.strip()) < min_len:
        doc = fitz.open(path)
        text_pymupdf = "\n".join(page.get_text() for page in doc)
        doc.close()
    # Merge both engines when available; each one may miss blocks the other keeps.
    merged = "\n".join(x for x in [text_pdfplumber, text_pymupdf] if x.strip())
    text = _norm(_maybe_fix_reversed_arabic(merged))
    return len(text) >= min_len, text


def pdf_pages_to_images(path: Path, dpi: int = 150):
    dpi = _normalize_dpi(dpi)
    doc = fitz.open(path)
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        yield pix.tobytes("png")
    doc.close()


def _preprocess_for_ocr(img_rgb: np.ndarray) -> np.ndarray:
    # Fast preprocessing: denoise + contrast enhancement.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def _ocr_lines_from_result(res, rtl: bool = True):
    if not res or not res[0]:
        return []
    ordered = []
    for item in res[0]:
        try:
            box = item[0]
            txt = item[1][0]
            if not txt:
                continue
            cx = sum(p[0] for p in box) / len(box)
            cy = sum(p[1] for p in box) / len(box)
            # Layout-aware ordering: top-to-bottom, then right-to-left for Arabic.
            ordered.append((round(cy / 18), -cx if rtl else cx, txt))
        except Exception:
            continue
    ordered.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in ordered]


def _norm_digits_only(s: str) -> str:
    if not s:
        return ""
    table = str.maketrans(
        {
            "\u0660": "0",
            "\u0661": "1",
            "\u0662": "2",
            "\u0663": "3",
            "\u0664": "4",
            "\u0665": "5",
            "\u0666": "6",
            "\u0667": "7",
            "\u0668": "8",
            "\u0669": "9",
            "\u06F0": "0",
            "\u06F1": "1",
            "\u06F2": "2",
            "\u06F3": "3",
            "\u06F4": "4",
            "\u06F5": "5",
            "\u06F6": "6",
            "\u06F7": "7",
            "\u06F8": "8",
            "\u06F9": "9",
        }
    )
    return "".join(ch for ch in s.translate(table) if ch.isdigit())


def _extract_text_from_ocr_any(res) -> str:
    if res is None:
        return ""
    if isinstance(res, str):
        return res
    if isinstance(res, (tuple, list)):
        if len(res) >= 1 and isinstance(res[0], str):
            return res[0]
        for x in res:
            t = _extract_text_from_ocr_any(x)
            if t:
                return t
    return ""


def _get_digit_ocr():
    global _OCR_DIGIT_INSTANCE, _OCR_DIGIT_INIT_ERROR
    if _OCR_DIGIT_INSTANCE is None and _OCR_DIGIT_INIT_ERROR is None:
        from paddleocr import PaddleOCR
        errs = []
        for kwargs in [
            {"lang": "en"},
            {"use_angle_cls": True, "lang": "en"},
        ]:
            try:
                _OCR_DIGIT_INSTANCE = PaddleOCR(**kwargs)
                break
            except Exception as e:
                errs.append(f"{kwargs}: {e}")
        if _OCR_DIGIT_INSTANCE is None:
            _OCR_DIGIT_INIT_ERROR = " | ".join(errs[-2:]) if errs else "digit ocr init failed"
    return _OCR_DIGIT_INSTANCE


def _boxed_id_candidates_from_result(res):
    if not res or not res[0]:
        return []
    # Collect small digit tokens (typical for boxed ID digits).
    tokens = []
    for item in res[0]:
        try:
            box = item[0]
            txt = item[1][0]
            d = _norm_digits_only(txt)
            if not d or len(d) > 3:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            w = max(xs) - min(xs)
            tokens.append((cy, cx, max(w, 8.0), d))
        except Exception:
            continue
    if not tokens:
        return []

    # Group by row and concatenate near-horizontal tokens.
    rows = {}
    for cy, cx, w, d in tokens:
        key = int(round(cy / 22))
        rows.setdefault(key, []).append((cx, w, d))

    candidates = []
    for _, row in rows.items():
        row.sort(key=lambda x: x[0])
        if not row:
            continue
        run = [row[0]]
        for cur in row[1:]:
            prev = run[-1]
            gap = cur[0] - prev[0]
            max_gap = max(prev[1], cur[1]) * 3.2
            if gap <= max_gap:
                run.append(cur)
            else:
                joined = "".join(x[2] for x in run)
                if 8 <= len(joined) <= 18:
                    candidates.append(joined)
                run = [cur]
        joined = "".join(x[2] for x in run)
        if 8 <= len(joined) <= 18:
            candidates.append(joined)

    # Prefer 14-digit Egyptian IDs first, then other plausible lengths.
    uniq = list(dict.fromkeys(candidates))
    uniq.sort(key=lambda x: (0 if len(x) == 14 else 1, abs(len(x) - 14)))
    return uniq[:4]


def _boxed_id_candidates_from_image(img_rgb: np.ndarray, ocr) -> list[str]:
    # Detect boxed digit rows and OCR each box independently.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
    )
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    rects = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww < 10 or hh < 14:
            continue
        if ww > 120 or hh > 120:
            continue
        ar = ww / float(hh)
        if not (0.35 <= ar <= 2.4):
            continue
        area = ww * hh
        if area < 160 or area > 6000:
            continue
        cy = y + hh / 2
        if not (0.18 * h <= cy <= 0.82 * h):
            continue
        rects.append((x, y, ww, hh))

    if not rects:
        return []

    # Group by row center.
    rows = []
    for r in sorted(rects, key=lambda z: z[1] + z[3] / 2):
        x, y, ww, hh = r
        cy = y + hh / 2
        placed = False
        for row in rows:
            if abs(cy - row["cy"]) <= 18:
                row["items"].append(r)
                row["cy"] = (row["cy"] * row["n"] + cy) / (row["n"] + 1)
                row["n"] += 1
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "items": [r], "n": 1})

    candidates = []
    digit_ocr = _get_digit_ocr() or ocr
    crops_budget = 28
    for row in rows:
        items = sorted(row["items"], key=lambda z: z[0])
        if len(items) < 8:
            continue
        # Build contiguous runs by x-gap.
        run = [items[0]]
        runs = []
        for cur in items[1:]:
            prev = run[-1]
            gap = cur[0] - (prev[0] + prev[2])
            max_gap = max(prev[2], cur[2]) * 2.2
            if gap <= max_gap:
                run.append(cur)
            else:
                runs.append(run)
                run = [cur]
        runs.append(run)

        for rr in runs:
            if len(rr) < 8:
                continue
            # OCR full ID row first; often better than per-box OCR on noisy scans.
            x0 = max(0, min(x for x, _, _, _ in rr) - 6)
            y0 = max(0, min(y for _, y, _, _ in rr) - 6)
            x1 = min(w, max(x + ww for x, _, ww, _ in rr) + 6)
            y1 = min(h, max(y + hh for _, y, _, hh in rr) + 6)
            row_crop = img_rgb[y0:y1, x0:x1]
            if row_crop.size != 0:
                row_up = cv2.resize(
                    row_crop,
                    (max(90, row_crop.shape[1] * 3), max(30, row_crop.shape[0] * 3)),
                    interpolation=cv2.INTER_CUBIC,
                )
                row_gray = cv2.cvtColor(row_up, cv2.COLOR_RGB2GRAY)
                row_bin = cv2.adaptiveThreshold(
                    row_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
                )
                row_rgb = cv2.cvtColor(row_bin, cv2.COLOR_GRAY2RGB)
                try:
                    rr_text = _extract_text_from_ocr_any(digit_ocr.ocr(row_rgb, det=False, cls=False))
                except Exception:
                    rr_text = ""
                rr_digits = _norm_digits_only(rr_text)
                if 10 <= len(rr_digits) <= 18:
                    candidates.append(rr_digits)

            digits = []
            # OCR each cell crop.
            for x, y, ww, hh in rr[:20]:
                if crops_budget <= 0:
                    break
                pad = 2
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(w, x + ww + pad)
                y1 = min(h, y + hh + pad)
                crop = img_rgb[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                # Upscale tiny box crop to improve digit recognition.
                scale = 4
                crop_up = cv2.resize(crop, (max(24, crop.shape[1] * scale), max(24, crop.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(crop_up, cv2.COLOR_RGB2GRAY)
                bw = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 4
                )
                bw = cv2.medianBlur(bw, 3)
                crop_ready = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
                try:
                    # Recognition-only works better on single boxed characters.
                    cr = digit_ocr.ocr(crop_ready, det=False, cls=False)
                except Exception:
                    cr = None
                txt = _extract_text_from_ocr_any(cr)
                if txt:
                    d = _norm_digits_only(txt)
                    if d:
                        digits.append(d[-1])  # keep single boxed digit
                crops_budget -= 1
            joined = "".join(digits)
            if 10 <= len(joined) <= 18:
                candidates.append(joined)

    uniq = list(dict.fromkeys(candidates))
    uniq.sort(key=lambda x: (0 if len(x) == 14 else 1, abs(len(x) - 14)))
    return uniq[:4]


def _id_region_candidates_from_result(res, img_rgb: np.ndarray, ocr) -> list[str]:
    if not res or not res[0]:
        return []
    h, w = img_rgb.shape[:2]
    keywords = (
        "\u0642\u0648\u0645",
        "\u0631\u0642\u0645",
        "\u0628\u0637\u0627\u0642",
        "nid",
        "national",
        " id",
        "\u0629\u064a\u0645\u0648\u0642",  # reversed "national"
        "\u0645\u0642\u0631",              # reversed "number"
        "\u0629\u0642\u0627\u0637\u0628",  # reversed "card"
    )
    anchors = []
    for item in res[0]:
        try:
            box = item[0]
            txt = _norm(item[1][0]).lower()
        except Exception:
            continue
        if not txt:
            continue
        if any(k in txt for k in keywords):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            anchors.append((max(0, int(min(xs) - 20)), int((min(ys) + max(ys)) / 2)))
    if not anchors:
        return []

    candidates = []
    for ax, ay in anchors[:4]:
        y0 = max(0, ay - 56)
        y1 = min(h, ay + 56)
        x0 = max(0, ax - 20)
        x1 = min(w, ax + int(w * 0.62))
        region = img_rgb[y0:y1, x0:x1]
        if region.size == 0:
            continue
        up = cv2.resize(
            region,
            (max(140, region.shape[1] * 2), max(48, region.shape[0] * 2)),
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
        )
        prep = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
        try:
            rr = ocr.ocr(prep, cls=False)
        except Exception:
            rr = None
        candidates.extend(_boxed_id_candidates_from_result(rr))
        if rr and rr[0]:
            toks = []
            for cell in rr[0]:
                try:
                    box = cell[0]
                    dd = _norm_digits_only(cell[1][0])
                    if not dd:
                        continue
                    cx = sum(p[0] for p in box) / len(box)
                    toks.append((cx, dd))
                except Exception:
                    continue
            if toks:
                toks.sort(key=lambda x: x[0])
                joined = "".join(d for _, d in toks)
                if 10 <= len(joined) <= 18:
                    candidates.append(joined)

    uniq = list(dict.fromkeys(candidates))
    uniq.sort(key=lambda x: (0 if len(x) == 14 else 1, abs(len(x) - 14)))
    return uniq[:4]


def _layout_regions(img_rgb: np.ndarray):
    h, w = img_rgb.shape[:2]
    regions = []
    if h >= 1000:
        regions.append(img_rgb[: int(h * 0.62), :])
        regions.append(img_rgb[int(h * 0.32):, :])
    if w >= 1400:
        regions.append(img_rgb[:, : int(w * 0.58)])
        regions.append(img_rgb[:, int(w * 0.42):])
    return regions


def run_ocr(path: Path) -> str:
    global _OCR_INSTANCE, _OCR_INIT_ERROR
    if _OCR_INSTANCE is None and _OCR_INIT_ERROR is None:
        from paddleocr import PaddleOCR
        init_errors = []
        for kwargs in [
            {"use_angle_cls": True, "lang": "ar"},
            {"lang": "ar"},
            {"use_angle_cls": True, "lang": "en"},
            {"lang": "en"},
        ]:
            try:
                _OCR_INSTANCE = PaddleOCR(**kwargs)
                break
            except Exception as e:
                init_errors.append(f"{kwargs}: {e}")
                continue
        if _OCR_INSTANCE is None:
            details = " | ".join(init_errors[-2:]) if init_errors else "unknown init error"
            _OCR_INIT_ERROR = f"PaddleOCR init failed: {details}"
    if _OCR_INSTANCE is None:
        raise RuntimeError(_OCR_INIT_ERROR or "PaddleOCR init failed")

    ocr = _OCR_INSTANCE
    texts = []
    for raw in pdf_pages_to_images(path, TARGET_DPI):
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        base = np.array(img)
        proc = _preprocess_for_ocr(base)

        res = ocr.ocr(proc, cls=True)
        lines = _ocr_lines_from_result(res, rtl=True)
        id_candidates = _boxed_id_candidates_from_result(res)
        if not id_candidates:
            id_candidates = _id_region_candidates_from_result(res, proc, ocr)
        if not id_candidates:
            id_candidates = _boxed_id_candidates_from_image(proc, ocr)

        # Layout-aware fallback: scan major regions when page-level text is weak.
        if len(lines) < 8:
            for region in _layout_regions(proc):
                rr = ocr.ocr(region, cls=True)
                lines.extend(_ocr_lines_from_result(rr, rtl=True))
                id_candidates.extend(_boxed_id_candidates_from_result(rr))

        if id_candidates:
            for cid in list(dict.fromkeys(id_candidates)):
                lines.append(f"NID {cid}")

        # Deduplicate while preserving order.
        uniq = list(dict.fromkeys(lines))
        if uniq:
            texts.append("\n".join(uniq))
    return _norm(_maybe_fix_reversed_arabic("\n".join(texts)))


def run_ocr_image(path: Path) -> str:
    global _OCR_INSTANCE, _OCR_INIT_ERROR
    if _OCR_INSTANCE is None and _OCR_INIT_ERROR is None:
        from paddleocr import PaddleOCR
        init_errors = []
        for kwargs in [
            {"use_angle_cls": True, "lang": "ar"},
            {"lang": "ar"},
            {"use_angle_cls": True, "lang": "en"},
            {"lang": "en"},
        ]:
            try:
                _OCR_INSTANCE = PaddleOCR(**kwargs)
                break
            except Exception as e:
                init_errors.append(f"{kwargs}: {e}")
                continue
        if _OCR_INSTANCE is None:
            details = " | ".join(init_errors[-2:]) if init_errors else "unknown init error"
            _OCR_INIT_ERROR = f"PaddleOCR init failed: {details}"
    if _OCR_INSTANCE is None:
        raise RuntimeError(_OCR_INIT_ERROR or "PaddleOCR init failed")

    ocr = _OCR_INSTANCE
    img = Image.open(path).convert("RGB")
    base = np.array(img)
    proc = _preprocess_for_ocr(base)

    res = ocr.ocr(proc, cls=True)
    lines = _ocr_lines_from_result(res, rtl=True)
    id_candidates = _boxed_id_candidates_from_result(res)
    if not id_candidates:
        id_candidates = _id_region_candidates_from_result(res, proc, ocr)
    if not id_candidates:
        id_candidates = _boxed_id_candidates_from_image(proc, ocr)
    if len(lines) < 8:
        for region in _layout_regions(proc):
            rr = ocr.ocr(region, cls=True)
            lines.extend(_ocr_lines_from_result(rr, rtl=True))
            id_candidates.extend(_boxed_id_candidates_from_result(rr))
    if id_candidates:
        for cid in list(dict.fromkeys(id_candidates)):
            lines.append(f"NID {cid}")
    uniq = list(dict.fromkeys(lines))
    return _norm(_maybe_fix_reversed_arabic("\n".join(uniq)))


def get_pdf_text_debug(path: Path) -> Dict[str, Any]:
    has_text, text = pdf_has_text(path, MIN_TEXT_LEN)
    text_norm = _norm(text)
    hint_hits = sum(1 for h in FIELD_HINTS if h.lower() in text_norm.lower())
    debug: Dict[str, Any] = {
        "has_pdf_text": has_text,
        "pdf_text_length": len(text_norm),
        "pdf_hint_hits": hint_hits,
        "ocr_attempted": False,
        "ocr_error": None,
        "ocr_text_length": 0,
        "source": "pdf_text",
        "pdf_preview": text_norm[:220],
        "ocr_preview": "",
    }

    if has_text and hint_hits >= 2:
        debug["source"] = "pdf_text"
        return {"text": text_norm, "debug": debug}

    debug["ocr_attempted"] = True
    try:
        ocr_text = run_ocr(path)
    except Exception as e:
        debug["ocr_error"] = str(e)
        if text_norm:
            debug["source"] = "pdf_text_fallback_after_ocr_error"
            return {"text": text_norm, "debug": debug}
        raise

    ocr_norm = _norm(ocr_text)
    ocr_hint_hits = sum(1 for h in FIELD_HINTS if h.lower() in ocr_norm.lower())
    debug["ocr_text_length"] = len(ocr_norm)
    debug["ocr_hint_hits"] = ocr_hint_hits
    debug["ocr_preview"] = ocr_norm[:220]

    if not text_norm:
        debug["source"] = "ocr_only"
        return {"text": ocr_norm, "debug": debug}
    if not ocr_norm:
        debug["source"] = "pdf_text_only"
        return {"text": text_norm, "debug": debug}

    merged = _norm(f"{text_norm}\n{ocr_norm}")
    debug["source"] = "merged_pdf_ocr"
    debug["merged_length"] = len(merged)
    debug["merged_hint_hits"] = sum(1 for h in FIELD_HINTS if h.lower() in merged.lower())
    return {"text": merged, "debug": debug}


def get_document_text_debug(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return get_pdf_text_debug(path)

    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        debug: Dict[str, Any] = {
            "has_pdf_text": False,
            "pdf_text_length": 0,
            "pdf_hint_hits": 0,
            "ocr_attempted": True,
            "ocr_error": None,
            "ocr_text_length": 0,
            "source": "image_ocr",
            "pdf_preview": "",
            "ocr_preview": "",
        }
        try:
            text = run_ocr_image(path)
        except Exception as e:
            debug["ocr_error"] = str(e)
            return {"text": "", "debug": debug}
        text = _norm(text)
        debug["ocr_text_length"] = len(text)
        debug["ocr_preview"] = text[:220]
        debug["ocr_hint_hits"] = sum(1 for h in FIELD_HINTS if h.lower() in text.lower())
        return {"text": text, "debug": debug}

    raise RuntimeError(f"Unsupported file type: {ext}")


def get_pdf_text(path: Path) -> str:
    return get_pdf_text_debug(path)["text"]
