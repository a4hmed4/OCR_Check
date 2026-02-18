"""
Extraction engine: regex + label:value pairs -> full_name, university, major, gpa, national_id.
"""
import re
from typing import Optional, Dict, Any


AR_NAME = "\u0627\u0644\u0627\u0633\u0645"
AR_STUDENT_NAME = "\u0627\u0633\u0645 \u0627\u0644\u0637\u0627\u0644\u0628"
AR_UNIVERSITY = "\u0627\u0644\u062c\u0627\u0645\u0639\u0629"
AR_UNI_SHORT = "\u062c\u0627\u0645\u0639\u0629"
AR_MAJOR = "\u0627\u0644\u062a\u062e\u0635\u0635"
AR_COLLEGE = "\u0627\u0644\u0643\u0644\u064a\u0629"
AR_DEPT = "\u0627\u0644\u0642\u0633\u0645"
AR_GPA = "\u0627\u0644\u0645\u0639\u062f\u0644"
AR_ID = "\u0631\u0642\u0645 \u0627\u0644\u0647\u0648\u064a\u0629"
AR_NATIONAL_ID = "\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0648\u0637\u0646\u064a"
AR_BACHELOR = "\u0628\u0643\u0627\u0644\u0648\u0631\u064a\u0648\u0633"
AR_MASTER = "\u0645\u0627\u062c\u0633\u062a\u064a\u0631"


def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return " ".join(str(s).split()).strip()


def _contains_arabic(s: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in s)


def _normalize_digits(s: str) -> str:
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
            "\u066B": ".",
            "\u066C": ",",
        }
    )
    return s.translate(table)


def _prepare_text_for_extraction(text: str) -> str:
    t = _normalize_digits(text or "")
    labels = [
        AR_STUDENT_NAME,
        AR_NAME,
        AR_UNIVERSITY,
        AR_UNI_SHORT,
        AR_MAJOR,
        "\u062a\u062e\u0635\u0635",
        AR_GPA,
        "\u0645\u0639\u062f\u0644",
        AR_NATIONAL_ID,
        AR_ID,
        "name",
        "university",
        "major",
        "gpa",
        "national id",
        AR_BACHELOR,
        AR_MASTER,
    ]
    for lb in labels:
        t = re.sub(rf"(?<!\n)\s+({re.escape(lb)})\s*:", r"\n\1: ", t, flags=re.I)
        t = re.sub(rf"(?<!\n)\s+({re.escape(lb)})\b", r"\n\1 ", t, flags=re.I)
    return _norm(t)


def _looks_like_person_name(value: str) -> bool:
    words = [w for w in _norm(value).split() if w]
    if not 2 <= len(words) <= 5:
        return False
    cleaned = [re.sub(r"[^A-Za-z\u0600-\u06FF]", "", w) for w in words]
    if any(not w for w in cleaned):
        return False
    if any(len(w) < 2 for w in cleaned):
        return False
    joined = " ".join(cleaned)
    if any(ch.isdigit() for ch in joined):
        return False
    weak_tokens = {"\u0645\u062c\u0644\u0633", "\u0641\u064a", "\u0645\u0646", "\u0627\u0644", "\u0648"}
    if sum(1 for w in cleaned if w in weak_tokens) >= 2:
        return False
    arabic_tokens = [w for w in cleaned if any("\u0600" <= c <= "\u06FF" for c in w)]
    if _contains_arabic(value) and len(arabic_tokens) < 2:
        return False
    blocked = [AR_UNIVERSITY, AR_UNI_SHORT, AR_COLLEGE, AR_DEPT, AR_MAJOR, "university", "college", "department"]
    return not any(b.lower() in value.lower() for b in blocked)


def _clean_university_value(value: str) -> str:
    v = _norm(value)
    if not v:
        return v
    stop_pattern = (
        r"(?:\s+|^)(?:\u0628\u0623\u0646|\u0623\u0646|\u0627\u0644\u0637\u0627\u0644\u0628|"
        r"\u0627\u0644\u0627\u0633\u0645|\u0627\u0633\u0645\s+\u0627\u0644\u0637\u0627\u0644\u0628|"
        r"\u0627\u0644\u062a\u062e\u0635\u0635|\u062a\u062e\u0635\u0635|"
        r"\u0627\u0644\u0645\u0639\u062f\u0644|\u0645\u0639\u062f\u0644|GPA|major|"
        r"\u0627\u0644\u0631\u0642\u0645(?:\s+\u0627\u0644\u0648\u0637\u0646\u064a)?)"
    )
    m = re.search(stop_pattern, v, flags=re.I)
    if m and m.start() > 0:
        v = v[:m.start()]
    return _norm(v.strip(" -:;, ."))


def _clean_ocr_field_value(value: str) -> str:
    v = _norm(value).strip(" -:;,.")
    v = re.sub(r"^[\u0621\u0627\u0623\u0625\u0622]\s+", "", v)
    return _norm(v)


def _clean_name_value(value: str) -> str:
    v = _clean_ocr_field_value(value)
    tokens = v.split()
    drop = {
        "\u0645\u0635\u0631\u064a", "\u0645\u0635\u0631\u064a\u0629", "\u0627\u0644\u062c\u0646\u0633\u064a\u0629", "\u0627\u0644\u0627\u0633\u0645",
        "\u0645\u062d\u0644", "\u0627\u0644\u0645\u064a\u0644\u0627\u062f", "\u062a\u0627\u0631\u064a\u062e",
    }
    tokens = [t for t in tokens if t not in drop]
    if len(tokens) > 4:
        tokens = tokens[:4]
    return _norm(" ".join(tokens))


def _is_university_noise(value: str) -> bool:
    v = _norm(value)
    if not v:
        return True
    months = {
        "\u064a\u0646\u0627\u064a\u0631", "\u0641\u0628\u0631\u0627\u064a\u0631", "\u0645\u0627\u0631\u0633", "\u0627\u0628\u0631\u064a\u0644",
        "\u0645\u0627\u064a\u0648", "\u064a\u0648\u0646\u064a\u0648", "\u064a\u0648\u0644\u064a\u0648", "\u0627\u063a\u0633\u0637\u0633",
        "\u0633\u0628\u062a\u0645\u0628\u0631", "\u0627\u0643\u062a\u0648\u0628\u0631", "\u0646\u0648\u0641\u0645\u0628\u0631", "\u062f\u064a\u0633\u0645\u0628\u0631",
    }
    if v in months:
        return True
    return any(ch.isdigit() for ch in v)


def _clean_major_value(value: str) -> str:
    v = _clean_ocr_field_value(value)
    v = re.sub(
        r".*(?:\u0627\u0644\u0631\u0642\u0645|\u0631\u0642\u0645|\u0627\u0644\u0642\u0648\u0645\u064a|\u0627\u0644\u0648\u0637\u0646\u064a)\s+",
        "",
        v,
    )
    return _clean_ocr_field_value(v)


def _extract_tail_phrase(value: str, blocked: set, max_words: int = 2) -> str:
    tokens = re.findall(r"[A-Za-z\u0600-\u06FF]{2,}", value)
    tokens = [_clean_ocr_field_value(t) for t in tokens]
    tokens = [t for t in tokens if t and t not in blocked]
    if not tokens:
        return ""
    return _norm(" ".join(tokens[-min(max_words, len(tokens)):]))


def _apply_field_ocr_corrections(result: Dict[str, Any]) -> Dict[str, Any]:
    uni = result.get("university")
    if isinstance(uni, str):
        u = _norm(uni)
        if u in {"\u0627\u0644\u062f\u0644\u062a", "\u062f\u0644\u062a\u0627"}:
            result["university"] = "\u0627\u0644\u062f\u0644\u062a\u0627"

    major = result.get("major")
    if isinstance(major, str):
        m = _norm(major)
        if m in {"\u0630\u0643\u0627", "\u0627\u0644\u0630\u0643\u0627"}:
            result["major"] = "\u0630\u0643\u0627\u0621"
        elif "\u0630\u0643\u0627\u0621" in m and ("\u0627\u0635\u0637\u0646\u0627" in m or "\u0625\u0635\u0637\u0646\u0627" in m):
            result["major"] = "\u0630\u0643\u0627\u0621"

    return result


def _extract_value_before_label_fallback(text: str) -> Dict[str, Any]:
    t = _normalize_digits(text)
    out: Dict[str, Any] = {}

    name_match = re.search(
        r"([^:\n]{4,60})\s*:\s*[^:\n]{0,50}(?:\u0627\u0644\u0627\u0633\u0645|\u0627\u0627\u0644\u0633\u0645|\u0627\u0633\u0645\s+\u0627\u0644\u0637\u0627\u0644\u0628|\u0627\u0633\u0645|name)",
        t,
        re.I | re.U,
    )
    if name_match:
        blocked = {
            "\u062c\u0627\u0645\u0639\u0629", "\u0627\u0644\u062c\u0627\u0645\u0639\u0629", "\u0643\u0644\u064a\u0629", "\u0627\u0644\u0643\u0644\u064a\u0629",
            "\u0628\u0623\u0646", "\u062f\u0631\u062c\u0629", "\u062a\u062e\u0635\u0635", "\u0627\u0644\u062a\u062e\u0635\u0635",
            "\u0639\u0644\u0648\u0645", "\u062a\u0643\u0646\u0648\u0644\u0648\u062c\u064a\u0627", "\u062a\u0643\u0646\u0648\u0644\u062c\u064a\u0627",
        }
        tail = _extract_tail_phrase(name_match.group(1), blocked, max_words=4)
        v = _clean_ocr_field_value(tail or name_match.group(1))
        if _looks_like_person_name(v):
            out["full_name"] = v

    uni_match = re.search(
        r"([^:\n]{2,60})\s*:\s*(?:\u0627\u0644\u062c\u0627\u0645\u0639\u0629|\u062c\u0627\u0645\u0639\u0629|university)",
        t,
        re.I | re.U,
    )
    if uni_match:
        uni_blocked = {
            "\u0627\u0627\u0644\u0643\u0627\u062f\u064a\u0645\u064a", "\u0627\u0644\u0627\u0643\u0627\u062f\u064a\u0645\u064a", "\u0627\u0643\u0627\u062f\u064a\u0645\u064a",
            "\u0627\u0644\u0645\u0639\u062f\u0644", "\u0645\u0639\u062f\u0644", "\u0627\u0644\u0631\u0642\u0645", "\u0631\u0642\u0645",
            "\u0627\u0644\u0642\u0648\u0645\u064a", "\u0627\u0644\u0648\u0637\u0646\u064a", "\u0627\u0644\u0627\u0633\u0645", "\u0627\u0633\u0645", "\u0627\u0644\u0643\u0627\u0645\u0644",
        }
        tail = _extract_tail_phrase(uni_match.group(1), uni_blocked, max_words=1)
        v = _clean_university_value(tail or _clean_ocr_field_value(uni_match.group(1)))
        if v and not _looks_like_person_name(v) and not _is_university_noise(v):
            out["university"] = v

    major_match = re.search(
        r"([^:\n]{2,60})\s*:\s*(?:\u0627\u0644\u062a\u062e\u0635\u0635|\u062a\u062e\u0635\u0635|major)",
        t,
        re.I | re.U,
    )
    if major_match:
        major_blocked = {
            "\u0627\u0644\u0631\u0642\u0645", "\u0631\u0642\u0645", "\u0627\u0644\u0642\u0648\u0645\u064a", "\u0627\u0644\u0648\u0637\u0646\u064a",
            "\u0627\u0644\u0627\u0633\u0645", "\u0627\u0633\u0645", "\u0627\u0644\u0643\u0627\u0645\u0644",
        }
        tail = _extract_tail_phrase(major_match.group(1), major_blocked, max_words=2)
        v = _clean_major_value(tail or major_match.group(1))
        if len(v) >= 2:
            out["major"] = v

    gpa_match = re.search(
        r"(\d[.,]\d{1,2})\s*:\s*(?:[A-Za-z\u0600-\u06FF\s]{0,25})?(?:\u0627\u0644\u0645\u0639\u062f\u0644|\u0645\u0639\u062f\u0644|gpa)",
        t,
        re.I | re.U,
    )
    if gpa_match:
        try:
            v = float(gpa_match.group(1).replace(",", "."))
            if 0 <= v <= 5:
                out["gpa"] = v
        except ValueError:
            pass

    return out


def _extract_label_value_pairs(text: str) -> Dict[str, Any]:
    labels = {
        "full_name": [AR_NAME, AR_STUDENT_NAME, "name"],
        "university": [AR_UNIVERSITY, AR_UNI_SHORT, "university"],
        "major": [AR_MAJOR, AR_COLLEGE, AR_DEPT, "\u062a\u062e\u0635\u0635", "major"],
        "gpa": [AR_GPA, "\u0645\u0639\u062f\u0644", "GPA", "gpa"],
        "national_id": [AR_ID, AR_NATIONAL_ID, "national id", "\u0627\u0644\u0631\u0642\u0645"],
        "degree": [AR_BACHELOR, AR_MASTER, "bachelor", "master", "degree"],
    }
    out: Dict[str, Any] = {}
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        parts = re.split(r"\s*:\s*", line, 1)
        if len(parts) != 2:
            continue
        left, right = _norm(parts[0]), _norm(parts[1])
        if not left or not right:
            continue

        for field, keywords in labels.items():
            if field in out:
                continue
            for kw in keywords:
                if kw not in left and kw not in right:
                    continue

                val = right if kw in left else left
                val = _norm(val)
                if len(val) < 2 or ":" in val or len(val) > 80:
                    break

                if field == "gpa":
                    try:
                        v = float(val.replace(",", "."))
                        if 0 <= v <= 5:
                            out[field] = v
                    except ValueError:
                        pass
                elif field == "national_id":
                    d = re.sub(r"\D", "", val)
                    if 7 <= len(d) <= 15:
                        out[field] = d
                elif field == "full_name":
                    if len(val) >= 3 and not val.isdigit():
                        out[field] = val
                elif field == "university":
                    digits = re.sub(r"\D", "", val)
                    if digits and len(digits) >= 7 and digits == val.replace(" ", ""):
                        break
                    if _contains_arabic(val) and _looks_like_person_name(val):
                        break
                    if _is_university_noise(val):
                        break
                    out[field] = val
                elif field == "major":
                    out[field] = val
                break
    return out


def extract_gpa(t: str) -> Optional[float]:
    txt = _normalize_digits(t)

    rev = re.search(r"[45]\s*/\s*(\d\.\d{1,2})\s*\u062a\u0631\u0627\u0643\u0645\u064a\s*\u0628\u0645\u0639\u062f\u0644", txt, re.I)
    if rev:
        try:
            v = float(rev.group(1))
            if 0 <= v <= 5:
                return v
        except ValueError:
            pass

    for p in [
        rf"(?:GPA|gpa|{AR_GPA}|\u0645\u0639\u062f\u0644)\s*[:\-]?\s*(\d\.\d{{1,2}})",
        rf"(?:GPA|gpa|{AR_GPA}|\u0645\u0639\u062f\u0644)\s*[:\-]?\s*(\d[.,]\d{{1,2}})",
        r"(\d\.\d{1,2})\s*/\s*[45]",
        r"(\d[.,]\d{1,2})\s*/\s*[45]",
        r"\u0628\u0645\u0639\u062f\u0644\s+\u062a\u0631\u0627\u0643\u0645\u064a\s+[45]\s*/\s*(\d\.\d{1,2})",
        r"(\d[.,]\d{1,2})\s*:\s*(?:[A-Za-z\u0600-\u06FF\s]{0,25})?(?:\u0627\u0644\u0645\u0639\u062f\u0644|\u0645\u0639\u062f\u0644|gpa)",
    ]:
        m = re.search(p, txt, re.I)
        if m:
            try:
                v = float(m.group(1).replace(",", "."))
                if 0 <= v <= 5:
                    return v
            except ValueError:
                pass
    return None


def extract_national_id(t: str) -> Optional[str]:
    txt = _normalize_digits(t)
    # First prefer explicit boxed-id marker injected by OCR pipeline.
    m_nid = re.search(r"\bNID\b\s*[:\-]?\s*((?:\d\D*){7,20})", txt, re.I)
    if m_nid:
        d = re.sub(r"\D", "", m_nid.group(1))
        if len(d) == 14:
            return d
        if 7 <= len(d) <= 15:
            return d

    # Then allow split digits only near ID labels (avoid random date concatenation).
    label_pat = rf"(?:national id|nid|{AR_ID}|{AR_NATIONAL_ID}|{AR_ID}|{AR_NATIONAL_ID}|\u0627\u0644\u0642\u0648\u0645\u064a|\u0642\u0648\u0645\u064a\u0629|\bID\b)"
    for m in re.finditer(rf"{label_pat}.{{0,48}}", txt, re.I):
        d = re.sub(r"\D", "", m.group(0))
        if len(d) == 14:
            return d
        if 7 <= len(d) <= 15:
            return d

    for p in [
        rf"(?:national id|nid|{AR_ID}|{AR_NATIONAL_ID}|\bID\b)\s*[:\-]?\s*(\d[\d\s\-]{{6,}})",
        r"(\d{9,15})",
    ]:
        m = re.search(p, txt, re.I)
        if m:
            d = re.sub(r"\D", "", m.group(1))
            if 7 <= len(d) <= 15:
                return d
    return None


def extract_name(t: str) -> Optional[str]:
    m0 = re.search(
        r"(?:\u0627\u0644\u0627\u0633\u0645|:\u0627\u0644\u0627\u0633\u0645)\s*[:\-]?\s*([A-Za-z\u0600-\u06FF]{2,}(?:\s+[A-Za-z\u0600-\u06FF]{2,}){1,3})",
        t,
        re.I | re.U,
    )
    if m0:
        v = _clean_name_value(m0.group(1))
        # Recover a missing last-name token from the token immediately before the nationality label.
        if len(v.split()) == 2:
            prev = re.search(
                r"([A-Za-z\u0600-\u06FF]{2,})\s*:\s*\u0627\u0644\u062c\u0646\u0633\u064a\u0629",
                t,
                re.I | re.U,
            )
            if prev:
                c = _clean_name_value(prev.group(1))
                # In many scanned certificates this token corresponds to the trailing name part.
                cand = _norm(f"{v} {c}")
                if _looks_like_person_name(cand):
                    v = cand
        if _looks_like_person_name(v):
            return v

    # Certificate-style pattern: "<name> ... <birth/place/date marker>"
    m1 = re.search(
        r"([A-Za-z\u0600-\u06FF]{2,}\s+[A-Za-z\u0600-\u06FF]{2,}\s+[A-Za-z\u0600-\u06FF]{2,}(?:\s+[A-Za-z\u0600-\u06FF]{2,})?)"
        r"\s+(?:\u0645\u0631\u0643\u0632|\u0627\u0644\u0645\u0648\u0644\u0648\u062f|\u0627\u0644\u0645\u0648\u0644\u0648\u062f\u0647|\u062a\u0627\u0631\u064a\u062e)",
        t,
        re.I | re.U,
    )
    if m1:
        v = _clean_name_value(m1.group(1))
        if _looks_like_person_name(v):
            return v

    m2 = re.search(
        r"\u0627\u0644\u062c\u0646\u0633\u064a\u0629\s*[:\-]?\s*[A-Za-z\u0600-\u06FF]{2,15}\s+(?:[A-Za-z\u0600-\u06FF]{2,15}\s+)?"
        r"([A-Za-z\u0600-\u06FF]{2,}\s+[A-Za-z\u0600-\u06FF]{2,}\s+[A-Za-z\u0600-\u06FF]{2,}(?:\s+[A-Za-z\u0600-\u06FF]{2,})?)\s+"
        r"(?:\u0627\u0644\u0627\u0633\u0645|\u0627\u0633\u0645)",
        t,
        re.I | re.U,
    )
    if m2:
        v = _norm(m2.group(1))
        if _looks_like_person_name(v):
            return v

    stop = rf"(?:\n|{AR_UNIVERSITY}|{AR_GPA}|GPA|major|{AR_MAJOR})"
    for p in [
        rf"(?:name|{AR_NAME}|{AR_STUDENT_NAME})\s*[:\-]\s*([A-Za-z\u0600-\u06FF\s\-\.]+?)(?={stop})",
        rf"([A-Za-z\u0600-\u06FF]{{2,}}\s+[A-Za-z\u0600-\u06FF\s\-\.]+?)(?=\s*{AR_UNI_SHORT}|\s*{AR_GPA})",
    ]:
        m = re.search(p, t, re.I | re.U)
        if m:
            v = _norm(m.group(1))
            bad = ["\u0628\u0623\u0646", AR_UNI_SHORT, AR_UNIVERSITY, AR_COLLEGE, "\u0639\u0644\u0648\u0645", "\u062a\u0643\u0646\u0648\u0644\u0648\u062c\u064a\u0627", "\u062a\u0643\u0646\u0648\u0644\u062c\u064a\u0627"]
            if any(x in v for x in bad):
                continue
            if len(v) >= 4 and sum(c.isalpha() or "\u0600" <= c <= "\u06FF" for c in v) >= 4:
                return v
    return None


def extract_university(t: str) -> Optional[str]:
    if re.search(r"\b(?:DELTA\s+UNIVERS\w*|UNIVERS\w*\s+DELTA)\b", t, re.I):
        return "\u0627\u0644\u062f\u0644\u062a\u0627"
    if re.search(r"\u062c\u0627\u0645\u0639\u0629\s+\u0627\u0644\u062f\u0644\u062a", t):
        return "\u0627\u0644\u062f\u0644\u062a\u0627"
    m_u = re.search(r"([A-Za-z\u0600-\u06FF]{2,20})\s+\u062c\u0627\u0645\u0639\u0629", t, re.I | re.U)
    if m_u:
        v = _clean_university_value(m_u.group(1))
        if v and not _is_university_noise(v):
            return v

    for p in [
        r"(?:\u0648?\u0627\u0644\u062a\u0643\u0646\u0648\u0644\u0648\u062c\u064a\u0627\s+\u0644\u0644\u0639\u0644\u0648\u0645|\u0644\u0644\u0639\u0644\u0648\u0645\s+\u0648?\u0627\u0644\u062a\u0643\u0646\u0648\u0644\u0648\u062c\u064a\u0627|\u0648?\u0627\u0644\u062a\u0643\u0646\u0648\u0644\u062c\u064a\u0627\s+\u0644\u0644\u0639\u0644\u0648\u0645|\u0644\u0644\u0639\u0644\u0648\u0645\s+\u0648?\u0627\u0644\u062a\u0643\u0646\u0648\u0644\u062c\u064a\u0627)\s+([A-Za-z\u0600-\u06FF]{2,20})\s+\u062c\u0627\u0645\u0639\u0629",
        rf"\u062a\u0634\u0647\u062f\s+{AR_UNI_SHORT}\s+([A-Za-z\u0600-\u06FF0-9\s\-\.]+?)\s+\u0628\u0623\u0646",
        rf"(?:{AR_UNI_SHORT}|{AR_UNIVERSITY}|university)\s*[:\-]?\s*([A-Za-z\u0600-\u06FF0-9\s\-\.]+?)(?:\n|{AR_MAJOR}|major)",
        rf"(?:{AR_UNI_SHORT})\s+([A-Za-z\u0600-\u06FF0-9\s\-\.]{{2,}})",
    ]:
        m = re.search(p, t, re.I | re.U)
        if m:
            v = _clean_university_value(m.group(1))
            if len(v) >= 2:
                d = re.sub(r"\D", "", v)
                if len(d) >= 7 and d == v.replace(" ", ""):
                    continue
                if _is_university_noise(v):
                    continue
                if _contains_arabic(v) and _looks_like_person_name(v):
                    continue
                return v
    return None


def extract_major(t: str) -> Optional[str]:
    if re.search(r"\u0630\u0643\u0627\w*", t) and re.search(r"\u0627\u0635\u0637|\u0627\u0635\u062d\u0637|\u0635\u0637\u0646\u0627", t):
        return "\u0630\u0643\u0627\u0621"
    m_m = re.search(r"([A-Za-z\u0600-\u06FF]{2,20})\s+\u0643\u0644\u064a\u0629", t, re.I | re.U)
    if m_m:
        v = _norm(m_m.group(1))
        if "\u062a\u0645\u0631\u064a\u0636" in v or "\u062a\u0631\u064a\u0636" in v:
            return "\u062a\u0645\u0631\u064a\u0636"
    for p in [
        rf"(?:{AR_MAJOR}|\u062a\u062e\u0635\u0635|major)\s*[:\-]?\s*([A-Za-z\u0600-\u06FF\s\-\.]+?)(?:\n|GPA|{AR_GPA})",
        r"\u0643\u0644\u064a\u0629\s+([A-Za-z\u0600-\u06FF\s\-\.]+?)(?:\n|\u062f\u0631\u062c\u0629)",
        r"\u0642\u0633\u0645\s+([A-Za-z\u0600-\u06FF\s\-\.]+?)(?:\n|GPA|\u0627\u0644\u0645\u0639\u062f\u0644)",
        r"(?:\u062a\u062e\u0635\u0635|major)\s+([A-Za-z\u0600-\u06FF\s\-\.]{2,})",
    ]:
        m = re.search(p, t, re.I | re.U)
        if m:
            v = _norm(m.group(1))
            if len(v) >= 2:
                return v
    return None


def extract_degree(t: str) -> Optional[str]:
    txt = _norm(t).lower()
    if AR_MASTER in txt:
        return AR_MASTER
    if AR_BACHELOR in txt:
        return AR_BACHELOR
    if re.search(r"\b(master|m\.?sc)\b", txt):
        return AR_MASTER
    if re.search(r"\b(bachelor|b\.?sc)\b", txt):
        return AR_BACHELOR
    # Common phrase in Arabic certificates
    if "\u062f\u0631\u062c\u0629" in txt and (
        "\u0628\u0643\u0627\u0644\u0648\u0631" in txt
        or "\u0628\u0643\u0627\u0644\u0631" in txt
        or "\u0627\u0644\u0628\u0643\u0648\u0631" in txt
        or "\u0627\u0644\u0627\u0644\u0628\u0643\u0648\u0631" in txt
    ):
        return AR_BACHELOR
    return None


def _postprocess_field_swaps(result: Dict[str, Any]) -> Dict[str, Any]:
    full_name = result.get("full_name")
    university = result.get("university")
    major = result.get("major")

    if not full_name:
        candidates = []
        if isinstance(university, str) and _contains_arabic(university) and _looks_like_person_name(university):
            candidates.append(("university", university))
        if isinstance(major, str) and _contains_arabic(major) and _looks_like_person_name(major):
            candidates.append(("major", major))
        if candidates:
            field, value = max(candidates, key=lambda x: len(_norm(x[1])))
            result["full_name"] = _norm(value)
            result[field] = None

    if result.get("full_name") and result.get("major"):
        if _norm(str(result["full_name"])) == _norm(str(result["major"])):
            result["major"] = None

    return result


def extract_all(text: str) -> Dict[str, Any]:
    prepared = _prepare_text_for_extraction(text)
    regex_result = {
        "full_name": extract_name(prepared),
        "university": extract_university(prepared),
        "major": extract_major(prepared),
        "gpa": extract_gpa(prepared),
        "national_id": extract_national_id(prepared),
        "degree": extract_degree(prepared),
    }
    pairs = _extract_label_value_pairs(prepared)
    before_label = _extract_value_before_label_fallback(text)
    for key in ["full_name", "university", "major", "gpa", "national_id", "degree"]:
        if regex_result.get(key) is None and before_label.get(key) is not None:
            regex_result[key] = before_label[key]
        if regex_result.get(key) is None and pairs.get(key) is not None:
            regex_result[key] = pairs[key]
    return _apply_field_ocr_corrections(_postprocess_field_swaps(regex_result))
