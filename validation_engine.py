"""
Validation engine: compare form input vs extracted data → is_valid, confidence, field_validation.
"""
from typing import Dict, Any, Tuple

from rapidfuzz import fuzz

GPA_TOLERANCE = 0.05
NAME_THRESHOLD = 0.85
UNI_THRESHOLD = 0.75
MAJOR_THRESHOLD = 0.75
DEGREE_THRESHOLD = 0.8
VALID_CONFIDENCE = 0.80
STRICT_NATIONAL_ID_REQUIRED = False


def _norm(s: str) -> str:
    return " ".join(str(s).split()).strip() if s else ""


def score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return max(fuzz.ratio(a, b), fuzz.partial_ratio(a, b)) / 100.0


def score_name(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    base = [
        fuzz.ratio(a, b),
        fuzz.partial_ratio(a, b),
        fuzz.token_sort_ratio(a, b),
        fuzz.token_set_ratio(a, b),
    ]
    # OCR may duplicate one token (e.g., "John John Smith").
    def dedup_tokens(s: str) -> str:
        out = []
        for t in s.split():
            if t not in out:
                out.append(t)
        return " ".join(out)
    ad = dedup_tokens(a)
    bd = dedup_tokens(b)
    if ad and bd:
        base.extend([fuzz.token_sort_ratio(ad, bd), fuzz.token_set_ratio(ad, bd)])
    return max(base) / 100.0


def _norm_degree(s: str) -> str:
    v = _norm(s).lower()
    if not v:
        return ""
    if any(x in v for x in ["\u0645\u0627\u062c\u0633\u062a\u064a\u0631", "master", "msc", "m.sc"]):
        return "master"
    if any(x in v for x in ["\u0628\u0643\u0627\u0644\u0648\u0631", "bachelor", "bsc", "b.sc"]):
        return "bachelor"
    return v


def validate_fields(inp: Dict[str, str], ext: Dict[str, Any]) -> Tuple[bool, float, Dict]:
    """
    inp: name, university, major, gpa, national_id, degree (from form)
    ext: full_name, university, major, gpa, national_id, degree (from extraction)
    Returns: (is_valid, confidence, field_validation)
    """
    field_validation = {}
    scores = []
    mapping = [
        ("name", "full_name"),
        ("university", "university"),
        ("major", "major"),
        ("gpa", "gpa"),
        ("national_id", "national_id"),
        ("degree", "degree"),
    ]
    for key, label in mapping:
        inv = _norm(inp.get(key) or "")
        exv = ext.get(label)
        if key == "gpa":
            try:
                iv = float(inv.replace(",", "."))
            except ValueError:
                iv = None
            if iv is None:
                ok, sc = True, 1.0
            elif exv is None:
                ok, sc = False, 0.0
            else:
                try:
                    ev = float(exv)
                    ok = abs(iv - ev) <= GPA_TOLERANCE
                    sc = 1.0 if ok else max(0, 1.0 - abs(iv - ev) / 0.5)
                except (TypeError, ValueError):
                    ok, sc = False, 0.0
        elif key == "national_id":
            fd = "".join(c for c in inv if c.isdigit())
            ed = "".join(c for c in (str(exv) if exv is not None else "") if c.isdigit())
            if not fd:
                ok, sc = True, 1.0
            elif not ed:
                ok, sc = False, 0.0
            else:
                match = sum(1 for c1, c2 in zip(fd, ed) if c1 == c2)
                sc = match / max(len(fd), len(ed))
                ok = fd == ed
        elif key == "degree":
            in_deg = _norm_degree(inv)
            ex_deg = _norm_degree(str(exv) if exv is not None else "")
            if not in_deg:
                ok, sc = True, 1.0
            elif not ex_deg:
                ok, sc = False, 0.0
            else:
                sc = score(in_deg, ex_deg)
                ok = sc >= DEGREE_THRESHOLD
        else:
            exs = _norm(str(exv)) if exv is not None else ""
            sc = score_name(inv, exs) if key == "name" else score(inv, exs)
            th = NAME_THRESHOLD if key == "name" else (UNI_THRESHOLD if key == "university" else MAJOR_THRESHOLD)
            ok = sc >= th
        scores.append(sc)
        field_validation[key] = {"match": ok, "score": round(sc, 2), "extracted": exv}
    confidence = sum(scores) / len(scores) if scores else 0.0
    is_valid = (
        confidence >= VALID_CONFIDENCE
        and field_validation["name"]["match"]
        and field_validation["gpa"]["match"]
        and (field_validation["national_id"]["match"] or not STRICT_NATIONAL_ID_REQUIRED)
    )
    return is_valid, round(confidence, 2), field_validation
