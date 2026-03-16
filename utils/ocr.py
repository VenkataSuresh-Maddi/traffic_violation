"""
OCR module for license plate text extraction.
Optimized for Indian license plates (TS, AP, KA, MH, DL formats).

All fixes applied:
1. District slots (3-4) ALWAYS forced to digit (J→0, U→0, etc.)
2. Series slot: dual-candidate approach — ambiguous 2nd letter (B,S,A,G,etc.)
   is tried BOTH as a series letter AND as a number digit; best score wins
3. Score function correctly detects series/number boundary dynamically
   (handles single-letter series like V, N, W as well as two-letter AB, FY)
4. 4-digit number strongly preferred over 3-digit (key for KA-03-V-8078)
5. K/Y disambiguation: K at series-end penalised (Y is more common)
6. 3x upscale + unsharp-mask sharpening before OCR
7. Multi-variant voting across 5 preprocessing modes
8. Fallback cap raised to 12 chars
"""

import re

import cv2
import numpy as np
import easyocr

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        try:
            _reader = easyocr.Reader(["en"], gpu=True)
        except Exception:
            _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


_INDIAN_PLATE_RE = re.compile(
    r"\b([A-Z]{2})\s*(\d{2})\s*([A-Z]{1,2})\s*(\d{1,4})\b",
    re.IGNORECASE,
)

_LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "J": "0",    # J → 0
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
    "A": "4",
}

_DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}

# Digits that visually resemble letters (safe to treat as letters in series slot)
# NOTE: "8" is intentionally excluded — it would be misread as "B" in series
DIGIT_LOOKS_LIKE_LETTER = {"0", "1", "4", "5", "6"}

# Series ambiguous letter swaps (for alternate candidate generation)
_SERIES_AMBIGUOUS = {
    "K": "Y", "Y": "K",
    "V": "U", "N": "M", "M": "N",
    "P": "R", "R": "P",
}


def _to_letter(c: str) -> str:
    c = c.upper()
    return c if c.isalpha() else _DIGIT_TO_LETTER.get(c, c)


def _to_digit(c: str) -> str:
    return c if c.isdigit() else _LETTER_TO_DIGIT.get(c.upper(), c)


def _build_correction(s: str, treat_ambiguous_series2_as_digit: bool) -> str:
    """
    Core correction builder. Called twice with different treatment of the
    2nd series character when it is ambiguous (in _LETTER_TO_DIGIT).
    """
    if len(s) < 5:
        return s
    result = []
    i = 0

    # State: 2 letters
    for _ in range(2):
        if i < len(s):
            result.append(_to_letter(s[i]))
            i += 1

    # District: 2 digits (ALWAYS — never letters)
    for _ in range(2):
        if i < len(s):
            result.append(_to_digit(s[i]))
            i += 1

    # Series: up to 2 letters
    letters_added = 0
    while i < len(s) and letters_added < 2:
        c = s[i]
        if c.isdigit():
            if c in DIGIT_LOOKS_LIKE_LETTER:
                result.append(_to_letter(c))
                letters_added += 1
                i += 1
            else:
                break  # raw digit (2,3,7,8,9) → number starts here
        elif c.isalpha():
            is_ambiguous = c in _LETTER_TO_DIGIT  # B, S, A, G, I, L, Z, T
            if letters_added == 1 and is_ambiguous and treat_ambiguous_series2_as_digit:
                break  # treat this char as the start of the number
            result.append(c.upper())
            letters_added += 1
            i += 1
        else:
            break

    # Number: up to 4 digits
    digits_added = 0
    while i < len(s) and digits_added < 4:
        result.append(_to_digit(s[i]))
        digits_added += 1
        i += 1

    return "".join(result)


def _correct_ocr_mistakes(raw: str) -> list[str]:
    """
    Returns 1 or 2 candidate corrected strings.
    When the 2nd series char is ambiguous (e.g. B, S, A), generates both
    interpretations — as a series letter and as a number digit.
    The caller scores all candidates and picks the best.
    """
    s = "".join(c for c in raw.upper() if c.isalnum())
    c1 = _build_correction(s, treat_ambiguous_series2_as_digit=False)
    c2 = _build_correction(s, treat_ambiguous_series2_as_digit=True)
    return list(dict.fromkeys([c1, c2]))  # deduplicated, order preserved


def _score_candidate(text: str) -> int:
    """
    Structural quality score for a candidate plate string.
    Dynamically detects where the series ends and number begins.

    Key scoring philosophy:
    - 4-digit number strongly preferred over 3-digit (prevents ambiguous
      2nd series char from eating the first number digit)
    - K at series-end penalised (Y is more common in Indian plates)
    """
    s = "".join(c for c in text.upper() if c.isalnum())
    if len(s) < 5:
        return 0

    score = 0
    if len(s) >= 8:  score += 3
    if len(s) >= 9:  score += 2
    if len(s) >= 10: score += 1
    if s[:2].isalpha():  score += 2
    if s[2:4].isdigit(): score += 3

    # Find series/number boundary dynamically
    j = 4
    while j < len(s) and s[j].isalpha() and j < 6:
        j += 1
    series = s[4:j]
    number = s[j:]

    if series and series.isalpha():
        score += 2
    if number and number.isdigit():
        score += 3
        if len(number) == 4: score += 3   # strong preference — key tiebreaker
        elif len(number) == 3: score += 1
        elif len(number) == 2: score -= 1

    # K tiebreaker
    if series.endswith("K"):
        score -= 1

    return score


def _generate_alternates(plate: str) -> list[str]:
    """Generate visually-similar alternates for ambiguous series letters."""
    candidates = [plate]
    s = "".join(c for c in plate if c.isalnum())
    if len(s) < 7:
        return candidates
    # Find series
    j = 4
    while j < len(s) and s[j].isalpha() and j < 6:
        j += 1
    series = s[4:j]
    for idx, ch in enumerate(series):
        if ch in _SERIES_AMBIGUOUS:
            alt = list(series)
            alt[idx] = _SERIES_AMBIGUOUS[ch]
            alt_plate = s[:4] + "".join(alt) + s[j:]
            candidates.append(alt_plate)
    return candidates


def _extract_from_string(corrected: str) -> str:
    """Try to extract a valid Indian plate from a corrected string."""
    s = "".join(c for c in corrected.upper() if c.isalnum())

    m = _INDIAN_PLATE_RE.search(s)
    if m:
        state, district, series, num = m.groups()
        return f"{state.upper()}{district}{series.upper()}{num}"

    # Structural fallback
    if len(s) >= 7 and s[:2].isalpha() and s[2:4].isdigit():
        rest = s[4:]
        series, number = "", ""
        j = 0
        while j < len(rest) and rest[j].isalpha() and len(series) < 2:
            series += rest[j]; j += 1
        while j < len(rest) and rest[j].isdigit() and len(number) < 4:
            number += rest[j]; j += 1
        if series and number:
            return f"{s[:2]}{s[2:4]}{series}{number}"

    return ""


def _preprocess(plate_img: np.ndarray) -> list:
    if plate_img is None or plate_img.size == 0:
        return []

    img = cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Unsharp mask
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    img  = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return [
        ("otsu",         otsu),
        ("otsu_inv",     cv2.bitwise_not(otsu)),
        ("adaptive",     adaptive),
        ("adaptive_inv", cv2.bitwise_not(adaptive)),
        ("raw",          denoised),
    ]


def read_plate(plate_img: np.ndarray) -> str:
    """
    Extract Indian license plate text from a cropped plate image.
    Returns plate string (e.g. KA03V8078, TS07FY2960) or '' if unreadable.
    """
    if plate_img is None or plate_img.size == 0:
        return ""

    variants = _preprocess(plate_img)
    if not variants:
        return ""

    reader = _get_reader()
    candidates: list[tuple[str, int, float]] = []
    raw_fallbacks: list[tuple[str, float]] = []

    for _name, proc_img in variants:
        for _, text, conf in reader.readtext(proc_img):
            if conf < 0.10:
                continue
            raw = "".join(c for c in text if c.isalnum()).upper()
            if len(raw) < 6:
                continue

            # Generate all corrected candidates (dual interpretation)
            for corrected in _correct_ocr_mistakes(raw):
                raw_fallbacks.append((corrected, conf))
                extracted = _extract_from_string(corrected)
                if extracted:
                    # Also generate visual alternates (K↔Y etc.)
                    for alt in _generate_alternates(extracted):
                        score = _score_candidate(alt)
                        candidates.append((alt, score, conf))

    if candidates:
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[0][0]

    if raw_fallbacks:
        raw_fallbacks.sort(key=lambda x: x[1], reverse=True)
        return raw_fallbacks[0][0][:12]

    return ""