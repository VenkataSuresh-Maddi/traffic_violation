"""
OCR module for license plate text extraction.
Optimized for Indian license plates (AP, TS, KA formats) using
image preprocessing and regex-based validation/filtering.
"""

import re

import cv2
import easyocr

# Lazy-load EasyOCR reader to avoid startup cost when not used
_reader = None

# Indian plate format: XX NN XX NNNN (e.g. AP01AB1234, KA01AA1234, TS09CD5678)
# State: 2 letters | District: 2 digits | Series: 1-2 letters | Number: 1-4 digits
_INDIAN_PLATE_PATTERN = re.compile(
    r"\b([A-Z]{2})\s*(\d{2})\s*([A-Z]{1,2})\s*(\d{1,4})\b",
    re.IGNORECASE,
)

# Common OCR substitutions (char → possible misreads)
_DIGIT_TO_LETTER = {"0": "O", "1": "IL", "5": "S", "6": "G", "8": "B", "2": "Z"}
_LETTER_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}


def _get_reader():
    """Get or create the EasyOCR reader instance.
    Tries GPU (CUDA/MPS) first; falls back to CPU if unavailable.
    """
    global _reader
    if _reader is None:
        try:
            _reader = easyocr.Reader(["en"], gpu=True)
        except Exception:
            _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _preprocess_for_indian_plate(plate_img):
    """
    Preprocess image for better OCR on Indian license plates.
    Applies grayscale, CLAHE contrast, bilateral denoising, and multiple thresholding.
    """
    if plate_img.size == 0:
        return []

    # 1. Upscale 2x for finer text
    img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization) for uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Bilateral filter: reduce noise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # 5. Multiple thresholding variants (different lighting/plate colors)
    preprocessed = []

    # Otsu's threshold
    _, otsu = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    preprocessed.append(otsu)

    # Adaptive threshold (handles shadow/gradient)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    preprocessed.append(adaptive)

    # Inverted Otsu (for dark text on light plate - standard Indian plates)
    preprocessed.append(otsu)

    # Inverted adaptive (sometimes helps)
    preprocessed.append(cv2.bitwise_not(adaptive))

    return preprocessed


def _correct_ocr_mistakes(raw_text):
    """
    Fix common OCR confusions in Indian plate format.
    Position-aware: slots 1-2 letters, 3-4 digits, 5-6 letters, 7-10 digits.
    """
    s = "".join(c for c in raw_text.upper() if c.isalnum())
    if len(s) < 5:
        return s

    # Expected layout: LLDDLLDDDD (2 letters, 2 digits, 1-2 letters, 1-4 digits)
    result = []
    i = 0

    def to_letter(c):
        """Convert digit misread as letter (0→O, 1→I)."""
        return _DIGIT_TO_LETTER.get(c, c)[0] if c in _DIGIT_TO_LETTER else c

    def to_digit(c):
        """Convert letter misread as digit (O→0, I→1)."""
        return _LETTER_TO_DIGIT.get(c, c) if c in _LETTER_TO_DIGIT else c

    # State (2 letters)
    for _ in range(2):
        if i < len(s):
            c = s[i]
            result.append(to_letter(c) if c.isdigit() else (c if c.isalpha() else ""))
            i += 1
    # District (2 digits)
    for _ in range(2):
        if i < len(s):
            c = s[i]
            result.append(to_digit(c) if c.isalpha() else (c if c.isdigit() else "0"))
            i += 1
    # Series (1-2 letters)
    for _ in range(2):
        if i < len(s):
            c = s[i]
            result.append(to_letter(c) if c.isdigit() else (c if c.isalpha() else ""))
            i += 1
    # Number (remaining digits)
    while i < len(s):
        c = s[i]
        result.append(to_digit(c) if c.isalpha() else (c if c.isdigit() else ""))
        i += 1

    return "".join(result)


def _extract_indian_plate(raw_text):
    """
    Extract or validate Indian plate format using regex.
    Returns formatted string (e.g. AP01AB1234) or empty if no match.
    """
    corrected = _correct_ocr_mistakes(raw_text)
    # Remove spaces for regex
    condensed = "".join(c for c in corrected.upper() if c.isalnum())

    match = _INDIAN_PLATE_PATTERN.search(condensed)
    if match:
        state, district, series, num = match.groups()
        return f"{state.upper()}{district}{series.upper()}{num}"

    # Try to build from condensed if length is right (8-10 chars)
    if 8 <= len(condensed) <= 10 and condensed[:2].isalpha() and condensed[2:4].isdigit():
        state = condensed[:2].upper()
        district = condensed[2:4]
        rest = condensed[4:]
        letters = "".join(c for c in rest if c.isalpha())[:2]
        digits = "".join(c for c in rest if c.isdigit())[:4]
        if letters and digits:
            return f"{state}{district}{letters}{digits}"

    return ""


def read_plate(plate_img):
    """
    Extract license plate text from a cropped plate image.
    Optimized for Indian formats (AP, TS, KA: XX-NN-XX-NNNN).

    Steps:
    1. Preprocess (grayscale, CLAHE, denoise, thresholding)
    2. Run EasyOCR on multiple preprocessed variants
    3. Validate/filter using Indian plate regex
    4. Apply OCR mistake corrections

    Args:
        plate_img: BGR image (numpy array) of the cropped license plate.

    Returns:
        Detected plate text string (e.g. AP01AB1234), or empty string if none found.
    """
    if plate_img is None or plate_img.size == 0:
        return ""

    preprocessed_images = _preprocess_for_indian_plate(plate_img)
    if not preprocessed_images:
        return ""

    reader = _get_reader()
    best_result = ""
    best_conf = 0.0

    for proc_img in preprocessed_images:
        results = reader.readtext(proc_img)
        for _, text, conf in results:
            if conf < 0.1:
                continue
            raw = "".join(c for c in text if c.isalnum()).upper()
            if len(raw) < 6:
                continue
            extracted = _extract_indian_plate(raw)
            if extracted and conf > best_conf:
                best_result = extracted
                best_conf = conf
            # Also try corrected raw if regex didn't match
            if not extracted:
                corrected = _correct_ocr_mistakes(raw)
                extracted = _extract_indian_plate(corrected)
                if extracted and conf > best_conf:
                    best_result = extracted
                    best_conf = conf

    if best_result:
        return best_result

    # Fallback: return highest-confidence raw text (cleaned) if no regex match
    all_texts = []
    for proc_img in preprocessed_images[:2]:  # Only Otsu and adaptive
        results = reader.readtext(proc_img)
        for _, text, conf in results:
            if conf > 0.1:
                raw = "".join(c for c in text if c.isalnum()).upper()
                if len(raw) >= 6:
                    all_texts.append((_correct_ocr_mistakes(raw), conf))
    if all_texts:
        best = max(all_texts, key=lambda x: x[1])
        return best[0][:10]  # Cap length for display

    return ""