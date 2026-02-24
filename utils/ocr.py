import easyocr
import cv2

reader = easyocr.Reader(['en'], gpu=False)

def read_plate(plate_img):
    if plate_img.size == 0:
        return ""
        
    # 1. Upscale the image 2x natively (3x combined with heavy blur washed out faint text)
    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # 3. Read directly on the crisp upscale without synthetic blurring
    results = reader.readtext(gray)

    texts = []
    for _, text, conf in results:
        # Keep threshold extremely low to capture faintly printed digits on the plate
        if conf > 0.05:
            # Filter out random symbols EasyOCR might hallucinate
            clean_text = ''.join(e for e in text if e.isalnum())
            if clean_text:
                texts.append(clean_text)

    # Join the texts and convert to uppercase
    plate_text = "".join(texts).upper()
    return plate_text