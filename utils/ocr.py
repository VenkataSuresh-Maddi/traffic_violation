import easyocr
import cv2

reader = easyocr.Reader(['en'], gpu=False)

def read_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)

    texts = []
    for _, text, conf in results:
        if conf > 0.3:
            texts.append(text)

    plate_text = "".join(texts).replace(" ", "").upper()
    return plate_text