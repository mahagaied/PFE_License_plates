import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"


def detect_haar_cascade(gray, image):
    detector = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
    detections = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

    # rectangle
    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plate = gray[y:y + h, x:x + w]

    return image, plate
