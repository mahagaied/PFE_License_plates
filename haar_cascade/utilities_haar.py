import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"


def detect_haar_cascade(gray, image):
    detector = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
    detections = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

    # rectangle
    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    coord = x, y, w, h
    return image, coord


def crop_plate(img, coordinates):
    x, y, w, h = coordinates
    crop = img[y:y + h, x:x + w]
    return crop


def recognition_haar_cascade(plate, detected, coord):
    height, width, _ = plate.shape
    grayscale = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    (T, thresh) = cv2.threshold(grayscale, 150, 255, cv2.THRESH_BINARY_INV)
    #blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    cv2.imshow("threshPlate", thresh)
    cv2.waitKey(0)

    number_plate = pytesseract.image_to_string(thresh,
                                               config='-l eng --oem 1 --psm 8')
    print(f"Plate Number : {number_plate}")
    print(coord)
    img = cv2.putText(detected, number_plate, (coord[0], coord[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                      cv2.LINE_AA)

    boxes = pytesseract.image_to_boxes(grayscale)
    for b in boxes.splitlines():
        b = b.split(' ')
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(plate, (x, height - y), (w, height - h), (0, 0, 255), 1)
    cv2.imshow('boxes', plate)
    cv2.waitKey(0)

    return img
