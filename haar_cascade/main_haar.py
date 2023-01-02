import cv2
import utilities_haar
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"

if __name__ == '__main__':
    path = "../license_images/4.png"
    image = cv2.imread(path)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    #gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)

    #filter
    gray_filter = cv2.bilateralFilter(gray, 40, 40, 30)
    cv2.imshow('gray filter', gray_filter)
    cv2.waitKey(0)

    # detect
    image,plate= utilities_haar.detect_haar_cascade(gray_filter, image)
    cv2.imshow("detection", image)
    cv2.waitKey(0)


    #recognition
    (T, thresh_plate) = cv2.threshold(plate, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("threshPlate", thresh_plate)
    print(f"Plate Number : {pytesseract.image_to_string(thresh_plate,config='--psm 11')}")
    cv2.waitKey(0)