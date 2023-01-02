import cv2
import pytesseract
import utilities_char
pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"


if __name__ == '__main__':
    # init Frozen east
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # detect
    path = '../license_images/2.png'
    img = cv2.imread(path)

    filter = cv2.bilateralFilter(img, 70,60,100)
    #filter = cv2.GaussianBlur(filter, (9, 9), 1)
    filter = cv2.medianBlur(filter, 5)

    image = cv2.resize(filter, (640, 320), interpolation=cv2.INTER_AREA)

    cv2.imshow("image", image)
    cv2.waitKey(0)

    char_Detected = utilities_char.char_detector(image, net)
    cv2.imshow("Detection", char_Detected)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    """
    east algo uses multple of 32 so we resize
    """