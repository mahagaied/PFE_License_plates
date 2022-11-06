import cv2


# function that loads and displays an image
# @input: image
# @output:
def display_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)


# function that converts RGB image to grayscale
# @input: RGB image
# @output : Gray image
def RGB_to_GrayScale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale


# function that applies binary threshold on an image (seuillage)
# @input : image, threshold value, maxvalue
# @output : image after applying thresholds (black and white)
def threshold(image, th, max_th):
    # convert the image to grayscale
    grayscale = RGB_to_GrayScale(image)

    (T, thresh) = cv2.threshold(grayscale, th, max_th, cv2.THRESH_BINARY)
    return thresh