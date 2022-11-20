import cv2
import imutils


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

    (T, thresh) = cv2.threshold(grayscale, th, max_th, cv2.THRESH_BINARY_INV)
    return thresh

# function that gives x, y for each point i from an numpy array
# @input : np array with all the points, index of the point
# @output : xi, yi
def xy_coordinates(poly_points, i):

    len_i = len(str(poly_points[i]))
    x, y = str(poly_points[i])[2:len_i - 2].split()

    return x, y

# function that checks if 2 parallel sides are almost equal
# @input : coordinates
# @output : boolean
def almost_equals(coord):
    y_left = abs(coord[0][1]-coord[3][1]) # longeur verticale
    y_right = abs(coord[1][1]-coord[2][1])
    if y_right - 5 <= y_left <= y_right + 5 :
        return True
    return False


# function that does the preprocess : noise reduce + noise reduce + canny + find contours
# @input : gray image
# @output : contours
def pre_process_contours(gray):
    # noise reduce
    gray_filter = cv2.bilateralFilter(gray, 40, 40, 30)  # 35, 25, 25 / 100, 70, 70)
    # gray_filter = cv2.bilateralFilter(gray, 100, 65, 65)
    # display_image(gray_filter)

    # canny : all contours
    gray_contours = cv2.Canny(gray_filter, 30, 200)
    # display_image(gray_contours)

    # find contours
    points = cv2.findContours(gray_contours.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # we want 4 points
    contours = imutils.grab_contours(points)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    return contours


# function for THE FIRST METHOD: preprocess + detect 4 point polygones + choose the one with almost equal sides
# @input : gray image
# @output : rectangle contour
def detect_polygone(gray):
    # noise reduce + canny + find contours
    contours = pre_process_contours(gray)

    rect_contours = None  # plate contour we are looking for

    for contour in contours:
        # change 10 for better results
        poly_points = cv2.approxPolyDP(contour, 10, True)  # approximate polygone, 10 fine enough for a rectangle
        print(len(poly_points))
        coord = []  # coordinates
        if len(poly_points) == 4:  # a rectangle has 4 key points
            for i in range(4):
                x, y = xy_coordinates(poly_points, i)
                coord.append([int(x), int(y)])
                print(coord)
            # check if parallel sides are almost equal
            if almost_equals(coord):
                rect_contours = poly_points
                break

    return rect_contours

