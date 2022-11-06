import cv2
import utilities
import imutils

if __name__ == '__main__':
    path = "license_images/5.png"
    image = cv2.imread(path)
    utilities.display_image(image)

    gray = utilities.RGB_to_GrayScale(image)
    utilities.display_image(gray)

    """#seuillage
    thresh = utilities.threshold(image, 130, 255)
    utilities.display_image(thresh)
    """
    # noise reduce
    gray_filter = cv2.bilateralFilter(gray, 40, 40, 30) # 35, 25, 25 / 100, 70, 70)
    utilities.display_image(gray_filter)

    # canny : all contours
    gray_contours = cv2.Canny(gray_filter, 30, 200)
    utilities.display_image(gray_contours)

    # find contours
    points = cv2.findContours(gray_contours.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # we want 4 points
    contours = imutils.grab_contours(points)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    rect_contours = None  # plate contour we are looking for
    for contour in contours:
        # change 10 for better results
        poly_points = cv2.approxPolyDP(contour, 10, True)  # approximate polygone, 10 fine enough for a rectangle
        print(len(poly_points))
        if len(poly_points) == 4:  # a rectangle has 4 key points
            rect_contours = poly_points
            break

    print(rect_contours)

    cv2.drawContours(image, [rect_contours], 0, (0, 255, 0), 3)
    cv2.imshow('Contours', image)
    cv2.waitKey(0)