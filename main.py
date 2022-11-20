import cv2
import utilities


if __name__ == '__main__':
    path = "license_images/3.png"
    image = cv2.imread(path)
    utilities.display_image(image)

    gray = utilities.RGB_to_GrayScale(image)
    utilities.display_image(gray)

    """ FIRST METHOD: noise reduce + canny + find contours+ assume 4 points contours
    problem : detects the glass cleaner as a 4 point polygone
    solution : verify the length of parallel sides to be almost equal
    
    2nd problem : detects another rectangle that is the lights and not the plate
    2nd solution : another method using findpossibleCharsinScene to detect rectangles with characters in it
    """

    rect_contours = utilities.detect_polygone(gray)

    cv2.drawContours(image, [rect_contours], 0, (0, 255, 0), 3)
    cv2.imshow('Contours', image)
    cv2.waitKey(0)

