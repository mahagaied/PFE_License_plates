import cv2
from pytesseract import pytesseract

import utilities

if __name__ == '__main__':
    path = "../license_images/2.png"
    image = cv2.imread(path)
    utilities.display_image(image)

    gray = utilities.RGB_to_GrayScale(image)
    utilities.display_image(gray)

    # detect contours by polygones : tested on 1,2,3,4,5,6

    rect_contours = utilities.detect_polygone(gray)
    cv2.drawContours(image, [rect_contours], 0, (0, 255, 0), 3)
    cv2.imshow('Contours', image)
    cv2.waitKey(0)

    # crop
    plate = utilities.crop_img(rect_contours, image)
    coord = cv2.boundingRect(rect_contours)

    text = utilities.recognition(plate, image, coord)
    cv2.imshow('text', text)
    cv2.waitKey(0)

    cv2.destroyAllWindows()








    # sans thresh: 1,5,6 NO, 3,4 YES
    # with thresh 170 : 3,4,6 NO, 1,5 YES
    # with thresh 150: 1,6 NO, 3,4,5 YES ; 
    # with thresh 150 and with config: all YES, 1,3 best

    # try detecting with letters for pic2, recognition, yolo
    # reduire taille images pour moins de pixels
    # check if yolo has training on chiffres et lettres

    ##############################################
    """ FIRST METHOD: noise reduce + canny + find contours+ assume 4 points contours
       problem : detects the glass cleaner as a 4 point polygone
       solution : verify the length of parallel sides to be almost equal

       2nd problem : detects another rectangle (the lights) and not the plate
       2nd solution : another method using findpossibleChars to detect rectangles with characters in it  
                     
       3rd problem: 2nd solution didnt work with image 2: can detect the characters but cant detect the rectangle plate, 
                    only detects some of it as 9 point polygone :(
       3rd solution: try HAAR CASCADE instead 
       
       4th problem: HAAR CASCADE sometimes detects more than one rectangle
       4th solution: add a noise filter, but even it still detects more than one its okay as long as one of them is correct, 
                     we can eliminate them when we are do the recognition 
       5th problem: YOLO google collab GPU limited
       5th solution: use more than 1 account and backup drive the weights
       
       TO DO: fix pic 2 with chars detection 
       """
