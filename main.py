import cv2
import utilities


if __name__ == '__main__':

    path = "license_images/8.png"
    image = cv2.imread(path)
    utilities.display_image(image)

    gray = utilities.RGB_to_GrayScale(image)
    utilities.display_image(gray)

    """"# detect contours by polygones : tested on 1,2,3,4,5,6

    rect_contours = utilities.detect_polygone(gray)

    cv2.drawContours(image, [rect_contours], 0, (0, 255, 0), 3)
    cv2.imshow('Contours', image)
    cv2.waitKey(0)"""


    # HAAR CASCADE
    gray_filter = cv2.bilateralFilter(gray, 40, 40, 30)
    utilities.display_image(gray_filter)
    image = utilities.detect_haar_cascade(gray_filter, image)
    cv2.imshow("detection", image)
    cv2.waitKey(0)




##############################################
    """ FIRST METHOD: noise reduce + canny + find contours+ assume 4 points contours
       problem : detects the glass cleaner as a 4 point polygone
       solution : verify the length of parallel sides to be almost equal

       2nd problem : detects another rectangle (the lights) and not the plate
       2nd solution : another method using findpossibleCharsinScene to detect rectangles with characters in it  
                     
       3rd problem: 2nd solution didnt work with image 2: can detect the characters but cant detect the rectangle plate, 
                    only detects some of it as 9 point polygone :(
       3rd solution: try HAAR CASCADE instead 
       
       4th problem: HAAR CASCADE sometimes detects more than one rectangle
       4th solution: add a noise filter, but even it still dectects more than one its okay as long as one of them is correct, 
                     we can eliminate them when we are do the recognition 
       """

    # mutiscale :
    '''the scale factor :your model has a fixed size defined during training, which is visible in the XML. 
    This means that this size is detected in the image if present. 
    However, by rescaling the input image, you can resize a larger face to a smaller one, making it detectable by the algorithm.

    1.05 is a good possible value for this, which means you use a small step for resizing, i.e. reduce the size by 5%, 
    you increase the chance of a matching size with the model for detection is found. 
    This also means that the algorithm works slower since it is more thorough. 
    You may increase it to as much as 1.4 for faster detection, with the risk of missing some features altogether.
 
    minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.'''