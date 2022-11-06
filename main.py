import cv2
import utilities

if __name__ == '__main__':
    path = "license_images/2.png"
    image = cv2.imread(path)
    utilities.display_image(image)

    grayscale = utilities.RGB_to_GrayScale(image)
    utilities.display_image(grayscale)

    thresh = utilities.threshold(image, 130, 255)
    utilities.display_image(thresh)