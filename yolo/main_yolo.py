import cv2
import numpy as np
import utilities_yolo


if __name__ == '__main__':
    # YOLO init
    path_weights = 'lapi.weights'
    path_cfg = 'darknet-yolov3.cfg'
    path_classes = 'classestxt.txt'

    net = cv2.dnn.readNet(path_weights, path_cfg)
    classes = []
    with open(path_classes, 'r') as f:
        classes = f.read().splitlines()


    # Detect
    # path = "license_images/8.png"
    path = "Data_labels/3.png"
    coord,lp_image = utilities_yolo.detect(path, net, classes)

    cv2.imshow("detect", lp_image)

    # crop LP
    img = cv2.imread(path)
    LP = utilities_yolo.crop_plate(img, coord)

    cv2.imshow("plate", LP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
