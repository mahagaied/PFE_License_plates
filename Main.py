import cv2
import yolo.utilities_yolo
import char_detection.utilities_char

if __name__ == '__main__':


    path = "license_images/8.png"
    #path = "yolo/Data_labels/100.png"
    try:
        # YOLO init
        path_weights = 'yolo\yolov3_training_final.weights'
        path_cfg = 'yolo\yolov3_testing.cfg'
        path_classes = 'yolo\classes.txt'

        net = cv2.dnn.readNet(path_weights, path_cfg)
        classes = []
        with open(path_classes, 'r') as f:
            classes = f.read().splitlines()

        # detection
        img = cv2.imread(path)
        cv2.imshow("image", img)
        cv2.waitKey(0)

        coord, lp_image = yolo.utilities_yolo.detection(path, net, classes)

        cv2.imshow("detect", lp_image)

        # crop LP
        img = cv2.imread(path)
        LP = yolo.utilities_yolo.crop_plate(img, coord)

        cv2.imshow("plate", LP)
        cv2.waitKey(0)

        # recognition
        text = yolo.utilities_yolo.recognition(LP, lp_image, coord)
        cv2.imshow("recognition", text)
        cv2.waitKey(0)

    except:
        # EAST init

        net = cv2.dnn.readNet('char_detection\\frozen_east_text_detection.pb')

        img = cv2.imread(path)

        # detect
        filter = cv2.bilateralFilter(img, 70, 60, 100)

        blur = cv2.medianBlur(filter, 5)
        image = cv2.resize(blur, (640, 320), interpolation=cv2.INTER_AREA)

        cv2.imshow("image", image)
        cv2.waitKey(0)

        lp_image, coord = char_detection.utilities_char.char_detect(image, net)

        # crop
        img = cv2.imread(path)
        image = cv2.resize(filter, (640, 320), interpolation=cv2.INTER_AREA)
        LP = char_detection.utilities_char.crop_plate(image, coord)
        cv2.imshow("plate", LP)
        cv2.waitKey(0)

        # recognition
        text = char_detection.utilities_char.char_recognition(LP, lp_image,coord)
        cv2.imshow("recognition", text)
        cv2.waitKey(0)





    cv2.destroyAllWindows()

    """ Free and open source libraries for deep neural networks """
# 14 feb apre midi
# refaire la recognnaince avec limage crop
# dedaline le 9 pour rapport
#