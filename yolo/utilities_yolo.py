import cv2
import numpy as np


def detect( img_path, net, classes):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layer_names)
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                print(f"{x},{y},{w}, {h}")
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            # label = str(classes[class_ids[i]])
            # confidence = str(round(confidences[i], 2))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.putText(img, label + ' ' + confidence, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    lp_image = img
    coordinates = (x, y, w, h)
    return coordinates , lp_image


def crop_plate(img, coordinates):
    x, y, w, h = coordinates
    crop = img[y:y + h, x:x + w]
    return crop

