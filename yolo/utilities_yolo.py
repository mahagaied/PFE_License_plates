import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"


def detection(img_path, net, classes):
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
    return coordinates, lp_image


def recognition(plate, detected, coord):
    height, width, _ = plate.shape
    #grayscale = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    #(T, thresh) = cv2.threshold(grayscale, 120, 255, cv2.THRESH_BINARY_INV)
    #blur = cv2.GaussianBlur(thresh, (5, 5), 0)

    imgFloat = plate.astype(np.float) / 255.

    # Calculate channel K:
    kChannel = 1 - np.max(imgFloat, axis=2)

    # Convert back to uint 8:
    kChannel = (255 * kChannel).astype(np.uint8)

    # Threshold image:
    binaryThresh = 150
    _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("binary", binaryImage)
    cv2.waitKey(0)
    #binaryImage = cv2.GaussianBlur(binaryImage, (3, 3), 0)

    # Use a little bit of morphology to clean the mask:
    # Set kernel (structuring element) size:
    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 0
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations,
                                   cv2.BORDER_REFLECT101)

    cv2.imshow("binaryImage [closed]", binaryImage)
    cv2.waitKey(0)



    number_plate = pytesseract.image_to_string(binaryImage, config='-l eng --oem 1 --psm 8')
    alnum_plate = ""
    for l in number_plate:
        if l.isalnum() :
            alnum_plate=alnum_plate+l
    print(f"Plate Number : {number_plate}")
    print(f"Plate Number : {alnum_plate}")
    print(coord)
    img = cv2.putText(detected, alnum_plate, (coord[0], coord[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                      cv2.LINE_AA)

    boxes = pytesseract.image_to_boxes(plate)
    for b in boxes.splitlines():
        b = b.split(' ')
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(binaryImage, (x, height - y), (w, height - h), (0, 0, 255), 1)
    cv2.imshow('boxes', binaryImage )
    cv2.waitKey(0)

    return img


def crop_plate(img, coordinates):
    x, y, w, h = coordinates
    crop = img[y:y + h, x:x + w]
    return crop
