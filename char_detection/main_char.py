import cv2
import pytesseract
import utilities_char


pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"


if __name__ == '__main__':
    # load Frozen east model
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    path = '../license_images/10.png'
    img = cv2.imread(path)

    # detect
    filter = cv2.bilateralFilter(img, 70, 60, 100)
    # filter = cv2.GaussianBlur(filter, (9, 9), 1)
    blur = cv2.medianBlur(filter, 5)

    image = cv2.resize(blur, (640, 320), interpolation=cv2.INTER_AREA)

    cv2.imshow("image", image)
    cv2.waitKey(0)

    char_Detected,coord = utilities_char.char_detect(image, net)
    cv2.imshow("Detection", char_Detected)
    cv2.waitKey(0)

    # crop
    img = cv2.imread(path)
    image = cv2.resize(filter, (640, 320), interpolation=cv2.INTER_AREA)
    LP = utilities_char.crop_plate(image, coord)
    cv2.imshow("plate", LP)
    cv2.waitKey(0)

    # recognition
    text = utilities_char.char_recognition(LP, char_Detected,coord)
    cv2.imshow("recognition", text)
    cv2.waitKey(0)


    cv2.destroyAllWindows()

    """
    EAST paper: https://arxiv.org/abs/1704.03155
    Efficient Accurate Scene Text Detector
    
It is a fast and accurate scene text detection method and consists of two stages:
1. It uses a complete convolutional network (FCN) model to directly generate pixel-based word or text line predictions
2. After generating text predictions ( Rotate a rectangle or quad) and the output is sent to the non-maximum suppression 
to produce the final result.

3 parts:
 -Feature Extractor (PVANet) :This part can be any convolutional neural network with convolutional layer and 
pooled layer interleaving pre-trained on Imagenet data for examples PVANet, VGG16 and RESNET50. From this network, 
four levels of feature maps f1, f2, f3 and f4 can be obtained. Because we are extracting features it is called a 
-Feature Merging Branch: In this part, 
the feature maps obtained from the feature extractor are first fed to the unpooling layer to double its size, 
and then concatenated with the current feature map in each merging state. Next, the 1X1 convolution is used where 
the conv bottleneck reduced the number of channels and reduced the amount of calculation also, followed by a 3X3 
convolution to fuse information to produce the final output of every merging stage as shown in fig. 
-output layer
Feature Extractor. east for detection only, uses multple of 32 so we resize"""
