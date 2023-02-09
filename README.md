# PFE_license_plates

In this project we used [OpenCV](https://opencv.org), [HaarCascade](https://www.researchgate.net/publication/3940582_Rapid_Object_Detection_using_a_Boosted_Cascade_of_Simple_Features), [EAST](https://www.researchgate.net/publication/316015737_EAST_An_Efficient_and_Accurate_Scene_Text_Detector) and [YOLOv3](https://pjreddie.com) to detect license plates, and we used Pytesseract for the recognition.


## Data 

- A [data source](https://platesmania.com) for license plates from different countries. 

* A [database](https://www.kaggle.com/datasets/elysian01/car-number-plate-detection) used for YOLO model training.

+ An image labeling tool [labelImg](https://github.com/mahagaied/labelImg.git).


## Detection Algorithms

1. **OpenCV: Contours detection**

- Convert the image from RGB to Grayscale.
* Detect contours in the image.
+ Select contours with four key points asuming it is forming a rectangle.

2. **HaarCascade**

- Convert the image from RGB to Grayscale.
* Load a pretrained Adaboost cascade classifier.
+ Extract the coordinates detected.

3. **EAST: Efficient Accurate Scene Text Detector**

- Load a pretrained FCN EAST model.
* Resize the image to a multiple of 32.
+ Apply non max suppression to select a bounding box.

4. **YOLOv3**

- Collect lience plate images, and label the data using labelImg. Or use a pre-labeled Database.
* Use a [training python script](https://colab.research.google.com/drive/1tZCu5A93KJAmH5xhO25sHArBHBxAQVFx#scrollTo=tzh3C8ndEq7w) running on Google Colab to train the model.
+ Load the trained model to extract the coordinates of the plates.

## Recognition

- Extract and crop the plate image using the coordinates detected.
* Extract the black pixels only using threshholding techniques.
+ Use morphology to clean the mask and remove blemishes.
- Use Pytesseract to recognise the text.
* Display the plate number on the image.

![License Plate](licence_plate_result.png?raw=true)



