"""
This script captures video from the webcam and detects hands in the frames.
Once hands are detected, it crops the image around the hand, 
resizes it to a fixed size, and uses a trained classifier to predict the hand gesture. 
The gesture is then overlayed on the original frame.
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Capturing video from the webcam
cap = cv2.VideoCapture(0)
# Using HandDetector to detect a single hand in the frames
detector = HandDetector(maxHands=1)
# Initializing a classifier from saved model and labels
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

"""
We use an offset to extend the area cropped around the hand to ensure the hand is fully included.
The image size is the standard size we will resize the cropped image to.
"""
offset = 20
imgSize = 300

# Define the directory where we will store the cropped images
folder = "Data/C"

# Initializing a counter to count the number of saved images
counter = 0

"""
We create a list of labels corresponding to the gestures the model can predict.
This list will be used to print the predicted gesture on the output frames.
"""
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

"""
Main loop to continuously capture video frames, detect and classify hand gestures, 
and display the result on the original frames.
"""
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    """
    If any hands are detected, we crop and resize the image around the hand 
    based on the hand's aspect ratio. 
    We then use the classifier to predict the hand gesture.
    """
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        """
        We draw a rectangle around the detected hand and overlay the predicted gesture 
        on the original frame.
        """
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # The cropped hand image and the white image with the hand at the center are shown.
        cv2.imshow("Cropped Hand Image", imgCrop)
        cv2.imshow("Hand-centered White Image", imgWhite)

    # The final output image is displayed.
    cv2.imshow("Output Image", imgOutput)

    # The frame is refreshed every 1 millisecond.
    cv2.waitKey(1)
