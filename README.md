# Real-time-Sign-Language-Recognition-Using-OpenCV-and-Deep-Learning
Employed OpenCV for video processing and hand-detection in real-time. Utilized Keras with TensorFlow backend to train a deep learning model for sign language classification on a dataset of 2900 300x300 images. This model offers instantaneous, precise sign language recognition. This repository contains a Python script that captures video from a webcam, detects hands in the video frames, classifies the detected hand gesture, and overlays the identified gesture onto the original video frame.

# Project Background
Hand gesture recognition has a wide array of applications, from sign language interpretation to human-computer interaction. In this project, we utilize computer vision and machine learning techniques to detect and classify 26 different hand gestures that correspond to the 26 letters of the English alphabet.

![The-26-hand-signs-of-the-ASL-Language](https://github.com/DevRishabhSinha/Real-time-Sign-Language-Recognition-Using-OpenCV-and-Deep-Learning/assets/127776575/f18d3cf6-7536-48f7-ad9d-cd3e2b785f9e)

# Underlying Technologies
The Python script relies on the following libraries:

1. OpenCV (cv2): An open-source computer vision library which includes several hundreds of computer vision algorithms.
2. cvzone: A computer vision package that simplifies common tasks, providing a higher level API to OpenCV.
3. NumPy: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
4. math: A Python module that provides mathematical functions.
5. TensorFlow (via cvzone.ClassificationModule): An end-to-end open-source platform for machine learning that provides comprehensive, flexible tools for developing and deploying machine learning models.
6. Keras (via cvzone.ClassificationModule): A high-level neural networks API, written in Python and capable of running on top of TensorFlow, Theano, or CNTK.

# Overview of the Code
The script begins by capturing video from the webcam using OpenCV's VideoCapture. It uses cvzone's HandDetector to detect hands in the video frames, with the maximum number of detected hands set to one.

The script initializes a classifier using a saved Keras model and label file from cvzone's ClassificationModule. These pre-trained models were trained on a dataset of images of hand gestures.

For each frame of the video, the script:

1. Detects hands and annotates the frame with bounding boxes and skeleton lines.
2. If a hand is detected, the script crops the image around the hand and resizes it to a fixed size of 300x300 pixels. The hand image is centered within a white canvas to maintain its aspect ratio.
3. It then uses the pre-trained Keras model to predict the hand gesture.
4. The predicted gesture is then overlaid on the original video frame.
5. This frame is displayed to the user with the hand detection and prediction.


# Model Architecture
The classifier used in this script is a convolutional neural network (CNN) created in Keras, a popular machine learning library. CNNs are a class of deep learning models that are highly effective for image classification tasks due to their ability to process visual data and detect hierarchical patterns.

The model architecture and training process aren't detailed in the script, as a pre-trained model is loaded from a file. But typically, CNNs for image classification tasks are composed of several layers including convolutional layers, pooling layers, and fully connected layers.

The network learns to detect features (like edges, corners, and other textures) from the images during the training process, where it adjusts its internal parameters to minimize a loss function using a method called backpropagation.

The trained model is then capable of predicting the class (or in this case, the hand gesture) of a new, unseen image.

# Running the Code
To run this script, clone the repository and ensure that you have the necessary dependencies installed (Python 3, OpenCV, cvzone, NumPy). Run the Python script. Ensure that your webcam is enabled and clear.
