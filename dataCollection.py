# Import the necessary libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(0)

# Initialize a hand detector with max 1 hand detection at a time
hand_detector = HandDetector(maxHands=1)

# Initialize an offset for bounding box
offset = 20
# Define a size for cropped image
cropped_img_size = 300

# Define a folder to store the saved images
folder_path = "Data/Z"
# Initialize a counter to count the number of saved images
img_counter = 0

# The main loop to continuously capture video frames
while True:
    # Read the image from video capture
    success, image = video_capture.read()
    # Detect hands in the image
    hands, image = hand_detector.findHands(image)

    # If any hands are detected
    if hands:
        # Get the first hand detected
        hand = hands[0]
        # Extract the bounding box of the hand
        x, y, w, h = hand['bbox']

        # Create a white image with size defined
        white_image = np.ones((cropped_img_size, cropped_img_size, 3), np.uint8) * 255
        # Crop the hand image from the original image
        cropped_image = image[y - offset:y + h + offset, x - offset:x + w + offset]

        # Calculate aspect ratio of the hand bounding box
        aspect_ratio = h / w

        # Resize the cropped image based on its aspect ratio
        if aspect_ratio > 1:
            scaling_factor = cropped_img_size / h
            width_calculated = math.ceil(scaling_factor * w)
            resized_image = cv2.resize(cropped_image, (width_calculated, cropped_img_size))
            width_gap = math.ceil((cropped_img_size - width_calculated) / 2)
            white_image[:, width_gap:width_calculated + width_gap] = resized_image
        else:
            scaling_factor = cropped_img_size / w
            height_calculated = math.ceil(scaling_factor * h)
            resized_image = cv2.resize(cropped_image, (cropped_img_size, height_calculated))
            height_gap = math.ceil((cropped_img_size - height_calculated) / 2)
            white_image[height_gap:height_calculated + height_gap, :] = resized_image

        # Show the cropped and the resized images
        cv2.imshow("Cropped Image", cropped_image)
        cv2.imshow("Resized Image", white_image)

    # Show the original image
    cv2.imshow("Original Image", image)

    # Wait for the user to press 's' key to save an image
    key_pressed = cv2.waitKey(1)
    if key_pressed == ord("s"):
        img_counter += 1
        cv2.imwrite(f'{folder_path}/Image_{time.time()}.jpg',white_image)
        print(img_counter)
