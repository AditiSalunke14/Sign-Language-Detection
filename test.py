import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)

# Update these paths to your actual model and labels file
model_path = r"C:\Users\salun\Downloads\converted_keras\keras_model.h5"
labels_path = r"C:\Users\salun\Downloads\converted_keras\labels.txt"

# Load classifier
if os.path.exists(model_path) and os.path.exists(labels_path):
    classifier = Classifier(model_path, labels_path)
else:
    raise FileNotFoundError("Model or labels file not found.")

offset = 20
imgSize = 300
labels = ["Hello", "Yes"]  # Make sure this matches your actual label file

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from webcam.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Check crop bounds
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue  # Skip if crop is empty

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hGap + hCal, :] = imgResize

        # Get prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        if index < len(labels):
            label = labels[index]
        else:
            label = "Unknown"

        # Display results
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                      (x - offset + 400, y - offset - 10), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label, (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Show cropped and white image
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Sign Detection', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
