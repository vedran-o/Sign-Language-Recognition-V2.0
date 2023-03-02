import time

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")

offset = 20
imgSize = 300
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M", "N", "O", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

previous_predictions = []
ispisText = []
najcescaGesta = ''
precisionGesta = 0
freezProg = 0
sec = 0.0

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    rijec = ''.join(ispisText)

    cv2.putText(imgOutput, "TEXT: ", (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(imgOutput, str(rijec), (130, 60), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 255), 2)

    cv2.putText(imgOutput, "MOST C GEST: ", (20, 124), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(imgOutput, str(najcescaGesta), (266, 124), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 127, 255), 2)

    cv2.putText(imgOutput, "PRECISION: ", (20, 184), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(imgOutput, str(round(precisionGesta, 2)), (206, 184), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

    if freezProg != 0 and sec < 600:
        cv2.putText(imgOutput, "PRITISNUTA TIPA [SPACE]", (100, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
        freezProg = 0
        sec = sec + 0.1

    try:
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset-55), (x - offset+120, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, str(round(precisionGesta, 2)), (x, y -30), cv2.FONT_HERSHEY_COMPLEX, 1.3, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w+offset, y + h+offset), (255, 0, 255), 4)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 110), (x - offset + 90, y - offset - 110 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y -88), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)


            # Precision capture
            precisionGesta = prediction[index]

            # Add the current prediction to the list of previous predictions
            previous_predictions.append(labels[index])

            # Keep the list of previous predictions to a fixed length (for example, the last 10 frames)
            previous_predictions = previous_predictions[-40:]

            # Find the most common prediction in the list of previous predictions
            most_common_prediction = max(set(previous_predictions), key=previous_predictions.count)

            # Check if the current prediction matches the most common prediction
            if previous_predictions.count(labels[index]) >= 30:
                ispisText.append(labels[index])
                previous_predictions = []

            else:
                najcescaGesta = most_common_prediction

            print(ispisText)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    except cv2.error as e:
        print(f"OpenCV error: {e}")
        print("Retrieving new image from live feed")
        continue

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)

    if key == ord(' '):
        ispisText.append(' ')
        freezProg = 1
        sec = 0

    if key == ord('\b'):
        ispisText = ispisText[:-1]

    if key == 27:
        break