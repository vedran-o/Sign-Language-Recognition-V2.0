import os
import copy
import cv2
import tensorflow
import keras
from keras.models import load_model
import numpy as np


def getPredictedClass(result):
    if result == 1:
        return "C"
    elif result == 2:
        return "4"
    elif result == 3:
        return "Thumbs Down"
    elif result == 4:
        return "2"
    elif result == 5:
        return "E"
    elif result == 6:
        return "I"
    elif result == 7:
        return "J"
    elif result == 8:
        return "Jedan"
    elif result == 9:
        return "L"
    elif result == 10:
        return "Thumbs Up"
    elif result == 11:
        return "O"
    elif result == 12:
        return "5"
    elif result == 13:
        return "3"
    elif result == 14:
        return "U"
    elif result == 15:
        return "A"


classifier = load_model("Moje_geste_CNN.h5")

cap = cv2.VideoCapture(0)  # Defaultni camera module

while True:

    ## Čitanje videa

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    ## Definiranje regije interesa ( ROI )

    ROI = frame[100:400, 320:620]
    cv2.imshow('Regija interesa', ROI)
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ROI = cv2.resize(ROI, (28, 28), interpolation=cv2.INTER_AREA)

    cv2.imshow('Skalirana regija interesa', ROI)
    kopija_frame = frame.copy()
    cv2.rectangle(kopija_frame, (320, 100), (620, 420), (255, 0, 0), 5)

    ROI = ROI.reshape(1, 28, 28, 1)
    ROI = ROI / 255

    predict_x = classifier.predict(ROI)
    result = np.argmax(predict_x, axis=1)
    predictedClass = getPredictedClass(result)

    cv2.putText(kopija_frame, str(predictedClass), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('Frame', kopija_frame)

    if cv2.waitKey(1) == 13:
        break

cap.relese()
cv2.destroyAllWindows()
