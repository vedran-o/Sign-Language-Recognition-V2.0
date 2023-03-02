import numpy as np
import math
import os
import cv2
import uuid
from cvzone.HandTrackingModule import HandDetector

offset = 20
imgSize = 300

## Funkcija za postavljanje direktroija

def Izradi_direktorij_geste(directory):
    global prva_gesta
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None
    else:
        prva_gesta


def Izradi_direktorij_train_test(link):
    global prva_gesta
    if not os.path.exists(link):
        os.makedirs(link)
        return None
    else:
        prva_gesta


def var_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


# Izgradnja dataseta pomoću OpenCV-a

i = 3
r = i
r = str(r)
ukupan_broj_slika = 0
ogranicenje_snimljenih_slika = 0

ogranicenje_snimljenih_slika_TRAIN = 0
ogranicenje_snimljenih_slika_TEST = 0

izlaz = ""
dialog = ""
prva_gesta = ""
resetiranje_dataseta = ""

temp_TRAIN = 0
temp_TEST = 0

glavniDIR = "./Dataset gestikulacije"
trainDS = "./Dataset gestikulacije/train"
testDS = "./Dataset gestikulacije/test"

print("Molim vas odredite velicinu TRAIN - DSa za sve klase: ")
ogranicenje_snimljenih_slika_TRAIN = int(input())

print("Molim vas odredite velicinu TEST - DSa za sve klase: ")
ogranicenje_snimljenih_slika_TEST = int(input())

Izradi_direktorij_train_test(trainDS)
Izradi_direktorij_train_test(testDS)

print("Unesite ime klase za koju zelite izraditi dataset: ")
dialog = input()


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

while (izlaz != "DA"):

    try:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        cv2.putText(img,
                    "BROJ PRITISAKA [ENTER]: " + str(i),
                    (120, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (140, 255, 140), 4)
        key = cv2.waitKey(1)

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
                ROI = imgWhite

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                ROI = imgWhite

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    except cv2.error as e:
        print(f"OpenCV error: {e}")
        print("Retrieving new image from live feed")
        continue

    if ((i == 0) and ((ogranicenje_snimljenih_slika_TRAIN == 0) and (ogranicenje_snimljenih_slika_TEST == 0))):
        print("Molim vas odredite velicinu TRAIN - DSa za sve klase: ")
        ogranicenje_snimljenih_slika_TRAIN = int(input())

        print("Molim vas odredite velicinu TEST - DSa za sve klase: ")
        ogranicenje_snimljenih_slika_TEST = int(input())

        print("Unesite ime klase za koju zelite izraditi dataset: ")
        dialog = input()
        prva_gesta = "./Dataset gestikulacije/train/" + dialog + "/"

        Izradi_direktorij_train_test(trainDS)
        Izradi_direktorij_train_test(testDS)

        if ((ogranicenje_snimljenih_slika_TRAIN != 0) and (ogranicenje_snimljenih_slika_TEST != 0)):
            i = 3
    if i == 3:
        ukupan_broj_slika = 0
        cv2.putText(img,
                    "Kada ste spremni pritisnite tipku [ENTER] kako bi zapoceli snimanje TRAIN slika za dataset " + dialog,
                    (100, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 255, 100), 4)

    if ((i == 4) and (ogranicenje_snimljenih_slika_TRAIN > ukupan_broj_slika)):
        ukupan_broj_slika = ukupan_broj_slika + 1
        cv2.putText(img, "Snimanje fotografija za TRAIN DATASET - Klasa: " + dialog, (100, 60),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(img, "Trenutni broj snimljenih slika: ", (150, 150), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 1)
        cv2.putText(img, str(ukupan_broj_slika), (750, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
        prva_gesta = "./Dataset gestikulacije/train/" + dialog + "/"
        Izradi_direktorij_geste(prva_gesta)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        if variance < 100:
            print('Zamucena slika')
        else:
            cv2.imwrite(prva_gesta + dialog + '_' + str(ukupan_broj_slika) + '_' + str(uuid.uuid4()) + ".jpg", ROI)

        if (ogranicenje_snimljenih_slika_TRAIN == ukupan_broj_slika):
            i = i + 1
            ukupan_broj_slika = 0

    if i == 5:
        ukupan_broj_slika = 0
        cv2.putText(img,
                    "Kada ste spremni pritisnite tipku [ENTER] kako bi započeli snimanje TEST slika za dataset " + dialog,
                    (100, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)

    if ((i == 6) and (ogranicenje_snimljenih_slika_TEST > ukupan_broj_slika)):
        ukupan_broj_slika = ukupan_broj_slika + 1
        cv2.putText(img, "Snimanje fotografija za TEST DATASET - Klasa: " + dialog, (100, 60),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(img, "Trenutni broj snimljenih slika: ", (150, 150), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 1)
        cv2.putText(img, str(ukupan_broj_slika), (750, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
        prva_gesta = "./Dataset gestikulacije/test/" + dialog + "/"
        Izradi_direktorij_geste(prva_gesta)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        if variance < 100:
            print('Zamucena slika')
        else:
            cv2.imwrite(prva_gesta + dialog + '_' + str(ukupan_broj_slika) + '_' + str(uuid.uuid4()) + ".jpg", ROI)

        if (ogranicenje_snimljenih_slika_TEST == ukupan_broj_slika):
            i = i + 1
            ukupan_broj_slika = 0

    if i == 7:
        cv2.putText(img, "Pritisni [ENTER] za izlaz", (100, 60),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('Frame', img)

    if i == 8:
        print("Želite li prekinuti ovaj program?")
        print("Jedino 'DA' će prekinuti program")
        izlaz = input()
        if izlaz == "DA":
            ukupan_broj_slika = 0
            i = 0
            ogranicenje_snimljenih_slika_TRAIN = 0
            ogranicenje_snimljenih_slika_TEST = 0

        else:
            temp_TRAIN = ogranicenje_snimljenih_slika_TRAIN
            temp_TEST = ogranicenje_snimljenih_slika_TEST
            print("\nDali zelite koristiti istu kolicniu ( broj slika ) za novi dataset?")
            print("Podsjetnik:\nTrain = {}\nTest = {}\n".format(ogranicenje_snimljenih_slika_TRAIN,
                                                                ogranicenje_snimljenih_slika_TEST))
            print("Molim Vas odgovorite pomocu DA ili NE: ")
            resetiranje_dataseta = input()
            if resetiranje_dataseta == "DA":
                ukupan_broj_slika = 0
                i = 3

                ogranicenje_snimljenih_slika_TRAIN = temp_TRAIN
                ogranicenje_snimljenih_slika_TEST = temp_TEST

                print("Unesite ime klase za koju zelite izraditi dataset: ")
                dialog = input()
                prva_gesta = "./Dataset gestikulacije/train/" + dialog + "/"

            else:
                ogranicenje_snimljenih_slika_TRAIN = 0
                ogranicenje_snimljenih_slika_TEST = 0

                ukupan_broj_slika = 0
                i = 0

    if cv2.waitKey(1) == 13:
        i = i + 1

cap.release()
cv2.destroyAllWindows()

