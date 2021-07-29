import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import cv2
from numberPlate.YoloV5 import Detector
from numberPlate.BBoxNpPoints import NpPointsCraft, getCvZoneRGB, convertCvZonesRGBtoBGR, reshapePoints
from numberPlate.TextRecognitor import TextRecognitor


NUMBER_PLATE_DIR = os.path.abspath('../../')
sys.path.append(NUMBER_PLATE_DIR)

detector = Detector()
detector.loadModel()

nPC = NpPointsCraft()
nPC.loadModel()

textDetector = TextRecognitor()
color = (0,120,255)
cap = cv2.VideoCapture(0)

while True:
    suc, img = cap.read()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        break
    imgResult = img.copy()

    tBoxes = detector.detect_bbox(img)
    tBoxes = tBoxes
    all_points = nPC.detect(img, tBoxes)
    all_points = [ps for ps in all_points if len(ps)]

     # cut
    toShowZones = [getCvZoneRGB(img, reshapePoints(rect, 1)) for rect in all_points]
    zones = convertCvZonesRGBtoBGR(toShowZones)
    for zone, points in zip(toShowZones, all_points):
        cv2.imshow('ZOne',zone)

    textArray = textDetector.predict(zones)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # print()
        # print("Number Plate: " + str(textArray))
        # draw rect and 4 points
    for tBox, points in zip(tBoxes, all_points):
        cv2.rectangle(imgResult,
                      (int(tBox[0]), int(tBox[1])),
                      (int(tBox[2]), int(tBox[3])),
                      color,
                      3)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            print()
            print("Number Plate: " + str(textArray))
            cv2.imwrite("Results/Scanned/NoPlate_" + str(textArray) + ".jpg", zone)
            #cv2.rectangle(imgResult, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgResult, str(textArray), (int(tBox[0])-30, int(tBox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
            cv2.imshow("Result", imgResult)
            cv2.waitKey(500)

    cv2.imshow("Number Recognition", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
