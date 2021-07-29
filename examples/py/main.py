import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import glob
import matplotlib.image as mpimg
import cv2
import copy
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

Dir = '../images/more/*.jpg'
imgs = [mpimg.imread(img_path) for img_path in glob.glob(Dir)]
for img in imgs:
    tBoxes = detector.detect_bbox(copy.deepcopy(img))
    tBoxes = tBoxes

    all_points = nPC.detect(img, tBoxes)
    all_points = [ps for ps in all_points if len(ps)]
    print(all_points)

     # cut
    toShowZones = [getCvZoneRGB(img, reshapePoints(rect, 1)) for rect in all_points]
    zones = convertCvZonesRGBtoBGR(toShowZones)
    for zone, points in zip(toShowZones, all_points):
        cv2.imshow('ZOne',zone)

    textArray = textDetector.predict(zones)
    print()
    print("Number Plate: " + str(textArray))
    # draw rect and 4 points
    for tBox, points in zip(tBoxes, all_points):
        cv2.rectangle(img,
                      (int(tBox[0]), int(tBox[1])),
                      (int(tBox[2]), int(tBox[3])),
                      (0,120,255),
                      3)
    cv2.imshow("Detected Image",img)
    k = cv2.waitKey(0)
