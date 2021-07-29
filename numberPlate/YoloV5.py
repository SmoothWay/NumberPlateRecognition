import os
#import time
import sys
import pathlib
import torch
import numpy as np

NUMBER_PLATE_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "../")
YOLOV5_DIR       = os.environ.get("YOLOV5_DIR", os.path.join(NUMBER_PLATE_DIR, 'yolov5'))
sys.path.append(YOLOV5_DIR)

# yolo
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device, load_classifier, time_synchronized

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Base')))


class Detector:
    @classmethod
    def get_classname(cls):
        return cls.__name__

    def loadModel(self,
                 weights=(NUMBER_PLATE_DIR + "/numberPlate/Base/models/Detector/yolov5x/yolov5s-2021-03-29.pt"),
                 device='0'):
        device = select_device(device)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())
        half = device.type != 'cpu'
        if half:
            model.half()  # to FP16
        
        self.model  = model
        self.device = device
        self.half   = half

    def detect_bbox(self, img, img_size=640, stride=32, min_accuracy=0.5):

        # normalize
        img_shape = img.shape
        img = letterbox(img, img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred)
        res = []
        for i, det in enumerate(pred): 
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_shape).round()
                res.append(det.cpu().detach().numpy())
        if len(res):
            return [[x1, y1, x2, y2, acc, b] for x1, y1, x2, y2, acc, b  in res[0] if acc > min_accuracy]
        else:
            return []
