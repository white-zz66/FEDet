from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


from torchvision.models import resnet50


from ultralytics import YOLO



# CFG = '/home/white/PycharmProjects/ultralytics-main/ultralytics/models/v8/yolov8x-dpe.yaml'
CFG = '/ultralytics/cfg/models/v8/yolov8x-SPPFImproveAndFAM.yaml'
# SOURCE = ROOT / 'assets/bus.jpg'
SOURCE = '/coco/images/val2017/KK-c7nX9lhV8e__0.jpg'


def test_model_forward():
    model = YOLO(CFG)
    model(SOURCE)
