from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


from torchvision.models import resnet50


from ultralytics import YOLO



# CFG = '/home/white/PycharmProjects/ultrayltics-main/ultralytics/models/v8/yolov8x-dpe.yaml'
CFG = '//home/white/sharedirs/论文模型/YOLOv8/ultrayltics-main/ultralytics/cfg/models/v8/yolov8x-SPPFImproveAndFAM.yaml'
# SOURCE = ROOT / 'assets/bus.jpg'
SOURCE = '/home/white/sharedirs/图片/images__5.jpg'


def test_model_forward():
    model = YOLO(CFG)
    model(SOURCE)
