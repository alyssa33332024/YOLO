# Ultralytics YOLO 🚀, AGPL-3.0 license


from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor


from ultralytics import RTDETR, YOLO
from ultralytics.data.build import load_inference_source
from ultralytics.utils import LINUX,ONLINE,ROOT,SETTINGS

CFG = '/mnt/c/Users/alyss/Desktop/yolo/ultralytics-main/ultralytics/cfg/models/v8/forest.yaml'
source = ROOT / '/mnt/c/Users/alyss/Desktop/yolo/ultralytics-main/ultralytics/assets/bus.jpg'
def test_model_forward():
    model = YOLO(CFG)
    model(source)  