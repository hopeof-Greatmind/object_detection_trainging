# Inference for ONNX model
import os
import cv2
cuda = True
w = "yolov7-tiny.onnx"
#img = cv2.imread('horses.jpg')  # image-based execute!

import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

providers = ['AzureExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)
