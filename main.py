import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import cv2
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


model = YOLO("yolov8n.pt")


# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
success = model.export(format="onnx")  # export the model to ONNX format

results = model.predict(source="0", show=True)
results